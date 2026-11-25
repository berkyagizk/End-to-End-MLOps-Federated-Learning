import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import flwr as fl
from flwr.common import parameters_to_ndarrays
import mlflow
import numpy as np

if __package__:
    from .evaluation import COUNT_KEYS, aggregate_validation_metrics
    from .experiment import child_run_path, localhost_address, parameter_digest
    from .global_evaluation import GlobalValidationContext, RoundParameters
    from .preprocessing import FEATURE_NAMES
else:
    from evaluation import COUNT_KEYS, aggregate_validation_metrics
    from experiment import child_run_path, localhost_address, parameter_digest
    from global_evaluation import GlobalValidationContext, RoundParameters
    from preprocessing import FEATURE_NAMES


def validate_count(count, name):
    if type(count) is not int or not 0 < count <= 2**63 - 1:
        raise ValueError(f"{name} must be a positive signed 64-bit integer")


def validate_parameters(parameters, source):
    if parameters.tensor_type != "numpy.ndarray" or len(parameters.tensors) != 2:
        raise ValueError(f"{source}: expected two NumPy arrays for coefficients and intercept")
    try:
        arrays = parameters_to_ndarrays(parameters)
    except (ValueError, OSError, EOFError) as error:
        raise ValueError(f"{source}: invalid NumPy parameter data") from error
    for name, array, shape in zip(
        ("coefficients", "intercept"), arrays, ((1, len(FEATURE_NAMES)), (1,)), strict=True,
    ):
        if (not isinstance(array, np.ndarray) or array.shape != shape
                or array.dtype != np.float64 or not np.isfinite(array).all()):
            raise ValueError(f"{source}: {name} must be a finite float64 array of shape {shape}")


def weighted_accuracy(results):
    # Retain the legacy accuracy-only library interface. New controlled runs
    # require the full diagnostic schema and never use this fallback.
    diagnostic_keys = (*COUNT_KEYS, "threshold", "validation_examples", "precision", "recall",
                       "f1", "specificity", "balanced_accuracy", "average_precision")
    if any(isinstance(metrics, dict) and any(key in metrics or f"{key}_defined" in metrics for key in diagnostic_keys)
           for _, metrics in results):
        return aggregate_validation_metrics(results)
    terms = []
    total = 0
    for count, metrics in results:
        validate_count(count, "Validation num_examples")
        if not isinstance(metrics, dict):
            raise ValueError("Validation metrics must be a dictionary")
        accuracy = metrics.get("accuracy")
        if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)) or not 0 <= accuracy <= 1:
            raise ValueError("Each validation result must include numeric accuracy in [0, 1]")
        terms.append(count * accuracy)
        total += count
    return {"accuracy": math.fsum(terms) / total} if total else {}


class MLflowStrategy(fl.server.strategy.FedAvg):
    def enrich_validation_metrics(self, server_round, metrics):
        return metrics

    def requires_metric_persistence(self):
        return False

    def aggregate_fit(self, server_round, results, failures):
        if results and (self.accept_failures or not failures):
            for client, result in results:
                source = f"Fit round {server_round}, client {client.cid}"
                count = result.num_examples
                validate_count(count, f"{source} num_examples")
                if not isinstance(result.metrics, dict):
                    raise ValueError(f"{source}: metrics must be a dictionary")
                if "post_smote_examples" in result.metrics:
                    resampled_count = result.metrics["post_smote_examples"]
                    validate_count(resampled_count, f"{source} post_smote_examples")
                    if resampled_count % 2 or not count <= resampled_count < 2 * count:
                        raise ValueError(f"{source}: post_smote_examples is inconsistent with num_examples")
                validate_parameters(result.parameters, source)
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is not None:
            validate_parameters(parameters, f"Fit round {server_round} aggregate")
        return parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        if results and (self.accept_failures or not failures):
            for client, result in results:
                source = f"Validation round {server_round}, client {client.cid}"
                validate_count(result.num_examples, f"{source} num_examples")
                if (isinstance(result.loss, bool) or not isinstance(result.loss, (int, float))
                        or not 0 <= result.loss <= sys.float_info.max):
                    raise ValueError(f"{source}: loss must be finite and nonnegative")
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if loss is not None and not math.isfinite(loss):
            raise ValueError(f"Validation round {server_round}: aggregated loss must be finite")
        if metrics:
            metrics = self.enrich_validation_metrics(server_round, metrics)

        print(f"\n>>> Tur {server_round} değerlendiriliyor...")
        if metrics and "accuracy" in metrics:
            acc = metrics["accuracy"]
            print(f">>> Başarı Oranı: {acc} - MLflow'a yazılıyor...")
            try:
                with mlflow.start_run(run_name=f"Round_{server_round}"):
                    persistence = {"synchronous": True} if self.requires_metric_persistence() else {}
                    for key, value in metrics.items():
                        if isinstance(value, (int, float, bool)):
                            mlflow.log_metric(key, float(value), step=server_round, **persistence)
                        elif self.requires_metric_persistence() and isinstance(value, str):
                            mlflow.set_tag(key, value, **persistence)
                    mlflow.log_metric("loss", loss if loss is not None else 0.0, step=server_round, **persistence)
                    for _, result in results:
                        client_id = result.metrics.get("client_id")
                        if type(client_id) is int and "average_precision_defined" in result.metrics:
                            mlflow.log_metric(f"client_{client_id}_average_precision_defined",
                                              float(result.metrics["average_precision_defined"]), step=server_round, **persistence)
                            if result.metrics["average_precision_defined"]:
                                mlflow.log_metric(f"client_{client_id}_average_precision",
                                                  result.metrics["average_precision"], step=server_round, **persistence)
                print(">>> MLflow yazma işlemi başarıyla tamamlandı.")
            except Exception as e:
                if self.requires_metric_persistence():
                    raise RuntimeError(f"Required MLflow persistence failed for round {server_round}") from e
                print(f">>> MLflow HATASI: {e}")
        return loss, metrics

def build_strategy(num_clients):
    return MLflowStrategy(
        fraction_fit=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_accuracy,
    )


class ControlledStrategy(MLflowStrategy):
    """Require a complete round and record the parameters sent and received."""

    def __init__(self, num_clients, rounds, report, *, global_evaluator=None):
        super().__init__(
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=aggregate_validation_metrics,
            on_fit_config_fn=lambda round_number: {"server_round": round_number},
            on_evaluate_config_fn=lambda round_number: {"server_round": round_number},
            accept_failures=False,
        )
        self.client_ids = set(range(num_clients))
        self.rounds = rounds
        self.report = report
        self.round_records = []
        self.global_evaluator = global_evaluator
        self._pending_round = None
        self._configured_round = None
        self._completed_round = 0
        self._evaluation_failed = False

    def configure_fit(self, server_round, parameters, client_manager):
        if self.global_evaluator is not None:
            if (self._evaluation_failed or type(server_round) is not int
                    or server_round != self._completed_round + 1 or server_round > self.rounds
                    or self._configured_round is not None or self._pending_round is not None):
                raise RuntimeError("Missing, stale, or duplicated global evaluation round state")
            self._configured_round = server_round
        self.sent_sha256 = parameter_digest(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def checked_replies(self, stage, server_round, results, failures, expected_sha256):
        if any(not isinstance(result.metrics, dict) for _, result in results):
            raise ValueError(f"{stage} round {server_round}: metrics must be a dictionary")
        ids = [result.metrics.get("client_id") for _, result in results]
        if (failures or len(ids) != len(self.client_ids)
                or any(type(client_id) is not int for client_id in ids) or set(ids) != self.client_ids):
            raise RuntimeError(
                f"Incomplete {stage} round {server_round}: expected clients {sorted(self.client_ids)}, "
                f"received {ids}, {len(failures)} failures: {failures!r}"
            )
        replies = []
        for _, result in results:
            if (result.metrics.get("server_round") != server_round
                    or result.metrics.get("received_sha256") != expected_sha256):
                raise RuntimeError(f"{stage} round {server_round}: client reported unexpected global parameters")
            if self.global_evaluator is not None:
                if type(result.metrics.get("server_round")) is not int:
                    raise RuntimeError("Invalid client round identity")
                client_id = result.metrics["client_id"]
                expected = self.global_evaluator.client_identity(client_id)
                if any(result.metrics.get(key) != value for key, value in expected.items()):
                    raise RuntimeError(f"{stage} round {server_round}: validation context identity mismatch")
                if stage == "validation" and result.num_examples != self.global_evaluator.client_validation_counts[client_id]:
                    raise RuntimeError("Client validation size differs from the frozen context")
            replies.append({**result.metrics, "num_examples": result.num_examples})
        return sorted(replies, key=lambda reply: reply["client_id"])

    def aggregate_fit(self, server_round, results, failures):
        if self.global_evaluator is not None:
            if (self._evaluation_failed or self._configured_round != server_round
                    or server_round != self._completed_round + 1 or self._pending_round is not None):
                raise RuntimeError("Missing, stale, or duplicated fit aggregation state")
        replies = self.checked_replies("fit", server_round, results, failures, self.sent_sha256)
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        self.aggregate_sha256 = parameter_digest(parameters)
        if self.global_evaluator is not None:
            self._pending_round = RoundParameters.capture(server_round, parameters)
        self.round_records.append({
            "round": server_round, "sent_sha256": self.sent_sha256,
            "aggregate_sha256": self.aggregate_sha256, "fit": replies,
        })
        return parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        try:
            if self._evaluation_failed or type(server_round) is not int or server_round != self._completed_round + 1:
                raise RuntimeError("Missing, stale, duplicated, or failed round state")
            if self.global_evaluator is not None:
                self._check_pending_round(server_round)
            replies = self.checked_replies("validation", server_round, results, failures, self.aggregate_sha256)
            loss, metrics = super().aggregate_evaluate(server_round, results, failures)
            if not self.round_records or self.round_records[-1]["round"] != server_round:
                raise RuntimeError("Missing round report state")
            completed = {**self.round_records[-1], "validation": replies, "loss": loss,
                         "accuracy": metrics["accuracy"], "validation_metrics": {**metrics, "loss": loss}}
            candidate = [*self.round_records[:-1], completed]
            if ([row["round"] for row in candidate] != list(range(1, server_round + 1))
                    or any("validation" not in row for row in candidate)):
                raise RuntimeError("Round report must contain exactly the completed round prefix")
            self._publish_round_report(candidate)
        except Exception:
            self._evaluation_failed = True
            raise
        self.round_records = candidate
        self._completed_round = server_round
        self._pending_round = None
        self._configured_round = None
        print(f"Completed round {server_round}/{self.rounds} with all clients", flush=True)
        return loss, metrics

    def _validate_published_report(self, expected):
        if self.report.is_symlink():
            raise RuntimeError("Round report must not be a symbolic link")
        published = json.loads(self.report.read_text(encoding="utf-8"))
        if (json.dumps(published, sort_keys=True, allow_nan=False)
                != json.dumps(expected, sort_keys=True, allow_nan=False)):
            raise RuntimeError("Published round report differs from completed in-memory rounds")

    def _publish_round_report(self, candidate):
        if self._completed_round:
            self._validate_published_report(self.round_records[:-1])
        elif self.report.exists() or self.report.is_symlink():
            raise FileExistsError("Refusing to replace a pre-existing round report")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=self.report.parent,
                prefix=f".{self.report.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(candidate, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            # Same-directory replacement publishes only the complete, closed file.
            os.replace(temporary, self.report)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _check_pending_round(self, server_round):
        pending = self._pending_round
        if (self._evaluation_failed or type(server_round) is not int or pending is None
                or pending.round_number != server_round or self._configured_round != server_round
                or server_round != self._completed_round + 1
                or pending.digest != self.aggregate_sha256 or not self.round_records
                or self.round_records[-1]["round"] != server_round
                or self.round_records[-1]["aggregate_sha256"] != pending.digest):
            raise RuntimeError("Missing, stale, duplicated, or mismatched global evaluation round state")
        return pending.packet()

    def enrich_validation_metrics(self, server_round, metrics):
        if self.global_evaluator is None:
            return metrics
        packet = self._check_pending_round(server_round)
        central_counts, diagnostics = self.global_evaluator.evaluate(packet)
        if any(central_counts[key] != metrics[key] for key in (*COUNT_KEYS, "validation_examples")):
            raise RuntimeError("Centralized and pooled client confusion counts disagree")
        if diagnostics["global_evaluation_parameters_sha256"] != self.aggregate_sha256:
            raise RuntimeError("Centralized evaluator used mismatched global parameters")
        return {**metrics, **diagnostics}

    def requires_metric_persistence(self):
        return self.global_evaluator is not None

    def finish(self, history):
        try:
            if self._evaluation_failed or self._pending_round is not None or self._completed_round != self.rounds:
                raise RuntimeError("Global evaluation or required metric/report persistence did not complete")
            if [row["round"] for row in self.round_records if "validation" in row] != list(range(1, self.rounds + 1)):
                raise RuntimeError("Flower returned before all fit and validation rounds completed")
            if len(history.losses_distributed) != self.rounds:
                raise RuntimeError("Flower history is missing validation rounds")
            self._validate_published_report(self.round_records)
        except Exception:
            self._evaluation_failed = True
            raise


def main():
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument("--num-clients", type=int, required=True, help="Beklenen toplam banka sayısı")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--round-timeout", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    try:
        run_dir = child_run_path(args.run_dir)
        address = localhost_address(args.port)
        if args.num_clients < 2 or args.rounds < 1 or not 0 < args.round_timeout < math.inf:
            raise ValueError("Expected at least two clients, one round, and a positive finite round timeout")
        for name in ("tracking.sqlite", "artifacts", "mlruns", "rounds.json"):
            path = run_dir / name
            if path.exists() or path.is_symlink():
                raise ValueError(f"Experiment output already exists: {path}")
        global_evaluator = GlobalValidationContext.from_snapshot(run_dir / "input.csv", args.num_clients, seed=args.seed)
    except ValueError as error:
        parser.error(str(error))

    db_path = run_dir / "tracking.sqlite"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    experiment_id = mlflow.create_experiment(
        "Credit_Card_Fraud_Federated_Learning", artifact_location=(run_dir / "artifacts").as_uri(),
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    print(f"MLflow database: {db_path}", flush=True)
    strategy = ControlledStrategy(args.num_clients, args.rounds, run_dir / "rounds.json", global_evaluator=global_evaluator)
    history = fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=args.rounds, round_timeout=args.round_timeout),
        strategy=strategy,
    )
    strategy.finish(history)


if __name__ == "__main__":
    main()
