import argparse
import json
import math
import sys

import flwr as fl
from flwr.common import parameters_to_ndarrays
import mlflow
import numpy as np

if __package__:
    from .experiment import child_run_path, localhost_address, parameter_digest
    from .preprocessing import FEATURE_NAMES
else:
    from experiment import child_run_path, localhost_address, parameter_digest
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

        print(f"\n>>> Tur {server_round} değerlendiriliyor...")
        if metrics and "accuracy" in metrics:
            acc = metrics["accuracy"]
            print(f">>> Başarı Oranı: {acc} - MLflow'a yazılıyor...")
            try:
                with mlflow.start_run(run_name=f"Round_{server_round}"):
                    mlflow.log_metric("accuracy", acc, step=server_round)
                    mlflow.log_metric("loss", loss if loss is not None else 0.0, step=server_round)
                print(">>> MLflow yazma işlemi başarıyla tamamlandı.")
            except Exception as e:
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

    def __init__(self, num_clients, rounds, report):
        super().__init__(
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=weighted_accuracy,
            on_fit_config_fn=lambda round_number: {"server_round": round_number},
            on_evaluate_config_fn=lambda round_number: {"server_round": round_number},
            accept_failures=False,
        )
        self.client_ids = set(range(num_clients))
        self.rounds = rounds
        self.report = report
        self.round_records = []

    def configure_fit(self, server_round, parameters, client_manager):
        self.sent_sha256 = parameter_digest(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def checked_replies(self, stage, server_round, results, failures, expected_sha256):
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
            replies.append({**result.metrics, "num_examples": result.num_examples})
        return sorted(replies, key=lambda reply: reply["client_id"])

    def aggregate_fit(self, server_round, results, failures):
        replies = self.checked_replies("fit", server_round, results, failures, self.sent_sha256)
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        self.aggregate_sha256 = parameter_digest(parameters)
        self.round_records.append({
            "round": server_round, "sent_sha256": self.sent_sha256,
            "aggregate_sha256": self.aggregate_sha256, "fit": replies,
        })
        return parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        replies = self.checked_replies("validation", server_round, results, failures, self.aggregate_sha256)
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        self.round_records[-1].update(validation=replies, loss=loss, accuracy=metrics["accuracy"])
        print(f"Completed round {server_round}/{self.rounds} with all clients", flush=True)
        return loss, metrics

    def finish(self, history):
        if [row["round"] for row in self.round_records if "validation" in row] != list(range(1, self.rounds + 1)):
            raise RuntimeError("Flower returned before all fit and validation rounds completed")
        if len(history.losses_distributed) != self.rounds:
            raise RuntimeError("Flower history is missing validation rounds")
        with self.report.open("x", encoding="utf-8") as stream:
            json.dump(self.round_records, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument("--num-clients", type=int, required=True, help="Beklenen toplam banka sayısı")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--round-timeout", type=float, required=True)
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
    except ValueError as error:
        parser.error(str(error))

    db_path = run_dir / "tracking.sqlite"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    experiment_id = mlflow.create_experiment(
        "Credit_Card_Fraud_Federated_Learning", artifact_location=(run_dir / "artifacts").as_uri(),
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    print(f"MLflow database: {db_path}", flush=True)
    strategy = ControlledStrategy(args.num_clients, args.rounds, run_dir / "rounds.json")
    history = fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=args.rounds, round_timeout=args.round_timeout),
        strategy=strategy,
    )
    strategy.finish(history)


if __name__ == "__main__":
    main()
