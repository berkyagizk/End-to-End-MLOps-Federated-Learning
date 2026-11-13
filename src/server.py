import argparse
import math
import os
import sys

import flwr as fl
from flwr.common import parameters_to_ndarrays
import mlflow
import numpy as np

if __package__:
    from .preprocessing import FEATURE_NAMES
else:
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


def main():
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument("--num-clients", type=int, required=True, help="Beklenen toplam banka sayısı")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(os.path.dirname(current_dir), "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("Credit_Card_Fraud_Federated_Learning")
    print(f"MLflow DB Yolu: {db_path}")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=build_strategy(args.num_clients),
    )


if __name__ == "__main__":
    main()
