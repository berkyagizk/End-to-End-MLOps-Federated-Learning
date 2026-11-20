import argparse
import sys

import flwr as fl
from flwr.common import ndarrays_to_parameters
import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

if __package__:
    from .evaluation import evaluate_binary
    from .experiment import child_run_path, localhost_address, parameter_digest
    from .utils import load_client_data
else:
    from evaluation import evaluate_binary
    from experiment import child_run_path, localhost_address, parameter_digest
    from utils import load_client_data


class FraudClient(fl.client.NumPyClient):
    def __init__(self, client_id, X_train, y_train, X_val, y_val, *, model_output=None):
        if not np.array_equal(np.unique(y_train), [0, 1]):
            raise ValueError("Client training labels must contain both classes 0 and 1 only")
        self.client_id = client_id
        self.model_output = model_output
        self.X_train = X_train
        self.y_train = np.asarray(y_train, dtype=np.int64)
        self.X_val, self.y_val = X_val, y_val
        self.n_features = X_train.shape[1]

        # Every client can supply the same initial global parameters to Flower.
        # Set the fitted attributes needed for prediction before the first fit.
        self.model = LogisticRegression(solver="lbfgs", max_iter=1, warm_start=True)
        self.model.classes_ = np.array([0, 1])
        self.model.n_features_in_ = self.n_features
        self.model.coef_ = np.zeros((1, self.n_features), dtype=np.float64)
        self.model.intercept_ = np.zeros(1, dtype=np.float64)

    def get_parameters(self, config):
        return [self.model.coef_.copy(), self.model.intercept_.copy()]

    def set_parameters(self, parameters):
        if not isinstance(parameters, (list, tuple)) or len(parameters) != 2:
            raise ValueError("Expected two parameter arrays: coefficients and intercept")

        validated = []
        for name, value, shape in (
            ("Coefficients", parameters[0], (1, self.n_features)),
            ("Intercept", parameters[1], (1,)),
        ):
            if not isinstance(value, np.ndarray) or value.dtype.kind != "f":
                raise ValueError(f"{name} must be a floating-point NumPy array")
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            with np.errstate(over="ignore", invalid="ignore"):
                array = value.astype(np.float64, copy=True)
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite float64 values")
            validated.append(array)
        self.model.coef_, self.model.intercept_ = validated

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        metrics = self.round_metrics(config)
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(self.X_train, self.y_train)

        self.model.fit(X_train_balanced, y_train_balanced)
        updated = self.get_parameters(config)
        if not all(np.isfinite(array).all() for array in updated):
            raise ValueError("Local optimization returned non-finite parameters")
        print(f"Client {self.client_id}: Eğitim tamamlandı (SMOTE ile dengelendi)...")

        metrics["post_smote_examples"] = len(X_train_balanced)
        return updated, len(self.X_train), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        probabilities = np.asarray(self.model.predict_proba(self.X_val))
        if (not np.array_equal(self.model.classes_, [0, 1])
                or probabilities.shape != (len(self.y_val), 2)
                or probabilities.dtype.kind not in "iuf"
                or not np.isfinite(probabilities).all()
                or not ((0 <= probabilities) & (probabilities <= 1)).all()
                or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12)):
            raise ValueError("Expected finite binary probabilities in class order [0, 1], summing to one")
        loss, metrics = evaluate_binary(self.y_val, probabilities[:, 1])
        accuracy = metrics["accuracy"]
        print(f"Client {self.client_id} - Doğrulama Başarımı: {accuracy}")

        if self.client_id == 0 and self.model_output is not None:
            joblib.dump(self.model, self.model_output)
            print(f"Saved validation-round model to {self.model_output}")

        return loss, len(self.X_val), {**metrics, **self.round_metrics(config)}

    def round_metrics(self, config):
        if "server_round" not in config:
            return {}
        return {
            "client_id": self.client_id,
            "server_round": config["server_round"],
            "received_sha256": parameter_digest(ndarrays_to_parameters(self.get_parameters({}))),
        }


def main():
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--client-id", type=int, required=True, help="Bankanın ID numarası (0, 1, 2...)")
    parser.add_argument("--num-clients", type=int, required=True, help="Toplam banka sayısı")
    parser.add_argument("--seed", type=int, default=42, help="Veri bölme tohumu (tüm istemcilerde aynı)")
    parser.add_argument("--run-dir", required=True, help="Isolated experiment directory")
    parser.add_argument("--port", type=int, required=True, help="Local Flower server port")
    args = parser.parse_args()
    try:
        run_dir = child_run_path(args.run_dir)
        address = localhost_address(args.port)
        model_output = run_dir / "model.pkl"
        if args.client_id == 0 and (model_output.exists() or model_output.is_symlink()):
            raise ValueError("Model output already exists in the experiment directory")
        data = load_client_data(run_dir / "input.csv", args.num_clients, args.client_id, seed=args.seed)
        client = FraudClient(args.client_id, *data, model_output=model_output)
    except ValueError as error:
        parser.error(str(error))

    if sys.platform == "win32":
        # Initialize the pinned sklearn controller before gRPC starts loading
        # modules on background threads during Windows' DLL enumeration.
        from sklearn.utils.parallel import _get_threadpool_controller

        _get_threadpool_controller()
    fl.client.start_numpy_client(server_address=address, client=client)


if __name__ == "__main__":
    main()
