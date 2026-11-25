"""Exact centralized validation AP for the single-machine simulation only."""

from dataclasses import dataclass, field
import re
from types import MappingProxyType

from flwr.common import Parameters, parameters_to_ndarrays
import numpy as np
from sklearn.linear_model import LogisticRegression

if __package__:
    from .evaluation import COUNT_KEYS, evaluate_binary
    from .experiment import parameter_digest
    from .preprocessing import FEATURE_NAMES
    from .utils import client_validation_identity, prepare_partitioned_data
else:
    from evaluation import COUNT_KEYS, evaluate_binary
    from experiment import parameter_digest
    from preprocessing import FEATURE_NAMES
    from utils import client_validation_identity, prepare_partitioned_data


MODE = "centralized_simulation_validation"


def _arrays(parameters):
    if not isinstance(parameters, Parameters) or parameters.tensor_type != "numpy.ndarray" or len(parameters.tensors) != 2:
        raise ValueError("Global evaluation requires two NumPy parameter arrays")
    try:
        arrays = parameters_to_ndarrays(parameters)
    except (ValueError, OSError, EOFError) as error:
        raise ValueError("Invalid global NumPy parameter data") from error
    for array, shape in zip(arrays, ((1, len(FEATURE_NAMES)), (1,)), strict=True):
        if array.shape != shape or array.dtype != np.float64 or not np.isfinite(array).all():
            raise ValueError("Global parameters must have finite float64 coefficient/intercept shapes")
    return arrays


@dataclass(frozen=True)
class RoundParameters:
    round_number: int
    digest: str
    tensors: tuple[bytes, ...] = field(repr=False)

    @classmethod
    def capture(cls, round_number, parameters):
        if type(round_number) is not int or round_number < 1:
            raise ValueError("Global evaluation round must be a positive integer")
        _arrays(parameters)
        snapshot = Parameters(tensors=[bytes(t) for t in parameters.tensors], tensor_type=parameters.tensor_type)
        return cls(round_number, parameter_digest(snapshot), tuple(snapshot.tensors))

    def packet(self):
        packet = Parameters(tensors=list(self.tensors), tensor_type="numpy.ndarray")
        if parameter_digest(packet) != self.digest:
            raise RuntimeError("Global evaluation parameter digest mismatch")
        return packet


def _frozen_array(values, dtype):
    array = np.asarray(values, dtype=dtype)
    # Immutable bytes backing prevents even setflags(write=True) from mutating the cache.
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


@dataclass(frozen=True)
class GlobalValidationContext:
    X_validation: np.ndarray = field(repr=False)
    y_validation: np.ndarray = field(repr=False)
    row_ids: np.ndarray = field(repr=False)
    metadata: dict = field(repr=False)
    client_validation_counts: tuple[int, ...]

    def __post_init__(self):
        X = np.asarray(self.X_validation)
        y = np.asarray(self.y_validation)
        rows = np.asarray(self.row_ids)
        if (y.ndim != 1 or not len(y) or X.shape != (len(y), len(FEATURE_NAMES))
                or X.dtype.kind not in "iuf" or not np.isfinite(X).all()
                or y.dtype.kind not in "biuf" or not np.isin(y, [0, 1]).all()
                or rows.shape != y.shape or rows.dtype.kind not in "iu"
                or (rows < 0).any()
                or len(np.unique(rows)) != len(rows)
                or any(type(n) is not int or n < 1 for n in self.client_validation_counts)
                or sum(self.client_validation_counts) != len(y)):
            raise ValueError("Invalid frozen global validation arrays or client counts")
        object.__setattr__(self, "X_validation", _frozen_array(X, np.float64))
        object.__setattr__(self, "y_validation", _frozen_array(y, np.int64))
        object.__setattr__(self, "row_ids", _frozen_array(rows, np.int64))
        metadata = dict(self.metadata)
        required = {"dataset_sha256", "preprocessing_sha256", "global_validation_membership_sha256",
                    "validation_context_sha256", "client_validation_sha256"}
        if set(metadata) != required or not isinstance(metadata["client_validation_sha256"], (tuple, list)):
            raise ValueError("Invalid validation identity metadata")
        metadata["client_validation_sha256"] = tuple(metadata["client_validation_sha256"])
        hashes = [value for key, value in metadata.items() if key != "client_validation_sha256"]
        hashes.extend(metadata["client_validation_sha256"])
        if (len(metadata["client_validation_sha256"]) != len(self.client_validation_counts)
                or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes)):
            raise ValueError("Invalid validation identity fingerprints")
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        object.__setattr__(self, "client_validation_counts", tuple(self.client_validation_counts))

    @classmethod
    def from_snapshot(cls, path, num_clients, *, seed):
        frame, plan, preprocessor, metadata = prepare_partitioned_data(
            path, num_clients, seed=seed, with_metadata=True,
        )
        rows = np.sort(np.concatenate([p.validation for p in plan.clients]))
        forbidden = np.concatenate([*(p.train for p in plan.clients), plan.final_test, plan.excluded_duplicates])
        if len(np.unique(rows)) != len(rows) or np.intersect1d(rows, forbidden).size:
            raise ValueError("Global validation overlaps another partition")
        X = preprocessor.transform(frame.drop(columns="Class").iloc[rows])
        y = frame["Class"].to_numpy()[rows]
        # The full frame, partition plan, and temporary training arrays are not retained.
        return cls(X, y, rows, metadata, tuple(len(p.validation) for p in plan.clients))

    @property
    def cache_bytes(self):
        return self.X_validation.nbytes + self.y_validation.nbytes + self.row_ids.nbytes

    def client_identity(self, client_id):
        return client_validation_identity(self.metadata, client_id)

    def evaluate(self, parameters):
        coefficients, intercept = _arrays(parameters)
        model = LogisticRegression()
        model.classes_ = np.array([0, 1])
        model.n_features_in_ = len(FEATURE_NAMES)
        model.coef_, model.intercept_ = coefficients.copy(), intercept.copy()
        probabilities = np.asarray(model.predict_proba(self.X_validation))
        if (probabilities.shape != (len(self.y_validation), 2) or probabilities.dtype.kind not in "iuf"
                or not np.isfinite(probabilities).all()
                or not ((0 <= probabilities) & (probabilities <= 1)).all()
                or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-12)):
            raise ValueError("Invalid centralized validation probabilities")
        _, metrics = evaluate_binary(self.y_validation, probabilities[:, 1])
        defined = metrics["average_precision_defined"]
        diagnostics = {
            "global_average_precision_computed": defined, "global_average_precision_defined": defined,
            "global_average_precision_status": "computed_exact" if defined else "undefined_no_positive_labels",
            "global_evaluation_mode": MODE,
            **{key: value for key, value in self.metadata.items() if key != "client_validation_sha256"},
            "global_evaluation_parameters_sha256": parameter_digest(parameters),
        }
        if defined:
            diagnostics["global_average_precision"] = metrics["average_precision"]
        return {key: metrics[key] for key in (*COUNT_KEYS, "validation_examples")}, diagnostics
