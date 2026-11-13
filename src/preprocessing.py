"""Shared feature transformation and JSON metadata for a trained model."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = ("Time", *(f"V{i}" for i in range(1, 29)), "Amount")
SCALED_FEATURES = ("Amount", "Time")
SCALED_INDICES = [FEATURE_NAMES.index(name) for name in SCALED_FEATURES]
SCHEMA_VERSION = 1


def _feature_array(features):
    if isinstance(features, pd.DataFrame):
        if tuple(features.columns) != FEATURE_NAMES:
            raise ValueError("Expected feature columns in order: Time, V1 through V28, Amount")
        if any(dtype.kind not in "fiu" for dtype in features.dtypes):
            raise ValueError("Features must be numeric")
        features = features.to_numpy(dtype=np.float64, na_value=np.nan)

    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES) or len(values) == 0:
        raise ValueError(f"Expected a non-empty batch with {len(FEATURE_NAMES)} features per row")
    if values.dtype.kind not in "fiu":
        raise ValueError("Features must be numeric")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Features must be finite")
    return values


@dataclass(frozen=True)
class FeaturePreprocessor:
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    def __post_init__(self):
        for field in ("mean", "scale"):
            values = getattr(self, field)
            if not isinstance(values, (list, tuple)) or len(values) != len(SCALED_FEATURES):
                raise ValueError(f"Preprocessing {field} must contain one value per scaled feature")
            try:
                valid = all(type(value) in (int, float) and math.isfinite(value) for value in values)
            except OverflowError:
                valid = False
            if not valid or (field == "scale" and any(value <= 0 for value in values)):
                raise ValueError(f"Invalid preprocessing {field}")
            object.__setattr__(self, field, tuple(float(value) for value in values))

    @classmethod
    def fit(cls, training_features):
        values = _feature_array(training_features)
        scaler = StandardScaler().fit(values[:, SCALED_INDICES])
        return cls(tuple(scaler.mean_.tolist()), tuple(scaler.scale_.tolist()))

    def transform(self, features):
        values = _feature_array(features).copy()
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = (values[:, SCALED_INDICES] - self.mean) / self.scale
        if not np.isfinite(scaled).all():
            raise ValueError("Preprocessing produced non-finite features")
        values[:, SCALED_INDICES] = scaled
        return values

    def save_for_model(self, model_path):
        """Write a new sidecar for the model trained with this fitted transformer."""
        model_path = Path(model_path)
        with model_path.open("rb") as stream:
            model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "feature_names": FEATURE_NAMES,
            "scaled_features": SCALED_FEATURES,
            "mean": self.mean,
            "scale": self.scale,
            "model_sha256": model_hash,
        }
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        sidecar = _sidecar_path(model_path)
        with sidecar.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        return sidecar


def _sidecar_path(model_path):
    return model_path.with_name(f"{model_path.name}.preprocessing.json")


def load_model_preprocessor(model_path, model_bytes):
    """Validate the sidecar against the same model bytes the caller will load."""
    sidecar = _sidecar_path(Path(model_path))
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read preprocessing metadata at {sidecar}") from error

    required = {"schema_version", "feature_names", "scaled_features", "mean", "scale", "model_sha256"}
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise ValueError("Incomplete or unexpected preprocessing metadata fields")
    if type(metadata["schema_version"]) is not int or metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported preprocessing schema version")
    if metadata["feature_names"] != list(FEATURE_NAMES) or metadata["scaled_features"] != list(SCALED_FEATURES):
        raise ValueError("Incompatible preprocessing feature names or order")
    if metadata["model_sha256"] != hashlib.sha256(model_bytes).hexdigest():
        raise ValueError("Preprocessing metadata does not match the model SHA-256")
    return FeaturePreprocessor(metadata["mean"], metadata["scale"])
