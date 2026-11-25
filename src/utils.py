from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from numbers import Integral

import numpy as np
import pandas as pd

if __package__:
    from .preprocessing import FEATURE_NAMES, SCALED_FEATURES, SCHEMA_VERSION, FeaturePreprocessor
else:
    from preprocessing import FEATURE_NAMES, SCALED_FEATURES, SCHEMA_VERSION, FeaturePreprocessor


MIN_TRAIN_PER_CLASS = 6  # SMOTE's default k_neighbors=5 needs at least six rows.


@dataclass(frozen=True)
class ClientPartition:
    train: np.ndarray
    validation: np.ndarray


@dataclass(frozen=True)
class PartitionPlan:
    clients: tuple[ClientPartition, ...]
    final_test: np.ndarray
    excluded_duplicates: np.ndarray


def validate_client_count(num_clients):
    if isinstance(num_clients, bool) or not isinstance(num_clients, Integral) or num_clients < 2:
        raise ValueError("num_clients must be an integer of at least 2")


def build_partition_plan(frame, num_clients=3, *, seed=42):
    """Assign zero-based CSV row positions to clients and one reserved holdout."""
    validate_client_count(num_clients)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if "Class" not in frame or set(frame["Class"].unique()) != {0, 1}:
        raise ValueError("Class must contain both binary labels 0 and 1, with no other values")

    # Ignore DataFrame index labels. Duplicate identity is the exact parsed feature vector.
    features = frame.drop(columns="Class")
    labels = frame["Class"].to_numpy()
    duplicate_features = features.duplicated().to_numpy()
    if not np.array_equal(duplicate_features, frame.duplicated().to_numpy()):
        raise ValueError("Duplicate feature vectors have conflicting Class labels")

    rng = np.random.default_rng(seed)
    train_by_class = []
    validation_by_class = []
    final_by_class = []
    for label in (0, 1):
        rows = rng.permutation(np.flatnonzero((labels == label) & ~duplicate_features))
        final_count = max(1, round(0.2 * len(rows)))
        remaining = rows[final_count:]
        validation_count = max(1, round(0.2 * len(remaining)))
        train = remaining[validation_count:]
        validation = remaining[:validation_count]
        for split, split_rows, minimum_per_client in (
            ("training", train, MIN_TRAIN_PER_CLASS),
            ("validation", validation, 1),
        ):
            minimum = num_clients * minimum_per_client
            if len(split_rows) < minimum:
                raise ValueError(
                    f"Class {label} has {len(split_rows)} unique rows in the global {split} split. "
                    f"{num_clients} clients need at least {minimum} "
                    f"({minimum_per_client} per client). Use fewer clients or more data."
                )
        final_by_class.append(rows[:final_count])
        train_by_class.append(train)
        validation_by_class.append(validation)

    # Distribute each global split by class. Assign validation remainders to the
    # last clients to balance the combined sizes against training remainders.
    train_chunks = [np.array_split(rows, num_clients) for rows in train_by_class]
    validation_chunks = [np.array_split(rows, num_clients)[::-1] for rows in validation_by_class]
    clients = tuple(
        ClientPartition(
            train=rng.permutation(np.concatenate([chunks[client_id] for chunks in train_chunks])),
            validation=rng.permutation(np.concatenate([chunks[client_id] for chunks in validation_chunks])),
        )
        for client_id in range(num_clients)
    )

    return PartitionPlan(
        clients=clients,
        final_test=np.concatenate(final_by_class),
        excluded_duplicates=np.flatnonzero(duplicate_features),
    )


def fit_partition_preprocessor(features, plan):
    """Fit the simulation's shared transformer on global training rows only."""
    training_rows = np.sort(np.concatenate([client.train for client in plan.clients]))
    return FeaturePreprocessor.fit(features.iloc[training_rows])


def membership_digest(rows):
    return hashlib.sha256(np.sort(rows).astype("<i8").tobytes()).hexdigest()


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def prepare_partitioned_data(file_path, num_clients, *, seed=42, with_metadata=False):
    """Prepare once per process; fingerprint exactly the bytes parsed when requested."""
    if with_metadata:
        with open(file_path, "rb") as stream:
            input_bytes = stream.read()
        dataset_hash = hashlib.sha256(input_bytes).hexdigest()
        frame = pd.read_csv(BytesIO(input_bytes))
        del input_bytes
    else:
        frame = pd.read_csv(file_path)
    plan = build_partition_plan(frame, num_clients, seed=seed)
    preprocessor = fit_partition_preprocessor(frame.drop(columns="Class"), plan)
    metadata = None
    if with_metadata:
        preprocessing_hash = _json_digest({
            "schema_version": SCHEMA_VERSION, "feature_names": FEATURE_NAMES,
            "scaled_features": SCALED_FEATURES, "mean": preprocessor.mean, "scale": preprocessor.scale,
        })
        validation_hash = membership_digest(np.concatenate([p.validation for p in plan.clients]))
        metadata = {
            "dataset_sha256": dataset_hash, "preprocessing_sha256": preprocessing_hash,
            "global_validation_membership_sha256": validation_hash,
            "validation_context_sha256": _json_digest({
                "dataset": dataset_hash, "preprocessing": preprocessing_hash,
                "validation": validation_hash, "seed": int(seed), "num_clients": int(num_clients),
            }),
            "client_validation_sha256": tuple(membership_digest(p.validation) for p in plan.clients),
        }
    return frame, plan, preprocessor, metadata


def client_validation_identity(metadata, client_id):
    return {**{key: value for key, value in metadata.items() if key != "client_validation_sha256"},
            "validation_membership_sha256": metadata["client_validation_sha256"][client_id]}


def load_client_data(file_path, num_clients, client_id, *, seed=42, with_metadata=False):
    """Return training and round-validation arrays for exactly one simulated client."""
    validate_client_count(num_clients)
    if isinstance(client_id, bool) or not isinstance(client_id, Integral) or not 0 <= client_id < num_clients:
        raise ValueError(f"client_id must be an integer in [0, {num_clients - 1}]")

    frame, plan, preprocessor, metadata = prepare_partitioned_data(
        file_path, num_clients, seed=seed, with_metadata=with_metadata,
    )
    partition = plan.clients[client_id]
    features = frame.drop(columns="Class")
    labels = frame["Class"].to_numpy()

    train = preprocessor.transform(features.iloc[partition.train])
    validation = preprocessor.transform(features.iloc[partition.validation])
    data = (train, labels[partition.train], validation, labels[partition.validation])
    return (*data, client_validation_identity(metadata, client_id)) if with_metadata else data
