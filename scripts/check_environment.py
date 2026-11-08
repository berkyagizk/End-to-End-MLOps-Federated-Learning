"""Check the Phase 0 environment without importing application entry points."""

import argparse
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import re
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "flwr": "flwr",
    "mlflow": "mlflow",
    "scikit-learn": "sklearn",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "imbalanced-learn": "imblearn",
    "numpy": "numpy",
    "scipy": "scipy",
    "joblib": "joblib",
}


def check_packages():
    expected_python = (ROOT / ".python-version").read_text().strip()
    if platform.python_implementation() != "CPython" or platform.python_version() != expected_python:
        raise RuntimeError(f"Expected CPython {expected_python}; got {platform.python_version()}")
    if sys.maxsize <= 2**32:
        raise RuntimeError("A 64-bit Python interpreter is required")
    print(f"PASS Python {platform.python_version()} ({sys.executable})")

    requirements = (ROOT / "requirements.txt").read_text().splitlines()
    pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s;]+)$", "\n".join(requirements), re.MULTILINE))
    if set(pins) != set(MODULES):
        raise RuntimeError("Update the smoke-check module map to match direct dependencies")
    modules = {}
    for package, module_name in MODULES.items():
        actual = importlib.metadata.version(package)
        if actual != pins[package]:
            raise RuntimeError(f"{package}: expected {pins[package]}, found {actual}")
        modules[module_name] = importlib.import_module(module_name)
        print(f"PASS import {module_name} ({actual})")

    flower = modules["flwr"]
    for api in (flower.client.NumPyClient, flower.client.start_numpy_client,
                flower.server.start_server, flower.server.ServerConfig,
                flower.server.strategy.FedAvg):
        if not callable(api):
            raise RuntimeError(f"Missing callable Flower API: {api}")
    flower.server.strategy.FedAvg(min_fit_clients=2, min_available_clients=2)
    print("PASS legacy Flower APIs used by the existing source")

    np = modules["numpy"]
    from scipy.linalg import solve
    from sklearn.metrics import log_loss

    np.testing.assert_allclose(solve(np.eye(2), np.array([1., 2.])), [1., 2.])
    if not np.isfinite(log_loss([0, 1], [[.9, .1], [.1, .9]])):
        raise RuntimeError("Scientific-library smoke check failed")
    print("PASS NumPy/SciPy/scikit-learn native interoperability (no model fit)")

    with tempfile.TemporaryDirectory(prefix="fraud-joblib-") as directory:
        target = Path(directory) / "smoke.joblib"
        modules["joblib"].dump({"phase": 0}, target)
        if modules["joblib"].load(target) != {"phase": 0}:
            raise RuntimeError("joblib round-trip failed")
    print("PASS joblib round-trip in a temporary directory")


def check_dataset(path):
    import numpy as np
    import pandas as pd

    expected = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
    frame = pd.read_csv(path)
    if list(frame.columns) != expected:
        raise RuntimeError(f"Expected columns in this order: {expected}")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise RuntimeError("All dataset columns must be numeric")
    if not np.isfinite(frame.to_numpy()).all():
        raise RuntimeError("Dataset contains missing or non-finite values")
    counts = frame["Class"].value_counts().to_dict()
    if len(frame) != 284807 or counts != {0: 284315, 1: 492}:
        raise RuntimeError(f"Unexpected benchmark size/class counts: {len(frame)}, {counts}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    print(f"PASS dataset schema: {len(frame)} rows, 30 features, class counts {counts}")
    print(f"Dataset SHA-256: {digest.hexdigest()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Optionally validate an unmodified creditcard.csv")
    args = parser.parse_args()
    # Keep import-time configuration local to this smoke-check process.
    os.environ["FLWR_TELEMETRY_ENABLED"] = "0"
    with tempfile.TemporaryDirectory(prefix="fraud-matplotlib-") as config:
        os.environ["MPLCONFIGDIR"] = config
        check_packages()
        if args.dataset is not None:
            check_dataset(args.dataset)
    print("PASS environment smoke checks; no application entry points or training executed")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
