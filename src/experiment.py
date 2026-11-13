"""Paths and parameter fingerprints for isolated local simulations."""

import hashlib
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def external_run_path(path):
    path = Path(path).resolve()
    if path.is_relative_to(PROJECT_ROOT) or PROJECT_ROOT.is_relative_to(path):
        raise ValueError("Experiment directory must be outside the project checkout")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", path.name):
        raise ValueError("Experiment directory name must contain only letters, digits, hyphens, and underscores")
    if str(path).startswith("\\\\") or "?" in str(path):
        raise ValueError("Experiment directory must be on a local drive with a SQLite-compatible path")
    return path


def child_run_path(path):
    path = external_run_path(path)
    if not path.is_dir() or Path.cwd().resolve() != path:
        raise ValueError("Run the child with the experiment directory as its working directory")
    return path


def localhost_address(port):
    if not 1024 <= port <= 65535:
        raise ValueError("Port must be between 1024 and 65535")
    return f"127.0.0.1:{port}"


def parameter_digest(parameters):
    digest = hashlib.sha256()
    for tensor in parameters.tensors:
        digest.update(len(tensor).to_bytes(8, "big"))
        digest.update(tensor)
    return digest.hexdigest()
