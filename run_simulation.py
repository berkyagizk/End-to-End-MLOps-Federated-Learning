"""Launch a bounded Flower simulation in a fresh directory outside the checkout."""

import argparse
from io import BytesIO
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import sysconfig
import time

import pandas as pd

from src.experiment import PROJECT_ROOT, external_run_path, localhost_address
from src.preprocessing import FEATURE_NAMES
from src.utils import build_partition_plan


def check_port(port):
    localhost_address(port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if os.name == "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise ValueError(f"Cannot bind localhost port {port}: {error}") from error


def prepare_run(run_dir, dataset, num_clients, seed, port):
    requested = Path(run_dir)
    if requested.exists() or requested.is_symlink():
        raise ValueError(f"Experiment directory must be new: {requested}")
    run_dir = external_run_path(requested)
    if not run_dir.parent.is_dir():
        raise ValueError("Experiment directory parent must already exist")
    dataset = Path(dataset).resolve(strict=True)
    if not dataset.is_file():
        raise ValueError("Dataset must be a CSV file")
    input_bytes = dataset.read_bytes()
    frame = pd.read_csv(BytesIO(input_bytes))
    if tuple(frame.columns) != (*FEATURE_NAMES, "Class"):
        raise ValueError("Dataset must use the ordered 30-feature schema followed by Class")
    build_partition_plan(frame, num_clients, seed=seed)
    check_port(port)
    # mkdir refuses an existing target, including one created after preflight.
    run_dir.mkdir(mode=0o700)
    with (run_dir / "input.csv").open("xb") as stream:
        stream.write(input_bytes)
    for name in ("tmp", "cache", "profile"):
        (run_dir / name).mkdir()
    return run_dir


def child_environment(run_dir):
    env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "SYSTEMDRIVE") if key in os.environ}
    profile = str(run_dir / "profile")
    cache = str(run_dir / "cache")
    tmp = str(run_dir / "tmp")
    system_root = env.get("SYSTEMROOT")
    paths = [sys.base_prefix]
    if system_root:
        paths.append(str(Path(system_root) / "System32"))
    env.update(
        PATH=os.pathsep.join(paths),
        HOME=profile, USERPROFILE=profile, APPDATA=profile, LOCALAPPDATA=profile,
        XDG_CACHE_HOME=cache, XDG_CONFIG_HOME=profile, XDG_DATA_HOME=profile,
        MPLCONFIGDIR=cache, JOBLIB_TEMP_FOLDER=tmp,
        TEMP=tmp, TMP=tmp, TMPDIR=tmp,
        PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
        FLWR_TELEMETRY_ENABLED="0", MLFLOW_DISABLE_TELEMETRY="true",
        MLFLOW_ENABLE_ASYNC_LOGGING="false", MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING="false",
        GIT_PYTHON_REFRESH="quiet", LOKY_MAX_CPU_COUNT="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
    )
    return env


def check_processes(processes):
    statuses = {name: process.poll() for name, process in processes.items()}
    for name, code in statuses.items():
        if code is not None and code != 0:
            raise RuntimeError(f"{name} exited with code {code}. See {name}.log")
    return statuses


def wait_for_server(server, log_path, deadline):
    while time.monotonic() < deadline:
        if check_processes({"server": server})["server"] is not None:
            raise RuntimeError("Server exited before reporting readiness. See server.log")
        if "Flower ECE: gRPC server running" in log_path.read_text(encoding="utf-8", errors="replace"):
            return
        time.sleep(0.05)
    raise TimeoutError("Timed out waiting for the Flower server. See server.log")


def wait_for_completion(processes, deadline):
    while time.monotonic() < deadline:
        statuses = check_processes(processes)
        if all(code == 0 for code in statuses.values()):
            return {name: process.wait() for name, process in processes.items()}
        time.sleep(0.05)
    active = [name for name, code in check_processes(processes).items() if code is None]
    raise TimeoutError(f"Simulation timed out. Still running: {', '.join(active)}")


def stop_processes(processes):
    for process in processes.values():
        if process.poll() is None:
            process.terminate()
    for process in processes.values():
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def run_simulation(run_dir, dataset, *, num_clients, seed, port, rounds, timeout):
    if rounds < 1 or not 0 < timeout < math.inf:
        raise ValueError("Expected at least one round and a positive finite timeout")
    run_dir = prepare_run(run_dir, dataset, num_clients, seed, port)
    print(f"Experiment directory: {run_dir}", flush=True)
    print(f"Flower address: {localhost_address(port)}", flush=True)
    env = child_environment(run_dir)
    processes = {}
    deadline = time.monotonic() + timeout
    common = ["--run-dir", str(run_dir), "--port", str(port), "--num-clients", str(num_clients)]

    def launch(name, script, args):
        # Use the base interpreter directly to avoid orphaning the Python
        # process behind Windows' virtual-environment redirector on timeout.
        packages = sysconfig.get_path("purelib")
        bootstrap = (
            "import runpy, site, sys; site.addsitedir(sys.argv.pop(1)); "
            "sys.path.insert(0, sys.argv.pop(1)); runpy.run_path(sys.argv.pop(1), run_name='__main__')"
        )
        with (run_dir / f"{name}.log").open("xb") as log:
            processes[name] = subprocess.Popen(
                [sys._base_executable, "-I", "-S", "-B", "-u", "-X", "utf8", "-c", bootstrap,
                 packages, str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / script), *args],
                cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
            )

    try:
        launch("server", "src/server.py", [
            *common, "--rounds", str(rounds), "--round-timeout", str(min(timeout, 30)),
        ])
        wait_for_server(processes["server"], run_dir / "server.log", deadline)
        for client_id in range(num_clients):
            launch(f"client-{client_id}", "src/client.py", [
                *common, "--client-id", str(client_id), "--seed", str(seed),
            ])
        statuses = wait_for_completion(processes, deadline)
        for name in ("rounds.json", "tracking.sqlite", "model.pkl"):
            if not (run_dir / name).is_file():
                raise RuntimeError(f"Simulation exited without producing {name}. See server.log and client logs")
        print(f"Completed {rounds} rounds. Exit codes: {statuses}", flush=True)
        return statuses
    finally:
        stop_processes(processes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="New experiment directory outside the checkout")
    parser.add_argument("--dataset", required=True, help="CSV to copy into the experiment directory")
    parser.add_argument("--num-clients", required=True, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--rounds", required=True, type=int)
    parser.add_argument("--timeout", required=True, type=float, help="Overall child-process deadline in seconds")
    try:
        run_simulation(**vars(parser.parse_args()))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Simulation failed: {error}\n")


if __name__ == "__main__":
    main()
