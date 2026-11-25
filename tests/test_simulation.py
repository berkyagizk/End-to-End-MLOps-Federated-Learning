from contextlib import closing, redirect_stdout
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.parse import unquote

import numpy as np
import pandas as pd

from run_simulation import PROJECT_ROOT, prepare_run, run_simulation, wait_for_completion
from src.experiment import external_run_path
from src.evaluation import aggregate_validation_metrics
from src.global_evaluation import GlobalValidationContext, MODE
from src.preprocessing import FEATURE_NAMES
from src.server import ControlledStrategy
from src.utils import build_partition_plan


def synthetic_csv(path):
    rng = np.random.default_rng(42)
    labels = np.r_[np.zeros(120, dtype=int), np.ones(40, dtype=int)]
    features = rng.normal(size=(len(labels), len(FEATURE_NAMES)))
    features[:, 1:4] += labels[:, None] * 0.5
    frame = pd.DataFrame(features, columns=FEATURE_NAMES).assign(Class=labels)
    frame.to_csv(path, index=False)
    return frame


class LauncherTests(unittest.TestCase):
    def test_rejects_checkout_and_existing_directory(self):
        with self.assertRaisesRegex(ValueError, "outside the project"):
            prepare_run(PROJECT_ROOT / "unsafe-test-target", "unused", 2, 42, 24081)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "preserve.txt"
            marker.write_text("original", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be new"):
                prepare_run(root, "unused", 2, 42, 24081)
            self.assertEqual(marker.read_text(encoding="utf-8"), "original")

    def test_port_conflict_is_detected_before_run_directory_creation(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as occupied:
            root = Path(directory)
            dataset = root / "synthetic.csv"
            synthetic_csv(dataset)
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            with self.assertRaisesRegex(ValueError, "Cannot bind localhost port"):
                prepare_run(root / "experiment", dataset, 2, 42, occupied.getsockname()[1])
            self.assertFalse((root / "experiment").exists())

    def test_launcher_reaps_successful_children(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            children = [Mock(**{"poll.return_value": 0, "wait.return_value": 0}) for _ in range(3)]
            for name in ("rounds.json", "tracking.sqlite", "model.pkl"):
                (root / name).touch()
            with patch("run_simulation.prepare_run", return_value=root), patch("run_simulation.wait_for_server"), \
                    patch("run_simulation.subprocess.Popen", side_effect=children) as spawn, redirect_stdout(StringIO()):
                statuses = run_simulation(root, "unused", num_clients=2, seed=42, port=24081, rounds=2, timeout=10)
            self.assertEqual(statuses, {"server": 0, "client-0": 0, "client-1": 0})
            for call in spawn.call_args_list:
                self.assertEqual(call.kwargs["cwd"], root)
                self.assertEqual(call.kwargs["env"]["TEMP"], str(root / "tmp"))
                self.assertEqual(call.kwargs["env"]["FLWR_TELEMETRY_ENABLED"], "0")
                self.assertNotIn("MLFLOW_TRACKING_URI", call.kwargs["env"])
            for child in children:
                child.wait.assert_called()
                child.terminate.assert_not_called()

    def test_launcher_cleans_up_after_a_failed_client(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = Mock(**{"poll.return_value": None})
            bad = Mock(**{"poll.return_value": 9})
            peer = Mock(**{"poll.return_value": None})
            with patch("run_simulation.prepare_run", return_value=root), patch("run_simulation.wait_for_server"), \
                    patch("run_simulation.subprocess.Popen", side_effect=[server, bad, peer]), redirect_stdout(StringIO()):
                with self.assertRaisesRegex(RuntimeError, "client-0 exited with code 9"):
                    run_simulation(root, "unused", num_clients=2, seed=42, port=24081, rounds=2, timeout=10)
            for child in (server, peer):
                child.terminate.assert_called_once()
                child.wait.assert_called_once_with(timeout=3)
            bad.wait.assert_called_once_with(timeout=3)

    def test_launcher_terminates_child_on_deadline_and_escalates_to_kill(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = Mock(**{"poll.return_value": None})
            child.wait.side_effect = [subprocess.TimeoutExpired("server", 3), 1]
            with patch("run_simulation.prepare_run", return_value=root), \
                    patch("run_simulation.subprocess.Popen", return_value=child), \
                    patch("run_simulation.time.monotonic", side_effect=[0, 2]), redirect_stdout(StringIO()):
                with self.assertRaisesRegex(TimeoutError, "waiting for the Flower server"):
                    run_simulation(root, "unused", num_clients=2, seed=42, port=24081, rounds=2, timeout=1)
            child.terminate.assert_called_once()
            child.kill.assert_called_once()
            self.assertEqual(child.wait.call_count, 2)

    def test_timeout_lists_processes_and_incomplete_round_reports_missing_clients(self):
        child = Mock(**{"poll.return_value": None})
        with self.assertRaisesRegex(TimeoutError, "Still running: client-1"):
            wait_for_completion({"client-1": child}, deadline=0)
        strategy = ControlledStrategy(2, 2, None)
        strategy.sent_sha256 = "expected"
        reply = Mock(metrics={"client_id": 0, "server_round": 1, "received_sha256": "expected"})
        with self.assertRaisesRegex(RuntimeError, r"Incomplete fit round 1.*received \[0\].*1 failures"):
            strategy.aggregate_fit(1, [(Mock(), reply)], [RuntimeError("disconnected")])
        strategy.aggregate_sha256 = "expected"
        with self.assertRaisesRegex(RuntimeError, "Incomplete validation round 1"):
            strategy.aggregate_evaluate(1, [], [])


class SyntheticSimulationTests(unittest.TestCase):
    def test_two_clients_two_rounds(self):
        # Keep evidence when an external destination is explicitly supplied.
        evidence = os.getenv("FRAUD_SMOKE_EVIDENCE_DIR")
        if evidence:
            root = Path(evidence).resolve()
            external_run_path(root).mkdir(mode=0o700)
        else:
            # Windows TEMP can use an 8.3 alias; compare canonical paths like MLflow.
            root = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="fraud-smoke-"))).resolve()
        dataset = root / "synthetic.csv"
        frame = synthetic_csv(dataset)
        digest = hashlib.sha256(dataset.read_bytes()).digest()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        run = root / "experiment"
        try:
            statuses = run_simulation(run, dataset, num_clients=2, seed=17, port=port, rounds=2, timeout=100)
        except (OSError, ValueError, RuntimeError) as error:
            logs = "\n".join(f"{p.name}:\n{p.read_text(encoding='utf-8', errors='replace')[-4000:]}"
                             for p in sorted(run.glob("*.log")))
            self.fail(f"{error}\n{logs}")
        self.assertEqual(statuses, {"server": 0, "client-0": 0, "client-1": 0})
        self.assertEqual(hashlib.sha256(dataset.read_bytes()).digest(), digest)
        self.assertEqual((run / "input.csv").read_bytes(), dataset.read_bytes())
        self.assertGreater((run / "model.pkl").stat().st_size, 0)
        self.assertTrue(all((run / f"{name}.log").stat().st_size > 0 for name in statuses))
        records = json.loads((run / "rounds.json").read_text(encoding="utf-8"))
        self.assertEqual([record["round"] for record in records], [1, 2])
        self.assertEqual(records[1]["sent_sha256"], records[0]["aggregate_sha256"])
        plan = build_partition_plan(frame, 2, seed=17)
        context = GlobalValidationContext.from_snapshot(dataset, 2, seed=17)
        print(f"Synthetic cached validation arrays: {context.cache_bytes} bytes")
        for record in records:
            self.assertNotEqual(record["sent_sha256"], record["aggregate_sha256"])
            for stage, expected in (("fit", record["sent_sha256"]), ("validation", record["aggregate_sha256"])):
                replies = record[stage]
                self.assertEqual([reply["client_id"] for reply in replies], [0, 1])
                self.assertTrue(all(reply["received_sha256"] == expected for reply in replies))
                for reply in replies:
                    for key, value in context.client_identity(reply["client_id"]).items():
                        self.assertEqual(reply[key], value)
                split = "train" if stage == "fit" else "validation"
                self.assertEqual([reply["num_examples"] for reply in replies],
                                 [len(getattr(client, split)) for client in plan.clients])
            pooled = aggregate_validation_metrics([(r["num_examples"], r) for r in record["validation"]])
            global_metrics = record["validation_metrics"]
            for key, value in pooled.items():
                if not key.startswith("global_average_precision_"):
                    self.assertEqual(global_metrics[key], value)
            self.assertEqual(global_metrics["loss"], record["loss"])
            self.assertTrue(global_metrics["global_average_precision_computed"])
            self.assertTrue(global_metrics["global_average_precision_defined"])
            self.assertEqual(global_metrics["global_average_precision_status"], "computed_exact")
            self.assertTrue(0 <= global_metrics["global_average_precision"] <= 1)
            self.assertEqual(global_metrics["global_evaluation_mode"], MODE)
            self.assertEqual(global_metrics["validation_context_sha256"], context.metadata["validation_context_sha256"])
            self.assertEqual(global_metrics["global_evaluation_parameters_sha256"], record["aggregate_sha256"])
            self.assertEqual(record["accuracy"], pooled["accuracy"])
            self.assertFalse(pooled["global_average_precision_computed"])
            self.assertNotIn("average_precision", pooled)
        # The centralized evaluator does not add score/label/transaction output files.
        self.assertEqual({p.name for p in run.iterdir() if p.is_file()},
                         {"input.csv", "model.pkl", "tracking.sqlite", "rounds.json",
                          "server.log", "client-0.log", "client-1.log"})
        uri = (run / "tracking.sqlite").resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            rows = connection.execute("SELECT name, status, artifact_uri FROM runs ORDER BY name").fetchall()
            self.assertEqual([(name, status) for name, status, _ in rows],
                             [("Round_1", "FINISHED"), ("Round_2", "FINISHED")])
            self.assertTrue(all(unquote(uri).startswith(unquote((run / "artifacts").as_uri()) + "/")
                                for _, _, uri in rows), (rows, (run / "artifacts").as_uri()))
            metrics = connection.execute(
                "SELECT r.name, m.key, m.value FROM metrics m JOIN runs r ON r.run_uuid=m.run_uuid"
            ).fetchall()
            for record in records:
                expected_metrics = {key: float(value) for key, value in record["validation_metrics"].items()
                                    if isinstance(value, (int, float, bool))}
                for reply in record["validation"]:
                    client_id = reply["client_id"]
                    expected_metrics[f"client_{client_id}_average_precision_defined"] = float(reply["average_precision_defined"])
                    if reply["average_precision_defined"]:
                        expected_metrics[f"client_{client_id}_average_precision"] = reply["average_precision"]
                actual_metrics = {key: value for name, key, value in metrics
                                  if name == f"Round_{record['round']}"}
                self.assertEqual(actual_metrics, expected_metrics)
                tags = dict(connection.execute(
                    "SELECT t.key, t.value FROM tags t JOIN runs r ON r.run_uuid=t.run_uuid WHERE r.name=?",
                    (f"Round_{record['round']}",),
                ).fetchall())
                for key, value in record["validation_metrics"].items():
                    if isinstance(value, str):
                        self.assertEqual(tags[key], value)
                self.assertNotIn("average_precision", actual_metrics)
                for key in ("accuracy", "loss"):
                    actual = [value for name, metric, value in metrics
                              if name == f"Round_{record['round']}" and metric == key]
                    self.assertEqual(actual, [record[key]])
