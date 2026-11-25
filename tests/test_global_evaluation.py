from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError, replace
import gc
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

from flwr.common import Code, EvaluateRes, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from src.client import FraudClient
from src.evaluation import COUNT_KEYS, evaluate_binary
from src.experiment import parameter_digest
from src.global_evaluation import GlobalValidationContext, MODE, RoundParameters
from src.preprocessing import FEATURE_NAMES, FeaturePreprocessor
from src.server import ControlledStrategy
from src.utils import build_partition_plan, fit_partition_preprocessor, load_client_data, prepare_partitioned_data
from test_simulation import synthetic_csv


OK = Status(Code.OK, "")


def weights(sign=1.0):
    coef = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float64)
    coef[0, 0] = sign
    return ndarrays_to_parameters([coef, np.zeros(1, dtype=np.float64)])


def toy_context(labels=(0, 0, 1, 1), tied=False):
    X = np.zeros((4, len(FEATURE_NAMES)))
    X[:, 0] = 0 if tied else logit([0.1, 0.4, 0.35, 0.8])
    digest = lambda name: hashlib.sha256(name.encode()).hexdigest()
    metadata = {key: digest(key) for key in (
        "dataset_sha256", "preprocessing_sha256", "global_validation_membership_sha256", "validation_context_sha256",
    )}
    metadata["client_validation_sha256"] = (digest("client0"), digest("client1"))
    return GlobalValidationContext(X, np.array(labels), np.arange(4), metadata, (2, 2))


class GlobalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(redirect_stdout(StringIO()))
        self.directory = self.enterContext(TemporaryDirectory())
        self.root = Path(self.directory)
        self.start = self.enterContext(patch("src.server.mlflow.start_run"))
        self.log = self.enterContext(patch("src.server.mlflow.log_metric"))
        self.tags = self.enterContext(patch("src.server.mlflow.set_tag"))

    def strategy(self, context=None, rounds=1):
        return ControlledStrategy(2, rounds, self.root / "rounds.json",
                                  global_evaluator=context if context is not None else toy_context())

    def prepare_round(self, strategy, number=1, sign=1.0):
        incoming = weights(0)
        manager = Mock()
        manager.num_available.return_value = 2
        manager.sample.return_value = [Mock(cid=str(i)) for i in (0, 1)]
        strategy.configure_fit(number, incoming, manager)
        results = []
        for client_id in (0, 1):
            metrics = {**strategy.global_evaluator.client_identity(client_id), "client_id": client_id,
                       "server_round": number, "received_sha256": parameter_digest(incoming), "post_smote_examples": 12}
            results.append((Mock(cid=str(client_id)), FitRes(OK, weights(sign), 12, metrics)))
        packet, _ = strategy.aggregate_fit(number, results, [])
        replies = []
        context = strategy.global_evaluator
        offset = 0
        for client_id, count in enumerate(context.client_validation_counts):
            client = FraudClient(client_id, np.zeros((12, len(FEATURE_NAMES))), np.arange(12) % 2,
                                 context.X_validation[offset:offset + count], context.y_validation[offset:offset + count],
                                 evaluation_identity=context.client_identity(client_id))
            loss, size, metrics = client.evaluate(parameters_to_ndarrays(packet), {"server_round": number})
            replies.append((Mock(cid=str(client_id)), EvaluateRes(OK, loss, size, metrics)))
            offset += count
        return packet, replies

    def assert_incomplete(self, strategy):
        self.assertEqual(strategy._completed_round, 0)
        self.assertTrue(all("validation" not in r for r in strategy.round_records))
        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            strategy.finish(SimpleNamespace(losses_distributed=[]))
        self.assertFalse((self.root / "rounds.json").exists())

    def test_exact_membership_nondefault_seed_and_client_preprocessing_parity(self):
        path = self.root / "synthetic.csv"
        synthetic_csv(path)
        frame = pd.read_csv(path)
        plan = build_partition_plan(frame, 2, seed=17)
        training = np.sort(np.concatenate([p.train for p in plan.clients]))
        with patch.object(FeaturePreprocessor, "fit", wraps=FeaturePreprocessor.fit) as fit:
            context = GlobalValidationContext.from_snapshot(path, 2, seed=17)
        self.assertEqual(fit.call_count, 1)
        pd.testing.assert_frame_equal(fit.call_args.args[0], frame.drop(columns="Class").iloc[training], check_exact=True)
        expected = np.sort(np.concatenate([p.validation for p in plan.clients]))
        np.testing.assert_array_equal(context.row_ids, expected)
        self.assertEqual(len(np.intersect1d(expected, training)), 0)
        self.assertEqual(len(np.intersect1d(expected, plan.final_test)), 0)
        transformer = fit_partition_preprocessor(frame.drop(columns="Class"), plan)
        np.testing.assert_array_equal(context.X_validation, transformer.transform(frame.drop(columns="Class").iloc[expected]))
        for client_id in (0, 1):
            data = load_client_data(path, 2, client_id, seed=17, with_metadata=True)
            positions = np.searchsorted(context.row_ids, plan.clients[client_id].validation)
            np.testing.assert_array_equal(data[2], context.X_validation[positions])
            np.testing.assert_array_equal(data[3], context.y_validation[positions])
            self.assertEqual(data[4], context.client_identity(client_id))
            original = load_client_data(path, 2, client_id, seed=17)
            self.assertEqual(len(original), 4)
            for old, new in zip(original, data[:4], strict=True):
                np.testing.assert_array_equal(old, new)
        other_seed = GlobalValidationContext.from_snapshot(path, 2, seed=42)
        self.assertNotEqual(context.metadata["validation_context_sha256"], other_seed.metadata["validation_context_sha256"])
        self.assertFalse(np.array_equal(context.row_ids, other_seed.row_ids))
        # Formatting changes alone must still change the identity of the source bytes.
        path.write_bytes(path.read_bytes() + b"\n")
        changed = GlobalValidationContext.from_snapshot(path, 2, seed=17)
        self.assertNotEqual(changed.metadata["dataset_sha256"], context.metadata["dataset_sha256"])

    def test_cached_context_is_immutable_releases_frame_and_never_reloads_per_round(self):
        path = self.root / "synthetic.csv"
        synthetic_csv(path)
        frames = []

        def observed(*args, **kwargs):
            result = prepare_partitioned_data(*args, **kwargs)
            frames.append(weakref.ref(result[0]))
            return result

        with patch("src.global_evaluation.prepare_partitioned_data", side_effect=observed) as prepare:
            context = GlobalValidationContext.from_snapshot(path, 2, seed=17)
            with patch("src.utils.pd.read_csv", side_effect=AssertionError("reload")), \
                    patch.object(FeaturePreprocessor, "fit", side_effect=AssertionError("refit")):
                context.evaluate(weights())
                context.evaluate(weights(-1))
        self.assertEqual(prepare.call_count, 1)
        gc.collect()
        self.assertIsNone(frames[0]())
        self.assertEqual(context.cache_bytes, len(context.y_validation) * (30 + 1 + 1) * 8)
        for array in (context.X_validation, context.y_validation, context.row_ids):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        with self.assertRaises(TypeError):
            context.metadata["dataset_sha256"] = "changed"
        self.assertNotIn("frame", vars(context))

    def test_known_global_ap_without_training_smote_or_saved_model_loading(self):
        context = toy_context()
        with patch.object(LogisticRegression, "fit", side_effect=AssertionError("training")), \
                patch("imblearn.over_sampling.SMOTE.fit_resample", side_effect=AssertionError("SMOTE")), \
                patch("joblib.load", side_effect=AssertionError("model load")), \
                patch("pickle.load", side_effect=AssertionError("pickle load")), \
                patch("joblib.dump", side_effect=AssertionError("model export")):
            counts, metrics = context.evaluate(weights())
        self.assertEqual([counts[k] for k in COUNT_KEYS], [1, 0, 2, 1])
        self.assertAlmostEqual(metrics["global_average_precision"], 5 / 6)
        self.assertAlmostEqual(metrics["global_average_precision"],
                               average_precision_score(context.y_validation, expit(context.X_validation[:, 0]), pos_label=1))
        self.assertEqual(metrics["global_evaluation_mode"], MODE)
        self.assertTrue(metrics["global_average_precision_computed"])
        self.assertNotIn("scores", metrics)
        self.assertNotIn("labels", metrics)

    def test_ties_undefined_ap_and_positive_only_support(self):
        counts, metrics = toy_context(tied=True).evaluate(weights())
        self.assertEqual([counts[k] for k in COUNT_KEYS], [0, 0, 2, 2])
        self.assertEqual(metrics["global_average_precision"], 0.5)
        strategy = self.strategy(toy_context(labels=(0, 0, 0, 0)))
        _, replies = self.prepare_round(strategy)
        _, metrics = strategy.aggregate_evaluate(1, replies, [])
        self.assertFalse(metrics["global_average_precision_computed"])
        self.assertFalse(metrics["global_average_precision_defined"])
        self.assertEqual(metrics["global_average_precision_status"], "undefined_no_positive_labels")
        self.assertNotIn("global_average_precision", metrics)
        self.assertFalse(any(c.args[0] == "global_average_precision" for c in self.log.call_args_list))
        _, metrics = toy_context(labels=(1, 1, 1, 1)).evaluate(weights())
        self.assertEqual(metrics["global_average_precision"], 1)

    def test_invalid_probabilities_and_parameters_are_rejected(self):
        context = toy_context()
        for probabilities in (np.full((4, 2), np.nan), np.full((4, 2), np.inf),
                              np.array([[-0.1, 1.1]] * 4), np.array([[0.1, 0.1]] * 4), np.ones((4, 1))):
            with self.subTest(shape=probabilities.shape), patch.object(LogisticRegression, "predict_proba", return_value=probabilities), \
                    self.assertRaisesRegex(ValueError, "probabilities"):
                context.evaluate(weights())
        for arrays in ([np.zeros(30), np.zeros(1)], [np.zeros((1, 30), dtype=int), np.zeros(1)],
                       [np.full((1, 30), np.nan), np.zeros(1)], [np.zeros((1, 30)), np.array([np.inf])],
                       [np.zeros((1, 30))]):
            with self.subTest(shapes=[a.shape for a in arrays]), self.assertRaises(ValueError):
                context.evaluate(ndarrays_to_parameters(arrays))

    def test_round_parameter_snapshot_owns_immutable_packets_and_digest(self):
        packet = weights()
        snapshot = RoundParameters.capture(1, packet)
        original = snapshot.digest
        packet.tensors[0] = weights(-1).tensors[0]
        self.assertEqual(parameter_digest(snapshot.packet()), original)
        returned = snapshot.packet()
        returned.tensors.clear()
        self.assertEqual(parameter_digest(snapshot.packet()), original)
        with self.assertRaises(FrozenInstanceError):
            snapshot.round_number = 2
        with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
            replace(snapshot, digest="invalid").packet()

    def test_two_rounds_use_corresponding_aggregates_and_preserve_json_metrics(self):
        strategy = self.strategy(rounds=2)
        expected_ap = []
        for number, sign in ((1, 1), (2, -1)):
            packet, replies = self.prepare_round(strategy, number, sign)
            loss, metrics = strategy.aggregate_evaluate(number, replies, [])
            context = strategy.global_evaluator
            expected = average_precision_score(context.y_validation, expit(sign * context.X_validation[:, 0]))
            self.assertAlmostEqual(metrics["global_average_precision"], expected)
            self.assertEqual(metrics["global_evaluation_parameters_sha256"], parameter_digest(packet))
            self.assertEqual(strategy._completed_round, number)
            self.assertIsNone(strategy._pending_round)
            expected_ap.append(expected)
            self.log.assert_any_call("global_average_precision", expected, step=number, synchronous=True)
            self.tags.assert_any_call("global_evaluation_mode", MODE, synchronous=True)
        self.assertNotEqual(expected_ap[0], expected_ap[1])
        strategy.finish(SimpleNamespace(losses_distributed=[(1, loss), (2, loss)]))
        records = json.loads((self.root / "rounds.json").read_text(encoding="utf-8"))
        self.assertEqual([r["round"] for r in records], [1, 2])
        for record in records:
            metrics = record["validation_metrics"]
            self.assertEqual(record["accuracy"], metrics["accuracy"])
            self.assertEqual(record["loss"], metrics["loss"])
            self.assertNotIn("average_precision", metrics)
            self.assertIn("average_precision_defined", record["validation"][0])

    def test_client_identity_and_incomplete_participation_fail_before_ap(self):
        for field in ("dataset_sha256", "preprocessing_sha256", "validation_context_sha256",
                      "validation_membership_sha256", "global_validation_membership_sha256", "missing_client"):
            with self.subTest(field=field):
                strategy = self.strategy()
                _, replies = self.prepare_round(strategy)
                if field == "missing_client":
                    replies.pop()
                else:
                    replies[0][1].metrics[field] = "different"
                self.start.reset_mock()
                with patch.object(GlobalValidationContext, "evaluate") as evaluate, self.assertRaises(RuntimeError):
                    strategy.aggregate_evaluate(1, replies, [])
                evaluate.assert_not_called()
                self.start.assert_not_called()
                self.assert_incomplete(strategy)

    def test_confusion_mismatch_aborts_before_mlflow(self):
        strategy = self.strategy()
        _, replies = self.prepare_round(strategy)
        replies[0][1].metrics.update(evaluate_binary([0, 0], [0.9, 0.9])[1])
        with self.assertRaisesRegex(RuntimeError, "confusion counts disagree"):
            strategy.aggregate_evaluate(1, replies, [])
        self.start.assert_not_called()
        self.assert_incomplete(strategy)

    def test_central_evaluation_failure_cannot_complete_round(self):
        strategy = self.strategy()
        _, replies = self.prepare_round(strategy)
        with patch.object(GlobalValidationContext, "evaluate", side_effect=ValueError("invalid scores")), \
                self.assertRaisesRegex(ValueError, "invalid scores"):
            strategy.aggregate_evaluate(1, replies, [])
        self.start.assert_not_called()
        self.assert_incomplete(strategy)

    def test_each_required_mlflow_failure_prevents_success(self):
        for operation in (self.start, self.log, self.tags):
            with self.subTest(operation=operation):
                strategy = self.strategy()
                _, replies = self.prepare_round(strategy)
                operation.side_effect = OSError("synthetic storage failure")
                with self.assertRaisesRegex(RuntimeError, "Required MLflow persistence failed"):
                    strategy.aggregate_evaluate(1, replies, [])
                operation.side_effect = None
                self.assert_incomplete(strategy)

    def test_required_writes_are_synchronous_even_with_async_environment(self):
        strategy = self.strategy()
        _, replies = self.prepare_round(strategy)
        with patch.dict(os.environ, {"MLFLOW_ENABLE_ASYNC_LOGGING": "true"}):
            strategy.aggregate_evaluate(1, replies, [])
        self.assertTrue(self.log.call_args_list and self.tags.call_args_list)
        self.assertTrue(all(call.kwargs.get("synchronous") is True
                            for call in self.log.call_args_list + self.tags.call_args_list))

    def test_missing_stale_duplicate_and_mismatched_round_state_is_rejected(self):
        strategy = self.strategy()
        with self.assertRaisesRegex(RuntimeError, "round state"):
            strategy.aggregate_evaluate(1, [], [])
        for corrupt in (lambda s: setattr(s, "_pending_round", None),
                        lambda s: setattr(s, "_configured_round", 2),
                        lambda s: setattr(s, "aggregate_sha256", "wrong"),
                        lambda s: setattr(s, "_pending_round", replace(s._pending_round, round_number=2))):
            strategy = self.strategy()
            _, replies = self.prepare_round(strategy)
            corrupt(strategy)
            with self.assertRaisesRegex(RuntimeError, "round state"):
                strategy.aggregate_evaluate(1, replies, [])
        strategy = self.strategy()
        _, replies = self.prepare_round(strategy)
        with self.assertRaisesRegex(RuntimeError, "duplicated"):
            strategy.configure_fit(1, weights(), Mock())
        with self.assertRaisesRegex(RuntimeError, "duplicated fit"):
            strategy.aggregate_fit(1, [], [])
        strategy.aggregate_evaluate(1, replies, [])
        with self.assertRaisesRegex(RuntimeError, "round state"):
            strategy.aggregate_evaluate(1, replies, [])
        self.assertEqual(len(strategy.round_records), 1)

    def test_no_final_test_predictions_and_no_context_refit_after_partitioning(self):
        path = self.root / "synthetic.csv"
        synthetic_csv(path)
        frame = pd.read_csv(path)
        plan = build_partition_plan(frame, 2, seed=17)
        transformer = fit_partition_preprocessor(frame.drop(columns="Class"), plan)
        validation = np.sort(np.concatenate([p.validation for p in plan.clients]))
        expected = transformer.transform(frame.drop(columns="Class").iloc[validation])
        context = GlobalValidationContext.from_snapshot(path, 2, seed=17)
        original = LogisticRegression.predict_proba

        def validate_rows(model, X):
            np.testing.assert_array_equal(X, expected)
            self.assertEqual(len(X), len(validation))
            return original(model, X)

        with patch.object(LogisticRegression, "predict_proba", autospec=True, side_effect=validate_rows) as predict:
            context.evaluate(weights())
        predict.assert_called_once()
        self.assertEqual(len(np.intersect1d(context.row_ids, plan.final_test)), 0)

    def test_fit_replies_require_context_identity_too(self):
        strategy = self.strategy()
        manager = Mock()
        manager.num_available.return_value = 2
        manager.sample.return_value = [Mock(), Mock()]
        strategy.configure_fit(1, weights(), manager)
        results = [(Mock(cid=str(cid)), FitRes(OK, weights(), 12,
                    {"client_id": cid, "server_round": 1, "received_sha256": strategy.sent_sha256})) for cid in (0, 1)]
        with self.assertRaisesRegex(RuntimeError, "context identity mismatch"):
            strategy.aggregate_fit(1, results, [])
        self.assertIsNone(strategy._pending_round)

    def test_atomic_round_publication_precedes_state_and_completion_message(self):
        strategy = self.strategy(rounds=2)
        real_replace = os.replace
        real_sync = os.fsync
        previous = None
        for number, sign in ((1, 1), (2, -1)):
            _, replies = self.prepare_round(strategy, number, sign)
            output = StringIO()

            def publish(source, destination):
                self.assertEqual(Path(source).parent, strategy.report.parent)
                self.assertNotEqual(Path(source), strategy.report)
                self.assertEqual(strategy._completed_round, number - 1)
                self.assertIsNotNone(strategy._pending_round)
                self.assertNotIn("validation", strategy.round_records[-1])
                self.assertNotIn(f"Completed round {number}/", output.getvalue())
                self.assertEqual(self.start.return_value.__exit__.call_count, number)
                self.log.assert_any_call("global_average_precision_computed", 1.0,
                                         step=number, synchronous=True)
                sync.assert_called_once()
                if previous is not None:
                    self.assertEqual(strategy.report.read_bytes(), previous)
                candidate = json.loads(Path(source).read_text(encoding="utf-8"))
                self.assertEqual([r["round"] for r in candidate], list(range(1, number + 1)))
                real_replace(source, destination)
                self.assertEqual(json.loads(strategy.report.read_text(encoding="utf-8")), candidate)
                # The durable-file publication still precedes in-memory completion.
                self.assertEqual(strategy._completed_round, number - 1)

            with redirect_stdout(output), patch("src.server.os.fsync", wraps=real_sync) as sync, \
                    patch("src.server.os.replace", side_effect=publish) as replace_file:
                strategy.aggregate_evaluate(number, replies, [])
            replace_file.assert_called_once()
            self.assertIn(f"Completed round {number}/2 with all clients", output.getvalue())
            self.assertEqual(strategy._completed_round, number)
            self.assertIsNone(strategy._pending_round)
            self.assertEqual(json.loads(strategy.report.read_text(encoding="utf-8")), strategy.round_records)
            self.assertEqual(list(self.root.glob(".rounds.json.*.tmp")), [])
            previous = strategy.report.read_bytes()

    def test_partial_write_and_replace_failures_preserve_published_rounds(self):
        for operation in ("write", "replace"):
            for number in (1, 2):
                with self.subTest(operation=operation, round=number):
                    strategy = self.strategy(rounds=number)
                    strategy.report = self.root / f"{operation}-{number}.json"
                    if number == 2:
                        _, first = self.prepare_round(strategy, 1)
                        strategy.aggregate_evaluate(1, first, [])
                    previous = strategy.report.read_bytes() if strategy.report.exists() else None
                    _, replies = self.prepare_round(strategy, number, -1)
                    pending = strategy._pending_round
                    output = StringIO()

                    def partial_write(records, stream, **kwargs):
                        stream.write('[{"round":')
                        raise OSError("synthetic partial JSON write failure")

                    failure = (patch("src.server.json.dump", side_effect=partial_write) if operation == "write"
                               else patch("src.server.os.replace", side_effect=OSError("synthetic replacement failure")))
                    with failure, redirect_stdout(output), self.assertRaises(OSError):
                        strategy.aggregate_evaluate(number, replies, [])
                    self.assertEqual(strategy._completed_round, number - 1)
                    self.assertTrue(strategy._evaluation_failed)
                    self.assertIs(strategy._pending_round, pending)
                    self.assertNotIn("validation", strategy.round_records[-1])
                    self.assertNotIn("Completed round", output.getvalue())
                    self.assertEqual(list(self.root.glob(f".{strategy.report.name}.*.tmp")), [])
                    if previous is None:
                        self.assertFalse(strategy.report.exists())
                    else:
                        self.assertEqual(strategy.report.read_bytes(), previous)
                        self.assertEqual([r["round"] for r in json.loads(previous)], [1])
                    calls_before_retry = self.start.call_count
                    with self.assertRaisesRegex(RuntimeError, "failed round state"):
                        strategy.aggregate_evaluate(number, replies, [])
                    self.assertEqual(self.start.call_count, calls_before_retry)
                    with self.assertRaisesRegex(RuntimeError, "did not complete"):
                        strategy.finish(SimpleNamespace(losses_distributed=[(n, 0.1) for n in range(1, number + 1)]))

    def test_report_sync_failure_prevents_replacement_and_cleans_temporary_file(self):
        strategy = self.strategy()
        _, replies = self.prepare_round(strategy)
        output = StringIO()
        with patch("src.server.os.fsync", side_effect=OSError("synthetic sync failure")), \
                patch("src.server.os.replace") as replace_file, redirect_stdout(output), \
                self.assertRaisesRegex(OSError, "sync failure"):
            strategy.aggregate_evaluate(1, replies, [])
        replace_file.assert_not_called()
        self.assertNotIn("Completed round", output.getvalue())
        self.assertEqual(list(self.root.glob(".rounds.json.*.tmp")), [])
        self.assertTrue(strategy._evaluation_failed)
        self.assert_incomplete(strategy)

    def test_finish_only_validates_the_already_published_report(self):
        strategy = self.strategy()
        _, replies = self.prepare_round(strategy)
        loss, _ = strategy.aggregate_evaluate(1, replies, [])
        original = strategy.report.read_bytes()
        modified = strategy.report.stat().st_mtime_ns
        with patch("src.server.tempfile.NamedTemporaryFile", side_effect=AssertionError("duplicate write")), \
                patch("src.server.json.dump", side_effect=AssertionError("duplicate JSON write")), \
                patch("src.server.os.replace", side_effect=AssertionError("duplicate replacement")):
            strategy.finish(SimpleNamespace(losses_distributed=[(1, loss)]))
            strategy.finish(SimpleNamespace(losses_distributed=[(1, loss)]))
        self.assertEqual(strategy.report.read_bytes(), original)
        self.assertEqual(strategy.report.stat().st_mtime_ns, modified)

    def test_finish_rejects_missing_or_changed_published_report(self):
        for action in ("missing", "changed"):
            with self.subTest(action=action):
                strategy = self.strategy()
                strategy.report = self.root / f"{action}.json"
                _, replies = self.prepare_round(strategy)
                loss, _ = strategy.aggregate_evaluate(1, replies, [])
                if action == "missing":
                    strategy.report.unlink()
                else:
                    strategy.report.write_text("[]", encoding="utf-8")
                with self.assertRaises((OSError, RuntimeError)):
                    strategy.finish(SimpleNamespace(losses_distributed=[(1, loss)]))
                self.assertTrue(strategy._evaluation_failed)

    def test_first_publication_does_not_replace_a_preexisting_report(self):
        strategy = self.strategy()
        original = b'{"preserve":"pre-existing"}'
        strategy.report.write_bytes(original)
        _, replies = self.prepare_round(strategy)
        with self.assertRaisesRegex(FileExistsError, "pre-existing"):
            strategy.aggregate_evaluate(1, replies, [])
        self.assertTrue(strategy._evaluation_failed)
        self.assertEqual(strategy._completed_round, 0)
        self.assertEqual(strategy.report.read_bytes(), original)
        self.assertEqual(list(self.root.glob(".rounds.json.*.tmp")), [])
