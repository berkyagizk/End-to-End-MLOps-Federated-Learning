from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from flwr.common import Code, EvaluateIns, EvaluateRes, Status, ndarrays_to_parameters
import numpy as np
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, log_loss, precision_recall_curve, precision_score,
    recall_score,
)

from src.client import FraudClient
from src.evaluation import COUNT_KEYS, aggregate_validation_metrics, confusion_metrics, evaluate_binary
from src.preprocessing import FEATURE_NAMES
from src.server import ControlledStrategy, build_strategy


OK = Status(Code.OK, "")


def unequal_clients():
    return ((np.array([1, 0]), np.array([0.9, 0.8])),
            (np.array([1, 1, 1, 1, 0, 0, 0, 0, 0]),
             np.array([0.9, 0.4, 0.3, 0.2, 0.1, 0.08, 0.07, 0.06, 0.05])))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(redirect_stdout(StringIO()))

    def test_manual_confusion_matrix_and_sklearn_reference(self):
        y = np.array([1, 1, 1, 0, 0, 0, 0])
        p = np.array([0.9, 0.8, 0.2, 0.7, 0.3, 0.1, 0.4])
        loss, metrics = evaluate_binary(y, p)
        self.assertEqual([metrics[k] for k in COUNT_KEYS], [2, 1, 3, 1])
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertAlmostEqual(metrics["specificity"], 3 / 4)
        self.assertAlmostEqual(metrics["balanced_accuracy"], (2 / 3 + 3 / 4) / 2)
        predicted = p > 0.5
        np.testing.assert_array_equal(confusion_matrix(y, predicted, labels=[0, 1]), [[3, 1], [1, 2]])
        for key, expected in (
            ("accuracy", accuracy_score(y, predicted)),
            ("precision", precision_score(y, predicted, pos_label=1, zero_division=0)),
            ("recall", recall_score(y, predicted, pos_label=1, zero_division=0)),
            ("f1", f1_score(y, predicted, pos_label=1, zero_division=0)),
            ("balanced_accuracy", balanced_accuracy_score(y, predicted)),
            ("average_precision", average_precision_score(y, p, pos_label=1)),
        ):
            self.assertAlmostEqual(metrics[key], expected)
        self.assertAlmostEqual(loss, log_loss(y, p, labels=[0, 1]))

    def test_average_precision_known_reference_is_not_trapezoidal_area(self):
        y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])
        _, metrics = evaluate_binary(y, p)
        precision, recall, _ = precision_recall_curve(y, p, pos_label=1)
        self.assertAlmostEqual(metrics["average_precision"], 5 / 6)
        self.assertNotAlmostEqual(metrics["average_precision"], auc(recall, precision))

    def test_no_predicted_frauds_and_probability_ties(self):
        _, metrics = evaluate_binary([0, 1, 0, 1], [0.5, 0.5, 0.1, 0.2])
        self.assertEqual([metrics[k] for k in COUNT_KEYS], [0, 0, 2, 2])
        self.assertEqual(metrics["precision"], 0)
        self.assertFalse(metrics["precision_defined"])
        self.assertEqual(metrics["recall"], 0)
        self.assertTrue(metrics["recall_defined"])
        self.assertEqual(metrics["f1"], 0)
        self.assertTrue(metrics["f1_defined"])
        self.assertEqual(metrics["specificity"], 1)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertTrue(metrics["average_precision_defined"])

    def test_no_positive_labels_omits_undefined_ap_and_balanced_accuracy(self):
        loss, metrics = evaluate_binary([0, 0, 0], [0.1, 0.9, 0.2])
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["recall"], 0)
        self.assertFalse(metrics["recall_defined"])
        self.assertFalse(metrics["average_precision_defined"])
        self.assertNotIn("average_precision", metrics)
        self.assertFalse(metrics["balanced_accuracy_defined"])
        self.assertNotIn("balanced_accuracy", metrics)
        _, none_predicted = evaluate_binary([0, 0], [0.1, 0.2])
        self.assertFalse(none_predicted["f1_defined"])
        self.assertEqual(none_predicted["f1"], 0)

    def test_only_positive_labels_and_extreme_probabilities(self):
        loss, metrics = evaluate_binary([1, 1], [0, 1])
        self.assertAlmostEqual(loss, log_loss([1, 1], [0, 1], labels=[0, 1]))
        self.assertEqual(metrics["average_precision"], 1)
        self.assertEqual(metrics["specificity"], 0)
        self.assertFalse(metrics["specificity_defined"])
        self.assertFalse(metrics["balanced_accuracy_defined"])

    def test_invalid_labels_and_probabilities_fail_clearly(self):
        for labels in ([], [[0, 1]], [0, 2], [0, np.nan], [0, np.inf], ["0", "1"]):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, "Validation labels"):
                evaluate_binary(labels, [0.1, 0.9])
        for probabilities in ([0.1], [[0.1, 0.9]], [np.nan, 0.1], [np.inf, 0.1],
                              [-0.1, 0.9], [0.1, 1.1], ["0.1", "0.9"], [0.1j, 0.9], [False, True]):
            with self.subTest(probabilities=probabilities), self.assertRaisesRegex(ValueError, "Fraud probabilities"):
                evaluate_binary([0, 1], probabilities)

    def test_pooled_unequal_clients_match_concatenated_reference_not_mean_f1(self):
        clients = unequal_clients()
        replies = [(len(y), evaluate_binary(y, p)[1]) for y, p in clients]
        metrics = aggregate_validation_metrics(replies)
        self.assertEqual([metrics[k] for k in COUNT_KEYS], [2, 1, 5, 3])
        self.assertEqual(metrics["validation_examples"], 11)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 5)
        self.assertEqual(metrics["f1"], 0.5)
        y = np.concatenate([y for y, _ in clients])
        p = np.concatenate([p for _, p in clients])
        _, reference = evaluate_binary(y, p)
        for key in ("accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy"):
            self.assertAlmostEqual(metrics[key], reference[key])
        self.assertNotAlmostEqual(metrics["f1"], np.mean([m["f1"] for _, m in replies]))
        self.assertNotAlmostEqual(metrics["f1"], sum(n * m["f1"] for n, m in replies) / 11)
        # Both local APs are 1, but inter-client score ordering makes pooled AP lower.
        self.assertTrue(all(m["average_precision"] == 1 for _, m in replies))
        self.assertLess(reference["average_precision"], 1)
        self.assertNotIn("average_precision", metrics)
        self.assertFalse(metrics["global_average_precision_computed"])
        self.assertEqual(metrics["global_average_precision_status"], "not_computed_requires_pooled_scores")

    def test_clients_missing_classes_can_have_a_valid_pooled_balanced_accuracy(self):
        results = [(2, evaluate_binary([0, 0], [0.1, 0.7])[1]),
                   (3, evaluate_binary([1, 1, 1], [0.9, 0.8, 0.2])[1])]
        metrics = aggregate_validation_metrics(results)
        self.assertTrue(metrics["balanced_accuracy_defined"])
        self.assertAlmostEqual(metrics["balanced_accuracy"], (2 / 3 + 1 / 2) / 2)

    def test_malformed_counts_rates_and_ap_are_rejected(self):
        _, valid = evaluate_binary([0, 1], [0.1, 0.9])
        for count, changes in (
            (0, {}), (True, {}), (2.5, {}), (3, {}),
            (2, {"true_positives": -1}), (2, {"false_positives": True}),
            (2, {"threshold": 0.6}), (2, {"threshold": True}),
            (2, {"accuracy": np.nan}), (2, {"accuracy": 0.5}),
            (2, {"recall": 0.1}), (2, {"precision_defined": 1}),
            (2, {"average_precision": np.nan}), (2, {"average_precision": 1.1}),
            (2, {"average_precision": 2**10000}),
            (2, {"average_precision_defined": False}),
        ):
            with self.subTest(count=count, changes=changes), self.assertRaises(ValueError):
                aggregate_validation_metrics([(count, {**valid, **changes})])
        for missing in (*COUNT_KEYS, "average_precision", "average_precision_defined", "f1"):
            incomplete = {k: v for k, v in valid.items() if k != missing}
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                aggregate_validation_metrics([(2, incomplete)])
        for counts in ((0, 0, 0, 0), (-1, 0, 1, 0), (True, 0, 1, 0), (2**63 - 1, 1, 0, 0)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                confusion_metrics(*counts)
        self.assertEqual(aggregate_validation_metrics([]), {})

    def test_undefined_metrics_cannot_be_silently_reported_as_defined(self):
        _, valid = evaluate_binary([0, 0], [0.1, 0.2])
        for extra in ({"average_precision": 0.0}, {"balanced_accuracy": 0.0},
                      {"average_precision_defined": True}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                aggregate_validation_metrics([(2, {**valid, **extra})])

    def test_actual_flower_client_preserves_accuracy_loss_and_fraud_column(self):
        X = np.zeros((4, len(FEATURE_NAMES)))
        y = np.array([0, 0, 0, 1])
        client = FraudClient(1, X, y, X, y)
        weights = ndarrays_to_parameters(client.get_parameters({}))
        result = client.to_client().evaluate(EvaluateIns(weights, {}))
        self.assertEqual(result.num_examples, 4)
        self.assertEqual(result.metrics["accuracy"], client.model.score(X, y))
        self.assertEqual(result.metrics["accuracy"], 0.75)
        self.assertAlmostEqual(result.loss, log_loss(y, client.model.predict_proba(X)))
        self.assertEqual(result.metrics["average_precision"], 0.25)
        for values in (np.array([[0.2, 0.9]] * 4), np.array([[np.nan, 0.5]] * 4), np.ones((4, 1))):
            with patch.object(client.model, "predict_proba", return_value=values), \
                    self.assertRaisesRegex(ValueError, "binary probabilities"):
                client.evaluate(client.get_parameters({}), {})
        client.model.classes_ = np.array([1, 0])
        with self.assertRaisesRegex(ValueError, "class order"):
            client.evaluate(client.get_parameters({}), {})

    def test_flower_strategy_logs_pooled_rates_and_separate_client_ap(self):
        with TemporaryDirectory() as directory:
            report = Path(directory) / "rounds.json"
            strategy = ControlledStrategy(2, 1, report)
            strategy.aggregate_sha256 = "expected"
            strategy.round_records = [{"round": 1}]
            results = []
            for client_id, (y, p) in enumerate(unequal_clients()):
                loss, metrics = evaluate_binary(y, p)
                metrics.update(client_id=client_id, server_round=1, received_sha256="expected")
                results.append((Mock(cid=str(client_id)), EvaluateRes(OK, loss, len(y), metrics)))
            with patch("src.server.mlflow.start_run") as start, patch("src.server.mlflow.log_metric") as log:
                loss, metrics = strategy.aggregate_evaluate(1, results, [])
            self.assertAlmostEqual(loss, sum(r.num_examples * r.loss for _, r in results) / 11)
            self.assertEqual(metrics["f1"], 0.5)
            start.assert_called_once_with(run_name="Round_1")
            log.assert_any_call("f1", 0.5, step=1)
            log.assert_any_call("true_positives", 2.0, step=1)
            log.assert_any_call("global_average_precision_computed", 0.0, step=1)
            for client_id in (0, 1):
                log.assert_any_call(f"client_{client_id}_average_precision", 1.0, step=1)
            self.assertFalse(any(c.args[0] == "average_precision" for c in log.call_args_list))
            strategy.finish(SimpleNamespace(losses_distributed=[(1, loss)]))
            saved = json.loads(report.read_text(encoding="utf-8"))[0]
            self.assertEqual(saved["accuracy"], metrics["accuracy"])
            self.assertEqual(saved["loss"], loss)
            self.assertEqual(saved["validation_metrics"]["f1"], 0.5)
            self.assertEqual(len(saved["validation"]), 2)

    def test_controlled_strategy_rejects_legacy_or_partial_diagnostics_before_logging(self):
        strategy = ControlledStrategy(2, 1, None)
        strategy.aggregate_sha256 = "expected"
        strategy.round_records = [{"round": 1}]
        results = [(Mock(), EvaluateRes(OK, 0.5, 2, {"accuracy": 0.5, "client_id": cid,
                    "server_round": 1, "received_sha256": "expected"})) for cid in (0, 1)]
        with patch("src.server.mlflow.start_run") as start, self.assertRaisesRegex(ValueError, "four confusion"):
            strategy.aggregate_evaluate(1, results, [])
        start.assert_not_called()
        # Even the legacy-compatible helper refuses mixed new/old replies.
        results[0][1].metrics.update(evaluate_binary([0, 1], [0.1, 0.9])[1])
        with patch("src.server.mlflow.start_run") as start, self.assertRaises(ValueError):
            build_strategy(2).aggregate_evaluate(1, results, [])
        start.assert_not_called()
        for diagnostic in ({"specificity": 1.0}, {"balanced_accuracy_defined": False}):
            legacy = EvaluateRes(OK, 0.5, 2, {"accuracy": 0.5, **diagnostic})
            with self.subTest(diagnostic=diagnostic), self.assertRaises(ValueError):
                build_strategy(2).aggregate_evaluate(1, [(Mock(), legacy)], [])
