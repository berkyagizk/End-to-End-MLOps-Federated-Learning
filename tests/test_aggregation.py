from dataclasses import replace
import unittest
from unittest.mock import Mock, patch
import warnings

from flwr.common import (
    Code, EvaluateIns, EvaluateRes, FitIns, FitRes, Status,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from src.client import FraudClient
from src.preprocessing import FEATURE_NAMES
from src.server import build_strategy


OK = Status(Code.OK, "")
N_FEATURES = len(FEATURE_NAMES)


def weights(coef, intercept):
    return ndarrays_to_parameters([np.full((1, N_FEATURES), coef, dtype=np.float64),
                                   np.array([intercept], dtype=np.float64)])


def proxy(cid):
    return Mock(cid=cid)


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.strategy = build_strategy(2)
        self.start_run = self.enterContext(patch("src.server.mlflow.start_run"))
        self.log_metric = self.enterContext(patch("src.server.mlflow.log_metric"))
        self.enterContext(patch("src.client.joblib.dump"))

    def assert_weights(self, aggregated, coef, intercept):
        arrays = parameters_to_ndarrays(aggregated)
        np.testing.assert_allclose(arrays[0], np.full((1, N_FEATURES), coef))
        np.testing.assert_allclose(arrays[1], [intercept])
        self.assertTrue(all(array.dtype == np.float64 and np.isfinite(array).all() for array in arrays))

    def test_fit_uses_flower_weighted_average_for_unequal_original_counts(self):
        results = [(proxy("a"), FitRes(OK, weights(2, -1), 20, {"post_smote_examples": 28})),
                   (proxy("b"), FitRes(OK, weights(8, 3), 60, {"post_smote_examples": 108}))]
        aggregated, _ = self.strategy.aggregate_fit(2, results, [])
        self.assert_weights(aggregated, 6.5, 2)

    def test_actual_fit_replies_keep_equal_weights_despite_different_smote_sizes(self):
        rng = np.random.default_rng(19)
        X = rng.normal(size=(20, N_FEATURES))
        results = []
        for client_id, positives in ((1, 6), (2, 10)):
            y = np.r_[np.zeros(20 - positives, dtype=int), np.ones(positives, dtype=int)]
            client = FraudClient(client_id, X, y, X[:2], np.array([0, 1])).to_client()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                result = client.fit(FitIns(weights(0, 0), {}))
            self.assertEqual(result.status.code, Code.OK)
            self.assertEqual(result.num_examples, 20)
            self.assertEqual(result.metrics, {"post_smote_examples": 2 * (20 - positives)})
            # Keep the actual fit counts and diagnostics with hand-calculable weights.
            results.append((proxy(str(client_id)), replace(result, parameters=weights(2 if client_id == 1 else 8, 0))))
        aggregated, _ = self.strategy.aggregate_fit(2, results, [])
        self.assert_weights(aggregated, 5, 0)

    def test_validation_weighting_matches_pooled_accuracy_and_logs_it(self):
        results = [(proxy("a"), EvaluateRes(OK, 0.5, 10, {"accuracy": 0.9})),
                   (proxy("b"), EvaluateRes(OK, 0.2, 30, {"accuracy": 0.5}))]
        loss, metrics = self.strategy.aggregate_evaluate(2, results, [])
        self.assertAlmostEqual(loss, 0.275)
        self.assertAlmostEqual(metrics["accuracy"], 24 / 40)
        self.start_run.assert_called_once_with(run_name="Round_2")
        self.log_metric.assert_any_call("accuracy", 0.6, step=2)
        self.log_metric.assert_any_call("loss", loss, step=2)

    def test_actual_evaluate_reply_counts_validation_rows(self):
        X = np.zeros((20, N_FEATURES))
        client = FraudClient(1, X, np.arange(20) % 2, X[:7], np.arange(7) % 2).to_client()
        result = client.evaluate(EvaluateIns(weights(0, -1), {}))
        self.assertEqual(result.status.code, Code.OK)
        self.assertEqual(result.num_examples, 7)
        self.assertAlmostEqual(result.metrics["accuracy"], 4 / 7)

    def test_invalid_training_counts_and_resampling_diagnostics(self):
        client = proxy("a")
        for count in (0, -1, True, 2.5, 2**63):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "num_examples.*positive"):
                self.strategy.aggregate_fit(2, [(client, FitRes(OK, weights(0, 0), count, {}))], [])
        for post_smote in (0, 18, 27, 40, 42, "28"):
            with self.subTest(post_smote=post_smote), self.assertRaisesRegex(ValueError, "post_smote_examples"):
                result = FitRes(OK, weights(0, 0), 20, {"post_smote_examples": post_smote})
                self.strategy.aggregate_fit(2, [(client, result)], [])

    def test_fit_rejects_broadcastable_shapes_nonfinite_values_and_wrong_dtype(self):
        good = (proxy("good"), FitRes(OK, weights(0, 0), 20, {}))
        for arrays in ([np.ones((1, 1)), np.zeros(1)],
                       [np.ones((1, N_FEATURES)), np.zeros((1, 1))],
                       [np.ones((1, N_FEATURES))],
                       [np.full((1, N_FEATURES), np.nan), np.zeros(1)],
                       [np.ones((1, N_FEATURES), dtype=int), np.zeros(1)]):
            with self.subTest(shapes=[a.shape for a in arrays]), self.assertRaisesRegex(ValueError, "client bad"):
                bad = (proxy("bad"), FitRes(OK, ndarrays_to_parameters(arrays), 20, {}))
                self.strategy.aggregate_fit(2, [good, bad], [])

    def test_validation_rejects_invalid_counts_losses_and_metrics_without_logging(self):
        client = proxy("a")
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "num_examples.*positive"):
                self.strategy.aggregate_evaluate(2, [(client, EvaluateRes(OK, 0.2, count, {}))], [])
        for metrics in ({}, {"accuracy": "0.5"}, {"accuracy": True}, {"accuracy": float("nan")},
                        {"accuracy": float("inf")}, {"accuracy": 1.01}, {"accuracy": -0.01}):
            with self.subTest(metrics=metrics), self.assertRaisesRegex(ValueError, "numeric accuracy"):
                self.strategy.aggregate_evaluate(2, [(client, EvaluateRes(OK, 0.2, 10, metrics))], [])
        for loss in (float("inf"), float("nan"), -0.1):
            with self.subTest(loss=loss), self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                self.strategy.aggregate_evaluate(2, [(client, EvaluateRes(OK, loss, 10, {"accuracy": 0.5}))], [])
        self.start_run.assert_not_called()

    def test_missing_results_and_partial_failures_keep_flower_policy(self):
        failed = [RuntimeError("client unavailable")]
        self.assertEqual(self.strategy.num_fit_clients(1), (2, 2))
        self.assertEqual(self.strategy.num_evaluation_clients(3), (3, 2))
        self.assertEqual(self.strategy.aggregate_fit(2, [], failed), (None, {}))
        self.assertEqual(self.strategy.aggregate_evaluate(2, [], failed), (None, {}))
        self.start_run.assert_not_called()

        fit = [(proxy("a"), FitRes(OK, weights(3, 1), 20, {}))]
        evaluation = [(proxy("a"), EvaluateRes(OK, 0.5, 10, {"accuracy": 0.9}))]
        aggregated, _ = self.strategy.aggregate_fit(2, fit, failed)
        self.assert_weights(aggregated, 3, 1)
        self.assertEqual(self.strategy.aggregate_evaluate(2, evaluation, failed), (0.5, {"accuracy": 0.9}))

        self.strategy.accept_failures = False
        self.assertEqual(self.strategy.aggregate_fit(2, fit, failed), (None, {}))
        self.assertEqual(self.strategy.aggregate_evaluate(2, evaluation, failed), (None, {}))
