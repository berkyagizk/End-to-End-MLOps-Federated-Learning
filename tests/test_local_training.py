from contextlib import redirect_stdout
from io import StringIO
import unittest
import warnings
from unittest.mock import patch

from flwr.common import (
    Code, EvaluateIns, FitIns, GetParametersIns,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from imblearn.over_sampling import SMOTE
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import _logistic
from sklearn.metrics import log_loss

from src.client import FraudClient


def parameters(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return [vector[:-1].reshape(1, -1), vector[-1:]]


def flatten(weights):
    return np.concatenate([weights[0].ravel(), weights[1]])


def reference_update(initial, X, y):
    """One L-BFGS-B iteration on the binary L2-regularized objective with C=1."""
    def loss_and_gradient(weights):
        scores = X @ weights[:-1] + weights[-1]
        error = expit(scores) - y
        loss = np.mean(np.logaddexp(0, scores) - y * scores)
        loss += np.dot(weights[:-1], weights[:-1]) / (2 * len(y))
        gradient = (X.T @ error + weights[:-1]) / len(y)
        return loss, np.r_[gradient, error.mean()]

    return minimize(
        loss_and_gradient, initial, method="L-BFGS-B", jac=True,
        options={"maxiter": 1, "maxls": 50, "gtol": 1e-4, "ftol": 64 * np.finfo(float).eps},
    ).x


class LocalTrainingTests(unittest.TestCase):
    def setUp(self):
        # The existing client messages include Turkish characters, which some
        # Windows console encodings cannot represent.
        self.enterContext(redirect_stdout(StringIO()))
        rng = np.random.default_rng(21)
        self.y = np.r_[np.zeros(24, dtype=int), np.ones(8, dtype=int)]
        self.X = rng.normal(size=(len(self.y), 4)) + self.y[:, None] * 0.8
        self.X_val = self.X[[0, 1, 24, 25]]
        self.y_val = self.y[[0, 1, 24, 25]]

    def client(self, client_id=1, y=None):
        return FraudClient(client_id, self.X, self.y if y is None else y, self.X_val, self.y_val)

    def fit_with_observed_start(self, flower_client, incoming):
        starts = []

        def observe(objective, initial, *args, **kwargs):
            starts.append(initial.copy())
            return minimize(objective, initial, *args, **kwargs)

        with patch.object(_logistic.optimize, "minimize", side_effect=observe):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                result = flower_client.fit(FitIns(ndarrays_to_parameters(incoming), {}))
        self.assertEqual(result.status.code, Code.OK)
        self.assertEqual(len(starts), 1)
        self.assertEqual(result.num_examples, 48)
        self.assertEqual(result.metrics, {})
        np.testing.assert_array_equal(starts[0], flatten(incoming))
        updated = parameters_to_ndarrays(result.parameters)
        self.assertEqual([array.shape for array in updated], [(1, 4), (1,)])
        self.assertTrue(all(array.dtype == np.float64 and np.isfinite(array).all() for array in updated))
        return flatten(updated)

    def test_same_zero_startup_parameters_and_binary_order_for_each_client(self):
        for client_id in (1, 2):
            client = self.client(client_id)
            result = client.to_client().get_parameters(GetParametersIns({}))
            self.assertEqual(result.status.code, Code.OK)
            initial = parameters_to_ndarrays(result.parameters)
            self.assertEqual([array.shape for array in initial], [(1, 4), (1,)])
            np.testing.assert_array_equal(flatten(initial), np.zeros(5))
            np.testing.assert_array_equal(client.model.classes_, [0, 1])
            np.testing.assert_array_equal(client.model.predict_proba(self.X_val), np.full((4, 2), 0.5))

    def test_different_global_weights_change_local_update_and_match_reference(self):
        X_balanced, y_balanced = SMOTE(random_state=42).fit_resample(self.X, self.y)
        starts = (np.zeros(5), np.array([1.5, -2.0, 0.4, 0.2, -1.25]))
        results = []
        for start in starts:
            client = self.client()
            incoming = parameters(start)
            result = self.fit_with_observed_start(client.to_client(), incoming)
            np.testing.assert_allclose(result, reference_update(start, X_balanced, y_balanced), atol=1e-12, rtol=1e-12)
            np.testing.assert_array_equal(flatten(incoming), start)
            np.testing.assert_array_equal(client.model.classes_, [0, 1])
            self.assertEqual(client.model.n_iter_[0], 1)
            results.append(result)
        self.assertGreater(np.max(np.abs(results[0] - results[1])), 0.1)

    def test_second_round_uses_new_server_weights_on_same_client(self):
        client = self.client().to_client()
        first = self.fit_with_observed_start(client, parameters(np.zeros(5)))
        second_start = np.array([-1.0, 0.7, 0.2, 1.1, -0.4])
        self.assertFalse(np.allclose(first, second_start))
        second = self.fit_with_observed_start(client, parameters(second_start))
        fresh = self.fit_with_observed_start(self.client().to_client(), parameters(second_start))
        np.testing.assert_array_equal(second, fresh)

    def test_invalid_parameters_are_rejected_without_partial_updates(self):
        client = self.client()
        valid = parameters([1, 2, 3, 4, 5])
        client.set_parameters(valid)
        for invalid, message in (
            ([], "two parameter arrays"),
            ([valid[0], valid[1], valid[1]], "two parameter arrays"),
            ([[[0.0] * 4], valid[1]], "floating-point NumPy array"),
            ([np.zeros((1, 4), dtype=complex), valid[1]], "floating-point NumPy array"),
            ([valid[0], np.array([0])], "floating-point NumPy array"),
            ([np.zeros(4), valid[1]], "Coefficients must have shape"),
            ([valid[0], np.zeros((1, 1))], "Intercept must have shape"),
            ([np.full((1, 4), np.nan), valid[1]], "finite float64"),
            ([np.zeros((1, 4)), np.array([np.inf])], "finite float64"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                client.set_parameters(invalid)
            np.testing.assert_array_equal(flatten(client.get_parameters({})), flatten(valid))

    def test_parameters_are_copied_and_float32_is_normalized(self):
        client = self.client()
        incoming = [np.ones((1, 4), dtype=np.float32), np.ones(1, dtype=np.float32)]
        client.set_parameters(incoming)
        incoming[0][:] = 5
        exported = client.get_parameters({})
        self.assertTrue(all(array.dtype == np.float64 for array in exported))
        exported[1][:] = 7
        np.testing.assert_array_equal(flatten(client.get_parameters({})), np.ones(5))

    def test_training_requires_the_two_expected_labels(self):
        for labels in (np.zeros_like(self.y), self.y + 1):
            with self.assertRaisesRegex(ValueError, "both classes 0 and 1 only"):
                self.client(y=labels)

    def test_evaluate_uses_received_parameters_and_existing_save_callback(self):
        client = self.client(client_id=0)
        incoming = parameters([0.5, -0.25, 0.9, 1.2, -0.6])
        prob_fraud = expit(self.X_val @ incoming[0][0] + incoming[1][0])
        with patch("src.client.joblib.dump") as save:
            result = client.to_client().evaluate(EvaluateIns(ndarrays_to_parameters(incoming), {}))
        self.assertEqual(result.status.code, Code.OK)
        self.assertEqual(result.num_examples, len(self.y_val))
        self.assertAlmostEqual(result.loss, log_loss(self.y_val, prob_fraud))
        self.assertEqual(result.metrics["accuracy"], np.mean((prob_fraud > 0.5) == self.y_val))
        np.testing.assert_array_equal(flatten(client.get_parameters({})), flatten(incoming))
        save.assert_called_once_with(client.model, "best_model.pkl")
