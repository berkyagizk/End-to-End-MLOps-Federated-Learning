import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from app import app, load_inference_bundle
from src.preprocessing import FEATURE_NAMES, FeaturePreprocessor, load_model_preprocessor
from src.utils import build_partition_plan, fit_partition_preprocessor, load_client_data
from test_partitions import make_frame


class PreprocessingTests(unittest.TestCase):
    def test_same_preprocessing_across_client_counts(self):
        frame = make_frame()
        features = frame.drop(columns="Class")
        reference = None
        for count in (2, 5, 10):
            with self.subTest(count=count):
                plan = build_partition_plan(frame, count, seed=42)
                fitted = fit_partition_preprocessor(features, plan)
                if reference is None:
                    reference = fitted
                self.assertEqual(fitted, reference)
                with patch("src.utils.pd.read_csv", return_value=frame):
                    train, _, validation, _ = load_client_data("unused", count, count - 1)
                partition = plan.clients[-1]
                np.testing.assert_array_equal(train, reference.transform(features.iloc[partition.train]))
                np.testing.assert_array_equal(validation, reference.transform(features.iloc[partition.validation]))
                final_test = features.iloc[plan.final_test]
                np.testing.assert_array_equal(fitted.transform(final_test), reference.transform(final_test))
                np.testing.assert_array_equal(train[:, 1:29], features.iloc[partition.train, 1:29])

    def test_fit_ignores_validation_and_final_test_values(self):
        frame = make_frame()
        plan = build_partition_plan(frame, 10, seed=42)
        features = frame.drop(columns="Class")
        training_rows = np.sort(np.concatenate([client.train for client in plan.clients]))
        pooled = features.iloc[training_rows][["Amount", "Time"]]
        fitted = fit_partition_preprocessor(features, plan)
        np.testing.assert_allclose(fitted.mean, pooled.mean())
        np.testing.assert_allclose(fitted.scale, pooled.std(ddof=0))

        held_out = np.concatenate([plan.final_test, *(client.validation for client in plan.clients)])
        features.iloc[held_out, [0, 29]] = np.nan
        self.assertEqual(fit_partition_preprocessor(features, plan), fitted)

    def test_schema_and_input_validation(self):
        frame = make_frame().drop(columns="Class")
        fitted = FeaturePreprocessor.fit(frame)
        for values, error in (
            (frame[frame.columns[::-1]], "columns in order"),
            (frame.iloc[:, :-1].to_numpy(), "30 features"),
            (np.zeros((1, 31)), "30 features"),
            (np.full((1, 30), np.nan), "finite"),
            (np.full((1, 30), np.inf), "finite"),
            (np.full((1, 30), "1"), "numeric"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                fitted.transform(values)

    def test_metadata_round_trip_is_deterministic_and_bound_to_model_bytes(self):
        frame = make_frame().drop(columns="Class")
        fitted = FeaturePreprocessor.fit(frame)
        with TemporaryDirectory() as directory:
            first = Path(directory) / "one.pkl"
            second = Path(directory) / "two.pkl"
            for path in (first, second):
                path.write_bytes(b"synthetic bytes for the metadata-only test")
            sidecar = fitted.save_for_model(first)
            self.assertEqual(sidecar.read_bytes(), fitted.save_for_model(second).read_bytes())
            with self.assertRaises(FileExistsError):
                fitted.save_for_model(first)
            loaded = load_model_preprocessor(first, first.read_bytes())
            np.testing.assert_array_equal(loaded.transform(frame), fitted.transform(frame))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_model_preprocessor(first, b"different bytes")

    def test_invalid_metadata_is_rejected_before_deserialization(self):
        fitted = FeaturePreprocessor.fit(make_frame().drop(columns="Class"))
        with TemporaryDirectory() as directory, patch("app.joblib.load") as deserialize:
            model_path = Path(directory) / "model.pkl"
            model_path.write_bytes(b"synthetic bytes")
            with self.assertRaisesRegex(ValueError, "Cannot read preprocessing metadata"):
                load_inference_bundle(model_path)
            sidecar = fitted.save_for_model(model_path)
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            for changes, error in (
                ({"schema_version": 2}, "schema version"),
                ({"feature_names": list(reversed(FEATURE_NAMES))}, "feature names or order"),
                ({"scaled_features": ["Time", "Amount"]}, "feature names or order"),
                ({"scale": [0, 1]}, "Invalid preprocessing scale"),
                ({"mean": [float("inf"), 0]}, "Invalid preprocessing mean"),
                ({"model_sha256": "0" * 64}, "SHA-256"),
            ):
                with self.subTest(error=error):
                    sidecar.write_text(json.dumps({**metadata, **changes}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        load_inference_bundle(model_path)
            del metadata["mean"]
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                load_inference_bundle(model_path)
            deserialize.assert_not_called()

    def test_estimator_schema_must_match_the_bundle(self):
        fitted = FeaturePreprocessor.fit(make_frame().drop(columns="Class"))
        with TemporaryDirectory() as directory, patch("app.joblib.load") as deserialize:
            path = Path(directory) / "schema-check.pkl"
            path.write_bytes(b"synthetic bytes")
            fitted.save_for_model(path)
            deserialize.return_value.n_features_in_ = 29
            with self.assertRaisesRegex(ValueError, "30 features"):
                load_inference_bundle(path)
            deserialize.return_value.n_features_in_ = 30
            deserialize.return_value.feature_names_in_ = FEATURE_NAMES[::-1]
            with self.assertRaisesRegex(ValueError, "Model feature names or order"):
                load_inference_bundle(path)


async def post_prediction(features):
    """Send an in-process ASGI request using the installed stack."""
    messages = []

    async def receive():
        return {"type": "http.request", "body": json.dumps({"features": features}).encode()}

    async def send(message):
        messages.append(message)

    await app({
        "type": "http", "method": "POST", "path": "/predict", "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
    }, receive, send)
    return messages[0]["status"], json.loads(messages[-1]["body"])


class InferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_unconfigured_api_does_not_load_the_legacy_model(self):
        with patch.dict(os.environ, {"FRAUD_MODEL_PATH": ""}), patch("app.joblib.load") as load:
            async with app.router.lifespan_context(app):
                status, _ = await post_prediction([0.0] * 30)
            self.assertEqual(status, 503)
            load.assert_not_called()

    async def test_prediction_without_metadata_returns_unavailable(self):
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "missing-sidecar.pkl"
            model_path.write_bytes(b"synthetic bytes")
            with patch.dict(os.environ, {"FRAUD_MODEL_PATH": str(model_path)}), patch("app.joblib.load") as load:
                with self.assertLogs("app", level="ERROR"):
                    async with app.router.lifespan_context(app):
                        status, result = await post_prediction([0.0] * 30)
                load.assert_not_called()
            self.assertEqual(status, 503)
            self.assertIn("preprocessing metadata", result["detail"])

    async def test_fastapi_uses_exact_training_transform_from_trusted_bundle(self):
        frame = make_frame()
        features = frame.drop(columns="Class")
        plan = build_partition_plan(frame, 5)
        fitted = fit_partition_preprocessor(features, plan)
        with patch("src.utils.pd.read_csv", return_value=frame):
            train, labels, _, _ = load_client_data("unused", 5, 0)
        model = DecisionTreeClassifier(max_depth=1, random_state=42).fit(
            pd.DataFrame(train, columns=FEATURE_NAMES), labels,
        )
        raw_row = features.iloc[plan.clients[0].train[0]].to_list()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trusted-synthetic.pkl"
            joblib.dump(model, path)
            fitted.save_for_model(path)
            with patch.dict(os.environ, {"FRAUD_MODEL_PATH": str(path)}):
                async with app.router.lifespan_context(app):
                    serving_model, _ = app.state.bundle
                    with patch.object(serving_model, "predict", wraps=serving_model.predict) as predict:
                        status, result = await post_prediction(raw_row)
                        self.assertEqual(status, 200)
                        self.assertIn(result["prediction"], (0, 1))
                        actual = predict.call_args.args[0]
                        self.assertEqual(tuple(actual.columns), FEATURE_NAMES)
                        np.testing.assert_array_equal(actual.to_numpy(), train[:1])
                        for invalid in (raw_row[:-1], [float("inf"), *raw_row[1:]], ["1", *raw_row[1:]]):
                            self.assertEqual((await post_prediction(invalid))[0], 422)
                        self.assertEqual(predict.call_count, 1)
