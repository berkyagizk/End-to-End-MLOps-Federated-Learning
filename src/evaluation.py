"""Validation-only binary diagnostics; no pooled score or label transport."""

import math

import numpy as np
from sklearn.metrics import average_precision_score, log_loss


THRESHOLD = 0.5
COUNT_KEYS = ("true_positives", "false_positives", "true_negatives", "false_negatives")


def _count(value, name, *, allow_zero=False):
    minimum = 0 if allow_zero else 1
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} signed 64-bit integer")


def _unit_interval(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not 0 <= value <= 1 or not math.isfinite(value)):
        raise ValueError(f"{name} must be finite and numeric in [0, 1]")


def confusion_metrics(true_positives, false_positives, true_negatives, false_negatives):
    """Derive fraud-positive rates from additive counts, with explicit support flags."""
    counts = dict(zip(COUNT_KEYS, (true_positives, false_positives, true_negatives, false_negatives), strict=True))
    for key, value in counts.items():
        _count(value, key, allow_zero=True)
    tp, fp, tn, fn = (counts[key] for key in COUNT_KEYS)
    total = tp + fp + tn + fn
    _count(total, "Validation num_examples")
    metrics = {**counts, "validation_examples": total, "threshold": THRESHOLD,
               "accuracy": (tp + tn) / total}
    for name, numerator, denominator in (
        ("precision", tp, tp + fp),
        ("recall", tp, tp + fn),
        ("f1", 2 * tp, 2 * tp + fp + fn),
        ("specificity", tn, tn + fp),
    ):
        metrics[name] = numerator / denominator if denominator else 0.0
        metrics[f"{name}_defined"] = denominator > 0
    both_classes = tp + fn > 0 and tn + fp > 0
    metrics["balanced_accuracy_defined"] = both_classes
    if both_classes:
        metrics["balanced_accuracy"] = (metrics["recall"] + metrics["specificity"]) / 2
    return metrics


def evaluate_binary(labels, fraud_probabilities):
    """Return log loss and local diagnostics for P(Class=1), predicting fraud at p > 0.5.

    AP is sklearn's non-interpolated Average Precision, not trapezoidal PR-AUC.
    With no positive labels, AP is omitted and its support flag is false.
    """
    labels = np.asarray(labels)
    scores = np.asarray(fraud_probabilities)
    if (labels.ndim != 1 or not labels.size or labels.dtype.kind not in "biuf"
            or not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all()):
        raise ValueError("Validation labels must be a nonempty vector of binary values 0 and 1")
    if (scores.shape != labels.shape or scores.dtype.kind not in "iuf"
            or not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all()):
        raise ValueError("Fraud probabilities must match labels and be finite numeric values in [0, 1]")
    scores = scores.astype(np.float64, copy=False)
    positive = labels == 1
    predicted = scores > THRESHOLD
    metrics = confusion_metrics(
        int(np.count_nonzero(positive & predicted)),
        int(np.count_nonzero(~positive & predicted)),
        int(np.count_nonzero(~positive & ~predicted)),
        int(np.count_nonzero(positive & ~predicted)),
    )
    loss = float(log_loss(labels, np.column_stack((1 - scores, scores)), labels=[0, 1]))
    if not math.isfinite(loss) or loss < 0:
        raise ValueError("Validation log loss must be finite and nonnegative")
    metrics["average_precision_defined"] = bool(positive.any())
    if metrics["average_precision_defined"]:
        ap = float(average_precision_score(labels, scores, pos_label=1))
        _unit_interval(ap, "Average Precision")
        metrics["average_precision"] = ap
    return loss, metrics


def aggregate_validation_metrics(results):
    """Pool confusion counts; deliberately do not average client AP or F1."""
    if not results:
        return {}
    pooled = {key: 0 for key in COUNT_KEYS}
    for count, metrics in results:
        _count(count, "Validation num_examples")
        if not isinstance(metrics, dict):
            raise ValueError("Validation metrics must be a dictionary")
        if not all(key in metrics for key in COUNT_KEYS):
            raise ValueError("Validation metrics must include all four confusion counts")
        expected = confusion_metrics(*(metrics[key] for key in COUNT_KEYS))
        if expected["validation_examples"] != count:
            raise ValueError("Confusion counts must sum to validation num_examples")
        if type(metrics.get("threshold")) is not float or metrics["threshold"] != THRESHOLD:
            raise ValueError("Validation threshold must be the fixed float 0.5")
        for key, value in expected.items():
            if key not in metrics:
                raise ValueError(f"Validation metrics are missing {key}")
            reported = metrics[key]
            if isinstance(value, bool):
                valid = type(reported) is bool and reported == value
            elif isinstance(value, int):
                valid = type(reported) is int and reported == value
            else:
                _unit_interval(reported, key)
                valid = math.isclose(reported, value, rel_tol=1e-12, abs_tol=1e-12)
            if not valid:
                raise ValueError(f"Validation {key} is inconsistent with confusion counts")
        if not expected["balanced_accuracy_defined"] and "balanced_accuracy" in metrics:
            raise ValueError("Balanced accuracy must be omitted when a validation class is missing")
        ap_defined = expected["true_positives"] + expected["false_negatives"] > 0
        if type(metrics.get("average_precision_defined")) is not bool or metrics["average_precision_defined"] != ap_defined:
            raise ValueError("Average Precision support flag is missing or inconsistent with positive labels")
        if ap_defined:
            _unit_interval(metrics.get("average_precision"), "Average Precision")
        elif "average_precision" in metrics:
            raise ValueError("Average Precision must be omitted when there are no positive labels")
        for key in COUNT_KEYS:
            pooled[key] += metrics[key]
    result = confusion_metrics(*(pooled[key] for key in COUNT_KEYS))
    result["global_average_precision_computed"] = False
    result["global_average_precision_status"] = "not_computed_requires_pooled_scores"
    return result
