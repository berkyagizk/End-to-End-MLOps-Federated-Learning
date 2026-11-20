# Phase 3C-A validation diagnostics

These diagnostics apply only to each client's established validation partition.
They do not read, score, or tune against the reserved final-test partition.
The Phase 3B reference report describes an earlier experiment and is unchanged.
No new reference-dataset experiment is part of this implementation.

## Inputs and fixed decision rule

`src/evaluation.py` accepts aligned, nonempty, one-dimensional vectors of binary
labels and probabilities for `Class=1`, the fraud class. Labels must be numeric
0 or 1. Probabilities must be numeric, finite, and within [0, 1]. Strings,
complex numbers, Boolean probabilities, shape mismatches, and non-finite inputs
are rejected. Before selecting the fraud probability column, the client also
checks the estimator's class order `[0, 1]`, its two-column probability shape,
and probability sums (absolute tolerance 1e-12).

The fixed threshold is 0.5: predict fraud when `P(Class=1) > 0.5`; an exact tie
predicts legitimate. This preserves the existing logistic-regression tie rule
at zero decision score. There is no threshold selection or optimization.

## Definitions and undefined cases

For fraud as the positive class:

| Metric | Definition |
| --- | --- |
| True positives (TP) | Fraud observations predicted as fraud |
| False positives (FP) | Legitimate observations predicted as fraud |
| True negatives (TN) | Legitimate observations predicted as legitimate |
| False negatives (FN) | Fraud observations predicted as legitimate |
| Precision | TP / (TP + FP) |
| Recall (sensitivity) | TP / (TP + FN) |
| F1 | 2 TP / (2 TP + FP + FN) |
| Specificity | TN / (TN + FP) |
| Balanced accuracy | (recall + specificity) / 2, when both classes are present |
| Accuracy | (TP + TN) / (TP + FP + TN + FN) |
| Log loss | Mean binary negative log likelihood, using scikit-learn's probability clipping |

Precision, recall, F1, and specificity use **zero_division=0** semantics: a zero
denominator gives numeric 0.0 and a false `<metric>_defined` flag. A defined
zero is distinguished from an unsupported rate by that flag. Balanced accuracy
is omitted when either validation class is absent, with
`balanced_accuracy_defined=false`. Empty validation sets are rejected.

Log loss explicitly supplies labels `[0, 1]`, so single-class validation remains
well-defined. Exact probabilities 0 and 1 use scikit-learn's float64 clipping
and produce finite loss. Invalid probabilities are rejected before this step.

**Average Precision (AP)** uses `sklearn.metrics.average_precision_score` on
unthresholded fraud probabilities with `pos_label=1`. It is the
non-interpolated sum `sum((R_n - R_(n-1)) * P_n)` over increasing recall. It is
not trapezoidal PR-AUC, which interpolates the precision-recall curve. AP uses
the full ranking, not only the fixed-threshold predictions. If no positive
labels exist, AP is omitted and `average_precision_defined=false`; it is not
reported as a measured zero. With only positive labels, AP is defined and is 1.

For labels `[0, 0, 1, 1]` and scores `[0.1, 0.4, 0.35, 0.8]`, AP is 5/6.
The tests independently compare this result with scikit-learn and distinguish
it from trapezoidal integration.

## Pooled aggregation and validation

Clients send only scalar counts, rates, support flags, and their local AP, plus
the existing client/round/parameter identifiers. No per-record scores, labels,
identifiers, or features are added to Flower messages.

The server adds TP, FP, TN, and FN across participating clients, then derives
the pooled rates from those totals. Neither an arithmetic nor a sample-weighted
mean of client F1 is the global F1. For unequal validation sizes of 2 and 9,
client counts `(TP, FP, TN, FN)` of `(1, 1, 0, 0)` and `(1, 0, 5, 3)` give pooled
counts `(2, 1, 5, 3)`: precision 2/3, recall 2/5, F1 1/2, accuracy 7/11,
specificity 5/6, and balanced accuracy `(2/5 + 5/6) / 2`.

Flower's existing validation-sample-weighted loss aggregation is unchanged.
Pooled count-based accuracy equals validation-sample-weighted client accuracy.
FedAvg model weighting, optimizer updates, preprocessing, SMOTE, and model saves
are unchanged. The controlled strategy still requires every configured client
to return fit and validation results with matching global parameter fingerprints.

Counts must be nonnegative Python integers, with a positive validation total
within Flower's signed 64-bit range. Each client's four counts must sum to its
reported `num_examples`. The server verifies the fixed threshold, support flags,
finite rate/AP ranges, and agreement of reported rates with confusion counts.
Missing or malformed diagnostics abort aggregation before MLflow logging.
As before, the server trusts clients to evaluate the assigned data honestly;
it cannot reconstruct or verify AP from their scalar reports.

Controlled simulations require the full metric schema. The existing
`build_strategy` library helper still accepts legacy accuracy-only replies for
compatibility; when any diagnostic fields are supplied it requires complete
diagnostics from every reply. Mixed old/new replies are rejected. Controlled
simulations never fall back to accuracy-only reporting.

## Reports and MLflow

`rounds.json` retains the existing list of round records and top-level `round`,
`sent_sha256`, `aggregate_sha256`, `fit`, `validation`, `accuracy`, and `loss`
fields. Client validation entries gain counts, rates, support flags, the fixed
threshold, and local AP where defined. A new `validation_metrics` object contains
pooled diagnostics and the existing aggregated loss. Values are finite JSON
scalars; unsupported AP and balanced accuracy values are omitted.

The existing per-round MLflow runs keep `accuracy` and `loss`. They additionally
log pooled counts/rates, the fixed threshold, validation total, and support
flags (Boolean flags become 0/1). Each available local AP is separately named
`client_<id>_average_precision`, with a corresponding support flag. No unqualified
global `average_precision` metric is logged. The round report records
`global_average_precision_computed=false` and
`global_average_precision_status="not_computed_requires_pooled_scores"`;
MLflow records the numeric computed flag as 0.

## Global AP: proposal requiring review

**Exact global AP is not implemented.** Confusion counts and local AP values do
not preserve the inter-client score ranking. For example, the test fixtures have
two local AP values of 1, yet AP on their concatenated synthetic score/label
vectors is below 1. Averaging the local APs would misrepresent the result.

The current Flower metric interface supports scalar values, not the pooled
score-label ranking required for exact AP. The following alternatives require
design approval before implementation:

1. **Simulation-only centralized validation:** evaluate each round's aggregate
   parameters against the fixed union of validation observations, using the
   identical training-fitted preprocessor. This would add a server-side data and
   preprocessing path and require strict row-membership/parity tests. It could
   give exact pooled AP without new per-record network messages, but must be
   explicitly described as centralized simulation evaluation.
2. **Explicit score/label collection:** transfer each validation observation's
   probability and label and compute AP on the pooled values. This changes the
   evaluation communication/privacy boundary and needs explicit authorization.
3. **Approximate histogram evaluation:** agree on fixed score bins and sum
   positive/negative counts per bin. Validate the resulting approximation against
   exact synthetic references, including ties and bin-resolution effects. Lost
   within-bin ranking prevents a claim of exact AP. Histogram sharing alone is
   not evidence of institutional privacy.

No option is implemented or implicitly selected here. Local AP plus pooled
fixed-threshold metrics can be reviewed now; global AP remains an explicit
limitation. Existing centralized preprocessing and scalar statistics likewise
do not establish privacy guarantees or regulatory compliance.

## Verification

Run the locked Phase 0 environment in UTF-8 mode:

```powershell
& .\.venv-phase0\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -v
```

Synthetic tests verify manual confusion counts, fraud-positive class handling,
unequal-client pooling, threshold ties, zero divisions, missing classes,
probability validation, AP references, and agreement with scikit-learn. Flower
integration tests check count-based pooling, unchanged loss averaging, strict
diagnostic validation, per-round JSON, and separated local AP logging. Existing
regressions, including the synthetic two-client/two-round transport test, remain
part of the suite. Model/MLflow writes in unit tests are mocked; integration
artifacts use temporary directories. No final-test scoring is performed.

The integration test resolves its temporary root before comparing MLflow artifact
URIs. Windows can expose the same temporary directory through an 8.3 alias and a
long path; comparing those literal strings previously rejected a correctly
isolated run. This change affects only test path comparison, not runtime output
destinations or MLflow configuration.
