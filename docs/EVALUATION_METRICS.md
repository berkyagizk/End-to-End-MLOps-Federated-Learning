# Validation diagnostics: Phase 3C-A and simulation-only global AP

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
global `average_precision` metric is logged. Without a centralized evaluator,
the legacy library interface records
`global_average_precision_computed=false` and
`global_average_precision_status="not_computed_requires_pooled_scores"`;
MLflow records the numeric computed flag as 0.

## Phase 3C-B: exact centralized simulation validation AP

Controlled launcher runs now enable `GlobalValidationContext` from
`src/global_evaluation.py`. This is explicitly **centralized single-machine
simulation evaluation**, not a privacy-preserving institutional protocol.
The server gains access to the snapshot and validation observations. Scalar
sharing and metadata fingerprints do not establish privacy or compliance.

At startup, server and clients parse and fingerprint the same snapshot bytes.
`prepare_partitioned_data` reuses the unchanged partition builder and shared
preprocessor. Both the server and clients receive the launcher's seed. Training
row positions are sorted before fitting the original scaling statistics. The
server transforms only the sorted union of client validation rows and caches
immutable, bytes-backed feature, label, and row-position arrays. It verifies
disjointness from training, final test, and excluded duplicates. Full frames,
training arrays, and the partition plan are released after initialization.
No dataset loading or preprocessing fitting occurs in per-round evaluation.

The optional metadata interface preserves the existing four-array client loader
return by default. Controlled clients additionally report SHA-256 fingerprints
of input bytes, the fitted preprocessing schema/means/scales, global validation
membership, and their assigned validation membership. The context fingerprint
also binds seed and client count. The server checks those identities on fit and
validation replies and checks each validation count against its prepared plan.
These hashes check consistency between trusted simulation processes; they are
neither authentication nor anonymization. No per-record IDs are transmitted.

After each complete fit aggregation, `RoundParameters` binds immutable copies of
the aggregate NumPy tensor bytes to the round number and parameter digest. Only
one round may be pending. Missing, stale, duplicate, out-of-order, or mismatched
state is rejected. Global AP is evaluated only after all required client
validation replies and their identities have passed checks. Round zero is not
centrally scored.

Evaluation constructs a prediction-only logistic-regression object from the
aggregate coefficients/intercept, class order `[0, 1]`, and 30-feature schema.
It calls `predict_proba` and the existing `evaluate_binary` helper, which calls
scikit-learn's `average_precision_score` with fraud class 1. There is no model
fitting, SMOTE, saved-model deserialization, preprocessing refit, or model export.
Ranking ties use scikit-learn's normal AP behavior without jitter or quantization.
The fixed-threshold central confusion counts must exactly match pooled client
counts before any AP is logged. A discrepancy aborts the round.

When defined, `validation_metrics` and the existing `Round_N` MLflow run gain
`global_average_precision`. Numeric computed/defined flags are logged as metrics.
Descriptive strings are MLflow tags and JSON fields:

- `global_evaluation_mode="centralized_simulation_validation"`
- `global_average_precision_status="computed_exact"`
- `validation_context_sha256`, dataset/preprocessing/membership fingerprints,
  and `global_evaluation_parameters_sha256` (matching `aggregate_sha256`)

Without positive validation labels, the AP value is omitted, computed/defined
flags are false, and status is `undefined_no_positive_labels`. It is never
replaced by zero. AP evaluation and required MLflow run/metric/tag persistence
failures propagate in AP-enabled controlled runs. Required metric/tag writes
explicitly use synchronous persistence, even if an ambient environment enables
async logging. The failed round is not marked
complete, pending state cannot be reused, and `finish` refuses a successful
report. Partial logs/database records remain isolated for diagnosis. Legacy
strategies without the evaluator retain their previous MLflow error policy.
Existing pooled rates, client AP names, and top-level report fields are retained.

Controlled round completion now also requires atomic JSON publication. After
validation, AP, and required MLflow writes succeed, the server prepares a report
containing the complete successful-round prefix. It writes a unique temporary
file beside `rounds.json`, flushes and synchronizes that file, closes it, then
publishes it with `os.replace`. Only after replacement succeeds does it update
completed-round state, clear pending parameters, and announce completion.
Temporary files are cleaned up on ordinary write/sync/replacement failures.
Those failures mark the experiment failed and prevent pending-state reuse.

A failed Round 2 write or replacement leaves the published Round 1 report
unchanged. The server refuses a pre-existing report on first publication and
checks the previous published content before replacing it in later rounds.
`finish()` verifies all expected rounds, Flower history, and the already-published
JSON; it never performs a second report write. Missing or changed final evidence
fails validation. MLflow and JSON are separate stores, so a failed JSON write
can leave an MLflow round record without matching JSON publication; the controlled
experiment still fails and retains the last successfully published JSON prefix.

Same-directory replacement avoids exposing a partially written report during
ordinary operation. Synchronizing the temporary file improves durability but
does not establish a cross-store transaction or guarantee recovery from every
power loss/filesystem failure. Directory metadata is not explicitly synchronized.
An abrupt process termination can leave a temporary file for later inspection.

Only scalar metrics and fingerprints are added to outputs. Per-observation
features, labels, scores, and row IDs stay transient in memory and are not added
to JSON, logs, MLflow, or new artifact files. The launcher's pre-existing
`input.csv` raw snapshot and validation-round `model.pkl` behavior are unchanged.
The checkpoint's missing inference preprocessing sidecar is not resolved here.

The unchanged stratified partition builder reads all source labels when defining
the splits. Thus final-test labels are still read for partition construction,
but are excluded from preprocessing fitting and all predictions/AP scoring.
A stronger no-read policy would require a separate split-storage design.

Collecting per-record scores/labels or aggregating approximate score histograms
is not implemented. Flower 1.25 supports bytes payloads, so those alternatives
are technically possible but would require a separately reviewed sharing design.
Local AP values cannot reconstruct exact global AP: synthetic fixtures with two
local APs of 1 have lower pooled AP because of inter-client score ordering.

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

Global evaluation tests cover snapshot parsing, exact validation membership,
nondefault seeds, train-only fitting, client/server identity and array parity,
frozen context reuse and frame release, parameter ownership and round binding,
central/pooled confusion equivalence, known/tied/undefined AP, invalid values,
and evaluation/MLflow failure propagation. The synthetic subprocess integration
uses a nondefault seed and verifies global AP, metadata tags, preserved client
metrics, complete participation, and no new observation-level output files.

The integration test resolves its temporary root before comparing MLflow artifact
URIs. Windows can expose the same temporary directory through an 8.3 alias and a
long path; comparing those literal strings previously rejected a correctly
isolated run. This change affects only test path comparison, not runtime output
destinations or MLflow configuration.
