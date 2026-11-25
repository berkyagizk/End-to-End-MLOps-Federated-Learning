# Phase 3C-C reference evaluation

The single controlled reference run started on **2026-11-25 at 11:41:22 UTC**
from source checkpoint `4be14fa62c268d96118ead3048d22cbd44f4d888`.
It extends the [Phase 3B engineering baseline](REFERENCE_BASELINE.md) with
validation confusion counts, per-client AP, and exact global AP.

## Configuration and data

The locked [Phase 0 environment](ENVIRONMENT.md) used Windows AMD64,
CPython 3.12.10, Flower 1.25.0, MLflow 3.7.0, scikit-learn 1.8.0,
imbalanced-learn 0.14.1, pandas 2.3.3, NumPy 2.3.5, and SciPy 1.16.3.
The run used two simulated clients, two rounds, partition seed 42, and
localhost port 24088. The launcher deadline was 600 seconds, the per-RPC
timeout was 30 seconds, and the external watchdog was 660 seconds.

The [reference CSV](DATASET.md) contained 284,807 rows and 150,828,752 bytes.
The source and isolated snapshot shared the dataset SHA-256 listed below.
The [partitioning policy](DATA_PARTITIONING.md) excluded 1,081 duplicate
rows. Both clients used the same training-fitted preprocessing statistics.

| Partition | Total | Legitimate | Fraud |
| --- | ---: | ---: | ---: |
| Eligible observations | 283,726 | 283,253 | 473 |
| Global training | 181,584 | 181,282 | 302 |
| Validation | 45,396 | 45,320 | 76 |
| Reserved final test | 56,746 | 56,651 | 95 |

Both clients completed fit and validation in both rounds. Each client started
with 90,792 training rows (90,641 legitimate and 151 fraud). SMOTE with seed 42
ran only on each client's training rows and yielded 181,282 rows per local fit.
Warm-started L-BFGS used one iteration per round. FedAvg weighted each update
by the original training count, giving each client weight 0.5. Each client
validated on 22,698 assigned rows (22,660 legitimate and 38 fraud).
See [preprocessing](PREPROCESSING.md),
[local training](LOCAL_TRAINING.md), and [aggregation](AGGREGATION.md).

## Validation results

Fraud is class 1. The fixed rule is `P(fraud) > 0.5`. A score of exactly 0.5
is classified as legitimate. Loss is weighted by validation row count. The
server pools confusion counts before calculating the threshold-based metrics.
For exact global AP, the server scores the combined validation membership
with the round's post-FedAvg weights and training-only preprocessing. It
calculates non-interpolated Average Precision over the pooled score ranking.
This evaluator operates centrally within the single-machine simulation.
See [evaluation definitions](EVALUATION_METRICS.md).

| Metric | Round 1 | Round 2 |
| --- | ---: | ---: |
| Fit participation | 2/2 | 2/2 |
| Validation participation | 2/2 | 2/2 |
| Validation log loss | 0.7040588855743408 | 0.35234734416007996 |
| Accuracy | 0.6740241430963081 | 0.9652832848709137 |
| Fraud precision | 0.004976462676529926 | 0.042682926829268296 |
| Fraud recall | 0.9736842105263158 | 0.9210526315789473 |
| Fraud F1 | 0.009902315000669076 | 0.08158508158508158 |
| Specificity | 0.673521624007061 | 0.9653574580759047 |
| Balanced accuracy | 0.8236029172666883 | 0.9432050448274261 |
| Exact global AP | 0.6889721591437376 | 0.7562923128758857 |
| Client 0 AP | 0.6590479660708343 | 0.7461699807600183 |
| Client 1 AP | 0.7305606235692321 | 0.7822723966082294 |

All values except loss and participation are proportions on a 0–1 scale.
All reported support flags were true. Global AP status was `computed_exact`
in mode `centralized_simulation_validation` in both rounds.

| Round | True positives | False positives | True negatives | False negatives | Correct predictions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 74 | 14,796 | 30,524 | 2 | 30,598 |
| 2 | 70 | 1,570 | 43,750 | 6 | 43,820 |

Centralized and pooled client confusion counts matched exactly in both rounds.
The reconstructed client memberships matched the recorded fingerprints. Both
matrices cover all 45,396 validation rows.

## Interpretation

Round 2 detected **70 of 76** validation fraud cases. Only **70 of 1,640**
flagged transactions were actual fraud, leaving about **22.4 false alerts per
true fraud alert**. False positives fell by **13,226** between rounds while
four more fraud cases were missed. Precision and F1 increased, but the alert
burden remains substantial at the fixed threshold.

Global AP increased from about **0.688972 to 0.756292**. The constant-score AP
reference equals the validation fraud prevalence, `76 / 45,396 ≈ 0.00167416`.
AP measures ranking quality across thresholds. Precision at 0.5 measures the
fraction of alerts that are correct at that one cutoff. The observed AP supports
promising validation ranking, alongside low precision at the chosen cutoff.
An all-legitimate classifier achieves approximately **99.832584% accuracy**
(45,320 of 45,396) while detecting no fraud.

Training-time SMOTE changes the class balance the optimizer sees. It may
affect probability calibration and the suitability of the fixed cutoff.
Calibration and threshold alternatives were neither implemented nor tested.
This run does not establish optimizer convergence, production readiness,
institutional privacy, or held-out test performance. The **56,746 reserved
final-test observations remain unscored**.

## Phase 3B reproducibility

| Round | Historical loss | Absolute loss difference | Accuracy difference (percentage points) | Correct predictions (Phase 3B → 3C-C) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.7040588855743408 | 0.0 | 0.0 | 30,598 → 30,598 |
| 2 | 0.35234734416007996 | 0.0 | 0.0 | 43,820 → 43,820 |

Loss matched exactly in both rounds and passed the diagnostic tolerances
`rel_tol=1e-8` and `abs_tol=1e-10`. Accuracy and integer correct-prediction
counts also matched exactly. The sent and aggregate parameter digests matched
Phase 3B for both rounds, including Round 1's aggregate as Round 2's
initialization. These findings apply to these two observed runs.

| Fingerprint | SHA-256 |
| --- | --- |
| Dataset | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` |
| Preprocessing | `85de5a26c3a89b4d0c8ef9f23148d7862001aec120976a559a3d70dc94187ec5` |
| Global validation membership | `b9c6dbc2cb0fe595e8e70cc2425c9362bfebbd83630407da674096358278fd2c` |
| Client 0 validation membership | `5fbd9813988e43e78df5c07108dda853a9250790ed8491a26e65ea98f52668d5` |
| Client 1 validation membership | `09941ab1e5a39c3a2be504f63459cf36fd3a68bd55a570dcd7842dad0d927733` |
| Validation context | `01717851c2ca2722f418109499aa94638e6e3241b4034319376154274450dd37` |
| Initial parameters | `43e1fb25241290f5407c907a7d8d631f3f9344814baaf0192b7285732c60bf97` |
| Round 1 aggregate and Round 2 sent parameters | `ee3757f3e42c28066eb6f8f8881446215768935cac56872009b8b9c743633ca0` |
| Round 2 aggregate | `4755a682f60c748c7177d1d5c8a9a9fe2318d009531a0b4a5aefe3d543482c0f` |

## Resources and persistence

The 12.19-second launch observation found no competing experiment process.
At the gate, 6.88 GiB RAM and 8.19 GiB commit headroom were available and the
output volume had 122.41 GiB free. The CSV snapshot completed at 6.63 s.
The launcher waited for server readiness before starting both clients. The
same monitoring sample at 19.94 s recorded readiness and both client launches.
Round 1 began at 28.63 s. Flower reported 2.14 s in its round loop.

| Measurement | Phase 3B | Phase 3C-C |
| --- | ---: | ---: |
| Wall time | 47.54 s | 31.91 s |
| Peak sampled combined working set | 1,182 MiB | 1,227 MiB |
| Peak sampled combined private commit | 1,851 MiB | 1,935 MiB |

Phase 3C-C used 36.89 CPU-seconds. Across 63 process-tree samples at about
0.5-second intervals, minimum available RAM was 5.88 GiB and minimum commit
headroom was 6.36 GiB. Paging briefly reached about 631 pages/s without triggering
a sustained-pressure stop. Sampled peaks can miss brief spikes. The shorter wall
time and higher sampled memory describe this comparison only. The run establishes
neither a general speedup nor capacity for five or ten clients.

Both rounds atomically published `rounds.json` before reporting completion.
Read-only SQLite inspection passed `PRAGMA quick_check` and found two
`FINISHED` MLflow runs. For each round, 25 numeric metrics and seven
evaluation tags matched the JSON exactly. The launcher, server, both clients,
and all other sampled processes exited with code zero. The watchdog's job
was empty afterward. No separate observation-level scores or labels were
persisted beyond the planned input snapshot. The source snapshot, pre-existing
model and databases, and prior experiment evidence were preserved.

Four L-BFGS convergence warnings were consistent with the one-iteration
budget. Flower emitted compatibility API deprecation warnings and the
notice about the missing fit-metrics aggregation callback. The run logs
contained no errors or tracebacks.

## Reproduction

Use the source checkpoint and the locked [environment](ENVIRONMENT.md).
Verify [the reference CSV](DATASET.md), choose a free localhost port, and
check memory, commit, and disk headroom. Run from the checkout root. The
example uses an existing temporary directory outside the checkout as the
parent. Choose a fresh destination for each authorized run.

```powershell
$expected = '76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89'
if ((Get-FileHash -LiteralPath .\data\creditcard.csv -Algorithm SHA256).Hash -ne $expected) {
    throw 'Reference CSV SHA-256 does not match'
}
$runDir = Join-Path ([System.IO.Path]::GetTempPath()) 'fl-reference-2c2r-s42'
if (Test-Path -LiteralPath $runDir) { throw 'Use a fresh run directory' }

& .\.venv-phase0\Scripts\python.exe -B -X utf8 .\run_simulation.py `
    --dataset .\data\creditcard.csv `
    --run-dir $runDir `
    --num-clients 2 --rounds 2 --seed 42 --port 24088 --timeout 600
if ($LASTEXITCODE -ne 0) { throw 'Simulation failed. Preserve its evidence.' }
```

The run used a separate 660-second process-tree watchdog and resource
sampler around this command. Use a separate fresh evidence directory for
that monitoring. Verify both round records, MLflow agreement, and process
cleanup before interpreting results. See [simulation isolation](SIMULATION.md).
