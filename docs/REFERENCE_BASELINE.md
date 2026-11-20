# Controlled reference-dataset baseline — Phase 3B

This engineering baseline completed on 2026-11-20 using checkpoint
`1137cd260f4e1bb2bfd7957f420c17db47fbe39a`. No source code changes were
required for the experiment. The results verify execution of the corrected
federated workflow; they do not establish meaningful fraud-detection quality.

## Configuration and dataset

The run used two simulated clients, two communication rounds, partition seed
42, localhost port 24087, and a 600-second launcher timeout. The launcher applies
a separate 30-second round RPC timeout. An external monitoring watchdog was
bounded at 660 seconds. The recorded start was 2026-11-20 at 13:15:09 UTC.

The environment was Windows AMD64 with CPython 3.12.10 and the dependencies in
`requirements.lock`, including Flower 1.25.0, scikit-learn 1.8.0,
imbalanced-learn 0.14.1, and MLflow 3.7.0. See [environment setup](ENVIRONMENT.md).

The unmodified credit-card fraud CSV contained 284,807 source rows and was
150,828,752 bytes. Its schema is 30 ordered input features followed by `Class`,
as documented in [dataset acquisition and validation](DATASET.md). Its SHA-256
was:

```text
76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89
```

The isolated `input.csv` snapshot had the same fingerprint. This locally measured
hash identifies the input bytes; it is not a publisher-signed checksum.

## Partitions and training policy

The existing [partitioning policy](DATA_PARTITIONING.md) retained the first
occurrence of each exact parsed feature vector and rejected conflicting labels.
It excluded 1,081 duplicate rows, leaving 283,726 eligible observations. Global
training, validation, and final-test memberships are deterministic and independent
of client count. Client assignments are mutually exclusive.

| Partition | Client 0 | Client 1 | Total |
| --- | ---: | ---: | ---: |
| Original training observations | 90,792 | 90,792 | 181,584 |
| Training legitimate / fraud | 90,641 / 151 | 90,641 / 151 | 181,282 / 302 |
| Post-SMOTE training observations, per round | 181,282 | 181,282 | 362,564 |
| Validation observations | 22,698 | 22,698 | 45,396 |
| Validation legitimate / fraud | 22,660 / 38 | 22,660 / 38 | 45,320 / 76 |

The final-test partition contains 56,746 reserved observations and was not
scored. No final-test outcomes were used for model selection or tuning.

The existing [preprocessor](PREPROCESSING.md) fitted shared scaling statistics
using global training observations only. Validation and final-test observations
did not contribute to fitting. SMOTE operated on client training data only,
using the unchanged `SMOTE(random_state=42)` configuration.

Each local fit used the existing warm-started logistic regression with
`solver="lbfgs"` and `max_iter=1`, initialized from the received global
parameters. [FedAvg](AGGREGATION.md) weighted updates by original assigned
training counts, giving each client weight 0.5. Post-SMOTE counts were diagnostic
values, not aggregation weights. Validation loss and accuracy were weighted by
actual validation counts.

## Observed results

| Round | Global validation loss | Global validation accuracy | Fit replies | Validation replies |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.7040588855743408 | 67.4024143096% | 2/2 | 2/2 |
| 2 | 0.35234734416007996 | 96.5283284871% | 2/2 | 2/2 |

Both clients completed training and validation in both rounds. Their received
parameter fingerprints matched the expected global parameters. Round 2 started
from the round 1 aggregate. The server, both clients, and the launcher all exited
with code zero, and the monitored processes terminated after completion.

**The observed 96.53% validation accuracy does not establish fraud-detection
quality.** An all-legitimate classifier would achieve
`45,320 / 45,396 = 99.8326%` accuracy on this validation set, exceeding the
observed result. Precision, recall, F1, PR-AUC, and confusion matrices remain
unmeasured.

MLflow recorded two `FINISHED` runs, `Round_1` and `Round_2`, in the
`Credit_Card_Fraud_Federated_Learning` experiment. Their four metric records
matched the values in the round report.

## Execution evidence and resources

Wall-clock duration was 47.54 seconds. Sampling the launcher and its descendant
processes every 0.5 seconds observed approximately 1,182 MiB combined working
memory, 1,851 MiB combined private commit, and 59.84 CPU-seconds. The host had
12 logical CPUs and approximately 3.47 GiB available RAM immediately before
execution. These are approximate resource observations; combined working memory
can count shared pages multiple times, and sampling can miss brief peaks.

Four convergence warnings occurred, consistent with the unchanged one-iteration
local optimizer. Existing Flower compatibility API deprecation warnings were
also present. No tracebacks were observed. Successful completion does not imply
optimizer convergence.

All generated outputs remained in a fresh experiment directory outside the
checkout. It contained `input.csv`, `rounds.json`, `tracking.sqlite`, `model.pkl`,
server and client logs, and isolated runtime destinations. MLflow artifact
locations were configured inside that directory. The retained external evidence
also includes `execution.json`, `verification.json`, resource samples, and the
regression log. Raw records, logs, databases, and model files are not included in
this report.

The verification report recorded matching hashes for 36 protected files,
matching snapshots for nine protected directories, and unchanged metadata for
33 pre-existing external output files. Existing project models, datasets,
databases, tracking artifacts, notes, backups, and environments were preserved.

The saved regression log records **40 passing tests in 56.929 seconds**,
including the isolated synthetic two-client, two-round integration test. These
are the recorded Phase 3B test results; creating this report did not rerun tests
or training.

## Reproduction

Follow [ENVIRONMENT.md](ENVIRONMENT.md) to create the locked Python environment
and [DATASET.md](DATASET.md) to acquire the CSV. Run the following PowerShell
commands from the checkout root. Check the fingerprint before comparing runs:

```powershell
$datasetHash = (Get-FileHash -LiteralPath data/creditcard.csv -Algorithm SHA256).Hash
if ($datasetHash -ne '76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89') {
    throw 'Dataset fingerprint differs from the recorded baseline.'
}
& .\.venv-phase0\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Regression tests failed.' }
```

The example output destination below is
`D:\FL-experiments\reference-2clients-2rounds-seed42`; it is not the original
machine's private evidence path. Choose an appropriate local drive and directory.
The destination must be fresh, outside the checkout, and its parent must already
exist. Recheck available memory, disk space for the dataset snapshot and outputs,
and localhost port availability before starting. See [SIMULATION.md](SIMULATION.md)
for isolation and failure handling.

```powershell
$runDir = 'D:\FL-experiments\reference-2clients-2rounds-seed42'
if (Test-Path -LiteralPath $runDir) { throw 'Choose a fresh output destination.' }
if (-not (Test-Path -LiteralPath (Split-Path -Parent $runDir) -PathType Container)) {
    throw 'The output parent directory must already exist.'
}
& .\.venv-phase0\Scripts\python.exe -B -X utf8 run_simulation.py `
    --dataset data/creditcard.csv `
    --run-dir $runDir `
    --num-clients 2 --rounds 2 --seed 42 `
    --port 24087 --timeout 600
if ($LASTEXITCODE -ne 0) { throw 'Simulation failed; preserve its isolated evidence.' }
```

The 660-second external monitoring watchdog and resource sampler were separate
from the launcher; the command above sets the launcher's 600-second deadline.
Runtime and floating-point results can vary across machines. Retain the isolated
evidence and verify participation and round completion before interpreting metrics.

## Limitations

- This is one two-round engineering baseline. It demonstrates neither convergence
  nor meaningful fraud-detection performance. Five- and ten-client experiments
  require separate resource preflight checks; their capacity remains unverified.
- Shared preprocessing statistics are fitted centrally in this simulation.
  Training-only fitting prevents evaluation leakage but does not establish
  institutional privacy. Simulated client partitions are not independent banks.
- SMOTE changes each client's training distribution and optimization objective.
  Original-count FedAvg weighting does not remove those effects.
- One local L-BFGS iteration produced convergence warnings. The global parameter
  initialization is verified, but this run does not establish optimizer convergence.
- The saved validation-round checkpoint lacks its exact fitted preprocessing
  sidecar. It is not a verified inference bundle. The pre-existing project model
  remains unchanged, and no end-to-end serving parity is claimed for either file.
- The reserved final-test set remains unscored. No final-test metric, model
  selection decision, or hyperparameter tuning was performed.
