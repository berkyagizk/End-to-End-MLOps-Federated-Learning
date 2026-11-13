# Phase 1A data partitioning

The existing launcher uses one shared benchmark CSV to simulate banks. Every
client process reconstructs the same row assignments from the CSV, client count,
and seed, then receives arrays for its own training and validation rows. The
launcher checks split feasibility before starting Flower. This setup does not
model separate institutions with private datasets or naturally different fraud
rates.

```powershell
# Validate assignments without launching Flower, training, or writing a model.
& .\.venv-phase0\Scripts\python.exe -B -m unittest discover -s tests -v
```

The tests use synthetic data and also check the local reference CSV when it is
present. The reference check reports counts per client and per split, in label
order `[legitimate, fraud]`. For a future simulation run, pass `--num-clients N`
and optionally `--seed SEED` to `run_simulation.py`. The seed defaults to 42 and
the launcher passes the same value to each client. When starting clients
manually, use the same CSV, client count, and seed for each process, with client
IDs from `0` to `N - 1`.

## Row identity and duplicate policy

Assignments use zero-based CSV data row positions, excluding the header. Exact
equality of all parsed feature columns defines a duplicate, regardless of label.
Keep the first source row for a feature vector and count subsequent copies as
excluded duplicates. If the same feature vector has conflicting labels, reject
the dataset so the conflict can be investigated. The raw file stays intact.
This policy does not group near duplicates or repeated transactions from the
same customer. The benchmark has no customer identifier in its schema.

The reference CSV has 284,807 rows, including 492 fraud rows. It has 1,081
duplicate copies (1,062 legitimate and 19 fraud) and no conflicting labels.
After exclusion, 283,726 observations are eligible (283,253 legitimate and 473
fraud). Each eligible observation appears exactly once in the split plan.

## Training, validation, and final test

1. Shuffle the unique rows separately within each class using the configured
   seed. Reserve 20% per class, rounded to the nearest row, for a single final
   holdout. Reserve 20% of the remaining rows in each class for global
   validation, again rounding to the nearest row with a minimum of one row.
   The rest form the global training set. All three sets have identical source
   row membership across supported client counts for a fixed CSV and seed.
   Freeze the seed before comparing client counts or models.
2. Distribute training and validation rows separately, evenly by class across
   clients. Per-client class counts in each split can differ by at most one.
3. Require at least six unique training rows and one validation row of each
   class per client, and at least one of each class in the final test. Six is
   the minimum needed for the existing SMOTE neighbor setting when a class is
   resampled. If either fixed split cannot support the requested client count,
   fail with the available and required counts. This can limit the maximum
   number of clients when fraud observations are scarce.

Client model initialization and SMOTE use training arrays. Flower's existing
round evaluation and MLflow accuracy/loss now describe **validation**. Client
0's existing save callback still runs during validation. The final holdout is
represented by row positions in the plan. The client loader returns no final
test arrays, and the current application does not evaluate or select models on
the final test. Reserve it until a model and threshold have been selected using
training and validation only.

## Preprocessing boundary and limitations

Clients still standardize `Time` and `Amount` and retain the existing feature
order. To keep validation and final-test values out of fitting, the shared
scaler fits only the union of the client training rows, in source-row order
for reproducible statistics across client counts. Each client transforms
its own training and validation rows using that common feature scale. This is
a central-simulation policy that relies on pooled training statistics. It does
not implement a scaler for independent institutions. Persistence of the scaler
and matching transformations in FastAPI remain open for later work.

Class-stratified random splits approximate an IID benchmark. They do not measure
performance under temporal drift or institutional differences. Partition checks
establish split integrity. The previously audited weight-update behavior,
resampling weights, metric aggregation, and model persistence still require
their own validation.
