# Phase 1B preprocessing

`src/preprocessing.py` defines the ordered schema `Time, V1, ..., V28, Amount`.
It standardizes `Amount` and `Time` with the existing `StandardScaler` mean
and scale and passes `V1`–`V28` through. Named inputs must match the schema
and order. Positional inputs must contain 30 numeric finite values per row.
The API accepts raw feature values. A positional payload gives the server no
way to detect two finite columns that the caller swapped.

`fit_partition_preprocessor(features, plan)` fits on the union of the Phase 1A
training rows in source-row order. The fixed global training membership gives
all simulated clients the same fitted parameters for a given dataset and seed.
The client loader applies the transformer to its training and validation rows.
For future final evaluation, call `transform(features.iloc[plan.final_test])`
on the frozen preprocessor that belongs to the selected model. The training
loader still reserves those rows for final evaluation and does not return them.

## Pairing a model and its preprocessing metadata

After training a model with this transformer, call
`fitted_preprocessor.save_for_model(model_path)` using the exact fitted object
from that training run. The method creates `<model filename>.preprocessing.json`
beside the saved model. The deterministic JSON records the schema version,
ordered features, scaled features, means, scales, and SHA-256 of the model
bytes. It refuses to replace an existing sidecar. Keep the two files together
as an immutable pair and retain the training configuration with the run.

Set `FRAUD_MODEL_PATH` to the trusted model path before starting FastAPI. The
server loads its sidecar at startup, validates the schema and numeric parameters,
and checks the SHA-256 against the same bytes it passes to joblib. It also checks
the estimator's feature count and feature names when present. SHA-256 detects a
stale or mismatched pair. It does not establish provenance or make a pickle safe
to deserialize. Only configure model files from a trusted source, and keep both
files under trusted operator control.

`POST /predict` returns HTTP 503 when the pair is unavailable or invalid and
HTTP 422 for malformed feature input. Send raw features in the documented
order. FastAPI applies the shared transformation before calling the estimator.

The existing client checkpoint callback still writes its legacy model without
a sidecar. There is no verified fitted preprocessor for `best_model.pkl`, so
FastAPI no longer implicitly loads it. Do not attach a newly fitted sidecar to
that file. Deploying a compatible project model requires a later training run
and an export step that pairs its model with the exact fitted transformer.
Phase 1B checks the interface with a locally generated synthetic estimator and
does not establish parity for the existing checkpoint.

## Simulation and institutional privacy

The benchmark CSV, labels, split plan, and pooled training rows are accessible
to every simulated client process. Each process independently computes the
same mean and scale from the pooled training rows. Excluding held-out values
from fitting prevents evaluation leakage through these statistics. It does
not isolate one institution's rows from another.

With independent institutions, each institution would keep its own rows and
held-out assignments local. The parties would need a common feature schema and
a reviewed protocol for computing common training-only statistics. A design
could aggregate counts and sufficient statistics without disclosing per-client
values, subject to a threat model for collusion, dropouts, small cohorts, and
what the released statistics reveal. The project implements no such protocol.

## Verification

```powershell
& .\.venv-phase0\Scripts\python.exe -B -m unittest discover -s tests -v
```

The suite checks the existing partition boundaries, exact parity for synthetic
training and FastAPI inputs, metadata validation, and fail-closed inference.
It creates synthetic model files only in temporary directories.
