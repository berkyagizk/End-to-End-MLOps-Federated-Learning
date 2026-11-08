# Phase 0 results - 2026-11-08

Phase 0 establishes the local environment only. No application source, model,
partitioning, preprocessing, SMOTE implementation, or API behavior was edited.
No training, federated simulation, database migration, commit, or push was run.

## Git and preservation

The starting branch was `main` at `0316f9b`. Existing modifications were present
in `app.py`, `best_model.pkl`, `requirements.txt`, `src/client.py`, and
`src/server.py`. Existing untracked items included `.venv/`, the app backup,
`komutlar.txt`, `mflow.db`, and `run_simulation.py`; ignored local data, tracking
artifacts, and `venv/` were also inspected.

Created `codex/phase-0-environment` at the same commit. All work stayed in place;
nothing was stashed, reset, staged, committed, or pushed. The SMOTE dependency
remains pinned to 0.14.1. Source hashes verify that the client selection/SMOTE
implementation and launcher remain intact. The new ignore rules hide virtual
environments from Git status; they do not delete their contents.

Final SHA-256 comparisons matched all 13 protected source, artifact, database,
launcher/backup, and original environment-configuration files against the initial
baseline. There were no staged changes and HEAD remained at `0316f9b`.

## Environment decisions

- Reference: CPython **3.12.10**, Windows AMD64, new `.venv-phase0/`.
- Bootstrap pip: **25.0.1**. Lock generator: **uv 0.9.18**.
- Twelve exact direct pins are in `requirements.txt`; all versions and rationale
  are documented in [ENVIRONMENT.md](ENVIRONMENT.md).
- `requirements.lock` contains **98** hash-locked distributions with platform
  markers. **97** apply to Windows; Gunicorn 23.0.0 is conditional on non-Windows.
- Every applicable locked version matches the previous training environment's
  metadata. Fresh Python-3.12 native wheels replace the ABI mismatch in the new
  environment; the original environments remain untouched.
- Flower 1.25.0 replaces the incompatible 1.5.0 declaration. It matches the
  existing runtime metadata and works with MLflow 3.7.0/cryptography 44.0.3.
  The application continues to use its existing legacy Flower entry points.

## Commands actually executed

The successful creation/install path was:

```powershell
python -m venv .venv-phase0
& .\.venv-phase0\Scripts\python.exe -m pip --isolated install --disable-pip-version-check --no-cache-dir --index-url https://pypi.org/simple -r requirements-tools.txt
& .\.venv-phase0\Scripts\python.exe -m uv pip compile requirements.txt --universal --python-version 3.12.10 --generate-hashes --no-strip-markers --no-python-downloads --no-build --exclude-newer 2026-11-08T00:00:00Z --default-index https://pypi.org/simple --cache-dir .phase0-cache --output-file requirements.lock --quiet
& .\.venv-phase0\Scripts\python.exe -m uv pip install --python .venv-phase0/Scripts/python.exe --require-hashes --only-binary :all: --no-python-downloads --default-index https://pypi.org/simple --cache-dir .phase0-cache -r requirements.lock
```

The initial resolution was replaced with a resolution preferring package versions
from the previous runtime metadata, avoiding unrelated transitive upgrades. The
generated lock, not the old environment, is the source for subsequent installs.

| Executed check | Result |
| --- | --- |
| Fresh hash-checked wheel installation | PASS: 97 runtime packages installed |
| `python -m pip check` using the new environment | PASS: `No broken requirements found.` |
| pip offline dry run against the complete lock | PASS: all applicable requirements already satisfied; non-Windows Gunicorn skipped |
| `python -B scripts/check_environment.py --dataset data/creditcard.csv` using the new environment | PASS |
| All 12 direct imports and exact versions | PASS |
| Existing Flower NumPyClient/start-server API availability | PASS; no network/server started |
| NumPy/SciPy/scikit-learn native interoperability | PASS; no model fitted |
| joblib temporary round-trip | PASS; existing model not used or overwritten |
| Dataset schema, numeric finiteness, row/class counts | PASS: 284,807 rows, 30 features; 284,315 legitimate / 492 fraud |
| Dataset SHA-256 | Matches the audited fingerprint in [DATASET.md](DATASET.md) |

The exact offline pip validation command was:

```powershell
& .\.venv-phase0\Scripts\python.exe -m pip --isolated install --dry-run --no-index --disable-pip-version-check --require-hashes --only-binary :all: -r requirements.lock
```

It validates the installed requirements and pip's handling of the lock. It does
not constitute a second fresh installation or re-hash already installed files.
The fresh installation was performed by uv with `--require-hashes`.

Initial sandbox attempts failed at `ensurepip`, PyPI DNS access, and temporary
directory cleanup. Authorized retries outside the sandbox succeeded. The first
smoke attempt passed imports/native checks but exited with a temporary-directory
permission error; the complete retry exited successfully. These failures were
environment permission restrictions, not hidden package-check passes.

## Files changed

- `.python-version`: exact interpreter version.
- `requirements.txt`: exact direct dependencies, preserving imbalanced-learn.
- `requirements.lock`: generated transitive pins and SHA-256 hashes.
- `requirements-tools.txt`: optional pinned lock-generation tool.
- `scripts/check_environment.py`: environment and optional raw-dataset checks.
- `README.md`: setup links and closure of the existing JSON code fence.
- `docs/ENVIRONMENT.md`, `docs/DATASET.md`, this results document: reproduction,
  provenance, decisions, and executed checks.
- `.gitignore`, `.dockerignore`: exclude retained/new local environments and cache.

Local `.venv-phase0/` and `.phase0-cache/` were created and are ignored. Existing
application changes, artifacts, backup, and launcher were preserved.

## Remaining limits

There is no remaining blocker for this Windows environment baseline. A separate
clean operating-system installation was not available; the tested environment
was freshly created on the current Windows machine. Other OSes and Docker are
not certified by these checks. The unchanged Dockerfile still uses Python 3.11
and does not install the hash lock.

The previously identified training/evaluation/inference defects are intentionally
open. No claim is made that the existing model is correct or that an end-to-end
federated run succeeds. `git diff --check` for the full working tree reports
pre-existing whitespace in `app.py` and `src/client.py`; those lines were left
untouched. The Phase 0 tracked-file diff passes its scoped whitespace check.

Phase 1 requires separate review and approval.
