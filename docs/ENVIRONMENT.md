# Phase 0 environment

The reference environment is **64-bit CPython 3.12.10 on Windows (AMD64)**.
`.python-version` records the interpreter version; it does not install Python.
Install that exact version from [Python.org](https://www.python.org/downloads/release/python-31210/).
Other operating systems are not runtime-tested in Phase 0. The universal lock
includes platform markers, but that alone does not certify other platforms.

## Reproduce on a clean checkout

These instructions assume the reviewed Phase 0 files are present. They are
currently uncommitted and have not been pushed; a fresh clone of the unchanged
remote branch will not contain them yet.

Run these PowerShell commands from the repository root. `python` must point to
the newly installed CPython 3.12.10, rather than an existing virtual environment.
If it does not, use the full path to that interpreter. The Windows `py` launcher
does not detect the interpreter on the audited machine, so these instructions do
not depend on it.

```powershell
python --version
python -c "import platform, sys; assert platform.python_version() == '3.12.10'; assert sys.maxsize > 2**32; print(sys.executable)"
if (Test-Path -LiteralPath .venv-phase0) { throw 'Choose a new environment directory; do not overwrite an existing environment.' }
python -m venv .venv-phase0
if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
& .\.venv-phase0\Scripts\python.exe -m pip --isolated install --disable-pip-version-check --index-url https://pypi.org/simple --require-hashes --only-binary :all: -r requirements.lock
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& .\.venv-phase0\Scripts\python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed' }
& .\.venv-phase0\Scripts\python.exe -B scripts/check_environment.py
if ($LASTEXITCODE -ne 0) { throw 'Environment smoke checks failed' }
```

No activation or PowerShell execution-policy changes are needed. Always invoke
the environment's Python explicitly, including when using existing application
commands. Python 3.12.10 supplies pip 25.0.1 in the tested setup. The runtime
lock pins application dependencies and their transitive dependencies with SHA-256
hashes. `--only-binary :all:` prevents an unexpected local native compilation.

The dataset is optional for package checks. Follow [DATASET.md](DATASET.md), then:

```powershell
& .\.venv-phase0\Scripts\python.exe -B scripts/check_environment.py --dataset data/creditcard.csv
```

## Dependency decisions

`requirements.txt` declares exact direct dependency versions;
`requirements.lock` is the installable, complete hash lock. Installing only
`requirements.txt` allows transitive dependency drift and is not the reproduction
procedure. Direct NumPy, SciPy, and joblib pins make the scientific/serialization
stack explicit.

| Distribution | Version |
| --- | --- |
| flwr | 1.25.0 |
| mlflow | 3.7.0 |
| scikit-learn | 1.8.0 |
| imbalanced-learn | 0.14.1 |
| numpy | 2.3.5 |
| scipy | 1.16.3 |
| pandas | 2.3.3 |
| joblib | 1.5.3 |
| matplotlib | 3.10.8 |
| fastapi | 0.125.0 |
| pydantic | 2.12.5 |
| uvicorn | 0.38.0 |

These versions match the pre-existing environments' package metadata for the
corresponding components. The lock prefers the existing runtime's transitive pins
where compatible; native wheels are installed afresh for Python 3.12. This does
not copy packages out of either old environment.

The original `flwr==1.5.0` declaration cannot coexist with MLflow 3.7.0:
[Flower 1.5.0](https://pypi.org/pypi/flwr/1.5.0/json) requires
`cryptography>=41.0.2,<42`, while
[MLflow 3.7.0](https://pypi.org/pypi/mlflow/3.7.0/json) requires
`cryptography>=43,<47`. Flower 1.25.0 matches the previous training environment
metadata and accepts cryptography 44.0.3. Its existing NumPyClient/start-server
APIs are checked without changing application source. No Flower optional extras
are requested: the subprocess launcher does not require Ray.

MLflow 3.7.0 and scikit-learn 1.8.0 retain the versions associated with the
existing tracking/model artifacts. Phase 0 does not migrate databases, rewrite
artifacts, or establish model correctness.

## Maintaining the lock

uv 0.9.18 is a lock-generation tool, pinned in `requirements-tools.txt`. It is
optional for ordinary pip-based reproduction and is not an application dependency.
To regenerate the lock intentionally after reviewing direct dependency changes:

```powershell
& .\.venv-phase0\Scripts\python.exe -m pip --isolated install --index-url https://pypi.org/simple -r requirements-tools.txt
& .\.venv-phase0\Scripts\python.exe -m uv pip compile requirements.txt --universal --python-version 3.12.10 --generate-hashes --no-strip-markers --no-python-downloads --no-build --exclude-newer 2026-11-08T00:00:00Z --default-index https://pypi.org/simple --cache-dir .phase0-cache --output-file requirements.lock
```

Keep the existing lock as the resolver's version preference; do not add `--upgrade`
unless a dependency update is intended. The date cutoff makes the resolution
horizon explicit. Review changes to the lock and repeat installation/import checks
in a new environment. Do not use a global `pip freeze` as a replacement.

## Existing work and limits

- Phase 0 uses `codex/phase-0-environment`, created from `main` at `0316f9b`.
  Existing unstaged and untracked work was carried across; nothing was stashed,
  reset, staged, committed, or pushed. A branch does not separately snapshot
  uncommitted work: it remains in this shared working tree.
- Existing `venv/` and `.venv/` are retained. The former has Python-3.14 NumPy
  binaries behind a Python-3.12 interpreter; the latter lacks training packages.
  Use `.venv-phase0/` for this baseline.
- The smoke script checks exact direct versions, imports, legacy Flower API
  availability, native scientific-library interoperability, temporary joblib
  serialization, and optionally the CSV schema. It does not fit a model, run SMOTE,
  start Flower/FastAPI, or import `src/client.py`, `src/server.py`, or `app.py`.
- Training correctness issues from the audit remain: overlapping partitions,
  scaling leakage, discarded global weights, and inference preprocessing mismatch.
- The existing Dockerfile still uses Python 3.11 and installs `requirements.txt`.
  It is not the locked Phase 0 environment; Docker alignment/build validation is
  deferred. Ignore rules only exclude the local environment/cache directories.
- `run_simulation.py` remains preserved and untracked until separately reviewed.
  The earlier uncommitted application changes also remain uncommitted.

See [PHASE0_RESULTS.md](PHASE0_RESULTS.md) for the checks actually executed.
