# Phase 3A controlled simulation

The previous launcher started the server and clients in the repository with
fixed delays, unbounded waits, and no exit-code checks. Client 0 saved over
`best_model.pkl` during evaluation. The server used the repository's `mlflow.db`,
and MLflow could resolve artifact paths relative to the working directory.
Flower telemetry and dependency defaults could use the user profile, caches,
or system temporary directories. The launcher terminated the server after the
client waits and reported success even when a client failed.

## Run directory and outputs

Use the Phase 0 environment. Supply `--dataset`, `--run-dir`, `--port`,
`--num-clients`, `--rounds`, and `--timeout` (seconds) to `run_simulation.py`.
The run directory must be new, on a local drive, and outside the checkout.
Its parent must exist. Use a parent directory under your control. The launcher
checks the dataset schema and partition feasibility, probes the port, and then
creates the run directory and copies the exact input bytes to `input.csv`.
The server binds to `127.0.0.1`. A competing bind between preflight and Flower
startup can cause startup to fail.

The child environment routes the working directory, profile, temporary files,
and caches into the run directory, disables telemetry and bytecode writes, and
limits numerical library threads. The Windows launcher uses the base Python
interpreter with the Phase 0 packages so its subprocess handle tracks the actual
child instead of the virtual environment's redirector. Direct server and
client entry points require an explicit run directory and port. Use the launcher
to establish the isolated environment for both.

| Path within the run directory | Contents |
| --- | --- |
| `input.csv` | Snapshot of the supplied CSV |
| `server.log`, `client-0.log`, `client-1.log`, etc. | Child stdout and stderr |
| `tracking.sqlite` | New local MLflow database and any transient SQLite files |
| `artifacts/` | Explicit artifact location for the experiment |
| `mlruns/` | MLflow's local default artifact root |
| `model.pkl` | Latest successful client-0 validation checkpoint for this run |
| `rounds.json` | Completed round evidence, written after Flower returns |
| `tmp/`, `cache/`, `profile/` | Child runtime files, when used |

The launcher retains the run directory on success or failure. It waits for
Flower's readiness message, checks child exit codes, and enforces an overall
process deadline after preflight, with a separate per-RPC timeout. On failure
it terminates and reaps its children. The controlled strategy requires a fit
and validation reply from every configured client each round. It checks the
clients' parameter fingerprints, delegates aggregation to the Phase 2B
strategy, and records the aggregate fingerprint for comparison with the next
round's input. Incomplete rounds raise an explicit error.

## Synthetic verification

```powershell
& .\.venv-phase0\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -v
```

The integration test generates a 160-row binary CSV using seed 42, starts two
clients for two rounds, and checks the round report and isolated SQLite database.
To retain evidence, set `FRAUD_SMOKE_EVIDENCE_DIR` to a new directory outside the
checkout before running the suite. Its parent must exist. The test writes the
generated CSV there and uses its `experiment` child for run outputs. The test
uses a 100-second process deadline. Launcher tests check unsafe destinations,
port conflicts, startup timeouts, nonzero exits, and incomplete participation.

The smoke test exercises transport, local updates, aggregation, and validation.
Final-test scoring and fraud detection performance are outside its scope. A
reference-data run needs a fresh directory with space for the snapshot and a
timeout appropriate for loading and training. Two initial Windows runs failed
in `threadpoolctl` during the first fit with `GetModuleFileNameEx failed`.
The client now initializes the pinned scikit-learn thread-pool controller before
opening its Flower connection on Windows. The next run completed both rounds.
Recheck this startup path when changing the scikit-learn or threadpoolctl pin.
The installed Flower 1.25
compatibility APIs are deprecated. A saved checkpoint needs its exact fitted
preprocessing sidecar before serving.
