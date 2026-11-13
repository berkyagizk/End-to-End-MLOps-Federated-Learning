# Phase 2A local training

The Phase 1B client copied incoming coefficients and intercepts into
`LogisticRegression(max_iter=1)` before calling `fit`. With the default
`warm_start=False`, scikit-learn discarded that starting point. On a controlled
binary dataset, two different starting vectors yielded identical fitted weights.

## Flower parameter exchange

The server uses the existing FedAvg startup flow. With no strategy-supplied
initial parameters, Flower requests parameters from one connected client. Every
fresh client now starts with zero weights, so the selected client's data and
connection order do not determine the starting vector. The previous two-sample
initial fit could make startup weights depend on the selected client.

Clients exchange two floating-point arrays in order: coefficients shaped
`(1, n_features)` and intercept shaped `(1,)`. The current preprocessing schema
provides 30 features. Incoming arrays must have those shapes and contain finite
values representable as `float64`. The client validates both arrays before
replacing either, and copies arrays when accepting or returning them. Class
order is `[0, 1]`. The coefficient row and intercept represent the log odds
of class `1`. The client requires both labels in its training set.

## Local update and evaluation

Each `fit` call replaces both arrays with the server's parameters, then runs
the existing train-only SMOTE operation and fits `LogisticRegression` with
`solver="lbfgs"`, `warm_start=True`, and `max_iter=1`. With the locked
scikit-learn 1.8.0, this solver concatenates the incoming coefficients and
intercept and passes that vector as `x0` to SciPy's L-BFGS-B. The solver starts
a fresh local optimization at those weights each round. It does not carry
L-BFGS history between rounds. The one-iteration budget is a local update
policy and may produce an iteration-limit convergence warning.

The fit reply keeps the existing post-SMOTE sample count. Evaluation applies
the received weights, uses the client's validation rows, and retains the
existing client-0 save callback. Server aggregation and checkpoint/export
policies are outside this phase.

## Verification

```powershell
& .\.venv-phase0\Scripts\python.exe -B -m unittest discover -s tests -v
```

Synthetic tests send requests through the installed Flower NumPyClient wrapper.
They capture the actual SciPy starting vector and compare the fitted weights
with a direct L-BFGS-B step on the binary logistic objective with an L2 penalty
on the coefficients. They check that distinct server weights yield distinct
updates and that a second round matches a fresh client given the same server
weights. The tests intercept the save callback and use no production artifacts.

The optimizer-start test intentionally checks a solver call in the locked
scikit-learn version. Reverify it when changing the solver or dependency pins.
These tests establish local update behavior. They do not establish full
federated convergence or validate the post-SMOTE aggregation weights.
