# Phase 2B aggregation

Each client reports the size of its eligible, deduplicated Phase 1A training
partition in `FitRes.num_examples`. `post_smote_examples` in the fit metrics
records the number of rows used by the local optimizer for diagnostics. Flower
1.25.0's FedAvg combines each parameter array as `sum(n_i * w_i) / sum(n_i)`,
where `n_i` is the reported original count. The server delegates this average
to Flower. With coefficients 2 and 8 and original counts 20 and 60, the
aggregate is 6.5. With 20 originals each, it is 5, even if SMOTE produces
28 rows for one client and 20 for the other. Previously, those resampled
counts produced an aggregate of 4.5.

SMOTE still changes the data and objective of each local fit. Original-count
weighting defines each client's influence on the average. It does not undo
resampling or make the federated updates equivalent to centralized fitting on
the original observations.

## Validation and participation

`EvaluateRes.num_examples` is the number of validation rows. Flower already
uses those counts for validation loss. Accuracy uses the same counts, which
gives the pooled fraction correct for the included validation rows. With
9 correct out of 10 and 15 correct out of 30, pooled accuracy is 24/40 (0.60).
The old mean of client accuracies was 0.70. MLflow still logs `accuracy` and
`loss` for each aggregated evaluation round. Final-test data remain reserved.

Fit and validation fractions are 1.0. The configured client count is the
minimum requested for sampling and availability. If extra clients are
connected when Flower computes the sample size, it can request more than that
minimum. Sampling does not guarantee successful replies. The inherited
`accept_failures=True` policy aggregates valid replies from successful clients
and renormalizes weights over that subset. With no successful fit replies, the
server retains its previous parameters. With no successful validation replies,
there is no aggregate or MLflow metric record for that evaluation.

The strategy rejects a malformed successful reply with an error. Counts must
be positive integers within Flower's signed 64-bit wire range. If present, a
post-SMOTE count must be even for the existing balanced binary resampling and
lie between the original count and strictly below twice that count. The server
checks the two finite `float64` parameter arrays against shapes `(1, 30)` and
`(1,)`, including the aggregate, to prevent accidental broadcasting. Each
validation reply must have finite nonnegative loss and numeric accuracy in
`[0, 1]`. Missing or malformed metrics prevent publishing an aggregate. These
checks reject invalid types, ranges, and shapes. The server trusts clients to
report the sizes of their assigned partitions accurately.

## Verification

```powershell
& .\.venv-phase0\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -v
```

Synthetic tests send client requests through Flower's NumPyClient wrapper and
exercise the project's strategy through Flower's actual FedAvg methods. Model
saves and MLflow logging are mocked in the aggregation tests. A full
federated simulation is outside this verification.
