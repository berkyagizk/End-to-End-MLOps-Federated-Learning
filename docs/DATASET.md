# Credit card dataset

Download the original **Credit Card Fraud Detection** dataset from its
[Kaggle publisher page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
Sign in and accept the publisher's terms if required. Extract the downloaded
archive locally and place the unmodified CSV at:

```text
BitirmeProjesi/
  data/
    creditcard.csv
```

`data/` is already excluded from Git and Docker. Do not commit the dataset or
account credentials. No automatic download or preprocessing is performed in Phase 0.
Run existing training commands from the repository root because the client uses
the relative path `data/creditcard.csv`.

## Expected schema

The CSV has a header, 284,807 records, and these **31 columns in exactly this order**:

```text
Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class
```

| Columns | Expected contents |
| --- | --- |
| `Time` | Numeric transaction time feature, unscaled |
| `V1` through `V28` | Numeric anonymized features |
| `Amount` | Numeric transaction amount, unscaled |
| `Class` | Binary target: `0` legitimate, `1` fraud |

There must be no extra exported index column and no missing or infinite values.
The expected class counts are **284,315 legitimate** and **492 fraud**. There are
30 model features; `Class` is excluded from model input. Keep the raw file intact:
do not normalize, resample, reorder columns, or remove duplicates for Phase 0.

## Validate and identify the file

```powershell
& .\.venv-phase0\Scripts\python.exe -B scripts/check_environment.py --dataset data/creditcard.csv
Get-FileHash -LiteralPath data/creditcard.csv -Algorithm SHA256
```

The locally audited file is 150,828,752 bytes with SHA-256:

```text
76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89
```

This is a locally measured fingerprint, not a publisher-signed checksum. The
checker validates schema, numeric finiteness, row/class counts, and reports the
fingerprint. It does not reject a file solely because formatting or line endings
produce a different checksum. Investigate any difference before comparing runs.
