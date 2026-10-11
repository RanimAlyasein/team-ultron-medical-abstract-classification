# Processed data

The processed files were generated from `data/raw/` by the scripts in `src/`.

| File | Rows | Description |
| --- | ---: | --- |
| `medical_abstracts_clean.csv` | 8,298 | Unique, single-label master dataset |
| `train.csv` | 5,808 | Training split |
| `validation.csv` | 1,245 | Model selection and tuning split |
| `test.csv` | 1,245 | Locked final evaluation split |

Columns:

- `abstract_id`: first 16 hexadecimal characters of the SHA-256 hash of the exact abstract text
- `medical_abstract`: original abstract text
- `condition_label`: integer label from 1 through 5
- `condition_name`: human-readable class name

The split uses seed 42 with stratification by label. There are no missing values, duplicate exact texts, or exact-text overlaps between splits.
