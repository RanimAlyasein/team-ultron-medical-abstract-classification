# Source scripts

Run scripts from the repository root.

| Script | Purpose |
| --- | --- |
| `01_dataset_audit.py` | Audit shape, labels, duplicates, conflicts, and original split overlap |
| `02_clean_dataset.py` | Remove every exact-text group associated with multiple labels |
| `03_create_splits.py` | Create the deterministic stratified 70/15/15 split |
| `run_classical.py` | Reproduce the four reported classical configurations |
| `calibration_analysis.py` | Recompute confidence coverage, ECE, and reliability bins from exported predictions |
| `make_figures.py` | Generate repository charts from canonical result CSVs |
| `verify_artifacts.py` | Check required files, split integrity, and final metric values |

The data pipeline uses only the Python standard library. The benchmark, calibration, and figure scripts require the packages in `requirements.txt`.

The canonical calibration summary is retained in `results/calibration_results.csv`. A reliability diagram can only be regenerated after exporting the prediction-level `true_label`, `predicted_label`, and `confidence` columns from the notebook; those raw probabilities are not committed.
