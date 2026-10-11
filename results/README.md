# Result tables

These CSV files are the canonical experiment records used by the README, benchmark report, figures, and presentation.

| File | Contents |
| --- | --- |
| `validation_benchmark.csv` | Validation ranking and the locked-test values retained for selected baselines |
| `final_test_results.csv` | Aggregate BiomedBERT locked-test metrics |
| `biomedbert_test_per_class.csv` | BiomedBERT locked-test F1 by class |
| `hierarchical_results.csv` | Flat, Stage 1, Stage 2, and combined hierarchical validation results |
| `calibration_results.csv` | Validation calibration and confidence-threshold summary |
| `efficiency_results.csv` | Parameter count and Colab T4 inference timing |

Blank cells indicate a metric that was not present in the retained project record. The repository does not contain a verified learning-curve result, prediction-level calibration bins, or locked-test confusion-matrix counts, so it does not provide fabricated CSVs for those artifacts.
