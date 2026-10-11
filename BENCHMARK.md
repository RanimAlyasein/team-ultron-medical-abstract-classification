# Benchmark and Evaluation Protocol

## Selection rule

Macro F1 was chosen before model comparison. It gives equal weight to all five classes and exposes failures on the smaller digestive and nervous-system categories that accuracy can hide.

All feature selection, hyperparameter tuning, epoch selection, and architecture comparisons used the validation split. The test split remained locked until final evaluation.

## Data contract

- Clean master: 8,298 unique, single-label abstracts
- Train: 5,808 abstracts
- Validation: 1,245 abstracts
- Test: 1,245 abstracts
- Split seed: 42
- Exact-text overlap between final splits: 0
- Primary metric: Macro F1

## Validation ranking

| Rank | Model | Track | Accuracy | Macro F1 |
| ---: | --- | --- | ---: | ---: |
| 1 | **BiomedBERT** | New method | 0.8683 | **0.8672** |
| 2 | BioBERT | New method | 0.8490 | 0.8456 |
| 3 | DeBERTa-v3 | New method | 0.8313 | 0.8322 |
| 4 | Logistic Regression | Baseline | 0.8217 | 0.8203 |
| 5 | Linear SVM | Baseline | 0.8241 | 0.8180 |
| 6 | BERT-base | Baseline | Not reported | 0.8124 |
| 7 | DistilBERT | Baseline | 0.8048 | 0.8027 |
| 8 | Naive Bayes, tuned | Baseline | 0.7920 | 0.7889 |
| 9 | TextCNN | Baseline | Not reported | 0.7882 |
| 10 | BiLSTM | Baseline | 0.7333 | 0.7127 |
| 11 | Naive Bayes, default | Baseline | 0.6755 | 0.4849 |

The baseline group follows the course coverage: Naive Bayes, Logistic Regression, Linear SVM, TextCNN, BiLSTM, BERT-base, and DistilBERT. The new-method group contains DeBERTa-v3, BioBERT, and BiomedBERT.

## Locked test result

BiomedBERT was selected exclusively by validation Macro F1. The saved checkpoint was then evaluated once on the locked test set.

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8635 |
| Macro precision | 0.8563 |
| Macro recall | 0.8651 |
| Macro F1 | **0.8603** |
| Weighted F1 | 0.8623 |

### Per-class test F1

| Class | F1 |
| --- | ---: |
| Neoplasms | 0.9367 |
| Digestive system diseases | 0.8774 |
| Nervous system diseases | 0.7975 |
| Cardiovascular diseases | 0.9151 |
| General pathological conditions | 0.7746 |

## Hierarchical experiment

Error analysis showed that most remaining mistakes involved the broad general-pathology class. A two-stage BiomedBERT system therefore routed each abstract through:

1. Specific disease versus general pathological condition
2. One of the four specific disease categories

Stage 2 reached Macro F1 0.9478 on its four-class subset, but routing errors reduced the combined five-class validation Macro F1 to **0.8599**. The flat BiomedBERT score of **0.8672** remained better and was retained.

## Calibration

The final BiomedBERT model was overconfident on validation.

| Metric | Value |
| --- | ---: |
| Validation accuracy | 0.8683 |
| Mean confidence | 0.9706 |
| Expected Calibration Error | **0.1041** |
| Coverage at confidence ≥ 0.90 | 0.9261 |
| Accuracy among accepted predictions | 0.8968 |

Confidence is useful for ranking cases for human review, but should not be treated as a calibrated probability without additional calibration.

## Efficiency

| Measure | Value |
| --- | ---: |
| Parameters | 109,486,085 |
| Validation examples | 1,245 |
| Inference time | 34.95 s |
| Time per abstract | 28.07 ms |
| Throughput | **35.62 abstracts/s** |
| Hardware | Google Colab T4 GPU |

## Interpretation

Domain-specific pretraining produced the largest improvement. BiomedBERT uses a BERT-base-sized architecture but scored about 5.5 Macro F1 points above general BERT on validation. Classical TF-IDF models remained strong: tuned Logistic Regression and Linear SVM matched or exceeded the general-purpose transformer baselines.

The main limitation is label ambiguity around general pathological conditions. The project used one deterministic split and did not run repeated seeds or cross-validation, so small differences should be interpreted cautiously. Biomedical pretraining may also have exposed models to some abstract text without labels.

## Machine-readable records

- `results/benchmark_validation.csv`
- `results/final_test_metrics.csv`
- `results/biomedbert_test_per_class.csv`
- `results/hierarchical_validation.csv`
- `results/calibration_summary.csv`
- `results/efficiency.csv`
