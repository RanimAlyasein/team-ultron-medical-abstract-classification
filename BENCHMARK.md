# Benchmark and Experiment Details

This document records the technical benchmark details that support the summary in the main [README](README.md). Model selection and tuning used the validation split only; the test split remained locked until the final BiomedBERT evaluation.

## Evaluation Contract

- Primary metric: Macro F1
- Supporting metrics: accuracy, macro precision, macro recall, weighted F1, per-class F1, and confusion matrices
- Final split: 5,808 training, 1,245 validation, and 1,245 test abstracts
- Split method: stratified with random seed 42
- Leakage control: zero abstract-ID and exact-text overlap between final splits

Macro F1 was selected because the cleaned dataset is imbalanced. The default Multinomial Naive Bayes model illustrates the issue: its validation accuracy was 0.6755, while its Macro F1 was only 0.4849.

## Architecture and Checkpoint Terminology

An architecture defines a model's computational design. A checkpoint is a set of learned weights used with that architecture. BERT-base and DeBERTa-v3 represent different transformer architectures, while BioBERT and BiomedBERT are biomedical-pretrained checkpoints built from BERT-family architectures. The benchmark therefore treats biomedical pretraining as a meaningful experimental factor rather than describing every checkpoint as a new architecture.

## Validation Benchmark

| Model | Family | Accuracy | Macro F1 |
|---|---|---:|---:|
| BiomedBERT | Biomedical transformer | 0.8683 | 0.8672 |
| BioBERT | Biomedical transformer | 0.8490 | 0.8456 |
| DeBERTa-v3 | Newer transformer | 0.8313 | 0.8322 |
| Linear SVM | Classical | 0.8241 | 0.8180 |
| Logistic Regression | Classical | 0.8217 | 0.8203 |
| BERT-base | General transformer | — | 0.8124 |
| DistilBERT | General transformer | — | 0.8027 |
| Tuned Naive Bayes | Classical | — | 0.7889 |
| TextCNN | Neural | — | 0.7882 |
| BiLSTM | Neural | 0.7333 | 0.7127 |
| Default Naive Bayes | Classical | 0.6755 | 0.4849 |

The machine-readable table is available in [`results/validation_benchmark.csv`](results/validation_benchmark.csv). Default and tuned Naive Bayes are reported separately because tuning materially changed the result.

## Classical Feature and Tuning Experiments

The classical benchmark examined:

- word-level unigram TF-IDF;
- word-level unigram and bigram TF-IDF;
- combined word and character TF-IDF, with character n-grams spanning lengths 3–5;
- validation-only tuning for Multinomial Naive Bayes, Logistic Regression, and Linear SVM.

The classical models provide transparent and computationally efficient reference points. Logistic Regression and Linear SVM remained strong baselines, but BiomedBERT produced the best validation Macro F1.

## Neural Experiments

TextCNN experiments compared convolution filter configurations, including `(3, 4, 5)` and `(2, 3, 4, 5)`, and used class-weighted loss. The final recorded TextCNN Macro F1 was 0.7882.

The BiLSTM implementation was corrected to use packed sequences so padded tokens did not influence recurrent processing. It also used class-weighted loss. Its final validation accuracy was 0.7333 and Macro F1 was 0.7127.

## Transformer Configuration

Transformer inputs used `max_length = 512`. The retained length analysis showed that approximately 1.36% of BiomedBERT-tokenized abstracts exceeded that limit, so truncation affected a small but nonzero portion of the dataset.

The transformer experiments used three training epochs, AdamW with a learning rate of `2e-5`, weight decay of `0.01`, and a 10% warmup proportion. Batch sizes were adapted to available GPU memory. DeBERTa-v3 exceeded the initial memory configuration on the Colab T4 GPU and was run with batch size 2 and gradient accumulation over 4 steps.

Biomedical pretraining was evaluated directly by comparing BioBERT and BiomedBERT against general transformer baselines. BiomedBERT achieved the strongest validation result.

## Learning-Curve Evidence

The project notes identify a learning-curve experiment, but no verified learning-curve output or source values are present in the submitted repository. A `learning_curve_results.csv` file is therefore not fabricated. Reproducing that experiment would require the original run record or a new, explicitly documented execution.

## Per-Class Validation Results

The retained BiomedBERT validation report contains the following F1 scores:

| Class | Validation F1 |
|---|---:|
| Neoplasms | 0.9235 |
| Digestive system diseases | 0.8826 |
| Nervous system diseases | 0.8150 |
| Cardiovascular diseases | 0.9228 |
| General pathological conditions | 0.7859 |

The broad General class was the most difficult category. Its boundary overlaps semantically with more specific disease classes, which was a recurring source of error.

## Hierarchical Experiment

| System | Scope | Macro F1 | General-class F1 |
|---|---|---:|---:|
| Stage 1 | Specific vs General | 0.8572 | — |
| Stage 2 | Four specific classes | 0.9478 | — |
| Combined hierarchy | Routed five-class prediction | 0.8599 | 0.7978 |
| Flat BiomedBERT | Direct five-class prediction | 0.8672 | 0.7859 |

The hierarchy slightly improved the General-class F1, but routing errors reduced the overall Macro F1. The flat BiomedBERT model was therefore retained. The corresponding data are in [`results/hierarchical_results.csv`](results/hierarchical_results.csv).

## Calibration and Confidence

| Measure | Value |
|---|---:|
| Validation accuracy | 0.8683 |
| Mean confidence | 0.9706 |
| Expected calibration error | 0.1041 |
| Confidence threshold | 0.90 |
| Coverage at threshold | 92.61% |
| Accepted samples | 1,153 / 1,245 |
| Accuracy on accepted predictions | 89.68% |

The mean confidence exceeds observed accuracy, so the model is overconfident. Its confidence scores should not be interpreted as perfectly calibrated probabilities. Only aggregate calibration values were retained; prediction-level confidence bins were not, so a reliability diagram cannot be reconstructed without rerunning inference. See [`results/calibration_results.csv`](results/calibration_results.csv) and [`src/calibration_analysis.py`](src/calibration_analysis.py).

## Efficiency

| Measure | Value |
|---|---:|
| Parameters | 109,486,085 |
| Validation samples | 1,245 |
| Total inference time | 34.95 seconds |
| Time per abstract | 28.07 ms |
| Throughput | 35.62 abstracts/second |
| Hardware | Google Colab T4 GPU |

These measurements demonstrate practical inference speed on a T4 GPU. They do not establish performance on other hardware or under production load. The machine-readable result is in [`results/efficiency_results.csv`](results/efficiency_results.csv).

## Locked Test Evaluation

The test split was not used for tuning or model selection. After BiomedBERT was selected using validation Macro F1, it was evaluated once on the locked test set.

| Metric | Test value |
|---|---:|
| Accuracy | 0.8635 |
| Macro Precision | 0.8563 |
| Macro Recall | 0.8651 |
| Macro F1 | 0.8603 |
| Weighted F1 | 0.8623 |

| Class | Test F1 |
|---|---:|
| Neoplasms | 0.9367 |
| Digestive system diseases | 0.8774 |
| Nervous system diseases | 0.7975 |
| Cardiovascular diseases | 0.9151 |
| General pathological conditions | 0.7746 |

Macro F1 changed from 0.8672 on validation to 0.8603 on test, a difference of 0.0069. This is a small validation-to-test gap indicating stable generalization. Aggregate and per-class values are retained in [`results/final_test_results.csv`](results/final_test_results.csv).

The repository does not contain the prediction-level locked-test output or the final trained checkpoint. Consequently, a final test confusion matrix cannot be reconstructed without a new evaluation run and is not presented as an existing artifact.

## Reproducibility Notes

The consolidated [`notebooks/04_baseline_benchmark.ipynb`](notebooks/04_baseline_benchmark.ipynb) contains the available classical, neural, transformer, hierarchy, calibration, efficiency, and locked-test workflow. Separate DistilBERT and final BiomedBERT notebooks were not retained, so placeholder notebooks are not added. The final Colab workflow is linked from the main README.
