# Final Model Report

## Selection

BiomedBERT was selected by validation Macro F1. It scored 0.8672, the highest result among the ten compared model families and the hierarchical alternative.

## Locked test evaluation

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8635 |
| Macro precision | 0.8563 |
| Macro recall | 0.8651 |
| Macro F1 | **0.8603** |
| Weighted F1 | 0.8623 |

## Reliability and efficiency

Validation Expected Calibration Error was 0.1041. Mean confidence, 0.9706, exceeded validation accuracy, 0.8683, so raw confidence should not be interpreted as a calibrated probability.

Inference over 1,245 validation abstracts took 34.95 seconds on a Google Colab T4 GPU. This equals 28.07 milliseconds per abstract and 35.62 abstracts per second.

## Error pattern

General pathological conditions had the lowest test F1 at 0.7746. Most remaining errors involved the boundary between that broad class and one of the four specific disease classes.

The hierarchical experiment did not solve the issue. Its combined validation Macro F1 was 0.8599, below the flat model's 0.8672, because Stage 1 routing errors propagated into the final decision.

## Limitations

- Results use one deterministic train/validation/test split rather than repeated seeds or cross-validation.
- Several abstracts remain semantically compatible with more than one disease category even after exact conflicting-label groups were removed.
- BiomedBERT is overconfident and needs calibration before confidence-based automation.
- Biomedical pretraining may have exposed the model to some abstract text without labels.
