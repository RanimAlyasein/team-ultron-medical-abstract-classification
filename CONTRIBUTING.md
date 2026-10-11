# Contributing

## Reproducibility rules

1. Do not modify files in `data/raw/`.
2. Use the frozen files in `data/processed/` for all comparisons.
3. Fit vocabularies, tokenizers that learn from data, and model parameters on the training split only.
4. Use validation Macro F1 for model and hyperparameter selection.
5. Keep the test split locked until the final evaluation of a selected model.
6. Add machine-readable result rows under `results/` and record the exact configuration in the notebook or script that produced them.
7. Run `python src/verify_artifacts.py` before opening a pull request.

## Style

- Prefer deterministic scripts with explicit random seeds.
- Preserve the original abstract text in shared dataset files.
- Do not commit model checkpoints, virtual environments, or notebook execution outputs.
- State unavailable metrics as not reported rather than estimating them.
