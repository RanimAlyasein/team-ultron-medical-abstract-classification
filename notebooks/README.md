# Notebook provenance

`04_baseline_benchmark.ipynb` is a repository snapshot of the shared Google Colab notebook:

<https://colab.research.google.com/drive/16WUP3B2rwgcXii_abqoziN7KphqATjCP?usp=sharing>

Repository preparation applied a narrow cleanup:

- removed execution counts and stored outputs
- removed one obsolete display-only status cell
- aligned the BiomedBERT benchmark constant with the final validation Macro F1 of 0.8672
- aligned the TextCNN benchmark constant with the final validation Macro F1 of 0.7882
- added a short Markdown header containing the canonical final metrics

The executable experiment cells remain in their original order. The notebook expects `train.csv`, `validation.csv`, and `test.csv` from `data/processed/` to be available in the Colab working directory.

The notebook documents the experimental workflow, but `results/*.csv` is the authoritative record for the final report values.
