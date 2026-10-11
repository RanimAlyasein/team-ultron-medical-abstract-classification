# Medical Abstract Disease Classification: Dataset Preparation

## 1. Project task

Our project is a single-label, multiclass text-classification task. Given the
text of a medical abstract, the model must assign exactly one of five disease
categories:

1. Neoplasms
2. Digestive system diseases
3. Nervous system diseases
4. Cardiovascular diseases
5. General pathological conditions

This document records how we audited, cleaned, and split the dataset before
model development. It is the shared reference for the dataset used in all
experiments.

## 2. Source data

The source data came from the Medical Abstracts Text Classification Corpus:

- Repository: <https://github.com/sebischair/Medical-Abstracts-TC-Corpus>
- Upstream commit: `70a2d9106c724729be8b3c4ddb00d1b14ec300c8`
- Retrieval date: 2026-10-09

The original data files were:

- `medical_tc_train.csv` — 11,550 labeled abstracts
- `medical_tc_test.csv` — 2,888 labeled abstracts
- `medical_tc_labels.csv` — mapping from numeric labels to disease categories

Together, the original training and test files contained **14,438 rows**. Both
data files used the columns `condition_label` and `medical_abstract`. The raw
files were retained unchanged as the source of truth.

### Original class distribution

| Label | Disease category | Original train | Original test | Combined |
| ---: | --- | ---: | ---: | ---: |
| 1 | Neoplasms | 2,530 | 633 | 3,163 |
| 2 | Digestive system diseases | 1,195 | 299 | 1,494 |
| 3 | Nervous system diseases | 1,540 | 385 | 1,925 |
| 4 | Cardiovascular diseases | 2,441 | 610 | 3,051 |
| 5 | General pathological conditions | 3,844 | 961 | 4,805 |
| **Total** |  | **11,550** | **2,888** | **14,438** |

## 3. Initial audit

We audited the raw files before making any changes. Duplicate, conflict, and
overlap checks compared the exact abstract strings stored in the CSV files; no
lowercasing, trimming, whitespace normalization, stemming, or other text
transformation was applied during these checks.

The audit confirmed:

- Expected columns were present in both input datasets.
- Missing `condition_label` values: **0**.
- Missing `medical_abstract` values: **0**.
- Total rows: **14,438**.
- Unique exact abstract texts: **11,227**.

### Exact duplicates and conflicting labels

| Scope | Unique abstracts | Repeated abstract groups | Rows in repeated groups | Excess rows |
| --- | ---: | ---: | ---: | ---: |
| Original train | 9,445 | 1,956 | 4,061 | 2,105 |
| Original test | 2,770 | 113 | 231 | 118 |
| Combined data | 11,227 | 2,929 | 6,140 | 3,211 |

The important finding was that all **2,929** repeated exact-text groups were
ambiguous: the same abstract text had been assigned more than one disease
label. These groups covered **6,140 source rows**. Therefore, they were not
ordinary same-label duplicates that could safely be reduced to one row; they
represented contradictory targets for a single-label classifier.

### Leakage in the original train/test split

The audit also found exact abstract text shared across the original training
and test files:

| Overlap measure | Count |
| --- | ---: |
| Unique abstract texts present in both files | 988 |
| Original train rows whose text appeared in original test | 1,097 |
| Original test rows whose text appeared in original train | 1,010 |
| Exact cross-split row pairs | 1,119 |

This meant that a model could encounter an abstract during training and then
be evaluated on the same text in the test set. Such leakage can inflate test
performance and prevents the test results from measuring generalization to
unseen abstracts.

## 4. Why we rejected the original split

We did not use the original train/test split for model evaluation for two
reasons:

1. **Exact-text leakage:** 988 unique abstracts appeared in both splits, so the
   original test set was not independent of the training set.
2. **Contradictory supervision:** repeated abstracts were associated with
   different labels, which is incompatible with our task definition of one
   correct label per abstract.

Keeping the original split would have made the benchmark unreliable. Instead,
we combined the original files for cleaning and generated new train,
validation, and test sets from the verified clean master dataset.

## 5. Cleaning decisions

We applied the following rules:

1. Combined the original training and test rows for dataset-level auditing.
2. Grouped records by the exact `medical_abstract` text.
3. Removed every abstract group associated with more than one distinct label.
   This removed all 2,929 ambiguous abstract texts and their 6,140 source rows.
4. Retained each remaining unambiguous abstract once with its single label.
5. Preserved the original abstract text. We did not lowercase it, remove
   punctuation or stop words, stem or lemmatize tokens, or otherwise apply
   aggressive natural-language preprocessing to the shared CSV files.

Removing ambiguous groups was chosen instead of selecting a majority label or
arbitrarily keeping the first label because either alternative would invent a
target that was not reliably supported by the source. Preserving the text
keeps the dataset suitable for different model families and prevents one
model-specific preprocessing choice from being permanently embedded in the
shared data. Any model-specific transformation must therefore occur inside the
model pipeline.

## 6. Final clean dataset

The clean master file, `medical_abstracts_clean.csv`, contains **8,298 unique,
single-label abstracts**.

| Label | Disease category | Final rows | Share |
| ---: | --- | ---: | ---: |
| 1 | Neoplasms | 2,195 | 26.45% |
| 2 | Digestive system diseases | 699 | 8.42% |
| 3 | Nervous system diseases | 1,049 | 12.64% |
| 4 | Cardiovascular diseases | 1,961 | 23.63% |
| 5 | General pathological conditions | 2,394 | 28.85% |
| **Total** |  | **8,298** | **100.00%** |

The final master dataset has:

- Missing values: **0**
- Duplicate exact abstract texts: **0**
- Abstracts with conflicting labels: **0**
- Unique single-label abstracts: **8,298**

## 7. Final split strategy

We created a new **70%/15%/15% train/validation/test split** using deterministic
stratification with random seed **42**. Stratification was performed by disease
label so that each split retained class proportions close to those of the clean
master dataset. The implementation is dependency-free and reproducible without
requiring scikit-learn.

### Split sizes

| Split | Rows | Percentage | Intended use |
| --- | ---: | ---: | --- |
| Train | 5,808 | 69.99% | Fit model parameters |
| Validation | 1,245 | 15.00% | Compare models and tune settings |
| Test | 1,245 | 15.00% | Final evaluation only |
| **Total** | **8,298** | **100.00%** |  |

### Class distribution by split

| Label | Disease category | Train | Validation | Test |
| ---: | --- | ---: | ---: | ---: |
| 1 | Neoplasms | 1,536 (26.45%) | 329 (26.43%) | 330 (26.51%) |
| 2 | Digestive system diseases | 489 (8.42%) | 105 (8.43%) | 105 (8.43%) |
| 3 | Nervous system diseases | 734 (12.64%) | 158 (12.69%) | 157 (12.61%) |
| 4 | Cardiovascular diseases | 1,373 (23.64%) | 294 (23.61%) | 294 (23.61%) |
| 5 | General pathological conditions | 1,676 (28.86%) | 359 (28.84%) | 359 (28.84%) |
| **Total** |  | **5,808** | **1,245** | **1,245** |

The split files are frozen for the modeling phase. All team members must use
the same files and must not generate independent splits or alter the shared
test set.

## 8. Verification checks

After splitting, we ran assertions covering dataset size, labels, missing
values, duplicates, and separation between splits.

| Split pair | ID overlap | Exact-text overlap |
| --- | ---: | ---: |
| Train–Validation | 0 | 0 |
| Train–Test | 0 | 0 |
| Validation–Test | 0 | 0 |

Additional checks confirmed:

- No missing values in any final split.
- No duplicate abstracts within any split.
- All five labels are present in every split.
- The three split sizes sum to 8,298.
- The clean master file was unchanged by the splitting stage.
- Clean master SHA-256 before and after splitting:
  `70c85bd37553dbea92871c64f3ed1d2b106e8ac48603a3440aed1f7f76fe2edf`.
- All automated assertions passed.

## 9. Reproducibility

The complete preparation process can be reproduced in order with:

1. `src/01_dataset_audit.py` — inspects source integrity, shape, missing values,
   class counts, exact duplicates, conflicting labels, and original split
   overlap.
2. `src/02_clean_dataset.py` — creates the clean master dataset and records the
   ambiguous groups removed.
3. `src/03_create_splits.py` — creates the deterministic stratified split with
   random seed 42 and runs split-verification checks.

The generated reports and supporting files are:

- `reports/dataset_audit_report.txt` — source statistics and audit findings.
- `reports/cleaning_report.txt` — cleaning rules and before/after counts.
- `reports/split_report.txt` — split sizes, class distributions, and checks.
- `reports/conflicting_abstracts.csv` — repeated texts carrying multiple labels.
- `reports/removed_ambiguous_abstracts.csv` — records excluded during cleaning.
- `reports/original_train_test_overlap.csv` — exact overlap found in the original split.

The raw files remain unchanged. Their integrity is independently recorded with
file sizes and SHA-256 checksums in the dataset audit output.

## 10. Final folder structure

```text
Medical-Abstracts-TC-Corpus-main/
├── DATASET_PREPARATION.md
├── data/
│   ├── raw/
│   │   ├── medical_tc_labels.csv
│   │   ├── medical_tc_test.csv
│   │   └── medical_tc_train.csv
│   └── processed/
│       ├── medical_abstracts_clean.csv
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
├── reports/
│   ├── dataset_audit_report.txt
│   ├── cleaning_report.txt
│   ├── split_report.txt
│   ├── conflicting_abstracts.csv
│   ├── removed_ambiguous_abstracts.csv
│   └── original_train_test_overlap.csv
├── src/
│   ├── 01_dataset_audit.py
│   ├── 02_clean_dataset.py
│   └── 03_create_splits.py
├── LICENSE
└── README.md
```

## 11. Why these decisions matter technically

- **Removing contradictory targets improves label validity.** A single text
  cannot provide reliable supervision when it is assigned several mutually
  exclusive labels in a single-label task.
- **Eliminating cross-split text overlap prevents leakage.** Test performance
  then reflects behavior on unseen abstracts instead of memorization of text
  encountered during training.
- **Stratification supports fair comparison.** Preserving class proportions is
  especially important because the five categories are imbalanced.
- **A separate validation set protects the final test result.** Model selection
  and tuning use validation data; the test set remains reserved for one final,
  unbiased evaluation.
- **Preserving source text keeps the benchmark flexible.** Traditional vector
  models and language models can apply their own appropriate preprocessing
  without inheriting irreversible dataset-level transformations.
- **A fixed seed and scripted workflow make results reproducible.** Every team
  member can work from identical data, and future experiments can be compared
  on the same benchmark.

## 12. Dataset-use rule

For all model experiments, use `train.csv` for training, `validation.csv` for
model selection and tuning, and `test.csv` only for final evaluation. Do not
modify these files, independently clean their text, rebalance them manually, or
create a different split. This keeps results comparable across the project.
