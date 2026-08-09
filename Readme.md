# Feature–Model Combination and Ensemble Strategy

This project evaluates complementary molecular representations across multiple regression models for molecular activity prediction.

### Molecular Representations

The following five molecular representations are evaluated:

* **RDKit descriptors**
* **ECFP4 fingerprints**
* **ChemBERTa embeddings**
* **MoLFormer embeddings**
* **ErG fingerprints**

These representations are evaluated across fourteen regression models to identify effective feature–model combinations for each dataset.

## Why Not Simply Combine All Features?

Concatenating all feature types is not used as the primary approach. Continuous-valued features, such as molecular descriptors and learned embeddings, can dominate sparse binary fingerprint features in many machine learning models. As a result, fingerprint information may be under-utilized rather than contributing equally to the prediction.

Furthermore, different regression algorithms—including linear, tree-based, and distance-based models—capture different relationships from the same molecular representation. Consequently, no single feature–model combination consistently provides the best predictions across all molecules or datasets.

Hence, evaluating feature–model combinations individually and subsequently exploring whether complementary models can be combined into ensembles might prove an attractive strategy especially with respect to small datasets that could be prone to overfitting.

---

## Ensemble Strategy

Model diversity alone does not guarantee that ensembling will improve performance. A simple average assumes that all models contribute equally. However, averaging can also weaken a strong prediction on a difficult molecule by pulling it toward the consensus of weaker models. Therefore, blindly combining multiple models can dilute accurate predictions rather than improve them. The key is to identify models that make different errors.

Two highly accurate models that consistently succeed and fail on the same molecules provide limited additional information when combined. In contrast, models whose errors occur on different molecules can complement one another, allowing one model to compensate for the weaknesses of another.

## Complementarity-Based Ensemble Selection

Rather than evaluating every possible model combination, candidate model pairs are selected based on prediction complementarity.

Three diversity measures are used:

* **Q-statistic** — Measures the correlation between the correct and incorrect predictions of two models. Values closer to −1 indicate greater complementarity, while values closer to +1 indicate greater redundancy.
* **Disagreement** — Fraction of molecules for which exactly one of the two models makes an acceptable prediction. Higher values indicate greater diversity.
* **Double-fault** — Fraction of molecules for which both models simultaneously make unacceptable predictions. Lower values indicate better complementarity.

These metrics are calculated using cross-validation (CV) training predictions to identify model pairs that are both accurate and complementary.

### Dynamic Error Threshold

The complementarity metrics require each prediction to be classified as either a correct or incorrect prediction based on its error relative to the experimental value. A fixed error threshold, such as 0.5, is not appropriate across datasets because prediction difficulty and activity distributions can vary substantially.

For example:

* In a highly homogeneous dataset where most models make small errors, a fixed threshold may classify nearly all predictions as correct, providing little meaningful information about model complementarity.
* In a difficult dataset where most models have large errors, the same threshold may classify nearly all predictions as incorrect, again making the diversity metrics uninformative.

To address this, a dynamic error threshold derived from the performance of the top three models on the cross-validation predictions is being computed. This provides a dataset-dependent definition of acceptable prediction error and allows the complementarity metrics to remain informative across datasets with different levels of prediction difficulty.

---

## Model and Ensemble Ranking

Individual models and selected ensembles are evaluated on a held-out scaffold split test set that is not used during model or ensemble selection. Models are ranked according to test RMSE, and the highest-performing models and ensembles are exported for downstream prediction.

---

## Installation

Create the required Conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate embedding-pipeline
```

---

## Training Ensemble Models

Use `ensem.py` to run the full feature/CV/ensemble training pipeline on a labeled dataset.
```bash
python ensem.py --data-path "train_val.csv" --smiles-col "Drug" --activity-col "Y" --output-dir "models1"
```
| Argument | Description |
|---|---|
| `--data-path` | CSV with training/validation data. |
| `--smiles-col` | SMILES column name. |
| `--activity-col` | Target/activity column name. |
| `--output-dir` | Directory to save models, features, and results. |

## Making Predictions

Use `prediction.py` to generate predictions for new molecules from a CSV of SMILES.
```bash
python prediction.py --model_path "model_path.pkl" --input_csv "input.csv" --smiles_col "column_name" --output_csv "output.csv"
```
| Argument | Description |
|---|---|
| `--model_path` | Path to trained `.pkl` model. |
| `--input_csv` | CSV with molecules to predict. |
| `--smiles_col` | SMILES column name. |
| `--output_csv` | Path to save predictions. |

## TDC Performance

Performance comparison against the top-performing model for each TDC ADMET benchmark. Only data as provided by the TDC train_val was used to build the models. Values are reported as mean ± standard deviation across the evaluated seeds.

| Dataset | Metric | Top Model | Our Model |
|---|---|---:|---:|
| Caco2 Wang | MAE ↓ | 0.256 ± 0.006 | 0.286 ± 0.005 |
| Lipophilicity AstraZeneca | MAE ↓ | 0.456 ± 0.008 | 0.514 ± 0.006 |
| Solubility AqSolDB | MAE ↓ | 0.741 ± 0.013 | 0.785 ± 0.011 |
| PPBR AZ | MAE ↓ | 7.440 ± 0.024 | 7.611 ± 0.083 |
| LD50 Zhu | MAE ↓ | 0.552 ± 0.009 | 0.581 ± 0.011 |
| Clearance Hepatocyte AZ | Spearman ↑ | 0.536 ± 0.020 | 0.460 ± 0.009 |
| Clearance Microsome AZ | Spearman ↑ | 0.630 ± 0.010 | 0.609 ± 0.016 |
| Half-Life Obach | Spearman ↑ | 0.576 ± 0.025 | 0.562 ± 0.028 |
| VDss Lombardo | Spearman ↑ | 0.713 ± 0.007 | 0.717 ± 0.005 |
