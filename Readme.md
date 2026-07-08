# Feature--Model Combination and Ensemble Strategy

This project evaluates four complementary molecular representations:

-   **RDKit descriptors**
-   **ECFP4 fingerprints**
-   **ChemBERTa embeddings**
-   **MoLFormer embeddings**

across nine regression models to predict molecular activity.

Simply concatenating all feature types is not used as the primary
approach. Continuous-valued features (descriptors and learned
embeddings) often dominate sparse binary fingerprint bits in many
machine learning models, causing fingerprint information to be
under-utilized rather than contributing equally. Likewise, different
regression algorithms (linear, tree-based, distance-based, etc.) capture
different relationships from the same representation, meaning that no
single feature--model combination performs best for every molecule.

This becomes especially apparent for **activity cliffs**, where small
structural changes produce large changes in biological activity. Models
frequently disagree on these difficult compounds, with one correctly
predicting molecules that another misses.

## Why Ensemble?

Although model diversity suggests that combining predictions could
improve performance, ensembling is **not automatically beneficial**. A
simple average assumes every model contributes equally, but averaging
can also reduce the impact of a model that is genuinely correct on
difficult molecules by pulling its predictions toward the consensus.
Consequently, blindly combining all models can dilute strong predictions
instead of improving them.

The important part is to identify modes that make different mistakes.

Two highly accurate models that consistently fail on the same molecules
contribute little when combined. In contrast, two models whose errors
occur on different compounds can complement each other, allowing one
model to compensate for the other's weaknesses.

## Complementarity-Based Ensemble Selection

Instead of ensembling every possible combination, candidate pairs are
selected based on **prediction complementarity**. Complementarity is
evaluated using three diversity measures:

-   **Q-statistic** -- Measures correlation between correct and
    incorrect predictions of two models (*−1 = highly complementary, +1
    = highly redundant*).
-   **Disagreement** -- Fraction of molecules where only one model
    predicts correctly.
-   **Double-fault** -- Fraction of molecules where both models fail
    simultaneously (*lower is better*).

These metrics are computed from fold-based training predictions and
analyzed separately for **activity cliff** and **non-cliff** molecules
to identify model pairs that are both **accurate** and **diverse**.

Single models and selected ensembles are then ranked according to their
**test RMSE**, and the highest-performing models can be exported as
serialized files for downstream prediction. This approach ensures that
ensembles are formed because they genuinely compensate for each other's
errors, rather than simply averaging together a collection of different
models.

## Installation

Create the required Conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate embedding-pipeline
```
---

## Training Ensemble Models

To reproduce the ensemble selection pipeline and generate the final serialized ensemble models, run the Jupyter notebook:

```text
ensemble_modelling.ipynb
```

The notebook evaluates individual models, computes complementarity metrics (Q-statistic, disagreement, and double-fault), ranks candidate ensembles, and exports the selected models as `.pkl` files.

---

## Making Predictions

Use `prediction.py` to generate predictions for new molecules from a CSV file containing SMILES strings.

### Command

```bash
python prediction.py \
    --model_path "model_path.pkl" \
    --input_csv "input_csv_path.csv" \
    --smiles_col "column_name" \
    --output_csv "output_csv_path.csv"
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--model_path` | Path to the trained `.pkl` model generated from the ensemble pipeline. |
| `--input_csv` | Input CSV file containing molecules to predict. |
| `--smiles_col` | Name of the column containing SMILES strings. |
| `--output_csv` | Path where the prediction results will be saved. |

### Example

```bash
python prediction.py \
    --model_path models/top1_model.pkl \
    --input_csv data/test_molecules.csv \
    --smiles_col SMILES \
    --output_csv predictions.csv
```