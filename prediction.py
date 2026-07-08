"""
predict_new_molecules.py

Loads a saved model/ensemble package (pickle) produced by your QSAR pipeline,
automatically computes only the feature types that package actually needs
(Descriptors / Fingerprints / Chemberta Embeddings / Molformer Embeddings),
applies the correct saved scaler per feature type, runs the model(s), and
writes predictions to a CSV.

-------------------------------------------------------------------------
REQUIRED INPUT: the package pickle must be a dict shaped like this
(this matches the `saved_top3` / `package_member` structure built earlier
in the pipeline):

    {
        "type": "single" | "ensemble",
        "members": [
            {
                "features": "Descriptors" | "Fingerprints" |
                            "Chemberta Embeddings" | "Molformer Embeddings",
                "model_name": str,
                "model": <fitted sklearn-like model, has .predict()>,
                "scaler": <fitted StandardScaler, has .transform()>,
                # only present / required for "Descriptors":
                "desc_columns_used": [list of column names, exact training order]
            },
            ...
        ],
        "test_rmse": float,   # optional, informational only
        "oof_rmse": float,    # optional, informational only
    }

If your saved pickle is a *dict of packages* (e.g. saved_top3 with multiple
entries), pass --package_key to select which one to use, otherwise the
script will use the single top entry (lowest test_rmse if multiple present,
or the only entry if there's just one).

-------------------------------------------------------------------------
USAGE:

    python predict_new_molecules.py \
        --model_path top3_deployable_models.pkl \
        --input_csv new_molecules.csv \
        --smiles_col SMILES \
        --output_csv predictions.csv \
        [--package_key "Fingerprints__RandomForest + Descriptors__Ridge"]

-------------------------------------------------------------------------
NOTES / THINGS TO DOUBLE-CHECK BEFORE TRUSTING THE OUTPUT:

1. Descriptors: this script recomputes the FULL RDKit descriptor set, then
   subsets down to the exact column list your training pipeline ended up
   keeping after its own NaN/constant-column filtering. That column list
   MUST be embedded in the package under "desc_columns_used" for any member
   using "Descriptors" features, or this script will raise an error rather
   than silently mis-align columns.

2. Fingerprints: uses Morgan fingerprints, radius=2, nBits=1024 (matches
   the training code you shared). If you changed these values later in
   your notebook without telling me, update RADIUS / N_BITS below to match.

3. Chemberta / Molformer: only loaded into memory if the package actually
   needs them (lazy loading) since these are large transformer models.
   Uses mean pooling for ChemBERTa (matching your training code) and the
   model's own pooler_output for MoLFormer (matching your training code).

4. Invalid SMILES: rows with unparseable SMILES get NaN features and will
   likely produce NaN or garbage predictions -- these rows are flagged in
   the output CSV with an "is_valid_smiles" column so you can filter them.

5. This script does NOT retrain or refit anything. It only transforms new
   molecules using scalers/models that were already fit during training.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------
# Feature computation settings -- MUST match your training pipeline
# ---------------------------------------------------------------
N_BITS = 1024
RADIUS = 2
CHEMBERTA_CHECKPOINT = "DeepChem/ChemBERTa-77M-MLM"
MOLFORMER_CHECKPOINT = "ibm/MoLFormer-XL-both-10pct"


def _lazy_import_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs, Descriptors
        from rdkit.ML.Descriptors import MoleculeDescriptors
        return Chem, AllChem, DataStructs, Descriptors, MoleculeDescriptors
    except ImportError as e:
        print("ERROR: rdkit is required but not installed. "
              "Install with: pip install rdkit --break-system-packages", file=sys.stderr)
        raise e


def build_descriptors_df(smiles_list, desc_columns_used):
    """Recomputes the full RDKit descriptor set, then subsets to the exact
    training-time surviving columns. Never recompute the NaN/constant-column
    filter on new data -- that filter is a property of the TRAINING set only."""
    Chem, AllChem, DataStructs, Descriptors, MoleculeDescriptors = _lazy_import_rdkit()

    descriptor_names = [name for name, _ in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    rows = []
    valid_flags = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append([np.nan] * len(descriptor_names))
            valid_flags.append(False)
            continue
        try:
            rows.append(calculator.CalcDescriptors(mol))
            valid_flags.append(True)
        except Exception:
            rows.append([np.nan] * len(descriptor_names))
            valid_flags.append(False)

    full_desc_df = pd.DataFrame(rows, columns=descriptor_names)
    full_desc_df = full_desc_df.replace([np.inf, -np.inf], np.nan)

    missing = [c for c in desc_columns_used if c not in full_desc_df.columns]
    if missing:
        raise ValueError(
            f"New descriptor computation is missing expected training columns: {missing}. "
            "This usually means your RDKit version differs from the one used during "
            "training, or desc_columns_used doesn't match the training pipeline."
        )

    return full_desc_df[desc_columns_used], valid_flags


def build_fingerprints_df(smiles_list, n_bits=N_BITS, radius=RADIUS):
    Chem, AllChem, DataStructs, Descriptors, MoleculeDescriptors = _lazy_import_rdkit()

    def get_morgan_fp(mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    rows = []
    valid_flags = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.full(n_bits, np.nan))
            valid_flags.append(False)
            continue
        rows.append(get_morgan_fp(mol))
        valid_flags.append(True)

    fp_array = np.array(rows)
    fp_df = pd.DataFrame(fp_array, columns=[f"FP_{i}" for i in range(n_bits)])
    return fp_df, valid_flags


def build_chemberta_df(smiles_list):
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_CHECKPOINT)
    model = AutoModel.from_pretrained(CHEMBERTA_CHECKPOINT)
    model.eval()

    embeddings = []
    valid_flags = []
    with torch.no_grad():
        for smi in smiles_list:
            try:
                inputs = tokenizer(smi, return_tensors="pt")
                out = model(**inputs)
                emb = out.last_hidden_state.mean(dim=1).squeeze().numpy()  # mean pooling
                embeddings.append(emb)
                valid_flags.append(True)
            except Exception:
                embeddings.append(np.full(model.config.hidden_size, np.nan))
                valid_flags.append(False)

    embeddings = np.array(embeddings)
    chemberta_df = pd.DataFrame(
        embeddings, columns=[f"chemb_{i}" for i in range(embeddings.shape[1])]
    )
    return chemberta_df, valid_flags


def build_molformer_df(smiles_list):
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(MOLFORMER_CHECKPOINT, trust_remote_code=True)
    model = AutoModel.from_pretrained(MOLFORMER_CHECKPOINT, trust_remote_code=True)
    model.eval()

    embeddings = []
    valid_flags = []
    with torch.no_grad():
        for smi in smiles_list:
            try:
                inputs = tokenizer(smi, return_tensors="pt", padding=True, truncation=True)
                out = model(**inputs)
                emb = out.pooler_output.squeeze().numpy()
                embeddings.append(emb)
                valid_flags.append(True)
            except Exception:
                embeddings.append(np.full(768, np.nan))
                valid_flags.append(False)

    embeddings = np.array(embeddings)
    molformer_df = pd.DataFrame(
        embeddings, columns=[f"molformer_{i}" for i in range(embeddings.shape[1])]
    )
    return molformer_df, valid_flags


FEATURE_BUILDERS = {
    "Descriptors": None,   # handled specially below (needs desc_columns_used)
    "Fingerprints": build_fingerprints_df,
    "Chemberta Embeddings": build_chemberta_df,
    "Molformer Embeddings": build_molformer_df,
}


def select_package(loaded_obj, package_key=None):
    """Handles either a single package dict, or a dict-of-packages
    (e.g. saved_top3 with multiple entries keyed by name)."""
    if isinstance(loaded_obj, dict) and "members" in loaded_obj:
        # it's already a single package
        return loaded_obj

    if isinstance(loaded_obj, dict):
        # it's a dict of packages
        if package_key is not None:
            if package_key not in loaded_obj:
                raise KeyError(
                    f"package_key '{package_key}' not found. Available keys: "
                    f"{list(loaded_obj.keys())}"
                )
            return loaded_obj[package_key]

        # no key given -- pick the best by test_rmse if available, else first entry
        try:
            best_key = min(loaded_obj, key=lambda k: loaded_obj[k].get("test_rmse", np.inf))
            print(f"No --package_key given. Auto-selected best entry by test_rmse: '{best_key}'")
            return loaded_obj[best_key]
        except Exception:
            first_key = next(iter(loaded_obj))
            print(f"No --package_key given and test_rmse not available. "
                  f"Using first entry: '{first_key}'")
            return loaded_obj[first_key]

    raise ValueError("Unrecognized package pickle structure -- expected a dict.")


def featurize_for_package(smiles_list, package):
    """Computes only the feature sets actually required by this package's members."""
    required_features = {m["features"] for m in package["members"]}
    X_new_dict = {}
    valid_flags_by_feature = {}

    for feat_name in required_features:
        print(f"Computing features: {feat_name} ...")
        if feat_name == "Descriptors":
            desc_columns_used = None
            for m in package["members"]:
                if m["features"] == "Descriptors":
                    desc_columns_used = m.get("desc_columns_used")
                    break
            if desc_columns_used is None:
                raise ValueError(
                    "This package uses 'Descriptors' but no 'desc_columns_used' "
                    "was found in the package. Re-save the package including the "
                    "exact training-time descriptor column list."
                )
            X_df, valid_flags = build_descriptors_df(smiles_list, desc_columns_used)
        else:
            builder = FEATURE_BUILDERS[feat_name]
            X_df, valid_flags = builder(smiles_list)

        X_new_dict[feat_name] = X_df
        valid_flags_by_feature[feat_name] = valid_flags

    # overall validity = valid across every feature type computed
    n = len(smiles_list)
    overall_valid = np.ones(n, dtype=bool)
    for flags in valid_flags_by_feature.values():
        overall_valid &= np.array(flags)

    return X_new_dict, overall_valid


def predict_with_package(package, X_new_dict):
    """Averages predictions across all members in the package (a single
    model has exactly one member, so this naturally covers both cases)."""
    member_preds = []
    for member in package["members"]:
        X_raw = X_new_dict[member["features"]]
        X_scaled = member["scaler"].transform(X_raw)
        pred = np.ravel(member["model"].predict(X_scaled))
        member_preds.append(pred)

    return np.mean(member_preds, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Featurize SMILES and predict using a saved model/ensemble package."
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to the saved package pickle "
                             "(e.g. top3_deployable_models.pkl).")
    parser.add_argument("--input_csv", required=True,
                        help="CSV containing a column of SMILES strings.")
    parser.add_argument("--smiles_col", required=True,
                        help="Name of the SMILES column in --input_csv.")
    parser.add_argument("--output_csv", required=True,
                        help="Path to write predictions to.")
    parser.add_argument("--package_key", default=None,
                        help="If --model_path contains multiple packages "
                             "(a dict of packages), specify which one to use. "
                             "If omitted, the best (lowest test_rmse) is used "
                             "automatically when possible.")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"ERROR: model_path does not exist: {args.model_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.input_csv):
        print(f"ERROR: input_csv does not exist: {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    with open(args.model_path, "rb") as f:
        loaded_obj = pickle.load(f)

    package = select_package(loaded_obj, args.package_key)

    required_features = sorted({m["features"] for m in package["members"]})
    member_names = [f"{m['features']}__{m['model_name']}" for m in package["members"]]
    print(f"Package type: {package.get('type', 'unknown')}")
    print(f"Members: {' + '.join(member_names)}")
    print(f"Feature sets required: {required_features}")

    input_df = pd.read_csv(args.input_csv)
    if args.smiles_col not in input_df.columns:
        print(f"ERROR: '{args.smiles_col}' not found in {args.input_csv}. "
              f"Available columns: {list(input_df.columns)}", file=sys.stderr)
        sys.exit(1)

    smiles_list = input_df[args.smiles_col].astype(str).tolist()
    print(f"Loaded {len(smiles_list)} molecules from {args.input_csv}")

    X_new_dict, overall_valid = featurize_for_package(smiles_list, package)

    n = len(smiles_list)
    predictions = np.full(n, np.nan)

    if overall_valid.any():
        valid_idx = np.where(overall_valid)[0]
        X_new_dict_valid = {
            feat_name: X_df.iloc[valid_idx].reset_index(drop=True)
            for feat_name, X_df in X_new_dict.items()
        }
        preds_valid = predict_with_package(package, X_new_dict_valid)
        predictions[valid_idx] = preds_valid

    n_invalid = int((~overall_valid).sum())
    if n_invalid > 0:
        print(f"WARNING: {n_invalid} molecule(s) had invalid SMILES or failed "
              f"featurization -- their predictions are set to NaN.")

    output_df = input_df.copy()
    output_df["is_valid_smiles"] = overall_valid
    output_df["predicted_value"] = predictions
    output_df["model_used"] = " + ".join(member_names)

    output_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()