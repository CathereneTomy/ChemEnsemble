import os
import json
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from joblib import Parallel, delayed
from tdc.benchmark_group import admet_group



from ensem import *
from prediction import (
    select_package,
    featurize_for_package,
    predict_with_package,
)


model_shortlisted = {
    "Caco2_Wang": {"Model": "Descriptors__CatBoost", "Transformation": False, "Metric": "MAE"},
    "Lipophilicity_AstraZeneca": {
        "Model": "Descriptors__SVR_RBF", "Transformation": False, "Metric": "MAE"},
    "Solubility_AqSolDB": {"Model": "Descriptors__XGBoost", "Transformation": False, "Metric": "MAE"},
    "PPBR_AZ": {
        "Model": "Descriptors__SVR_RBF + Descriptors__XGBoost + Erg Fingerprints__SVR_RBF + Molformer Embeddings__Lasso",
        "Transformation": True, "Metric": "MAE"
    },
    "VDss_Lombardo": {
        "Model": "Descriptors__SVR_RBF + Erg Fingerprints__KernelRidge + Erg Fingerprints__XGBoost + Fingerprints__XGBoost + Molformer Embeddings__SVR_RBF",
        "Transformation": True, "Metric": "Spearman"
    },
    "Half_Life_Obach": {"Model": "Descriptors__CatBoost", "Transformation": True, "Metric": "Spearman"},
    "Clearance_Hepatocyte_AZ": {
        "Model": "Chemberta Embeddings__SVR_RBF + Descriptors__SVR_RBF + Descriptors__XGBoost + Fingerprints__XGBoost + Molformer Embeddings__BayesianRidge",
        "Transformation": True, "Metric": "Spearman"
    },
    "Clearance_Microsome_AZ": {
        "Model": "Descriptors__SVR_RBF + Fingerprints__RandomForest + Molformer Embeddings__BayesianRidge",
        "Transformation": True, "Metric": "Spearman"
    },
    "LD50_Zhu": {"Model": "Chemberta Embeddings__KNN + Chemberta Embeddings__SVR_RBF + Descriptors__KNN + Descriptors__SVR_RBF + Fingerprints__CatBoost", "Transformation": False, "Metric": "MAE"},
}

SMILES_COL = "Drug"
ACTIVITY_COL = "Y"
SEEDS = [1, 2, 3, 4, 5]
DATA_PATH = os.path.abspath("data/")
OUT_DIR = "tdc_splits"


def fetch_tdc_data(benchmark_name, seeds, data_path="data/", out_dir="tdc_splits"):
    """Fetch train_val/test + per-seed splits for a benchmark directly via
    the tdc library (no subprocess / separate environment needed)."""
    os.makedirs(out_dir, exist_ok=True)
    group = admet_group(path=data_path)
    benchmark = group.get(benchmark_name)
    name = benchmark["name"]

    benchmark["train_val"].to_csv(os.path.join(out_dir, f"{name}_train_val.csv"), index=False)
    benchmark["test"].to_csv(os.path.join(out_dir, f"{name}_test.csv"), index=False)

    for seed in seeds:
        train, valid = group.get_train_valid_split(benchmark=name, split_type="default", seed=seed)
        train.to_csv(os.path.join(out_dir, f"{name}_train_seed{seed}.csv"), index=False)
        valid.to_csv(os.path.join(out_dir, f"{name}_valid_seed{seed}.csv"), index=False)

    print(f"Saved train_val/test + {len(seeds)} seed splits for {name} to {out_dir}")


def save_predictions_json(predictions_list, out_dir="tdc_splits", output_csv="test_pred_metrics.csv"):
    """Serialize predictions to a JSON file (matching the format TDC's
    group.evaluate_many expects) and stop -- no evaluation is performed."""
    preds_dir = os.path.join(out_dir, "model_preds")
    os.makedirs(preds_dir, exist_ok=True)
    pred_json_path = os.path.join(preds_dir, f"_predictions_{os.path.splitext(os.path.basename(output_csv))[0]}.json")
    serializable = [
        {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
        for d in predictions_list
    ]
    with open(pred_json_path, "w") as f:
        json.dump(serializable, f)
    print(f"Saved predictions JSON to {pred_json_path}")
    return pred_json_path


def process_benchmark(name, seeds=SEEDS, data_path=DATA_PATH, out_dir=OUT_DIR):
    """Run the full fetch -> feature -> fit -> ensemble -> save -> predict
    pipeline for a single TDC benchmark. Saves a predictions JSON file per
    benchmark (no evaluation is run)."""

    cfg = model_shortlisted[name]
    model = cfg["Model"]
    Transformation = cfg["Transformation"]
    metric = cfg["Metric"]

    # MAE benchmarks follow TDC's official 5-seed protocol.
    # Spearman benchmarks are run once with a single seed.
    run_seeds = seeds if metric == "MAE" else [seeds[0]]
    print(f"[{name}] Metric={metric} -> running seeds={run_seeds}")

    # ---- Fetch data for this benchmark ----
    fetch_tdc_data(name, run_seeds, data_path=data_path, out_dir=out_dir)

    train_val = pd.read_csv(os.path.join(out_dir, f"{name}_train_val.csv"))
    test = pd.read_csv(os.path.join(out_dir, f"{name}_test.csv"))

    df = train_val.copy()
    df["mol"] = [Chem.MolFromSmiles(s) for s in df[SMILES_COL]]

    feature_calc = {
        "Chemberta Embeddings": lambda: compute_chemberta(df),
        "Descriptors": lambda: compute_desc(df),
        "Fingerprints": lambda: compute_fingerprints(df),
        "Molformer Embeddings": lambda: compute_molformer(df),
        "Erg Fingerprints": lambda: compute_erg(df),
    }

    model_df = pd.DataFrame(
        [item.split("__", 1) for item in model.split(" + ")],
        columns=["Feature", "Model"],
    )

    feature_dfs = {}
    for _, row in model_df.iterrows():
        feat = row["Feature"]
        if feat not in feature_dfs:
            feature_dfs[feat] = feature_calc[feat]()

    predictions_list = []

    for seed in run_seeds:
        predictions = {}
        reset_ensemble_registry()

        train = pd.read_csv(os.path.join(out_dir, f"{name}_train_seed{seed}.csv"))
        valid = pd.read_csv(os.path.join(out_dir, f"{name}_valid_seed{seed}.csv"))

        if Transformation:
            print(f"[{name}] Y variable to be transformed")
            transformed_df, _, transform_registry = auto_transform_df(df, [ACTIVITY_COL])
            y = transformed_df[ACTIVITY_COL].values
        else:
            print(f"[{name}] No transformation applied to the Y-variable")
            transform_registry = None
            y = df[ACTIVITY_COL].values

        idx_train = df.index[df["Drug_ID"].isin(train["Drug_ID"])]
        idx_test = df.index[df["Drug_ID"].isin(valid["Drug_ID"])]

        y_train, y_test = y[idx_train], y[idx_test]

        feature_test_sets = {}
        for feat in model_df["Feature"]:
            feat_train, feat_test, feat_scaler = scale_split(feature_dfs[feat], idx_train, idx_test)
            feature_test_sets[feat] = (feat_train, feat_test, feat_scaler)

        tasks = []
        for _, row in model_df.iterrows():
            feature_name = row["Feature"]
            model_name = row["Model"]
            X_train, X_test, _ = feature_test_sets[feature_name]
            tasks.append((feature_name, model_name, X_train, X_test))

        results_list = Parallel(n_jobs=5, prefer="processes")(
            delayed(fit_one)(feature_name, model_name, X_train, y_train, X_test, y_test)
            for feature_name, model_name, X_train, X_test in tasks
        )

        results_map = {}
        for feature_name, model_name, result in results_list:
            results_map.setdefault(feature_name, {})[model_name] = result
            print(
                f"[{name}][{feature_name}] {model_name}: "
                f"RMSE={result['rmse']:.3f} MAE={result['mae']:.3f} R2={result['r2']:.3f}"
            )

        ensemble_members = [m.strip() for m in model.split(" + ")]

        train_pred, train_true = ensemble_predict(ensemble_members, "train", results_map, idx_train, y)
        train_stats = compute_stats(train_true, train_pred)

        test_pred, test_true = ensemble_predict(ensemble_members, "test", results_map, idx_test, y)
        test_stats = compute_stats(test_true, test_pred)

        final_model_dict = {
            "source": "ensemble" if "+" in model else "single",
            "members": model,
            "test_rmse": test_stats["rmse"],
            "test_r2": test_stats["r2"],
        }
        final_model_df = pd.DataFrame(final_model_dict, index=[0])

        # ---- Saving Models ----
        desc_columns_used = None
        if "Descriptors" in model_df["Feature"].values:
            desc_columns_used = list(feature_dfs["Descriptors"].columns)

        if Transformation:
            target_transform_info = transform_registry[ACTIVITY_COL]
        else:
            target_transform_info = {"method": "none", "param": None}

        scaler_map = {feat: feature_test_sets[feat][2] for feat in model_df["Feature"]}

        print(f"[{name}] Seed {seed}\n")
        print(final_model_df)

        top1_package = build_package(
            final_model_df.iloc[0],
            target_transform_info=target_transform_info,
            scaler_map=scaler_map,
            results_map=results_map,
            desc_columns=desc_columns_used,
        )

        top1_path = os.path.join(OUTPUT_DIR, f"{name}_{seed}_model.pkl")
        with open(top1_path, "wb") as f:
            pickle.dump(top1_package, f)
        print(f"Saved {name}_{seed}_model.pkl")

        # ------------------------------ Predicting for held-out test set ------------------------------
        package = select_package(top1_package)

        smiles_list_test = test["Drug"].astype(str).tolist()
        X_new_dict, overall_valid = featurize_for_package(smiles_list_test, package)

        n = len(smiles_list_test)
        y_pred_test = np.full(n, np.nan)

        if overall_valid.any():
            valid_idx = np.where(overall_valid)[0]
            X_new_dict_valid = {
                feat: fdf.iloc[valid_idx].reset_index(drop=True)
                for feat, fdf in X_new_dict.items()
            }
            _, preds_raw = predict_with_package(package, X_new_dict_valid)
            y_pred_test[valid_idx] = preds_raw

        n_invalid = int((~overall_valid).sum())
        if n_invalid > 0:
            invalid_smiles = [s for s, v in zip(smiles_list_test, overall_valid) if not v]
            print(
                f"WARNING [{name}] seed {seed}: {n_invalid} molecule(s) failed featurization "
                f"(invalid SMILES or feature computation error): {invalid_smiles}"
            )

        if np.isnan(y_pred_test).any():
            fallback_value = float(np.nanmean(df.loc[idx_train, ACTIVITY_COL]))
            nan_mask = np.isnan(y_pred_test)
            print(
                f"WARNING [{name}] seed {seed}: imputing {int(nan_mask.sum())} NaN prediction(s) "
                f"with training-target mean ({fallback_value:.3f})."
            )
            y_pred_test[nan_mask] = fallback_value

        predictions[name] = y_pred_test
        predictions_list.append(predictions)

    benchmark_out_csv = f"test_pred_metrics_{name}.csv"
    pred_json_path = save_predictions_json(predictions_list, out_dir=out_dir, output_csv=benchmark_out_csv)
    return pred_json_path


if __name__ == "__main__":
    saved_json_paths = {}

    for name in model_shortlisted:
        print(f"\n{'=' * 80}\nRunning benchmark: {name}\n{'=' * 80}")
        try:
            pred_json_path = process_benchmark(name, seeds=SEEDS, data_path=DATA_PATH, out_dir=OUT_DIR)
            saved_json_paths[name] = pred_json_path
        except Exception as e:
            print(f"FAILED on {name}: {e}")
            continue

    print("\nSaved predictions JSON files:")
    for name, path in saved_json_paths.items():
        print(f"  {name}: {path}")