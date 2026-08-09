import os
import argparse

os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # HuggingFace tokenizer fork warning
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")       # OpenMP duplicate-lib crashes (rdkit + numpy/sklearn)

import warnings
warnings.filterwarnings("ignore")
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

import pickle
from collections import defaultdict
from itertools import product, combinations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdReducedGraphs
from rdkit.ML.Descriptors import MoleculeDescriptors

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, ARDRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.cross_decomposition import PLSRegression

from scipy.stats import skew, kurtosis, normaltest, boxcox

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from joblib import Parallel, delayed

pd.set_option("display.max_columns", 50)

#Config

TRANSFORMATION = True
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_TEST_SIZE = 150
N_SPLITS = 5
N_JOBS = 12
TOP_K_PER_FEATURE = 5
TOP_N_OVERALL = 20
MAX_ENSEMBLE_SIZE = 5
N_DIVERSITY_PAIRS = 10
MAX_FEATURE_PAIR_OCCURRENCES = 2

PAIRWISE_COLS = [
    "model_a", "model_b", "n_total", "n11_both_correct", "n00_both_wrong",
    "n10_a_correct_b_wrong", "n01_a_wrong_b_correct", "q_statistic",
    "disagreement", "double_fault", "pct_a_rescues_b", "pct_b_rescues_a",
    "passes_double_fault_filter",
]

TRAIN_PRED_KEY = "preds"
TEST_PRED_KEY = "test_preds"

KF = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Registry of every unique ensemble built during the diversity search.
# key: tuple(sorted(members)) -> members list
all_built_ensembles = {}


# =============================================================================
# Models
# =============================================================================

def get_model_dict():
    return {
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=6, random_state=RANDOM_STATE
        ),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=10000),
        "SVR_RBF": SVR(kernel="rbf", C=10.0, epsilon=0.1),
        "KNN": KNeighborsRegressor(n_neighbors=5, weights="distance"),
        "GaussianProcess": GaussianProcessRegressor(
            kernel=ConstantKernel() * RBF(), alpha=1e-2, normalize_y=True, random_state=RANDOM_STATE
        ),
        "PLS": PLSRegression(n_components=10),  # n_components overridden per-call where X shape is known
        "KernelRidge": KernelRidge(alpha=1.0, kernel="rbf", gamma=None),
        "BayesianRidge": BayesianRidge(),
        "ARD": ARDRegression(),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            alpha=1e-2,
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=3.0,
            random_state=RANDOM_STATE,
            verbose=False,
            thread_count=4,
        ),
    }


# =============================================================================


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)



def profile_column(s: pd.Series) -> dict:
    """Pure statistics, no domain knowledge."""
    s = s.dropna()
    stats = {
        "n": len(s),
        "min": s.min(),
        "max": s.max(),
        "mean": s.mean(),
        "std": s.std(),
        "skew": skew(s),
        "excess_kurtosis": kurtosis(s),
        "has_negative": (s < 0).any(),
        "has_zero": (s == 0).any(),
    }
    if stats["min"] > 0:
        stats["range_ratio"] = stats["max"] / stats["min"]
        stats["cv"] = stats["std"] / stats["mean"] if stats["mean"] != 0 else np.nan
    else:
        stats["range_ratio"] = np.nan
        stats["cv"] = np.nan
    try:
        stats["normality_p"] = normaltest(s)[1]
    except ValueError:
        stats["normality_p"] = np.nan
    return stats


def pick_best_transform(s: pd.Series, skew_improve_threshold: float = 0.3) -> dict:
    stats = profile_column(s)
    candidates = {"none": (s.values, None)}

    if stats["min"] > 0:
        candidates["log1p"] = (np.log1p(s.values), None)
        candidates["sqrt"] = (np.sqrt(s.values), None)
        try:
            bc, lam = boxcox(s.values)
            candidates["boxcox"] = (bc, lam)
        except ValueError:
            pass
    elif stats["min"] >= 0:
        candidates["log1p"] = (np.log1p(s.values), None)
        candidates["sqrt"] = (np.sqrt(s.values), None)

    pt = PowerTransformer(method="yeo-johnson")
    yj_vals = pt.fit_transform(s.values.reshape(-1, 1)).flatten()
    candidates["yeo-johnson"] = (yj_vals, pt)

    scored = {}
    for name, (vals, _) in candidates.items():
        try:
            sk = abs(skew(vals))
            _, p = normaltest(vals)
        except Exception:
            continue
        scored[name] = {"abs_skew": sk, "normality_p": p}

    scored_df = pd.DataFrame(scored).T.sort_values("abs_skew")
    baseline_skew = scored_df.loc["none", "abs_skew"]
    best_name = scored_df.index[0]
    best_skew = scored_df.loc[best_name, "abs_skew"]

    if best_name != "none" and (baseline_skew - best_skew) < skew_improve_threshold:
        best_name = "none"

    chosen_values, chosen_param = candidates[best_name]

    return {
        "stats": stats,
        "scoreboard": scored_df,
        "chosen": best_name,
        "chosen_param": chosen_param,
        "chosen_values": chosen_values,
    }


def auto_transform_df(df: pd.DataFrame, cols: list, **kwargs):
    out_df = df.copy()
    report = {}
    transform_registry = {}

    for col in cols:
        s = df[col].dropna()
        result = pick_best_transform(s, **kwargs)
        out_df.loc[s.index, col] = result["chosen_values"]

        report[col] = {
            "chosen": result["chosen"],
            "raw_skew": result["stats"]["skew"],
            "raw_kurtosis": result["stats"]["excess_kurtosis"],
            "range_ratio": result["stats"]["range_ratio"],
            "cv": result["stats"]["cv"],
            "normality_p_raw": result["stats"]["normality_p"],
        }
        transform_registry[col] = {
            "method": result["chosen"],
            "param": result["chosen_param"],
        }

    return out_df, pd.DataFrame(report).T, transform_registry


# =============================================================================

def compute_desc(df):
    descriptor_names = [name for name, _ in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    desc_rows = []
    for m in df["mol"]:
        try:
            desc_rows.append(calculator.CalcDescriptors(m))
        except Exception:
            desc_rows.append([np.nan] * len(descriptor_names))

    desc_df = pd.DataFrame(desc_rows, columns=descriptor_names)
    desc_df = desc_df.replace([np.inf, -np.inf], np.nan)
    desc_df = desc_df.dropna(axis=1, how="any")
    desc_df = desc_df.loc[:, desc_df.std() > 0]

    print(f"Descriptor matrix shape: {desc_df.shape}")

    save_desc = pd.concat(
        [df[[SMILES_COL, ACTIVITY_COL]].reset_index(drop=True), desc_df.reset_index(drop=True)],
        axis=1,
    )
    save_desc.to_csv(os.path.join(FEAT_DIR, "descriptors.csv"), index=False)
    return desc_df


def compute_fingerprints(df):
    n_bits, radius = 1024, 2

    def get_morgan_fp(mol, radius=radius, n_bits=n_bits):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    fp_array = np.array([get_morgan_fp(m) for m in df["mol"]])
    fp_df = pd.DataFrame(fp_array, columns=[f"FP_{i}" for i in range(n_bits)])
    print(f"Fingerprint matrix shape: {fp_df.shape}")

    save_fp = pd.concat(
        [df[[SMILES_COL, ACTIVITY_COL]].reset_index(drop=True), fp_df.reset_index(drop=True)],
        axis=1,
    )
    save_fp.to_csv(os.path.join(FEAT_DIR, "fingerprints.csv"), index=False)
    return fp_df


def compute_erg(df):
    erg = [rdReducedGraphs.GetErGFingerprint(mol) for mol in df["mol"]]
    erg_fp = np.array(erg)
    erg_df = pd.DataFrame(erg_fp, columns=[f"erg_{i}" for i in range(erg_fp.shape[1])])
    print(f"ErG fingerprint matrix shape: {erg_df.shape}")

    save_erg = pd.concat(
        [df[[SMILES_COL, ACTIVITY_COL]].reset_index(drop=True), erg_df.reset_index(drop=True)],
        axis=1,
    )
    save_erg.to_csv(os.path.join(FEAT_DIR, "erg_fp.csv"), index=False)
    return erg_df


def compute_chemberta(df):
    from transformers import AutoTokenizer, AutoModel
    import torch

    print("Loading ChemBERTa-77M-MLM ...")
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    model.eval()

    embeddings = []
    with torch.no_grad():
        for smi in df[SMILES_COL]:
            inputs = tokenizer(smi, return_tensors="pt")
            out = model(**inputs)
            emb = out.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(emb)

    embeddings = np.array(embeddings)
    chemberta_df = pd.DataFrame(embeddings, columns=[f"chemb_{i}" for i in range(embeddings.shape[1])])
    print(f"ChemBERTa embedding matrix shape: {chemberta_df.shape}")

    save_chemberta = pd.concat(
        [df[[SMILES_COL, ACTIVITY_COL]].reset_index(drop=True), chemberta_df.reset_index(drop=True)],
        axis=1,
    )
    save_chemberta.to_csv(os.path.join(FEAT_DIR, "chemberta.csv"), index=False)
    return chemberta_df


def compute_molformer(df):
    from transformers import AutoTokenizer, AutoModel
    import torch

    print("Loading MoLFormer-XL-both-10pct ...")
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for smi in df[SMILES_COL]:
            inputs = tokenizer(smi, return_tensors="pt", padding=True, truncation=True)
            out = model(**inputs)
            emb = out.pooler_output.squeeze().numpy()
            embeddings.append(emb)

    embeddings = np.array(embeddings)
    print(f"MoLFormer embedding matrix shape: {embeddings.shape}")

    molformer_df = pd.DataFrame(embeddings, columns=[f"molformer_{i}" for i in range(embeddings.shape[1])])
    save_molformer = pd.concat(
        [df[[SMILES_COL, ACTIVITY_COL]].reset_index(drop=True), molformer_df.reset_index(drop=True)],
        axis=1,
    )
    save_molformer.to_csv(os.path.join(FEAT_DIR, "molformer.csv"), index=False)
    return molformer_df



# =============================================================================

def generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def scaffold_split(df, smiles_col="smiles", test_size=0.2, random_state=42, max_test_size=150):
    scaffolds = defaultdict(list)
    for idx, smiles in zip(df.index, df[smiles_col]):
        scaffolds[generate_scaffold(smiles)].append(idx)

    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)

    rng = np.random.RandomState(random_state)
    rng.shuffle(scaffold_sets)

    n_total = len(df)
    n_test = min(max_test_size, int(np.floor(test_size * n_total)))

    idx_test, idx_train = [], []
    for group in scaffold_sets:
        if len(idx_test) + len(group) <= n_test:
            idx_test.extend(group)
        else:
            idx_train.extend(group)

    return np.array(idx_train), np.array(idx_test)


def scale_split(X_df, idx_train, idx_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_df.loc[idx_train])
    X_test = scaler.transform(X_df.loc[idx_test])
    return X_train, X_test, scaler



# =============================================================================

def precompute_fold_splits(feature_dfs, idx_train_arr, kf):
    """Precompute scaled train/val splits once per feature, reused across all models on that feature."""
    fold_cache = {}
    for label, X_df in feature_dfs.items():
        folds = []
        for fold_train_pos, fold_val_pos in kf.split(idx_train_arr):
            fold_train_idx = idx_train_arr[fold_train_pos]
            fold_val_idx = idx_train_arr[fold_val_pos]

            scaler = StandardScaler()
            X_fold_train = scaler.fit_transform(X_df.loc[fold_train_idx])
            X_fold_val = scaler.transform(X_df.loc[fold_val_idx])

            folds.append((fold_train_idx, fold_val_idx, X_fold_train, X_fold_val, fold_val_pos))
        fold_cache[label] = folds
    return fold_cache


def fit_one_model(label, model_name, fold_cache, y, idx_train_arr):
    folds = fold_cache[label]
    oof_preds = np.full(len(idx_train_arr), np.nan)
    fold_rmses = []

    for fold_train_idx, fold_val_idx, X_fold_train, X_fold_val, fold_val_pos in folds:
        y_fold_train = y[fold_train_idx]
        y_fold_val = y[fold_val_idx]

        models = get_model_dict()
        if model_name == "PLS":
            models["PLS"] = PLSRegression(
                n_components=min(10, X_fold_train.shape[1], X_fold_train.shape[0] - 1)
            )
        model = models[model_name]
        model.fit(X_fold_train, y_fold_train)

        fold_preds = np.ravel(model.predict(X_fold_val))
        oof_preds[fold_val_pos] = fold_preds
        fold_rmses.append(np.sqrt(mean_squared_error(y_fold_val, fold_preds)))

    oof_rmse = np.sqrt(mean_squared_error(y[idx_train_arr], oof_preds))
    fold_rmses = np.array(fold_rmses)

    result = {
        "preds": oof_preds,
        "oof_rmse": oof_rmse,
        "fold_rmses": fold_rmses,
        "cv_rmse_mean": fold_rmses.mean(),
        "cv_rmse_std": fold_rmses.std(),
    }
    return label, model_name, result


def generate_oof_predictions_fast(top_models_df, feature_dfs, y, idx_train_arr, kf, n_jobs=-1):
    fold_cache = precompute_fold_splits(feature_dfs, idx_train_arr, kf)
    tasks = [(row["features"], row["model"]) for _, row in top_models_df.iterrows()]

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(fit_one_model)(label, model_name, fold_cache, y, idx_train_arr)
        for label, model_name in tasks
    )

    oof_results_map = {}
    for label, model_name, result in results:
        oof_results_map.setdefault(label, {})[model_name] = result
        print(
            f"[CV] {label} / {model_name}: "
            f"OOF_RMSE={result['oof_rmse']:.4f} CV_mean={result['cv_rmse_mean']:.4f} "
            f"CV_std={result['cv_rmse_std']:.4f}"
        )

    return oof_results_map



# =============================================================================

def pairwise_diversity_table(wide_df, model_labels):
    if wide_df is None or wide_df.empty or len(model_labels) < 2:
        reason = "no rows" if wide_df is None or wide_df.empty else "fewer than 2 models"
        print(f"pairwise_diversity_table: skipping ({reason}).")
        return pd.DataFrame(columns=PAIRWISE_COLS)

    rows = []
    for m_a, m_b in combinations(model_labels, 2):
        col_a, col_b = f"correct_{m_a}", f"correct_{m_b}"
        if col_a not in wide_df.columns or col_b not in wide_df.columns:
            continue  # model had zero rows in this slice; pivoted column never created

        a, b = wide_df[col_a].values, wide_df[col_b].values

        n11 = int(((a == 1) & (b == 1)).sum())
        n00 = int(((a == 0) & (b == 0)).sum())
        n10 = int(((a == 1) & (b == 0)).sum())
        n01 = int(((a == 0) & (b == 1)).sum())
        n_total = n11 + n00 + n10 + n01

        denom_q = n11 * n00 + n01 * n10
        q_stat = ((n11 * n00 - n01 * n10) / denom_q) if denom_q != 0 else np.nan

        rows.append({
            "model_a": m_a, "model_b": m_b, "n_total": n_total,
            "n11_both_correct": n11, "n00_both_wrong": n00,
            "n10_a_correct_b_wrong": n10, "n01_a_wrong_b_correct": n01,
            "q_statistic": q_stat,
            "disagreement": (n10 + n01) / n_total if n_total else np.nan,
            "double_fault": n00 / n_total if n_total else np.nan,
            "pct_a_rescues_b": 100 * n10 / n_total if n_total else np.nan,
            "pct_b_rescues_a": 100 * n01 / n_total if n_total else np.nan,
        })

    if not rows:
        print("pairwise_diversity_table: no valid pairs after column checks.")
        return pd.DataFrame(columns=PAIRWISE_COLS)

    df = pd.DataFrame(rows)
    double_fault_cutoff = df["double_fault"].quantile(0.75)
    df["passes_double_fault_filter"] = (
        df["double_fault"] <= double_fault_cutoff if not df["double_fault"].isna().all() else True
    )

    return df.sort_values(
        by=["passes_double_fault_filter", "disagreement", "q_statistic"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def long_to_wide(long_df):
    if long_df is None or long_df.empty:
        print("long_to_wide: input is empty - returning empty wide frame.")
        return pd.DataFrame(columns=["SMILES"])

    correct_wide = long_df.pivot_table(index=["SMILES"], columns="model_label", values="correct")
    correct_wide.columns = [f"correct_{c}" for c in correct_wide.columns]

    err_wide = long_df.pivot_table(index=["SMILES"], columns="model_label", values="abs_error")
    err_wide.columns = [f"err_{c}" for c in err_wide.columns]

    return correct_wide.join(err_wide).reset_index()


def build_model_prediction_long(top_models_df, results_map, y, idx_test, test_smiles, threshold):
    empty_cols = ["SMILES", "model_label", "actual", "pred", "abs_error", "correct"]
    if top_models_df is None or top_models_df.empty:
        print("build_model_prediction_long: no models in top_models_df.")
        return pd.DataFrame(columns=empty_cols)
    if len(test_smiles) == 0:
        print("build_model_prediction_long: no molecules in this split.")
        return pd.DataFrame(columns=empty_cols)

    y_test = y[idx_test]
    rows = []
    for _, row in top_models_df.iterrows():
        label, model_name = row["features"], row["model"]
        model_label = f"{label}__{model_name}"

        r = results_map[label][model_name]
        preds = np.ravel(r["preds"])
        abs_err = np.abs(y_test - preds)
        correct = (abs_err < threshold).astype(int)

        for smi, actual, pred, err, corr in zip(test_smiles, y_test, preds, abs_err, correct):
            rows.append({
                "SMILES": smi, "model_label": model_label,
                "actual": actual, "pred": pred, "abs_error": err, "correct": corr,
            })

    return pd.DataFrame(rows)


def extract_feature(model_label):
    return model_label.split("__", 1)[0]


def limit_feature_pair_occurrences(df, model_a_col="model_a", model_b_col="model_b", max_occurrences=2):
    df = df.copy()
    feat_a = df[model_a_col].apply(extract_feature)
    feat_b = df[model_b_col].apply(extract_feature)
    df["_feature_pair_key"] = [tuple(sorted((fa, fb))) for fa, fb in zip(feat_a, feat_b)]
    df["_occurrence_rank"] = df.groupby("_feature_pair_key").cumcount()

    return (
        df[df["_occurrence_rank"] < max_occurrences]
        .drop(columns=["_feature_pair_key", "_occurrence_rank"])
        .reset_index(drop=True)
    )


def pairs_from_table(pairwise_df, prefix, n=3):
    pairs = []
    for i, row in pairwise_df.head(n).iterrows():
        pairs.append((f"{prefix}_pair{i + 1}", [row["model_a"], row["model_b"]]))
    return pairs


def ensemble_predict(model_labels_list, split, results_map, idx, y_full):
    preds_stack = []
    for label in model_labels_list:
        features, model_name = label.split("__", 1)
        r = results_map[features][model_name]
        key = TEST_PRED_KEY if split == "test" else TRAIN_PRED_KEY
        if key not in r:
            raise KeyError(
                f"'{key}' not found for {label}. Check how results_map stores {split} predictions."
            )
        preds_stack.append(np.ravel(r[key]))
    preds_stack = np.vstack(preds_stack)
    ensemble_pred = preds_stack.mean(axis=0)
    return ensemble_pred, y_full[idx]


def compute_stats(y_true, y_pred, threshold):
    abs_err = np.abs(y_true - y_pred)
    return {
        "n": len(y_true),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "accuracy_at_thresh": float((abs_err < threshold).mean()),
    }


def fit_one(feature_name, model_name, X_train, y_train, X_test, y_test, sample_weight=None):
    models = get_model_dict()
    models["PLS"] = PLSRegression(n_components=min(10, X_train.shape[1], X_train.shape[0] - 1))
    model = models[model_name]

    fit_kwargs = {}
    if sample_weight is not None and "sample_weight" in model.fit.__code__.co_varnames:
        fit_kwargs["sample_weight"] = sample_weight

    model.fit(X_train, y_train, **fit_kwargs)

    train_preds = np.ravel(model.predict(X_train))
    preds = np.ravel(model.predict(X_test))

    result = {
        "model": model,
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds),
        "preds": train_preds,
        "test_preds": preds,
    }
    return feature_name, model_name, result


def compute_correctness_df(preds, y, idx, smiles_arr, label, threshold):
    actual = y[idx]
    abs_err = np.abs(actual - preds)
    correct = (abs_err < threshold).astype(int)
    return pd.DataFrame({
        "SMILES": smiles_arr,
        f"correct_{label}": correct,
        f"err_{label}": abs_err,
    })


def dedup_by_member_set(ensemble_list):
    seen = {}
    deduped = []
    for name, members in ensemble_list:
        key = tuple(sorted(members))
        if key not in seen:
            seen[key] = name
            deduped.append((name, members))
        else:
            print(f"Dropping duplicate '{name}' - same members as '{seen[key]}'.")
    return deduped


def register(members):
    key = tuple(sorted(members))
    is_new = key not in all_built_ensembles
    if is_new:
        all_built_ensembles[key] = list(members)
    return key, is_new


def grow_ensemble_lineages(initial_pool, model_labels, oof_results_map, y, idx_train_arr,
                            wide_all_oof, train_smiles_arr, error_threshold,
                            max_size=MAX_ENSEMBLE_SIZE):
    """Breadth-first growth of each seed pair into larger, diversity-selected ensembles."""
    queue = []
    for name, members in initial_pool:
        _, is_new = register(members)
        if is_new:
            queue.append(list(members))

    print(f"Starting queue with {len(queue)} deduped pairs.")

    while queue:
        current_members = queue.pop(0)
        unused_models = [m for m in model_labels if m not in current_members]

        if not unused_models or len(current_members) >= max_size:
            continue

        ens_oof_pred, _ = ensemble_predict(current_members, "train", oof_results_map, idx_train_arr, y)
        ens_label = "ENS[" + "+".join(sorted(current_members)) + "]"

        ens_corr_df = compute_correctness_df(
            ens_oof_pred, y, idx_train_arr, train_smiles_arr, ens_label, threshold=error_threshold
        )
        wide_with_ens = wide_all_oof.merge(ens_corr_df, on=["SMILES"], how="inner")

        candidate_labels = [ens_label] + unused_models
        pairwise_vs_candidates = pairwise_diversity_table(wide_with_ens, candidate_labels)
        if pairwise_vs_candidates.empty:
            continue

        pairwise_vs_candidates = pairwise_vs_candidates[
            (pairwise_vs_candidates["model_a"] == ens_label)
            | (pairwise_vs_candidates["model_b"] == ens_label)
        ].reset_index(drop=True)
        if pairwise_vs_candidates.empty:
            continue

        top3_rows = pairwise_vs_candidates.head(3)
        candidate_models = [
            (r["model_b"] if r["model_a"] == ens_label else r["model_a"])
            for _, r in top3_rows.iterrows()
        ]

        for cand in candidate_models:
            new_members = current_members + [cand]
            _, is_new = register(new_members)
            if is_new:
                queue.append(new_members)
                print(f"Built new ensemble: {' + '.join(new_members)}")

    print(f"Total unique ensembles built: {len(all_built_ensembles)}")



# =============================================================================

def package_member(features, model_name, results_map, scaler_map, desc_columns):
    member = {
        "features": features,
        "model_name": model_name,
        "model": results_map[features][model_name]["model"],
        "scaler": scaler_map[features],
    }
    if features == "Descriptors":
        member["desc_columns_used"] = desc_columns
    return member


def build_package(row, target_transform_info, scaler_map, results_map, desc_columns):
    if row["source"] == "single":
        features, model_name = row["members"].split("__", 1)
        members_packaged = [package_member(features, model_name, results_map, scaler_map, desc_columns)]
    else:
        member_labels = row["members"].split(" + ")
        members_packaged = [
            package_member(*m.split("__", 1), results_map, scaler_map, desc_columns) for m in member_labels
        ]
    return {
        "type": row["source"],
        "members": members_packaged,
        "members_label": row["members"],
        "test_rmse": row["test_rmse"],
        "test_r2": row["test_r2"],
        "target_transform": target_transform_info,
    }



# =============================================================================

def load_and_clean_data():
    print("Stage 1: Load and clean data")

    df_raw = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df_raw)} rows")

    df_raw = df_raw[[SMILES_COL, ACTIVITY_COL]].dropna().reset_index(drop=True)
    print(f"After dropping missing values: {len(df_raw)} rows")

    df_raw["can_smiles"] = df_raw[SMILES_COL].apply(canonicalize)
    n_invalid = df_raw["can_smiles"].isna().sum()
    print(f"Invalid / unparseable SMILES dropped: {n_invalid}")

    df_clean = df_raw.dropna(subset=["can_smiles"]).reset_index(drop=True)
    n_before = len(df_clean)
    dup_counts = df_clean["can_smiles"].value_counts()
    n_dup_groups = (dup_counts > 1).sum()
    print(f"Duplicate structure groups found: {n_dup_groups}")

    df_dedup = df_clean.groupby("can_smiles", as_index=False).agg(
        {ACTIVITY_COL: "mean", SMILES_COL: "first"}
    )
    print(f"Rows before dedup: {n_before}, after dedup: {len(df_dedup)}")

    df = df_dedup.rename(columns={"can_smiles": "SMILES_canonical"}).reset_index(drop=True)
    mols = [Chem.MolFromSmiles(s) for s in df["SMILES_canonical"]]
    df["mol"] = mols
    assert all(m is not None for m in mols), "Unexpected parsing failure after cleaning"

    return df


def compute_all_features(df):
    print("Stage 2: Feature extraction")
    os.makedirs(FEAT_DIR, exist_ok=True)

    print("Computing physicochemical descriptors ...")
    desc_df = compute_desc(df)

    print("Computing Morgan fingerprints ...")
    fp_df = compute_fingerprints(df)

    print("Computing ChemBERTa embeddings ...")
    chemberta_df = compute_chemberta(df)

    print("Computing MoLFormer embeddings ...")
    molformer_df = compute_molformer(df)

    print("Computing ErG fingerprints ...")
    erg_df = compute_erg(df)

    return {
        "Chemberta Embeddings": chemberta_df,
        "Descriptors": desc_df,
        "Fingerprints": fp_df,
        "Molformer Embeddings": molformer_df,
        "Erg Fingerprints": erg_df,
    }


def transform_target(df):
    print("Stage 3: Target transformation")
    if TRANSFORMATION:
        print("Y variable will be auto-transformed to reduce skew ...")
        transformed_df, report_df, transform_registry = auto_transform_df(df, [ACTIVITY_COL])
        chosen = transform_registry[ACTIVITY_COL]["method"]
        print(f"Chosen transform for '{ACTIVITY_COL}': {chosen}")
        y = transformed_df[ACTIVITY_COL].values
    else:
        print("No transformation applied to the Y-variable.")
        transform_registry = None
        y = df[ACTIVITY_COL].values
    return y, transform_registry


def make_splits_and_scale(df, feature_dfs, y):
    print("Stage 4: Scaffold split + feature scaling")

    idx_train, idx_test = scaffold_split(
        df, smiles_col=SMILES_COL, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, max_test_size=MAX_TEST_SIZE,
    )
    print(f"Train molecules: {len(idx_train)} | Test molecules: {len(idx_test)}")

    y_train, y_test = y[idx_train], y[idx_test]

    scaled = {}
    for label, X_df in feature_dfs.items():
        X_train, X_test, scaler = scale_split(X_df, idx_train, idx_test)
        scaled[label] = (X_train, X_test, scaler)
        print(f"Scaled '{label}': train={X_train.shape}, test={X_test.shape}")

    return idx_train, idx_test, y_train, y_test, scaled


def run_cv_screening(feature_dfs, y, idx_train):
    print("Stage 5: Cross-validated model screening")

    model_dict = get_model_dict()
    combinations_df = pd.DataFrame(
        product(feature_dfs.keys(), model_dict.keys()), columns=["features", "model"]
    )
    print(f"Screening {len(combinations_df)} (feature, model) combinations with {N_SPLITS}-fold CV ...")

    idx_train_arr = np.array(idx_train)
    oof_results_map = generate_oof_predictions_fast(
        combinations_df, feature_dfs, y, idx_train_arr, KF, n_jobs=N_JOBS
    )

    cv_results = []
    for feature, models in oof_results_map.items():
        for model_name, data in models.items():
            preds = data["preds"]
            cv_results.append({
                "features": feature,
                "model": model_name,
                "CV R2": r2_score(y[idx_train], preds),
                "CV RMSE": np.sqrt(mean_squared_error(y[idx_train], preds)),
                "CV MAE": mean_absolute_error(y[idx_train], preds),
                "CV RMSE std": data["cv_rmse_std"],
            })

    cv_results_df = pd.DataFrame(cv_results).sort_values("CV RMSE").reset_index(drop=True)
    return cv_results_df, oof_results_map, idx_train_arr


def select_top_models(cv_results_df):
    print("Stage 6: Select stable, top-ranked models")

    std_cutoff = cv_results_df["CV RMSE std"].quantile(0.75)
    stable_df = cv_results_df[cv_results_df["CV RMSE std"] <= std_cutoff].copy()
    stable_df = stable_df.sort_values("CV RMSE").reset_index(drop=True)
    print(f"Kept {len(stable_df)}/{len(cv_results_df)} models within the 75th-percentile CV-std cutoff.")

    stable_df["rank_in_feature"] = stable_df.groupby("features").cumcount()
    capped_df = stable_df[stable_df["rank_in_feature"] < TOP_K_PER_FEATURE].drop(columns="rank_in_feature")
    capped_df = capped_df.sort_values("CV RMSE").reset_index(drop=True)
    print(f"Capped to top {TOP_K_PER_FEATURE} per feature -> {len(capped_df)} candidates.")

    top20_keys = capped_df.head(TOP_N_OVERALL)
    model_labels = [f"{r.features}__{r.model}" for r in top20_keys.itertuples()]
    print(f"Selected overall top {len(top20_keys)} models for ensembling.")

    return top20_keys, model_labels


def build_diversity_pairs(top20_keys, model_labels, oof_results_map, y, idx_train_arr, df):
    print("Stage 7: Pairwise diversity analysis (out-of-fold)")

    top5_mae = top20_keys.sort_values("CV MAE").head(5)["CV MAE"]
    error_threshold = 0.75 * top5_mae.mean()
    print(f"Error threshold for 'correct' prediction: {error_threshold:.4f}")

    train_smiles_arr = df.loc[idx_train_arr]["SMILES_canonical"].values
    long_df_oof = build_model_prediction_long(
        top20_keys, oof_results_map, y, idx_train_arr, train_smiles_arr, threshold=error_threshold
    )

    wide_all_oof = long_to_wide(long_df_oof)
    pairwise_all_oof = pairwise_diversity_table(wide_all_oof, model_labels)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wide_all_oof.to_csv(os.path.join(OUTPUT_DIR, "wide_matrix_all_molecules_OOF.csv"), index=False)
    pairwise_all_oof.to_csv(os.path.join(OUTPUT_DIR, "pairwise_diversity_all_molecules_OOF.csv"), index=False)

    pairwise_filtered = limit_feature_pair_occurrences(
        pairwise_all_oof, max_occurrences=MAX_FEATURE_PAIR_OCCURRENCES
    )
    all_pairs = pairs_from_table(pairwise_filtered, "all", n=N_DIVERSITY_PAIRS)

    print(f"Seed pairs selected for ensemble growth ({len(all_pairs)}):")
    for name, members in all_pairs:
        print(f"    {name} -> {members}")

    return all_pairs, wide_all_oof, train_smiles_arr, error_threshold


def train_top_models_on_full_train(top20_keys, scaled_features, y_train, y_test):
    print("Stage 8: Fit top models on the full train/test split")

    tasks = []
    for feature_name, group in top20_keys.groupby("features"):
        X_train, X_test, _ = scaled_features[feature_name]
        for model_name in group["model"].tolist():
            tasks.append((feature_name, model_name, X_train, X_test))

    print(f"Fitting {len(tasks)} (feature, model) pairs in parallel ...")
    results_list = Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(fit_one)(feature_name, model_name, X_train, y_train, X_test, y_test)
        for feature_name, model_name, X_train, X_test in tasks
    )

    results_map = {}
    for feature_name, model_name, result in results_list:
        results_map.setdefault(feature_name, {})[model_name] = result
        print(
            f"[{feature_name}] {model_name}: "
            f"RMSE={result['rmse']:.3f} MAE={result['mae']:.3f} R2={result['r2']:.3f}"
        )

    return results_map


def rank_singles_and_ensembles(top20_keys, results_map, y, idx_train_arr, idx_test, error_threshold):
    print("Stage 9: Rank single models and ensembles")

    ensemble_rows = []
    for _, members in all_built_ensembles.items():
        train_pred, train_true = ensemble_predict(members, "train", results_map, idx_train_arr, y)
        train_stats = compute_stats(train_true, train_pred, error_threshold)

        test_pred, test_true = ensemble_predict(members, "test", results_map, idx_test, y)
        test_stats = compute_stats(test_true, test_pred, error_threshold)

        ensemble_rows.append({
            "source": "ensemble",
            "members": " + ".join(sorted(members)),
            "n_members": len(members),
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
        })
    ensemble_full_df = pd.DataFrame(ensemble_rows)
    print(f"Ranked {len(ensemble_full_df)} ensembles.")

    single_rows = []
    for _, row in top20_keys.iterrows():
        label, model_name = row["features"], row["model"]
        r = results_map[label][model_name]

        train_pred = np.ravel(r["preds"])
        test_pred = np.ravel(r["test_preds"])

        train_stats = compute_stats(y[idx_train_arr], train_pred, error_threshold)
        test_stats = compute_stats(y[idx_test], test_pred, error_threshold)

        single_rows.append({
            "source": "single",
            "members": f"{label}__{model_name}",
            "n_members": 1,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
        })
    single_full_df = pd.DataFrame(single_rows)
    print(f"Ranked {len(single_full_df)} single models.")

    combined_df = pd.concat([ensemble_full_df, single_full_df], ignore_index=True)
    combined_df = combined_df.sort_values("test_rmse").reset_index(drop=True)
    combined_df.to_csv(os.path.join(OUTPUT_DIR, "combined_models_and_ensembles_ranked.csv"), index=False)
    print(f"Saved combined ranking -> combined_models_and_ensembles_ranked.csv")

    return combined_df


def prompt_and_save_models(combined_df, desc_df, transform_registry, scaled_features, results_map):
    print("Stage 10: Save selected models")

    desc_columns_used = list(desc_df.columns)
    target_transform_info = (
        transform_registry[ACTIVITY_COL] if TRANSFORMATION else {"method": "none", "param": None}
    )
    scaler_map = {label: scaler for label, (_, _, scaler) in scaled_features.items()}

    top10 = combined_df.head(10).reset_index(drop=True)

    print("Top 10 models/ensembles (ranked by test RMSE):\n")
    for rank, row in enumerate(top10.itertuples(), start=1):
        member_list = row.members.split(" + ")
        print(f"  #{rank}  [{row.source}]  test_rmse={row.test_rmse:.4f}  test_r2={row.test_r2:.4f}")
        for m in member_list:
            print(f"        - {m}")
        print()

    user_input = input(
        "Enter the number(s) of the models you want to save (e.g. '1,3,5'), "
        "or press Enter to save the top 3 by default: "
    ).strip()

    if user_input == "":
        chosen_ranks = [1, 2, 3]
        print("No selection made - defaulting to top 3.")
    else:
        try:
            chosen_ranks = [int(x.strip()) for x in user_input.split(",") if x.strip() != ""]
            chosen_ranks = [r for r in chosen_ranks if 1 <= r <= len(top10)]
            if not chosen_ranks:
                raise ValueError
        except ValueError:
            print("Invalid input - defaulting to top 3.")
            chosen_ranks = [1, 2, 3]

    print(f"Saving ranks: {chosen_ranks}")

    for rank in chosen_ranks:
        row = top10.iloc[rank - 1]
        package = build_package(
            row,
            target_transform_info=target_transform_info,
            scaler_map=scaler_map,
            results_map=results_map,
            desc_columns=desc_columns_used,
        )
        out_path = os.path.join(OUTPUT_DIR, f"top{rank}_model.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(package, f)
        print(f"Saved top{rank}: {row['members']} (test_rmse={row['test_rmse']:.4f}) -> {out_path}")



# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and ensemble ADMET property prediction models."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=r"data\admet_group\clearance_hepatocyte_az\train_val.csv",
        help="Path to the training/validation CSV file.",
    )
    parser.add_argument(
        "--smiles-col",
        type=str,
        default="Drug",
        help="Column name containing SMILES strings.",
    )
    parser.add_argument(
        "--activity-col",
        type=str,
        default="Y",
        help="Column name containing the target/activity values.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"data\admet_group\clearance_hepatocyte_az\models1_1",
        help="Directory to save models, features, and results.",
    )

    args = parser.parse_args()

    # Derived path, kept consistent with OUTPUT_DIR
    args.feat_dir = os.path.join(args.output_dir, "Features")

    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_and_clean_data(args.data_path, args.smiles_col, args.activity_col)
    feature_dfs = compute_all_features(df)
    y, transform_registry = transform_target(df)
    idx_train, idx_test, y_train, y_test, scaled_features = make_splits_and_scale(df, feature_dfs, y)

    cv_results_df, oof_results_map, idx_train_arr = run_cv_screening(feature_dfs, y, idx_train)
    top20_keys, model_labels = select_top_models(cv_results_df)

    all_pairs, wide_all_oof, train_smiles_arr, error_threshold = build_diversity_pairs(
        top20_keys, model_labels, oof_results_map, y, idx_train_arr, df
    )

    results_map = train_top_models_on_full_train(top20_keys, scaled_features, y_train, y_test)

    print("Stage 8b: Grow diverse ensembles")
    initial_pool = dedup_by_member_set(all_pairs)
    grow_ensemble_lineages(
        initial_pool, model_labels, oof_results_map, y, idx_train_arr,
        wide_all_oof, train_smiles_arr, error_threshold,
    )

    combined_df = rank_singles_and_ensembles(
        top20_keys, results_map, y, idx_train_arr, idx_test, error_threshold
    )

    prompt_and_save_models(
        combined_df, feature_dfs["Descriptors"], transform_registry, scaled_features, results_map, args.output_dir
    )

    print("Pipeline complete")


if __name__ == "__main__":
    main()
