import os
import sys
import warnings
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

# Fix matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils import resample

# Optional libs with graceful fallbacks
HAS_IMBLEARN = True
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    HAS_IMBLEARN = False

HAS_SHAP = True
try:
    import shap
except Exception:
    HAS_SHAP = False

HAS_STRAT_GROUP = True
try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    HAS_STRAT_GROUP = False

# Plot style & constants
sns.set_style("whitegrid")
PLOT_FORMAT = "pdf"
RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5
N_BOOTSTRAP = 1000

# -----------------------
# User configuration
# -----------------------
# Edit this path to point to your cleaned CSV before running
INPUT_CSV = r'C:\Users\white\OneDrive\Desktop\Coding\My Work\Paper 3\Master\cleaned_water_quality_data.csv'

# Columns required in the CSV
REQUIRED_COLS = [
    "station_code", "location_name", "year",
    "bod", "dissolved_oxygen", "nitrate", "fecal_coliform", "hotspot"
]
# -----------------------
# Utility functions
# -----------------------
def ensure_columns(df: pd.DataFrame, req_cols):
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")


def make_dirs(base="outputs"):
    out = os.path.abspath(base)
    plots = os.path.join(out, "plots")
    os.makedirs(plots, exist_ok=True)
    return out, plots


def bootstrap_ci(values: np.ndarray, func=np.mean, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    n = len(values)
    if n == 0:
        return (np.nan, np.nan)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boots.append(func(values[idx]))
    return np.percentile(boots, [2.5, 97.5])


# -----------------------
# Data loading & preprocessing
# -----------------------
def load_and_prepare(path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    ensure_columns(df, REQUIRED_COLS)

    # Drop rows with missing core parameters
    df = df.dropna(subset=["bod", "dissolved_oxygen", "nitrate", "fecal_coliform", "year"])

    # Optional engineered features: detect common names
    # Accept either 'WQI' or 'wqi'; 'Pollution_Index' or 'pollution_index'
    if 'wqi' in df.columns and 'WQI' not in df.columns:
        df.rename(columns={'wqi': 'WQI'}, inplace=True)
    if 'pollution_index' in df.columns and 'Pollution_Index' not in df.columns:
        df.rename(columns={'pollution_index': 'Pollution_Index'}, inplace=True)

    # Build feature list: only hydro-chemical & engineered features; DO NOT include station identifiers
    candidate = ['year', 'bod', 'dissolved_oxygen', 'nitrate', 'fecal_coliform', 'WQI', 'Pollution_Index']
    features = [c for c in candidate if c in df.columns]
    if not features:
        raise ValueError("No predictor features found in input file. Ensure hydro-chemical columns exist.")

    X = df[features].copy()
    y = df['hotspot'].astype(int).copy()
    groups = df['station_code'].copy()  # use station_code strictly for grouping

    print(f"Loaded data: {len(df)} rows, using features: {features}")
    return X, y, groups, df


# -----------------------
# Cross-validation chooser
# -----------------------
def get_cv(n_splits: int, groups: pd.Series):
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    if n_groups < 2:
        raise ValueError("Need at least 2 unique stations for grouped CV.")
    if n_groups < n_splits:
        n_splits = max(2, n_groups)
        print(f"Warning: reduced n_splits to {n_splits} because there are only {n_groups} stations.")
    if HAS_STRAT_GROUP:
        print("Using StratifiedGroupKFold.")
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        print("StratifiedGroupKFold not available; using GroupKFold (no stratification).")
        return GroupKFold(n_splits=n_splits)


# -----------------------
# Model evaluation per model (grouped CV)
# -----------------------
def evaluate_model_cv(model, X: pd.DataFrame, y: pd.Series, groups: pd.Series, model_name: str):
    cv = get_cv(DEFAULT_N_SPLITS, groups)
    fold_metrics = []
    agg_true = []
    agg_prob = []
    featimp_frames = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # SMOTE on training fold if available
        if HAS_IMBLEARN:
            sm = SMOTE(random_state=RANDOM_STATE)
            X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
        else:
            X_tr_res, y_tr_res = X_tr, y_tr

        model.fit(X_tr_res, y_tr_res)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)[:, 1]
        else:
            probs = model.predict(X_val)
            # ensure numeric array
            probs = np.asarray(probs, dtype=float)

        preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        roc = roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else np.nan
        pr_auc = average_precision_score(y_val, probs) if len(np.unique(y_val)) > 1 else np.nan

        fold_metrics.append({
            "fold": fold_idx + 1, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "roc_auc": roc, "pr_auc": pr_auc
        })

        agg_true.extend(y_val.tolist())
        agg_prob.extend(probs.tolist())

        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({
                "feature": X.columns,
                "importance": model.feature_importances_,
                "model": model_name,
                "fold": fold_idx + 1
            })
            featimp_frames.append(fi)

    folds_df = pd.DataFrame(fold_metrics)
    featimp_df = pd.concat(featimp_frames, ignore_index=True) if featimp_frames else pd.DataFrame(columns=["feature", "importance", "model", "fold"])
    return folds_df, featimp_df, np.array(agg_true), np.array(agg_prob)


# -----------------------
# Final pipeline
# -----------------------
def run_pipeline(input_csv: str):
    out_dir, plots_dir = make_dirs("outputs")
    X, y, groups, full_df = load_and_prepare(input_csv)

    # create grouped holdout (20% stations)
    stations = np.unique(full_df['station_code'])
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(stations)
    n_hold = max(1, int(0.2 * len(stations)))
    hold_stations = set(stations[:n_hold])
    mask_hold = full_df['station_code'].isin(hold_stations)

    X_train = X.loc[~mask_hold].reset_index(drop=True)
    y_train = y.loc[~mask_hold].reset_index(drop=True)
    groups_train = full_df.loc[~mask_hold, 'station_code'].reset_index(drop=True)

    X_hold = X.loc[mask_hold].reset_index(drop=True)
    y_hold = y.loc[mask_hold].reset_index(drop=True)

    print(f"Train rows: {len(X_train)}, Holdout rows: {len(X_hold)}")
    print(f"Train stations: {len(np.unique(groups_train))}, Holdout stations: {len(hold_stations)}")

    # Models (class-weighted or pos-weighted)
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200, class_weight='balanced', n_jobs=-1),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE, n_estimators=200,
            use_label_encoder=False, eval_metric="logloss",
            scale_pos_weight=(len(y_train) - y_train.sum()) / max(1, y_train.sum())
        )
    }

    summary_rows = []
    all_featimps = []
    preds_for_plots = {}

    for name, model in models.items():
        print(f"\n--- Evaluating {name} ---")
        folds_df, featimp_df, agg_true, agg_prob = evaluate_model_cv(model, X_train, y_train, groups_train, name)

        # Save fold-level CSV
        folds_path = os.path.join(out_dir, f"{name}_cv_folds.csv")
        folds_df.to_csv(folds_path, index=False)

        # Collect average CV metrics
        mean_metrics = folds_df.mean(numeric_only=True).to_dict()
        ci_low, ci_high = bootstrap_ci(folds_df['f1'].values) if not folds_df['f1'].empty else (np.nan, np.nan)

        row = {
            "Model": name,
            "CV_mean_f1": mean_metrics.get('f1', np.nan),
            "CV_f1_ci_low": ci_low,
            "CV_f1_ci_high": ci_high,
            "CV_pr_auc": mean_metrics.get('pr_auc', np.nan),
            "CV_roc_auc": mean_metrics.get('roc_auc', np.nan)
        }

        # Retrain on full training set (with SMOTE if available)
        if HAS_IMBLEARN:
            sm = SMOTE(random_state=RANDOM_STATE)
            X_tr_res, y_tr_res = sm.fit_resample(X_train, y_train)
        else:
            X_tr_res, y_tr_res = X_train, y_train

        model.fit(X_tr_res, y_tr_res)

        # Evaluate on holdout if available
        if len(X_hold) > 0:
            if hasattr(model, "predict_proba"):
                hold_probs = model.predict_proba(X_hold)[:, 1]
            else:
                hold_probs = model.predict(X_hold)
            hold_preds = (hold_probs >= 0.5).astype(int)

            row.update({
                "Hold_accuracy": accuracy_score(y_hold, hold_preds),
                "Hold_precision": precision_score(y_hold, hold_preds, zero_division=0),
                "Hold_recall": recall_score(y_hold, hold_preds, zero_division=0),
                "Hold_f1": f1_score(y_hold, hold_preds, zero_division=0),
                "Hold_roc_auc": roc_auc_score(y_hold, hold_probs) if len(np.unique(y_hold)) > 1 else np.nan,
                "Hold_pr_auc": average_precision_score(y_hold, hold_probs) if len(np.unique(y_hold)) > 1 else np.nan
            })

            # bootstrap CI on holdout predictions for F1
            n_hold = len(y_hold)
            rng2 = np.random.RandomState(RANDOM_STATE)
            boot_f1 = []
            for _ in range(500):
                idx = rng2.choice(n_hold, size=n_hold, replace=True)
                yb = y_hold.iloc[idx].values
                pb = hold_preds[idx]
                boot_f1.append(f1_score(yb, pb, zero_division=0))
            row['Hold_f1_ci_low'], row['Hold_f1_ci_high'] = np.percentile(boot_f1, [2.5, 97.5])
        else:
            row.update({
                "Hold_accuracy": np.nan, "Hold_precision": np.nan, "Hold_recall": np.nan, "Hold_f1": np.nan,
                "Hold_roc_auc": np.nan, "Hold_pr_auc": np.nan, "Hold_f1_ci_low": np.nan, "Hold_f1_ci_high": np.nan
            })

        summary_rows.append(row)
        if not featimp_df.empty:
            all_featimps.append(featimp_df)
        preds_for_plots[name] = (agg_true, agg_prob)

        # SHAP (safe, sample X_train)
        try:
            if HAS_SHAP:
                Xs = X_train.sample(n=min(200, len(X_train)), random_state=RANDOM_STATE)
                shap_exp = shap.TreeExplainer(model)
                shap_vals = shap_exp.shap_values(Xs)
                plt.figure(figsize=(8, 6))
                shap.summary_plot(shap_vals, Xs, show=False)
                plt.tight_layout()
                shp_path = os.path.join(plots_dir, f"shap_{name}.png")
                plt.savefig(shp_path, dpi=300)
                plt.close()
                print(f"Saved SHAP: {shp_path}")
            else:
                print("shap not installed: SKIPPING SHAP plots.")
        except Exception as e:
            print(f"SHAP generation failed for {name}: {e}")

    # Save summary and feature importances
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "model_metrics_summary.csv"), index=False)
    if all_featimps:
        feat_all = pd.concat(all_featimps, ignore_index=True)
    else:
        feat_all = pd.DataFrame(columns=["feature", "importance", "model", "fold"])
    feat_all.to_csv(os.path.join(out_dir, "feature_importances_summary.csv"), index=False)

    # ROC / PR comparison plots
    # ROC
    plt.figure(figsize=(8, 6))
    for name, (yt, yp) in preds_for_plots.items():
        if len(yt) == 0:
            continue
        try:
            fpr, tpr, _ = roc_curve(yt, yp)
            auc_v = roc_auc_score(yt, yp)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc_v:.3f})")
        except Exception:
            continue
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Comparison (aggregated CV folds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"roc_comparison.{PLOT_FORMAT}"))
    plt.close()

    # PR
    plt.figure(figsize=(8, 6))
    for name, (yt, yp) in preds_for_plots.items():
        if len(yt) == 0:
            continue
        try:
            prec, rec, _ = precision_recall_curve(yt, yp)
            ap = average_precision_score(yt, yp)
            plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
        except Exception:
            continue
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Comparison (aggregated CV folds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"pr_comparison.{PLOT_FORMAT}"))
    plt.close()

    print("\nOutputs written to:", out_dir)
    print("Files: model_metrics_summary.csv, feature_importances_summary.csv, plots/*")

# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    if INPUT_CSV == "":
        print("Please set INPUT_CSV path at the top of the script.")
        sys.exit(1)
    run_pipeline(INPUT_CSV)