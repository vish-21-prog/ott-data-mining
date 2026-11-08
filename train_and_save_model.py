"""train_and_save_model.py
Trains and saves the Unweighted Model C with Tenure (15 features) on mature cohort (N=293,936).
DO NOT RUN AUTOMATICALLY: Wait for user confirmation before executing.

Model Configuration:
- Algorithm: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
- Weighting: Unweighted (natural class prevalence, churn rate 8.76%)
- Evaluates on an 80/20 stratified holdout split, prints cross-check ROC-AUC and PR-AUC, then fits on full mature cohort.
- Serializes model and metadata to model_c_unweighted.joblib.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

PROJECT_ROOT = Path(r"E:\ott video paltform")
ACTIVITY_FILE = PROJECT_ROOT / "monthly_user_activity.csv"
FEATURES_V3_FILE = PROJECT_ROOT / "user_features_v3.csv"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "model_c_unweighted.joblib"

SIX_MONTHS = [201609, 201610, 201611, 201612, 201701, 201702]
SIX_MONTHS_SET = set(SIX_MONTHS)

FEATURE_NAMES = [
    "mean_active_days",
    "slope_active_days",
    "zero_months",
    "mean_songs_total",
    "mean_num_unq",
    "mean_total_secs",
    "active_days_diff",
    "is_auto_renew",
    "payment_plan_days",
    "is_cancel",
    "actual_amount_paid",
    "tx_count",
    "cancel_count",
    "has_transaction",
    "months_since_registration"
]

def find_file_or_partition(target_name, search_dirs=None):
    if search_dirs is None:
        search_dirs = [PROJECT_ROOT, PROJECT_ROOT.parent]
    for d in search_dirs:
        candidates = [
            d / target_name,
            d / f"{target_name}.csv",
            d / f"{target_name}_v2.csv",
            d / f"{target_name}_v2",
            d / f"{target_name}_v3.csv",
            d / f"{target_name}_v3",
        ]
        for c in candidates:
            if c.exists():
                if c.is_file():
                    return [c]
                elif c.is_dir():
                    files = [p for p in sorted(c.rglob("*")) if p.is_file() and not p.name.startswith(("_SUCCESS", ".", "_commit"))]
                    if files:
                        return files
    return []

def main():
    total_start = time.time()
    print("=================================================================")
    print("TRAIN & SAVE UNWEIGHTED MODEL C (WITH TENURE, 15 FEATURES)")
    print(f"Target Output: {MODEL_OUTPUT_PATH}")
    print("=================================================================\n")

    # 1. Cohort identification (201501 to 201608)
    print("Step 1: Identifying mature cohort subscribers...")
    train_files = find_file_or_partition("train_v2")
    if not train_files:
        raise FileNotFoundError("Could not find train_v2 dataset.")
    train_df = pd.concat([pd.read_csv(f, usecols=["msno", "is_churn"]) for f in train_files], ignore_index=True).drop_duplicates(subset=["msno"], keep="last")
    train_df["is_churn"] = train_df["is_churn"].astype("int8")

    members_files = find_file_or_partition("members_v3")
    if not members_files:
        raise FileNotFoundError("Could not find members_v3 dataset.")
    target_msnos = set(train_df["msno"])
    mem_chunks = []
    for f in members_files:
        for chunk in pd.read_csv(f, usecols=["msno", "registration_init_time"], chunksize=1000000):
            m = chunk[chunk["msno"].isin(target_msnos)].copy()
            if not m.empty:
                mem_chunks.append(m)
    members_df = pd.concat(mem_chunks, ignore_index=True).drop_duplicates(subset=["msno"], keep="last")
    reg_numeric = pd.to_numeric(members_df["registration_init_time"], errors="coerce")
    members_df["reg_month"] = (reg_numeric // 100).fillna(0).astype("int64")

    cohort_users = pd.merge(train_df, members_df[["msno", "reg_month"]], on="msno", how="inner")
    cohort_users = cohort_users[(cohort_users["reg_month"] >= 201501) & (cohort_users["reg_month"] <= 201608)].reset_index(drop=True)
    if len(cohort_users) != 293936:
        raise ValueError(f"Cohort size mismatch: {len(cohort_users):,} != 293,936")
    print(f"Confirmed mature cohort: {len(cohort_users):,} users.\n")

    # 2. 7 Listening Features (201609 to 201702)
    print("Step 2: Aggregating 6-month listening features...")
    cohort_set = set(cohort_users["msno"])
    act_cols = ["msno", "month", "active_days", "songs_total", "num_unq", "total_secs"]
    act_dtypes = {"msno": "string", "month": "int64", "active_days": "int16", "songs_total": "int32", "num_unq": "int32", "total_secs": "float32"}
    act_chunks = []
    for chunk in pd.read_csv(ACTIVITY_FILE, usecols=act_cols, dtype=act_dtypes, chunksize=1000000):
        matched = chunk[chunk["month"].isin(SIX_MONTHS_SET) & chunk["msno"].isin(cohort_set)]
        if not matched.empty:
            act_chunks.append(matched)
    six_act = pd.concat(act_chunks, ignore_index=True) if act_chunks else pd.DataFrame(columns=act_cols)

    piv_active = six_act.pivot(index="msno", columns="month", values="active_days").fillna(0).reindex(index=cohort_users["msno"], columns=SIX_MONTHS, fill_value=0).to_numpy(dtype=np.float32)
    piv_songs = six_act.pivot(index="msno", columns="month", values="songs_total").fillna(0).reindex(index=cohort_users["msno"], columns=SIX_MONTHS, fill_value=0).to_numpy(dtype=np.float32)
    piv_unq = six_act.pivot(index="msno", columns="month", values="num_unq").fillna(0).reindex(index=cohort_users["msno"], columns=SIX_MONTHS, fill_value=0).to_numpy(dtype=np.float32)
    piv_secs = six_act.pivot(index="msno", columns="month", values="total_secs").fillna(0).reindex(index=cohort_users["msno"], columns=SIX_MONTHS, fill_value=0).to_numpy(dtype=np.float32)

    weights = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32) / 17.5
    listening_df = pd.DataFrame({
        "msno": cohort_users["msno"].values,
        "mean_active_days": np.mean(piv_active, axis=1),
        "slope_active_days": np.dot(piv_active, weights),
        "zero_months": np.sum(piv_active == 0, axis=1).astype(np.int8),
        "mean_songs_total": np.mean(piv_songs, axis=1),
        "mean_num_unq": np.mean(piv_unq, axis=1),
        "mean_total_secs": np.mean(piv_secs, axis=1),
        "active_days_diff": piv_active[:, 5] - piv_active[:, 0]
    })

    # 3. 7 Subscription Features (user_features_v3.csv)
    print("Step 3: Loading subscription features...")
    v3_cols = ["msno", "is_auto_renew", "payment_plan_days", "is_cancel", "actual_amount_paid", "tx_count", "cancel_count"]
    v3_df = pd.read_csv(FEATURES_V3_FILE, usecols=v3_cols)
    sub_df = pd.merge(cohort_users[["msno", "reg_month", "is_churn"]], v3_df, on="msno", how="left")
    sub_df["has_transaction"] = sub_df["is_auto_renew"].notna().astype(np.int8)
    for col in ["is_auto_renew", "payment_plan_days", "is_cancel", "actual_amount_paid", "tx_count", "cancel_count"]:
        sub_df[col] = sub_df[col].fillna(0).astype(np.float32)

    # 4. Tenure Feature
    reg_m_vals = sub_df["reg_month"].values
    sub_df["months_since_registration"] = np.array([
        (2017 - (ym // 100)) * 12 + (2 - (ym % 100)) for ym in reg_m_vals
    ], dtype=np.float32)

    # 5. Assemble Full Dataset
    full_df = pd.merge(listening_df, sub_df, on="msno", how="inner")
    X = full_df[FEATURE_NAMES]
    y = full_df["is_churn"].values

    # Compute and save medians for the predictor default values
    feature_medians = X.median().to_dict()
    print(f"\nFeature Medians for Predictor Defaults:")
    for k, v in feature_medians.items():
        print(f"  {k:<28}: {v}")

    # 6. Stratified 80/20 Holdout Split Check
    print("\nStep 4: Evaluating on Stratified 80/20 Holdout Split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    clf_check = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
    clf_check.fit(X_train, y_train)
    probs_test = clf_check.predict_proba(X_test)[:, 1]

    holdout_auc = roc_auc_score(y_test, probs_test)
    holdout_pr = average_precision_score(y_test, probs_test)
    holdout_brier = brier_score_loss(y_test, probs_test)

    print("-----------------------------------------------------------------")
    print(f"CROSS-CHECK HOLDOUT TEST METRICS (Unweighted Model C + Tenure):")
    print(f"  ROC-AUC:     {holdout_auc:.4f}")
    print(f"  PR-AUC:      {holdout_pr:.4f}")
    print(f"  Brier Score: {holdout_brier:.5f}")
    print("-----------------------------------------------------------------\n")

    # 7. Fit Final Model on Full Cohort and Serialize
    print("Step 5: Fitting final estimator on full mature cohort (N=293,936)...")
    final_model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
    final_model.fit(X, y)

    payload = {
        "model": final_model,
        "feature_names": FEATURE_NAMES,
        "feature_medians": feature_medians,
        "holdout_metrics": {
            "roc_auc": holdout_auc,
            "pr_auc": holdout_pr,
            "brier_score": holdout_brier
        },
        "training_metadata": {
            "cohort_size": len(full_df),
            "churn_rate": float(np.mean(y)),
            "algorithm": "HistGradientBoostingClassifier",
            "weighting": "unweighted"
        }
    }

    joblib.dump(payload, MODEL_OUTPUT_PATH)
    print(f"SUCCESS: Model saved to {MODEL_OUTPUT_PATH.resolve()}")
    print(f"Execution complete in {(time.time()-total_start)/60:.2f} minutes.")
    print("=================================================================")

if __name__ == "__main__":
    main()
