"""evaluate_models.py
Rigorous Statistical Evaluation & Calibration Analysis of Churn Models.
Project Directory: E:\\ott video paltform

Reuses exact data prep, stratified 80/20 split (random_state=42), and model configurations from build_model.py.

Key Tasks:
1. Re-train models A (Listening), B (Subscription), and C (Combined), and obtain test-set predicted probabilities.
2. Bootstrap the test set (1,000 resamples, fixed random seed):
   - 95% percentile confidence intervals for ROC-AUC and PR-AUC for Models A, B, and C.
3. In the same bootstrap:
   - 95% confidence intervals for the paired differences: (Model C - Model B) for ROC-AUC and PR-AUC.
   - Empirical probability / share of resamples where Model C > Model B.
4. Compute and print the PR-AUC baseline (test set empirical churn rate).
5. Calibration analysis for Model C:
   - 10 probability bins: predicted churn rate vs. actual churn rate.
   - Save calibration curve to calibration_plot.png.
   - Compute and print Brier score for each model (A, B, C).
6. Permutation Importance for Model C:
   - Test set ROC-AUC drop over 5 repeats.
   - Print top 10 features and save permutation_importance.png.
7. Save all printed results to evaluation_results.txt.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve
)
from sklearn.calibration import calibration_curve

PROJECT_ROOT = Path(r"E:\ott video paltform")
ACTIVITY_FILE = PROJECT_ROOT / "monthly_user_activity.csv"
FEATURES_V3_FILE = PROJECT_ROOT / "user_features_v3.csv"
OUTPUT_REPORT_TXT = PROJECT_ROOT / "evaluation_results.txt"
PLOT_CALIBRATION = PROJECT_ROOT / "calibration_plot.png"
PLOT_PERM_IMPORTANCE = PROJECT_ROOT / "permutation_importance.png"

SIX_MONTHS = [201609, 201610, 201611, 201612, 201701, 201702]
SIX_MONTHS_SET = set(SIX_MONTHS)

class Logger:
    """Tee logger to print to console and write to file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

def find_file_or_partition(target_name, search_dirs=[PROJECT_ROOT, PROJECT_ROOT.parent]):
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
    logger = Logger(OUTPUT_REPORT_TXT)
    sys.stdout = logger

    print("=================================================================")
    print("EVALUATE_MODELS: STATISTICAL BOOTSTRAP, CALIBRATION & IMPORTANCE")
    print(f"Working Directory: {PROJECT_ROOT}")
    print("=================================================================\n")

    # Step 1: Load train_v2 and members_v3 (Cohort: reg_month <= 201608)
    print("Loading train_v2 and members_v3 metadata...")
    train_files = find_file_or_partition("train_v2")
    if not train_files:
        print("ERROR: Could not locate train_v2!")
        sys.exit(1)
    train_dfs = [pd.read_csv(f, usecols=["msno", "is_churn"]) for f in train_files]
    train_df = pd.concat(train_dfs, ignore_index=True).drop_duplicates(subset=["msno"], keep="last")
    train_df["is_churn"] = train_df["is_churn"].astype("int8")
    total_train = len(train_df)

    members_files = find_file_or_partition("members_v3")
    if not members_files:
        print("ERROR: Could not locate members_v3!")
        sys.exit(1)
    target_msno_set = set(train_df["msno"])
    members_chunks = []
    for f in members_files:
        for chunk in pd.read_csv(f, usecols=["msno", "registration_init_time"], chunksize=500000):
            matched = chunk[chunk["msno"].isin(target_msno_set)].copy()
            if not matched.empty:
                members_chunks.append(matched)
    members_df = pd.concat(members_chunks, ignore_index=True).drop_duplicates(subset=["msno"], keep="last")
    members_df["reg_date"] = pd.to_numeric(members_df["registration_init_time"], errors="coerce")
    members_df["reg_month"] = (members_df["reg_date"] // 100).fillna(0).astype("int64")

    cohort_users = pd.merge(train_df, members_df[["msno", "reg_month"]], on="msno", how="inner")
    cohort_users = cohort_users[(cohort_users["reg_month"] >= 201501) & (cohort_users["reg_month"] <= 201608)].copy()
    cohort_users = cohort_users.reset_index(drop=True)
    cohort_msno_set = set(cohort_users["msno"])
    print(f"Eligible mature cohort users (201501 <= reg_month <= 201608): {len(cohort_users):,}")

    # Step 2: Compute Listening Features (Exact identical logic to build_model.py)
    print("Computing 6-month listening features (201609 to 201702)...")
    act_cols = ["msno", "month", "active_days", "songs_total", "num_unq", "total_secs"]
    act_dtypes = {
        "msno": "string",
        "month": "int64",
        "active_days": "int16",
        "songs_total": "int32",
        "num_unq": "int32",
        "total_secs": "float32"
    }

    act_chunks = []
    for chunk in pd.read_csv(ACTIVITY_FILE, usecols=act_cols, dtype=act_dtypes, chunksize=1000000):
        matched = chunk[chunk["month"].isin(SIX_MONTHS_SET) & chunk["msno"].isin(cohort_msno_set)]
        if not matched.empty:
            act_chunks.append(matched)

    six_act = pd.concat(act_chunks, ignore_index=True) if act_chunks else pd.DataFrame(columns=act_cols)

    piv_active = six_act.pivot(index="msno", columns="month", values="active_days").fillna(0).reindex(index=cohort_users["msno"], columns=SIX_MONTHS, fill_value=0).to_numpy(dtype=np.float32)
    piv_songs = six_act.pivot(index="msno", columns="month", values="songs_total").fillna(0).reindex(index=cohort_users["msno"], columns=SIX_MONTHS, fill_value=0).to_numpy(dtype=np.float32)
    piv_unq = six_act.pivot(index="msno", columns="month", values="num_unq").fillna(0).reindex(index=cohort_users["msno"], columns=SIX_MONTHS, fill_value=0).to_numpy(dtype=np.float32)
    piv_secs = six_act.pivot(index="msno", columns="month", values="total_secs").fillna(0).reindex(index=cohort_users["msno"], columns=SIX_MONTHS, fill_value=0).to_numpy(dtype=np.float32)

    weights = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32) / 17.5
    listening_features = pd.DataFrame({
        "msno": cohort_users["msno"],
        "mean_active_days": np.mean(piv_active, axis=1),
        "slope_active_days": np.dot(piv_active, weights),
        "zero_months": np.sum(piv_active == 0, axis=1).astype(np.int8),
        "mean_songs_total": np.mean(piv_songs, axis=1),
        "mean_num_unq": np.mean(piv_unq, axis=1),
        "mean_total_secs": np.mean(piv_secs, axis=1),
        "active_days_diff": piv_active[:, 5] - piv_active[:, 0]
    })

    # Step 3: Load Subscription Features from user_features_v3.csv
    print("Loading subscription features from user_features_v3.csv...")
    if not FEATURES_V3_FILE.exists():
        print(f"ERROR: {FEATURES_V3_FILE.name} not found!")
        sys.exit(1)

    v3_cols = ["msno", "is_auto_renew", "payment_plan_days", "is_cancel", "actual_amount_paid", "tx_count", "cancel_count"]
    v3_df = pd.read_csv(FEATURES_V3_FILE, usecols=v3_cols)
    sub_df = pd.merge(cohort_users[["msno", "is_churn"]], v3_df, on="msno", how="left")
    sub_df["has_transaction"] = sub_df["is_auto_renew"].notna().astype(np.int8)
    for col in ["is_auto_renew", "payment_plan_days", "is_cancel", "actual_amount_paid", "tx_count", "cancel_count"]:
        sub_df[col] = sub_df[col].fillna(0).astype(np.float32)

    full_dataset = pd.merge(listening_features, sub_df, on="msno", how="inner")

    listening_feature_names = [
        "mean_active_days",
        "slope_active_days",
        "zero_months",
        "mean_songs_total",
        "mean_num_unq",
        "mean_total_secs",
        "active_days_diff"
    ]
    subscription_feature_names = [
        "is_auto_renew",
        "payment_plan_days",
        "is_cancel",
        "actual_amount_paid",
        "tx_count",
        "cancel_count",
        "has_transaction"
    ]
    all_feature_names = listening_feature_names + subscription_feature_names

    X_all = full_dataset[all_feature_names]
    y = full_dataset["is_churn"].values

    # Step 4: Stratified 80/20 Split (random_state=42)
    indices = np.arange(len(full_dataset))
    idx_train, idx_test = train_test_split(indices, test_size=0.20, random_state=42, stratify=y)

    y_train = y[idx_train]
    y_test = y[idx_test]

    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos = num_neg / num_pos if num_pos > 0 else 1.0

    # Step 5: PR-AUC Baseline
    test_churn_rate = y_test.mean()
    print("-----------------------------------------------------------------")
    print(f"TEST SET PR-AUC BASELINE (Empirical Churn Rate): {test_churn_rate:.4f} ({test_churn_rate*100:.2f}%)")
    print("-----------------------------------------------------------------\n")

    # Step 6: Train Models A, B, C
    use_lgb = False
    try:
        import lightgbm as lgb
        use_lgb = True
        print("Using LightGBM Classifier.")
    except ImportError:
        print("Falling back to sklearn.ensemble.HistGradientBoostingClassifier.")
        from sklearn.ensemble import HistGradientBoostingClassifier

    model_configs = {
        "Model A": listening_feature_names,
        "Model B": subscription_feature_names,
        "Model C": all_feature_names
    }

    trained_models = {}
    test_probs = {}

    print("Re-training Models A, B, and C with identical parameters...")
    for m_key, feat_cols in model_configs.items():
        X_tr = X_all.iloc[idx_train][feat_cols].values
        X_te = X_all.iloc[idx_test][feat_cols].values

        if use_lgb:
            clf = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                scale_pos_weight=scale_pos,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
        else:
            clf = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_depth=6,
                class_weight="balanced",
                random_state=42
            )
        clf.fit(X_tr, y_train)
        probs_te = clf.predict_proba(X_te)[:, 1]
        trained_models[m_key] = clf
        test_probs[m_key] = probs_te

    # Point estimates on full test set
    print("\n--- FULL TEST SET POINT ESTIMATES ---")
    for m_key in ["Model A", "Model B", "Model C"]:
        auc_pt = roc_auc_score(y_test, test_probs[m_key])
        pr_pt = average_precision_score(y_test, test_probs[m_key])
        brier_pt = brier_score_loss(y_test, test_probs[m_key])
        print(f"{m_key:<8}: ROC-AUC = {auc_pt:.4f} | PR-AUC = {pr_pt:.4f} | Brier Score = {brier_pt:.4f}")
    print()

    # Step 7: Paired Bootstrap (1,000 resamples, fixed seed)
    N_BOOTSTRAP = 1000
    RNG_SEED = 42
    rng = np.random.RandomState(RNG_SEED)
    n_test = len(y_test)

    print(f"Running Paired Bootstrap ({N_BOOTSTRAP:,} resamples, seed={RNG_SEED})...")
    boot_start = time.time()

    boot_auc = {k: np.empty(N_BOOTSTRAP, dtype=np.float64) for k in model_configs}
    boot_pr = {k: np.empty(N_BOOTSTRAP, dtype=np.float64) for k in model_configs}
    diff_auc_CB = np.empty(N_BOOTSTRAP, dtype=np.float64)
    diff_pr_CB = np.empty(N_BOOTSTRAP, dtype=np.float64)

    probs_A = test_probs["Model A"]
    probs_B = test_probs["Model B"]
    probs_C = test_probs["Model C"]

    for b in range(N_BOOTSTRAP):
        sample_indices = rng.randint(0, n_test, size=n_test)
        y_b = y_test[sample_indices]

        # In extreme rare case of homogeneous sample, resample
        while y_b.sum() == 0 or y_b.sum() == n_test:
            sample_indices = rng.randint(0, n_test, size=n_test)
            y_b = y_test[sample_indices]

        auc_a = roc_auc_score(y_b, probs_A[sample_indices])
        pr_a = average_precision_score(y_b, probs_A[sample_indices])

        auc_b = roc_auc_score(y_b, probs_B[sample_indices])
        pr_b = average_precision_score(y_b, probs_B[sample_indices])

        auc_c = roc_auc_score(y_b, probs_C[sample_indices])
        pr_c = average_precision_score(y_b, probs_C[sample_indices])

        boot_auc["Model A"][b] = auc_a
        boot_pr["Model A"][b] = pr_a

        boot_auc["Model B"][b] = auc_b
        boot_pr["Model B"][b] = pr_b

        boot_auc["Model C"][b] = auc_c
        boot_pr["Model C"][b] = pr_c

        diff_auc_CB[b] = auc_c - auc_b
        diff_pr_CB[b] = pr_c - pr_b

    print(f"Completed bootstrap in {time.time()-boot_start:.1f} s.\n")

    # Print 95% Percentile Confidence Intervals
    print("=================================================================")
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS (1,000 Resamples)")
    print("=================================================================")
    for m_key in ["Model A", "Model B", "Model C"]:
        auc_low, auc_high = np.percentile(boot_auc[m_key], [2.5, 97.5])
        pr_low, pr_high = np.percentile(boot_pr[m_key], [2.5, 97.5])
        print(f"{m_key:<8} ROC-AUC: {np.mean(boot_auc[m_key]):.4f} (95% CI: [{auc_low:.4f}, {auc_high:.4f}])")
        print(f"{m_key:<8} PR-AUC:  {np.mean(boot_pr[m_key]):.4f} (95% CI: [{pr_low:.4f}, {pr_high:.4f}])")
        print()

    # Model C minus Model B comparisons
    diff_auc_low, diff_auc_high = np.percentile(diff_auc_CB, [2.5, 97.5])
    diff_pr_low, diff_pr_high = np.percentile(diff_pr_CB, [2.5, 97.5])
    p_c_greater_b_auc = np.mean(diff_auc_CB > 0) * 100
    p_c_greater_b_pr = np.mean(diff_pr_CB > 0) * 100

    print("--- PAIRED DIFFERENCE (MODEL C minus MODEL B) ---")
    print(f"ROC-AUC Difference (C - B): {np.mean(diff_auc_CB):+.4f} (95% CI: [{diff_auc_low:+.4f}, {diff_auc_high:+.4f}])")
    print(f"Share of resamples where Model C > Model B (ROC-AUC): {p_c_greater_b_auc:.2f}%\n")

    print(f"PR-AUC Difference (C - B):  {np.mean(diff_pr_CB):+.4f} (95% CI: [{diff_pr_low:+.4f}, {diff_pr_high:+.4f}])")
    print(f"Share of resamples where Model C > Model B (PR-AUC):  {p_c_greater_b_pr:.2f}%\n")

    # Step 8: Calibration Check for Model C & Brier Scores
    print("=================================================================")
    print("CALIBRATION ANALYSIS & BRIER SCORES")
    print("=================================================================")
    print("Brier Scores (Lower is Better):")
    for m_key in ["Model A", "Model B", "Model C"]:
        bs = brier_score_loss(y_test, test_probs[m_key])
        print(f"   {m_key:<8}: {bs:.5f}")
    print()

    # 10 Decile probability bins for Model C
    prob_c = test_probs["Model C"]
    # Compute calibration curve with 10 bins
    prob_true, prob_pred = calibration_curve(y_test, prob_c, n_bins=10, strategy="uniform")

    # Also build detailed table using pd.cut
    bins = np.linspace(0.0, 1.0, 11)
    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(10)]
    bin_assignments = pd.cut(prob_c, bins=bins, include_lowest=True, labels=bin_labels)
    calib_df = pd.DataFrame({"y_true": y_test, "y_prob": prob_c, "bin": bin_assignments})

    print(f"{'Bin':<12} | {'Count (n)':<10} | {'Mean Pred Prob':<16} | {'Actual Churn Rate':<18} | {'Gap (Pred - Actual)':<20}")
    print("-" * 84)
    calib_table = []
    for b_label in bin_labels:
        sub_b = calib_df[calib_df["bin"] == b_label]
        n_b = len(sub_b)
        if n_b > 0:
            m_pred = sub_b["y_prob"].mean()
            m_act = sub_b["y_true"].mean()
            gap = m_pred - m_act
            print(f"{b_label:<12} | {n_b:10,d} | {m_pred:16.4f} | {m_act:18.4f} | {gap:+20.4f}")
            calib_table.append({"bin": b_label, "n": n_b, "pred": m_pred, "actual": m_act})
        else:
            print(f"{b_label:<12} | {0:10,d} | {'N/A':<16} | {'N/A':<18} | {'N/A':<20}")

    # Calibration Plot
    plt.figure(figsize=(7.5, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
    plt.plot(prob_pred, prob_true, "s-", color="#2ca02c", linewidth=2.2, label=f"Model C (Brier = {brier_score_loss(y_test, prob_c):.4f})")
    plt.title("Calibration Curve: Model C (10 Probability Bins)", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Observed Churn Proportion", fontsize=11)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=11, loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOT_CALIBRATION, dpi=200)
    plt.close()
    print(f"\nSaved Calibration Plot: {PLOT_CALIBRATION.name}\n")

    # Step 9: Permutation Importance for Model C (ROC-AUC drop, 5 repeats)
    print("=================================================================")
    print("PERMUTATION IMPORTANCE: MODEL C (Test Set, 5 Repeats)")
    print("=================================================================")
    N_REPEATS = 5
    baseline_auc_c = roc_auc_score(y_test, prob_c)
    clf_c = trained_models["Model C"]
    X_te_c = X_all.iloc[idx_test][all_feature_names].values.copy()

    perm_drops = {f_name: np.empty(N_REPEATS, dtype=np.float64) for f_name in all_feature_names}

    print(f"Evaluating {len(all_feature_names)} features across {N_REPEATS} repeats on test set...")
    perm_rng = np.random.RandomState(42)

    for col_idx, f_name in enumerate(all_feature_names):
        col_orig = X_te_c[:, col_idx].copy()
        for r in range(N_REPEATS):
            # Shuffle feature values
            X_te_c[:, col_idx] = perm_rng.permutation(col_orig)
            perm_prob = clf_c.predict_proba(X_te_c)[:, 1]
            shuffled_auc = roc_auc_score(y_test, perm_prob)
            perm_drops[f_name][r] = baseline_auc_c - shuffled_auc
        # Restore original column
        X_te_c[:, col_idx] = col_orig

    perm_summary = []
    for f_name in all_feature_names:
        mean_drop = np.mean(perm_drops[f_name])
        std_drop = np.std(perm_drops[f_name])
        f_type = "Listening" if f_name in listening_feature_names else "Subscription"
        perm_summary.append({
            "feature": f_name,
            "type": f_type,
            "mean_auc_drop": mean_drop,
            "std_auc_drop": std_drop
        })

    perm_df = pd.DataFrame(perm_summary).sort_values("mean_auc_drop", ascending=False).reset_index(drop=True)

    print("\n--- TOP 10 FEATURES BY TEST ROC-AUC DROP ---")
    print(f"{'Rank':<5} | {'Feature':<22} | {'Type':<13} | {'Mean AUC Drop':<14} | {'Std Dev':<10}")
    print("-" * 72)
    for rank in range(min(10, len(perm_df))):
        row = perm_df.iloc[rank]
        print(f"{rank+1:<5} | {row['feature']:<22} | {row['type']:<13} | {row['mean_auc_drop']:14.5f} | {row['std_auc_drop']:10.5f}")

    # Plot Permutation Importance (Top 10)
    top10_df = perm_df.head(10).iloc[::-1]  # Invert for horizontal bar plot
    plt.figure(figsize=(9.5, 6))
    bar_colors = ["#1f77b4" if t == "Listening" else "#ff7f0e" for t in top10_df["type"]]
    plt.barh(top10_df["feature"], top10_df["mean_auc_drop"], xerr=top10_df["std_auc_drop"], capsize=4, color=bar_colors, edgecolor="black", linewidth=0.6)
    plt.title("Top 10 Features: Test Set Permutation Importance (ROC-AUC Drop)", fontsize=12.5, fontweight="bold")
    plt.xlabel("Mean ROC-AUC Drop Upon Permutation (5 Repeats)", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", edgecolor="black", label="Listening / Decay Feature"),
        Patch(facecolor="#ff7f0e", edgecolor="black", label="Subscription / Transaction Feature")
    ]
    plt.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=10.5)
    plt.tight_layout()
    plt.savefig(PLOT_PERM_IMPORTANCE, dpi=200)
    plt.close()
    print(f"\nSaved Permutation Importance Plot: {PLOT_PERM_IMPORTANCE.name}\n")

    total_time = time.time() - total_start
    print("=================================================================")
    print(f"EVALUATION COMPLETE IN {total_time/60:.2f} MINUTES")
    print(f"Saved complete log to: {OUTPUT_REPORT_TXT.name}")
    print("=================================================================")

    logger.close()
    sys.stdout = logger.terminal

if __name__ == "__main__":
    main()
