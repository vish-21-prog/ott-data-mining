# Empirical Fact Sheet: User Retention Decay Profiling and Churn Analysis in Subscription Streaming Platforms Using KKBox Listening Logs

**Location:** `E:\ott video paltform\report_assets\report_fact_sheet.md`  
**Generated At:** 2026-10-19 18:17:04  

All figures below are extracted directly from empirical datasets, computed tables, and statistical validation logs without interpolation or hand-typed values.

---

### 1. Dataset Dimensions & Subscriber Cohorts
- **Historical Listening Logs (user_logs.csv)**: 392,106,543 records (Scope: 201501 to 201612, 26 months) [Population: Total Historical Logs, Source: dataset_summary.csv]
- **March 2017 Listening Logs (user_logs_v2.csv)**: 18,396,362 records (Scope: 20170301 to 20170331, 1 month) [Population: March 2017 Logs, Source: dataset_summary.csv]
- **Total Combined Listening Logs**: 410,502,905 records (Scope: 201501 to 201703, 27 calendar months) [Population: Combined Logs, Source: dataset_summary.csv]
- **Total Training Ground Truth Users (train_v2.csv)**: 970,960 subscribers [Population: Total Ground Truth, Source: dataset_summary.csv]
- **Training Users with Listening History**: 855,160 subscribers [Population: train_v2 Subscribers, Source: dataset_summary.csv]
- **Training Users with No Listening History**: 115,800 subscribers [Population: train_v2 Subscribers, Source: dataset_summary.csv]
- **New Subscriber Cohorts (Registered 201501 to 201703)**: 382,789 subscribers [Population: train_v2 Subscribers, Source: dataset_summary.csv]
- **Mature Subscriber Cohorts (Registered 201501 to 201608)**: 293,936 subscribers [Population: train_v2 Subscribers, Source: dataset_summary.csv]

---

### 2. Longitudinal Retention Decay
- **Retention at k = 1**: Average 83.63% across 26 cohorts (Range: [74.52%, 93.75%]) [Population: Mature Cohorts, Source: retention_summary.csv]
- **Retention at k = 3**: Average 79.49% across 24 cohorts (Range: [71.01%, 89.36%]) [Population: Mature Cohorts, Source: retention_summary.csv]
- **Retention at k = 6**: Average 78.51% across 21 cohorts (Range: [70.26%, 89.35%]) [Population: Mature Cohorts, Source: retention_summary.csv]
- **Retention at k = 12**: Average 79.09% across 15 cohorts (Range: [72.15%, 82.43%]) [Population: Mature Cohorts, Source: retention_summary.csv]

---

### 3. Decay Profiles (Mature Cohort N=293,936)
- **Inactive Profile**: 23,129 subscribers (7.87%) | Churn Rate: 5.72% (95% CI: [5.42%, 6.02%]) [Population: Mature Cohort, Source: profile_summary_final.csv]
- **Fast Decay Profile**: 94,531 subscribers (32.16%) | Churn Rate: 9.31% (95% CI: [9.13%, 9.50%]) [Population: Mature Cohort, Source: profile_summary_final.csv]
- **Stable Profile**: 97,038 subscribers (33.01%) | Churn Rate: 7.67% (95% CI: [7.50%, 7.84%]) [Population: Mature Cohort, Source: profile_summary_final.csv]
- **Growing Profile**: 79,238 subscribers (26.96%) | Churn Rate: 10.33% (95% CI: [10.12%, 10.54%]) [Population: Mature Cohort, Source: profile_summary_final.csv]
- **Cramér's V (Profile vs. Churn)**: 0.0484 [Population: Mature Cohort N=293,936, Source: profile_summary_final.csv]

---

### 4. Auto-Renew Contractual Interactions
- **Overall Non-Auto-Renew (is_auto_renew = 0)**: 41,459 subscribers | Churn Rate: 37.82% (95% CI: [37.36%, 38.29%]) [Population: Mature Cohort with Tx, Source: subscription_summary.csv]
- **Overall Auto-Renew (is_auto_renew = 1)**: 252,234 subscribers | Churn Rate: 3.92% (95% CI: [3.84%, 4.00%]) [Population: Mature Cohort with Tx, Source: subscription_summary.csv]
- **Inactive Subgroup**: Total Tx n=22,896 | Auto-Renew=0: 149 (0.65%) | Churn (AR=0): 83.22% (95% CI: [76.40%, 88.37%]) | Churn (AR=1): 4.52% (95% CI: [4.26%, 4.80%]) [Population: Inactive Profile with Tx, Source: profile_autorenew_share.csv]
- **Fast Decay Subgroup**: Total Tx n=94,529 | Auto-Renew=0: 13,197 (13.96%) | Churn (AR=0): 39.36% (95% CI: [38.53%, 40.19%]) | Churn (AR=1): 4.43% (95% CI: [4.29%, 4.58%]) [Population: Fast Decay Profile with Tx, Source: profile_autorenew_share.csv]
- **Stable Subgroup**: Total Tx n=97,033 | Auto-Renew=0: 11,664 (12.02%) | Churn (AR=0): 37.35% (95% CI: [36.48%, 38.24%]) | Churn (AR=1): 3.6% (95% CI: [3.48%, 3.73%]) [Population: Stable Profile with Tx, Source: profile_autorenew_share.csv]
- **Growing Subgroup**: Total Tx n=79,235 | Auto-Renew=0: 16,449 (20.76%) | Churn (AR=0): 36.51% (95% CI: [35.78%, 37.25%]) | Churn (AR=1): 3.46% (95% CI: [3.32%, 3.61%]) [Population: Growing Profile with Tx, Source: profile_autorenew_share.csv]

---

### 5. Subscribers Without Listening Logs
- **Subscribers with No Listening Rows (Any Month 201501-201703)**: 115,800 users | Churn Rate: 5.53% (95% CI: [5.40%, 5.66%]) [Population: Total Training Users N=970,960, Source: no_listening_analysis.csv]
- **No-Listening Subscribers (0 logs in Sep 16 - Feb 17)**: 49,333 users | Churn Rate: 10.82% (95% CI: [10.55%, 11.10%]) | Auto-Renew=1 Share: 99.32% [Population: Registered Training Users N=860,967, Source: no_listening_analysis.csv]
- **Active-Listening Subscribers (>=1 logs in Sep 16 - Feb 17)**: 811,634 users | Churn Rate: 9.38% (95% CI: [9.31%, 9.44%]) | Auto-Renew=1 Share: 84.41% [Population: Registered Training Users N=860,967, Source: no_listening_analysis.csv]
- **No-Listening with Auto-Renew = 0**: 176 users | Churn Rate: 81.25% (95% CI: [74.84%, 86.33%]) [Population: No-Listening with Known Tx, Source: no_listening_analysis.csv]
- **No-Listening with Auto-Renew = 1**: 25,574 users | Churn Rate: 4.17% (95% CI: [3.93%, 4.42%]) [Population: No-Listening with Known Tx, Source: no_listening_analysis.csv]

---

### 6. Model Predictive Performance: Cutoff 20170228 (1-Month Horizon)
- **Model A (Listening Only, 7 features)**: ROC-AUC = 0.6316 [0.6235, 0.6393] | PR-AUC = 0.1599 [0.1523, 0.1683] [Population: Test Set N=58,788, Source: model_results_final.csv]
- **Model B (Subscription Only, 7 features)**: ROC-AUC = 0.8816 [0.8766, 0.8867] | PR-AUC = 0.5964 [0.5838, 0.6104] [Population: Test Set N=58,788, Source: model_results_final.csv]
- **Model C (Combined Features, 14 features)**: ROC-AUC = 0.8843 [0.8792, 0.8893] | PR-AUC = 0.6125 [0.5995, 0.6265] [Population: Test Set N=58,788, Source: model_results_final.csv]
- **Incremental ROC Lift (C - B)**: +0.0027 [+0.0010, +0.0043] (100% win) [Population: Test Set N=58,788, Source: model_results_final.csv]
- **Incremental PR Lift (C - B)**: +0.0160 [+0.0125, +0.0197] (100% win) [Population: Test Set N=58,788, Source: model_results_final.csv]
- **PR-AUC Baseline**: 0.0876 (8.76%) [Population: Test Set N=58,788, Source: model_results_final.csv]

---

### 7. Temporal Sensitivity: Cutoff 20170131 (2-Month Horizon)
- **Model A (Listening Only)**: ROC-AUC = 0.6316 [0.6235, 0.6393] | PR-AUC = 0.1599 [0.1523, 0.1683] [Population: Test Set N=58,788, Source: model_results_final.csv]
- **Model B (Subscription Only)**: ROC-AUC = 0.8461 [0.8406, 0.8522] | PR-AUC = 0.4923 [0.4790, 0.5075] [Population: Test Set N=58,788, Source: model_results_final.csv]
- **Model C (Combined Features)**: ROC-AUC = 0.8531 [0.8472, 0.8589] | PR-AUC = 0.5117 [0.4978, 0.5276] [Population: Test Set N=58,788, Source: model_results_final.csv]
- **Incremental ROC Lift (C - B)**: +0.0070 [+0.0049, +0.0090] (100% win) [Population: Test Set N=58,788, Source: model_results_final.csv]
- **Incremental PR Lift (C - B)**: +0.0194 [+0.0155, +0.0237] (100% win) [Population: Test Set N=58,788, Source: model_results_final.csv]

---

### 8. 5-Fold Stratified Cross-Validation (Mature Cohort N=293,936)
- **Model A (Listening Only)**: ROC-AUC = 0.6249 +/- 0.0043 | PR-AUC = 0.1551 +/- 0.0035 [Population: Mature Cohort N=293,936, Source: cv_results.csv]
- **Model B (Subscription Only)**: ROC-AUC = 0.8780 +/- 0.0030 | PR-AUC = 0.5916 +/- 0.0061 [Population: Mature Cohort N=293,936, Source: cv_results.csv]
- **Model C (Combined Features)**: ROC-AUC = 0.8811 +/- 0.0029 | PR-AUC = 0.6096 +/- 0.0048 [Population: Mature Cohort N=293,936, Source: cv_results.csv]

---

### 9. Probability Calibration & Brier Scores
- **Prevalence Constant Baseline**: Brier Score = 0.07993 [Population: Test Set N=58,788, Source: model_unweighted.csv]
- **Model B (Weighted, scale_pos_weight=10.42)**: Brier Score = 0.12291 [Population: Test Set N=58,788, Source: model_unweighted.csv]
- **Model B (Unweighted)**: Brier Score = 0.05136 (58.2% error reduction vs weighted) [Population: Test Set N=58,788, Source: model_unweighted.csv]
- **Model C (Weighted, scale_pos_weight=10.42)**: Brier Score = 0.11856 [Population: Test Set N=58,788, Source: model_unweighted.csv]
- **Model C (Unweighted)**: Brier Score = 0.05028 (57.6% error reduction vs weighted) [Population: Test Set N=58,788, Source: model_unweighted.csv]

---

### 10. Test-Set Permutation Feature Importance (Model C, Top 10)
- Rank 1: `is_auto_renew` (Subscription) | ROC-AUC Drop = 0.11073 +/- 0.00090 (Population: Test Set N=58,788, Source: evaluation_results.txt)
- Rank 2: `tx_count` (Subscription) | ROC-AUC Drop = 0.03906 +/- 0.00176 (Population: Test Set N=58,788, Source: evaluation_results.txt)
- Rank 3: `actual_amount_paid` (Subscription) | ROC-AUC Drop = 0.03874 +/- 0.00112 (Population: Test Set N=58,788, Source: evaluation_results.txt)
- Rank 4: `is_cancel` (Subscription) | ROC-AUC Drop = 0.03106 +/- 0.00016 (Population: Test Set N=58,788, Source: evaluation_results.txt)
- Rank 5: `cancel_count` (Subscription) | ROC-AUC Drop = 0.01075 +/- 0.00040 (Population: Test Set N=58,788, Source: evaluation_results.txt)
- Rank 6: `mean_active_days` (Listening) | ROC-AUC Drop = 0.00447 +/- 0.00041 (Population: Test Set N=58,788, Source: evaluation_results.txt)
- Rank 7: `payment_plan_days` (Subscription) | ROC-AUC Drop = 0.00427 +/- 0.00024 (Population: Test Set N=58,788, Source: evaluation_results.txt)
- Rank 8: `zero_months` (Listening) | ROC-AUC Drop = 0.00332 +/- 0.00028 (Population: Test Set N=58,788, Source: evaluation_results.txt)
- Rank 9: `mean_total_secs` (Listening) | ROC-AUC Drop = 0.00254 +/- 0.00036 (Population: Test Set N=58,788, Source: evaluation_results.txt)
- Rank 10: `slope_active_days` (Listening) | ROC-AUC Drop = 0.00184 +/- 0.00027 (Population: Test Set N=58,788, Source: evaluation_results.txt)

---

### 11. Ground Truth Membership Expiry Verification (train_v2 Users)
- **Primary Expiry Distribution**: Most membership expiry dates fall in 201703 (951,186 subscribers, 98.22% of all train_v2 users with transactions before cutoff).
- Expiry 201603: 1,142 subscribers (0.12%) (Population: train_v2 Subscribers with Transactions <= 20170228, Source: train_expiry_check.csv)
- Expiry 201701: 334 subscribers (0.03%) (Population: train_v2 Subscribers with Transactions <= 20170228, Source: train_expiry_check.csv)
- Expiry 201702: 13,820 subscribers (1.43%) (Population: train_v2 Subscribers with Transactions <= 20170228, Source: train_expiry_check.csv)
- Expiry 201703: 951,186 subscribers (98.22%) (Population: train_v2 Subscribers with Transactions <= 20170228, Source: train_expiry_check.csv)
- Expiry Other Months Combined: 1,951 subscribers (0.2%) (Population: train_v2 Subscribers with Transactions <= 20170228, Source: train_expiry_check.csv)

---

### 12. Longitudinal Retention Trajectory Check (v2 vs. Old)
- k = 0: v2 Non-Churn Ret = 82.91% (n=268,191), v2 Churn Ret = 69.64% (n=25,745) | Old Non-Churn Ret = 82.91% (n=268,191), Old Churn Ret = 69.64% (n=25,745) (Population: Mature Cohorts 201501-201608, Source: retention_v2_check.csv)
- k = 1: v2 Non-Churn Ret = 81.46% (n=268,191), v2 Churn Ret = 65.58% (n=25,745) | Old Non-Churn Ret = 81.46% (n=268,191), Old Churn Ret = 65.58% (n=25,745) (Population: Mature Cohorts 201501-201608, Source: retention_v2_check.csv)
- k = 3: v2 Non-Churn Ret = 77.66% (n=268,191), v2 Churn Ret = 57.08% (n=25,745) | Old Non-Churn Ret = 77.66% (n=268,191), Old Churn Ret = 57.08% (n=25,745) (Population: Mature Cohorts 201501-201608, Source: retention_v2_check.csv)
- k = 6: v2 Non-Churn Ret = 77.97% (n=268,191), v2 Churn Ret = 49.12% (n=25,745) | Old Non-Churn Ret = 77.97% (n=268,191), Old Churn Ret = 49.12% (n=25,745) (Population: Mature Cohorts 201501-201608, Source: retention_v2_check.csv)
- k = 12: v2 Non-Churn Ret = 78.96% (n=181,602), v2 Churn Ret = 33.16% (n=18,167) | Old Non-Churn Ret = 79.09% (n=197,442), Old Churn Ret = 33.22% (n=20,610) (Population: Mature Cohorts 201501-201608, Source: retention_v2_check.csv)

---

### 13. Monthly Activity Profile (Mature Cohort N=293,936: 201609 to 201702)
- Month 201609 | Non-Churned (All (Overall Group)): Active Share = 80.78% (n=216,632 of 268,191) | Mean Active Days = 12.94 (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
- Month 201609 | Churned (All (Overall Group)): Active Share = 76.38% (n=19,665 of 25,745) | Mean Active Days = 13.47 (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
- Month 201610 | Non-Churned (All (Overall Group)): Active Share = 81.25% (n=217,898 of 268,191) | Mean Active Days = 13.82 (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
- Month 201610 | Churned (All (Overall Group)): Active Share = 76.99% (n=19,820 of 25,745) | Mean Active Days = 14.42 (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
- Month 201611 | Non-Churned (All (Overall Group)): Active Share = 81.02% (n=217,298 of 268,191) | Mean Active Days = 13.55 (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
- Month 201611 | Churned (All (Overall Group)): Active Share = 78.09% (n=20,103 of 25,745) | Mean Active Days = 14.09 (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
- Month 201612 | Non-Churned (All (Overall Group)): Active Share = 82.37% (n=220,897 of 268,191) | Mean Active Days = 14.24 (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
- Month 201612 | Churned (All (Overall Group)): Active Share = 80.04% (n=20,607 of 25,745) | Mean Active Days = 14.9 (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
- Month 201701 | Non-Churned (All (Overall Group)): Active Share = 83.14% (n=222,963 of 268,191) | Mean Active Days = 14.12 (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
- Month 201701 | Churned (All (Overall Group)): Active Share = 81.68% (n=21,029 of 25,745) | Mean Active Days = 14.82 (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
- Month 201702 | Non-Churned (All (Overall Group)): Active Share = 82.89% (n=222,299 of 268,191) | Mean Active Days = 12.66 (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
- Month 201702 | Churned (All (Overall Group)): Active Share = 88.0% (n=22,656 of 25,745) | Mean Active Days = 13.73 (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
- Month 201701 | Non-Churned (is_auto_renew = 0): Active Share = 92.76% (n=23,911 of 25,778) | Mean Active Days = 19.61 (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
- Month 201701 | Churned (is_auto_renew = 0): Active Share = 84.89% (n=13,311 of 15,681) | Mean Active Days = 16.57 (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
- Month 201701 | Non-Churned (is_auto_renew = 1): Active Share = 82.13% (n=199,052 of 242,348) | Mean Active Days = 13.54 (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
- Month 201701 | Churned (is_auto_renew = 1): Active Share = 78.04% (n=7,715 of 9,886) | Mean Active Days = 12.32 (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
- Month 201702 | Non-Churned (is_auto_renew = 0): Active Share = 99.51% (n=25,652 of 25,778) | Mean Active Days = 18.69 (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
- Month 201702 | Churned (is_auto_renew = 0): Active Share = 96.97% (n=15,206 of 15,681) | Mean Active Days = 16.1 (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
- Month 201702 | Non-Churned (is_auto_renew = 1): Active Share = 81.14% (n=196,646 of 242,348) | Mean Active Days = 12.02 (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
- Month 201702 | Churned (is_auto_renew = 1): Active Share = 75.32% (n=7,446 of 9,886) | Mean Active Days = 10.22 (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)

---

### 14. Zero-Activity Subscribers Across Entire 6-Month Window
- Non-Churned Subscribers with Zero Activity in all 6 Months: 21,807 of 268,191 (8.13%) (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
- Churned Subscribers with Zero Activity in all 6 Months: 1,322 of 25,745 (5.13%) (Population: Mature Cohort N=293,936, Source: activity_by_month_check.csv)
