# User Retention Decay Profiling and Churn Analysis in Subscription Streaming Platforms Using KKBox Listening Logs

**Location:** `E:\ott video paltform\report_assets`  
**Execution Timestamp:** 2026-10-19 18:17:04

---

## 1. Index of Tables

| Asset File | Format | One-Line Factual Summary |
| :--- | :--- | :--- |
| **`dataset_summary`** | [CSV](dataset_summary.csv) / [MD](dataset_summary.md) | Summarizes 410,502,905 combined listening logs across 27 months, 970,960 training users, and 293,936 mature cohort subscribers (201501-201608). |
| **`train_expiry_check`** | [CSV](train_expiry_check.csv) / [MD](train_expiry_check.md) | Verifies membership expiry date distribution across train_v2 subscribers using original transactions <= 20170228 (Most membership expiry dates fall in 201703 (951,186 subscribers, 98.22% of all train_v2 users with transactions before cutoff).). |
| **`retention_summary`** | [CSV](retention_summary.csv) / [MD](retention_summary.md) | Reports cohort retention decay averaging 83.63% at $k=1$ (26 cohorts), 79.49% at $k=3$ (24 cohorts), 78.51% at $k=6$ (21 cohorts), and 79.09% at $k=12$ (15 cohorts). |
| **`retention_v2_check`** | [CSV](retention_v2_check.csv) / [MD](retention_v2_check.md) | Compares retention at $k=0, 1, 3, 6, 12$ between v2 (activity <= 201702) and old (<= 201703) versions with contributing non-churn and churn user counts. |
| **`profile_summary_final`** | [CSV](profile_summary_final.csv) / [MD](profile_summary_final.md) | Details decay profile distribution across mature subscribers with churn rates: Inactive (5.72%), Fast decay (9.31%), Stable (7.67%), Growing (10.33%), and Cramér's V = 0.0484. |
| **`subscription_summary`** | [CSV](subscription_summary.csv) / [MD](subscription_summary.md) | Confirms non-auto-renew churn of 37.82% (95% CI: [37.36%, 38.29%]) vs. auto-renew churn of 3.92% (95% CI: [3.84%, 4.00%]), with full cell distributions. |
| **`profile_autorenew_share`** | [CSV](profile_autorenew_share.csv) / [MD](profile_autorenew_share.md) | Reports auto-renew=0 subscriber share across profiles and corresponding churn rates within each auto-renew subgroup. |
| **`model_results_final`** | [CSV](model_results_final.csv) / [MD](model_results_final.md) | Compares Models A, B, and C at both temporal cutoffs: Feb 28 ROC-AUC is 0.6316 [0.6235, 0.6393] (A), 0.8816 [0.8766, 0.8867] (B), 0.8843 [0.8792, 0.8893] (C) with PR lift of +0.0160 [+0.0125, +0.0197] (100% win); Jan 31 ROC-AUC is 0.8461 [0.8406, 0.8522] (B) vs. 0.8531 [0.8472, 0.8589] (C) with PR lift of +0.0194 [+0.0155, +0.0237] (100% win). |
| **`model_unweighted`** | [CSV](model_unweighted.csv) / [MD](model_unweighted.md) | Contrasts weighted vs. unweighted Model C, showing unweighted Brier score drops from 0.11856 to 0.05028 compared to the 0.07993 baseline. |
| **`cv_results`** | [CSV](cv_results.csv) / [MD](cv_results.md) | Evaluates unweighted 5-fold cross-validation on mature subscribers (N=293,936): ROC-AUC is 0.6249 +/- 0.0043 (A), 0.8780 +/- 0.0030 (B), and 0.8811 +/- 0.0029 (C). |
| **`activity_by_month_check`** | [CSV](activity_by_month_check.csv) / [MD](activity_by_month_check.md) | Analyzes monthly active listening day share and mean active days for mature cohort subscribers across 201609 to 201702 by churn outcome and auto-renew status. |
| **`no_listening_analysis`** | [CSV](no_listening_analysis.csv) / [MD](no_listening_analysis.md) | Identifies 49,333 non-listening subscribers (churn: 10.82%) vs. 811,634 active subscribers (churn: 9.38%), revealing 81.25% churn when auto-renew is 0 vs. 4.17% when auto-renew is 1. |

---

## 2. Index of High-Resolution Figures (300 DPI)

| Figure File | Preview / Filename | One-Line Factual Caption |
| :--- | :--- | :--- |
| **Figure 1** | `figure_retention_cohorts.png` | Active retention trajectories across cohorts 201501, 201507, 201601, and 201607 plotted strictly through valid observed calendar months without zero imputation. |
| **Figure 2** | `figure_retention_churn_vs_nonchurn_v2.png` | Longitudinal retention decay curves comparing non-churned and churned subscribers using activity months strictly through 201702, truncated where contributing cohort user sample falls below 1,000 users. |
| **Figure 3** | `figure_profile_churn.png` | Bar chart of subscriber churn rates across the 4 decay profiles with 95% Wilson confidence interval error bars (Fast decay: 9.31%, Growing: 10.33%). |
| **Figure 4** | `figure_roc.png` | Test-set ROC curve comparison under cutoff 20170228: Model A (AUC=0.6316), Model B (AUC=0.8816), and Model C (AUC=0.8843). |
| **Figure 5** | `figure_calibration_unweighted.png` | Calibration reliability diagram across 10 probability deciles for unweighted Model C showing close agreement with small deviations and Brier score = 0.05028. |
| **Figure 6** | `figure_permutation_importance.png` | Horizontal bar chart of the Top 10 features ranked by test-set ROC-AUC drop over 5 shuffling repeats, highlighting `is_auto_renew` (-0.1107) and `mean_active_days` (-0.0045). |
| **Figure 7** | `figure_cutoff_comparison.png` | Grouped bar chart showing Model B and C ROC-AUC fall from 0.8816 to 0.8461 and 0.8843 to 0.8531 when transactions are cut at 20170131, and that Model A is unchanged at 0.6316. |

---

## 3. Methodological Limitations

1. **Survivor Bias in `train_v2` Target Population**:
   - The KKBox `train_v2.csv` dataset contains subscribers whose memberships were active and scheduled to expire in March 2017. Users who churned during 2015 or 2016 are systematically excluded from this target set, resulting in survivor bias.
2. **Music Streaming Proxy Limitation**:
   - KKBox is a music streaming service, so listening features are only a proxy for video streaming engagement. Whether the findings transfer to video platforms was not tested.
3. **Cross-Validation Scope**:
   - 5-fold cross-validation on the mature cohort gave results consistent with the single test split. Temporal validation across different cutoff periods was not performed.
4. **Single-Month Churn Window (March 2017)**:
   - The binary target indicates churn strictly during March 2017 (no renewal within 30 days of expiration). Seasonal dynamics or platform pricing adjustments specific to Q1 2017 could influence the empirical churn rates.
5. **Untuned Fixed Heuristic Decay Thresholds ($\pm 0.5$ Days/Month)**:
   - Profile classification thresholds ($-0.5$ days/month for Fast Decay, $+0.5$ for Growing) were established as intuitive fixed heuristic boundaries rather than optimized hyperparameter cutoffs derived from downstream business utility.
6. **Approximate Nature of Permutation Importance**:
   - Permutation importance assesses marginal feature impact by shuffling one feature at a time on the test set. When features are correlated (such as `mean_active_days`, `mean_songs_total`, and `zero_months`, or `is_auto_renew` and `tx_count`), permutation can break realistic multivariate feature structures, slightly understating the joint predictive importance of correlated feature subsets.
7. **Observational Nature of Analysis**:
   - The analysis is observational, so results show association, not causation.
8. **Subscribers Lacking Registration Metadata**:
   - Approximately 110,000 training users lack a registration date and are excluded from cohort analyses.
9. **Single-Window Definition of No-Listening Subscribers**:
   - The no-listening group is defined by one 6-month window, and its high auto-renew share means low churn there does not show those users will keep subscribing.
