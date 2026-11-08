"""app.py
Interactive Streamlit Dashboard for:
"User Retention Decay Profiling and Churn Analysis in Subscription Streaming Platforms Using KKBox Listening Logs"

Usage:
    streamlit run app.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# -------------------------------------------------------------
# Configuration & Paths
# -------------------------------------------------------------
st.set_page_config(
    page_title="KKBox Churn & Retention Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(r"E:\ott video paltform")
REPORT_DIR = PROJECT_ROOT / "report_assets"
MODEL_JOB_FILE = PROJECT_ROOT / "model_c_unweighted.joblib"

# Data file paths
CHECKPOINTS_CSV = REPORT_DIR / "retention_checkpoints_reconciled.csv"
COHORT_TRAJ_CSV = REPORT_DIR / "cohort_calendar_trajectories.csv"
PROFILE_CSV = REPORT_DIR / "profile_summary_final.csv"
PROFILE_AR_CSV = REPORT_DIR / "profile_autorenew_share.csv"
MODEL_RESULTS_CSV = REPORT_DIR / "model_results_final.csv"
TENURE_EXP_CSV = REPORT_DIR / "tenure_experiment.csv"
EVAL_RESULTS_TXT = PROJECT_ROOT / "evaluation_results.txt"


# -------------------------------------------------------------
# Cached Data Loaders
# -------------------------------------------------------------
@st.cache_data
def load_checkpoints():
    if not CHECKPOINTS_CSV.exists():
        return None
    return pd.read_csv(CHECKPOINTS_CSV)

@st.cache_data
def load_cohort_trajectories():
    if not COHORT_TRAJ_CSV.exists():
        return None
    return pd.read_csv(COHORT_TRAJ_CSV)

@st.cache_data
def load_profiles():
    if not PROFILE_CSV.exists():
        return None
    return pd.read_csv(PROFILE_CSV)

@st.cache_data
def load_profile_autorenew():
    if not PROFILE_AR_CSV.exists():
        return None
    return pd.read_csv(PROFILE_AR_CSV)

@st.cache_data
def load_model_results():
    if not MODEL_RESULTS_CSV.exists():
        return None
    return pd.read_csv(MODEL_RESULTS_CSV)

@st.cache_data
def load_tenure_experiment():
    if not TENURE_EXP_CSV.exists():
        return None
    return pd.read_csv(TENURE_EXP_CSV)

@st.cache_data
def load_feature_importance():
    if not EVAL_RESULTS_TXT.exists():
        return None
    # Parse Top 10 feature importance table from evaluation_results.txt
    lines = EVAL_RESULTS_TXT.read_text(encoding="utf-8").splitlines()
    start_idx = -1
    for i, line in enumerate(lines):
        if "--- TOP 10 FEATURES BY TEST ROC-AUC DROP ---" in line:
            start_idx = i + 3
            break
    if start_idx == -1 or start_idx >= len(lines):
        return None
    
    rows = []
    for line in lines[start_idx:start_idx + 10]:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 5 and parts[0].isdigit():
            rows.append({
                "Rank": int(parts[0]),
                "Feature": parts[1],
                "Type": parts[2],
                "Mean AUC Drop": float(parts[3]),
                "Std Dev": float(parts[4])
            })
    return pd.DataFrame(rows) if rows else None

@st.cache_resource
def load_model_payload():
    if not MODEL_JOB_FILE.exists():
        return None
    try:
        return joblib.load(MODEL_JOB_FILE)
    except Exception as e:
        return None


# -------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------
st.sidebar.title("🎵 KKBox Analytics")
st.sidebar.markdown(
    "**Project:** User Retention Decay Profiling and Churn Analysis in Subscription Streaming Platforms Using KKBox Listening Logs"
)
st.sidebar.markdown("---")

PAGES = [
    "1. Overview",
    "2. Retention (Corrected)",
    "3. Decay Profiles",
    "4. Model Comparison",
    "5. Churn Risk Predictor",
    "6. Methodological Limitations"
]

selected_page = st.sidebar.radio("Navigate Pages", PAGES)

st.sidebar.markdown("---")
st.sidebar.caption("E:\\ott video paltform • College Project Dashboard")


# -------------------------------------------------------------
# PAGE 1: Overview
# -------------------------------------------------------------
if selected_page == "1. Overview":
    st.header("Project Overview & Executive Summary")
    st.markdown(
        "### User Retention Decay Profiling and Churn Analysis in Subscription Streaming Platforms Using KKBox Listening Logs"
    )
    st.markdown(
        "This project investigates whether granular user listening patterns provide early churn warning signals beyond subscription billing records in recurring digital streaming services."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mature Cohort Size (N)", "293,936", help="Subscribers registered 201501 to 201608 with full 6-month pre-churn observation window")
    with col2:
        st.metric("Empirical Churn Rate", "8.76%", help="25,745 churners out of 293,936 mature subscribers in March 2017")
    with col3:
        st.metric("Observation Scope", "26 Months", help="Historical listening logs spanning 201501 to 201702")
    with col4:
        st.metric("Top Feature Gain", "is_auto_renew", help="Accounts for over 11.0 pp of ROC-AUC explanatory power")

    st.markdown("---")
    st.subheader("Key Empirical Findings")
    st.markdown(
        """
        - **1. Auto-Renew Dominates Churn Prediction:**
          A single subscription flag (`is_auto_renew`) serves as the dominant churn determinant. Non-auto-renew users experience a **37.82%** churn rate, compared to just **3.92%** for auto-renew subscribers—a nearly 10-fold relative difference.
        - **2. Listening Behavior Alone Has Limited Discriminative Power:**
          Classifiers built solely on 7 listening activity features achieve only moderate performance (**ROC-AUC 0.6249 / PR-AUC 0.1551**), confirming that listening decay in isolation is insufficient to reliably forecast subscription termination.
        - **3. Incremental Lift from Combining Streams:**
          Adding listening signals on top of subscription metadata yields a small but statistically significant lift (ROC-AUC improves from **0.8780 to 0.8811**, PR-AUC improves from **0.5971 to 0.6139**, winning in 100% of bootstrap resamples).
        - **4. Subscriber Tenure Moderates Baseline Engagement:**
          Adding subscriber tenure (`months_since_registration`) raises listening-only model ROC-AUC by **+0.0432** (0.6249 → 0.6681) and combined model ROC-AUC by **+0.0082** (0.8811 → 0.8893).
        - **5. Conditioning & Selection Artifact in Churn Datasets:**
          Fresh longitudinal recomputations demonstrate that activity rates among training subscribers remain steady (~80%) rather than steeply decaying, because `train_v2` users are conditioned on surviving until the early 2017 renewal window.
        """
    )


# -------------------------------------------------------------
# PAGE 2: Retention (Corrected)
# -------------------------------------------------------------
elif selected_page == "2. Retention (Corrected)":
    st.header("Longitudinal Activity Presence (Corrected Recomputation)")
    
    st.info(
        "**Methodological Note:** Longitudinal curves in this study measure **activity presence within a renewal-conditioned subscriber sample**, "
        "not unbiased customer survival. Because all users in `train_v2` were conditioned on having an active membership scheduled to expire around March 2017, "
        "subscribers who churned early in 2015 or 2016 were naturally excluded. Calendar month 201702 is excluded from trend curves due to end-of-window transaction artifacts."
    )

    df_check = load_checkpoints()
    df_traj = load_cohort_trajectories()

    tab1, tab2 = st.tabs(["Checkpoints Across Maturing Cohorts", "Single Cohort Calendar Trajectories"])

    with tab1:
        st.subheader("Activity Presence Checkpoints (k = 0, 1, 3, 6, 12, 18 Months Post-Registration)")
        if df_check is None:
            st.warning("File `retention_checkpoints_reconciled.csv` not found in `report_assets/`.")
        else:
            c1, c2 = st.columns([3, 2])
            with c1:
                # Plotly grouped bar
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=df_check["Post-Reg Month (k)"],
                    y=df_check["Non-Churn Retention (%)"],
                    name="Non-Churners (Retained)",
                    marker_color="#2b5c8f",
                    hovertemplate="%{x}<br>Active: %{y:.2f}%<extra></extra>"
                ))
                fig_bar.add_trace(go.Bar(
                    x=df_check["Post-Reg Month (k)"],
                    y=df_check["Churn Retention (%)"],
                    name="Churners (March 2017)",
                    marker_color="#d95f02",
                    hovertemplate="%{x}<br>Active: %{y:.2f}%<extra></extra>"
                ))
                fig_bar.update_layout(
                    title="Monthly Activity Rate by Post-Registration Month (k)",
                    yaxis_title="Active Listening Share (%)",
                    yaxis=dict(range=[50, 100]),
                    barmode="group",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with c2:
                st.markdown("**Empirical Checkpoint Summary**")
                display_cols = [
                    "Post-Reg Month (k)", 
                    "Non-Churn Retention (%)", 
                    "Non-Churn Contributing Users (n)", 
                    "Churn Retention (%)", 
                    "Churn Contributing Users (n)"
                ]
                st.dataframe(df_check[display_cols], hide_index=True, use_container_width=True)

            st.markdown(
                """
                **Key Observation:**  
                Notice that non-churners maintain flat activity presence across all tenure checkpoints: **81.20%** at $k=0$, **82.59%** at $k=1$, **78.96%** at $k=3$, **79.12%** at $k=6$, and **79.20%** at $k=12$.  
                There is **no early steep retention drop** in months 1–3, directly refuting earlier assumptions of early decay.
                """
            )

    with tab2:
        st.subheader("Single Cohort Calendar Month Trajectories (Excluding 201702 Artifact)")
        if df_traj is None:
            st.warning("File `cohort_calendar_trajectories.csv` not found in `report_assets/`.")
        else:
            cohort_options = sorted(df_traj["Cohort"].unique())
            selected_cohort = st.selectbox("Select Registration Cohort:", cohort_options, index=0)

            # Filter cohort and strictly EXCLUDE 201702
            c_data = df_traj[(df_traj["Cohort"] == selected_cohort) & (df_traj["Calendar Month"] < 201702)].copy()
            c_data["Month_Str"] = c_data["Calendar Month"].astype(str)

            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=c_data["Month_Str"],
                y=c_data["Non-Churn Active Share (%)"],
                name="Non-Churn Subscribers",
                mode="lines+markers",
                line=dict(color="#2b5c8f", width=2.5),
                hovertemplate="Month: %{x}<br>Non-Churn Active: %{y:.2f}%<extra></extra>"
            ))
            fig_line.add_trace(go.Scatter(
                x=c_data["Month_Str"],
                y=c_data["Churn Active Share (%)"],
                name="Churn Subscribers",
                mode="lines+markers",
                line=dict(color="#d95f02", width=2.5, dash="dash"),
                hovertemplate="Month: %{x}<br>Churn Active: %{y:.2f}%<extra></extra>"
            ))

            fig_line.update_layout(
                title=f"Calendar Month Active Share for Cohort {selected_cohort} (Through 201701)",
                xaxis_title="Calendar Year-Month",
                yaxis_title="Subscribers with Active Days > 0 (%)",
                yaxis=dict(range=[50, 100]),
                hovermode="x unified"
            )
            st.plotly_chart(fig_line, use_container_width=True)

            st.caption(
                f"Cohort {selected_cohort} population: {c_data['Non-Churn Total (N)'].iloc[0]:,} Non-Churners, {c_data['Churn Total (N)'].iloc[0]:,} Churners. "
                "Notice the upward drift as calendar months approach the March 2017 renewal window—an expected consequence of conditioning the sample on survival up to that date."
            )


# -------------------------------------------------------------
# PAGE 3: Profiles
# -------------------------------------------------------------
elif selected_page == "3. Decay Profiles":
    st.header("Subscriber Decay Profiling & Auto-Renew Interaction")
    
    st.markdown(
        "Subscribers in the mature cohort were classified into 4 behavioral engagement profiles based on their active listening day trajectories over the 6-month observation window (201609 to 201702)."
    )

    df_prof = load_profiles()
    df_prof_ar = load_profile_autorenew()

    if df_prof is None or df_prof_ar is None:
        st.warning("Profile datasets (`profile_summary_final.csv` or `profile_autorenew_share.csv`) not found in `report_assets/`.")
    else:
        col1, col2 = st.columns([3, 2])
        with col1:
            # Bar chart of profile churn rates
            fig_prof = px.bar(
                df_prof,
                x="profile",
                y="churn_rate_pct",
                color="profile",
                error_y=df_prof["ci_upper_95"] - df_prof["churn_rate_pct"],
                error_y_minus=df_prof["churn_rate_pct"] - df_prof["ci_lower_95"],
                labels={"churn_rate_pct": "Churn Rate (%)", "profile": "Engagement Profile"},
                title="Empirical Churn Rate by Decay Profile (with 95% Wilson CIs)",
                color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
            )
            fig_prof.update_layout(showlegend=False, yaxis=dict(range=[0, 15]))
            st.plotly_chart(fig_prof, use_container_width=True)

        with col2:
            st.markdown("### Profile Definitions & Churn")
            st.dataframe(
                df_prof[["profile", "user_count", "user_pct", "churn_rate_pct", "mean_active_days", "mean_slope"]].rename(
                    columns={
                        "profile": "Profile",
                        "user_count": "Users (n)",
                        "user_pct": "Share (%)",
                        "churn_rate_pct": "Churn (%)",
                        "mean_active_days": "Mean Days",
                        "mean_slope": "Mean Slope"
                    }
                ),
                hide_index=True,
                use_container_width=True
            )
            st.caption("Cramér's V = 0.0484 (weak correlation between profile and churn).")

        st.markdown("---")
        st.subheader("The Critical Interaction: Auto-Renew Status vs. Profile Churn")
        st.markdown(
            "Analyzing churn within profiles separated by `is_auto_renew` demonstrates that payment mechanics heavily supersede engagement trajectory."
        )

        selected_prof_filter = st.selectbox("Filter Profile Breakdown:", ["All Profiles"] + list(df_prof_ar["Profile"].unique()))

        if selected_prof_filter == "All Profiles":
            display_ar_df = df_prof_ar
        else:
            display_ar_df = df_prof_ar[df_prof_ar["Profile"] == selected_prof_filter]

        st.dataframe(display_ar_df, hide_index=True, use_container_width=True)

        # Plotly comparison
        fig_ar_comp = go.Figure()
        fig_ar_comp.add_trace(go.Bar(
            x=df_prof_ar["Profile"],
            y=df_prof_ar["Churn Rate (Auto-Renew=0) (%)"],
            name="Manual Pay (Auto-Renew = 0)",
            marker_color="#d95f02",
            hovertemplate="Profile: %{x}<br>Churn: %{y:.2f}%<extra></extra>"
        ))
        fig_ar_comp.add_trace(go.Bar(
            x=df_prof_ar["Profile"],
            y=df_prof_ar["Churn Rate (Auto-Renew=1) (%)"],
            name="Auto-Renew Active (Auto-Renew = 1)",
            marker_color="#2b5c8f",
            hovertemplate="Profile: %{x}<br>Churn: %{y:.2f}%<extra></extra>"
        ))
        fig_ar_comp.update_layout(
            title="Churn Rate Across Profiles: Manual Pay vs. Auto-Renew",
            yaxis_title="Churn Rate (%)",
            barmode="group",
            yaxis=dict(range=[0, 100]),
            hovermode="x unified"
        )
        st.plotly_chart(fig_ar_comp, use_container_width=True)

        st.markdown(
            "> **Insight:** For every profile, manual payers churn at **36% to 83%**, while auto-renew subscribers churn at only **3.4% to 4.5%**. "
            "Listening activity has almost zero predictive power when auto-renew is active."
        )


# -------------------------------------------------------------
# PAGE 4: Model Comparison
# -------------------------------------------------------------
elif selected_page == "4. Model Comparison":
    st.header("Predictive Churn Modeling: Listening vs. Subscription")

    df_models = load_model_results()
    df_tenure = load_tenure_experiment()
    df_imp = load_feature_importance()

    tab1, tab2, tab3 = st.tabs(["Models A, B, and C Benchmarks", "Tenure Experiment (5-Fold CV)", "Permutation Feature Importance"])

    with tab1:
        st.subheader("Model Performance Comparison on Test Set (N = 58,788)")
        st.markdown(
            """
            - **Model A**: 7 Listening Features Only (Activity days, slope, zero months, total secs, songs, unique songs, diff)
            - **Model B**: 7 Subscription Features Only (Auto-renew, plan days, cancel flag, amount paid, transaction count, cancel count, transaction presence)
            - **Model C**: Combined 14 Features (All listening + subscription signals)
            """
        )
        if df_models is None:
            st.warning("File `model_results_final.csv` not found in `report_assets/`.")
        else:
            st.dataframe(df_models, hide_index=True, use_container_width=True)

    with tab2:
        st.subheader("Tenure Experiment: Unweighted 5-Fold Stratified Cross-Validation")
        st.markdown(
            "Evaluates the impact of incorporating customer tenure (`months_since_registration` as of Feb 2017) into Models A and C."
        )
        if df_tenure is None:
            st.warning("File `tenure_experiment.csv` not found in `report_assets/`.")
        else:
            st.dataframe(df_tenure, hide_index=True, use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.info(
                    "**Model A Impact:** Adding tenure provides a substantial lift in listening-only modeling:  \n"
                    "• ROC-AUC: **0.6249 → 0.6681** (+0.0432 ± 0.0047)  \n"
                    "• PR-AUC: **0.1551 → 0.1694** (+0.0143 ± 0.0015)"
                )
            with c2:
                st.info(
                    "**Model C Impact:** Adding tenure to combined features provides a small, stable incremental gain:  \n"
                    "• ROC-AUC: **0.8811 → 0.8893** (+0.0082 ± 0.0010)  \n"
                    "• PR-AUC: **0.6096 → 0.6141** (+0.0045 ± 0.0008)"
                )

    with tab3:
        st.subheader("Top 10 Feature Importance (Test Set ROC-AUC Drop over 5 Repeats)")
        if df_imp is None:
            st.warning("Permutation importance table could not be parsed from `evaluation_results.txt`.")
        else:
            fig_imp = px.bar(
                df_imp.sort_values("Mean AUC Drop", ascending=True),
                x="Mean AUC Drop",
                y="Feature",
                color="Type",
                orientation="h",
                error_x="Std Dev",
                title="Top 10 Features by Test ROC-AUC Drop (Permutation Importance)",
                color_discrete_map={"Subscription": "#2b5c8f", "Listening": "#d95f02"}
            )
            fig_imp.update_layout(height=450)
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption("Notice that `is_auto_renew` accounts for ~0.1107 drop, overshadowing all listening metrics by an order of magnitude.")


# -------------------------------------------------------------
# PAGE 5: Churn Risk Predictor
# -------------------------------------------------------------
elif selected_page == "5. Churn Risk Predictor":
    st.header("Interactive Subscriber Churn Risk Predictor")
    st.markdown(
        "Estimate the probability that a subscriber will churn in the target renewal window based on subscription and listening indicators."
    )

    st.warning(
        "**Sample Representation Notice:** Predicted probabilities reflect the training population "
        "(mature subscribers surviving to the renewal window, with a **8.76% base churn rate**). "
        "These are calibrated risk estimates for this target population, not an unconditional population forecast."
    )

    payload = load_model_payload()

    if payload is None:
        st.error(
            "⚠️ **Trained model file `model_c_unweighted.joblib` not found.**  \n\n"
            "To enable live predictions, run the training script from your terminal:  \n"
            "`python train_and_save_model.py`  \n"
            "This will train and serialize the unweighted Model C with tenure, then refresh this page."
        )
    else:
        model = payload["model"]
        feature_names = payload["feature_names"]
        medians = payload["feature_medians"]

        st.markdown(
            "Provide the primary subscriber parameters below. Secondary features (such as song counts and cancellations) "
            "are set to their empirical training-set medians."
        )

        with st.form("churn_prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Billing & Subscription Features")
                in_auto_renew = st.selectbox(
                    "Auto-Renew Active (is_auto_renew):",
                    options=[1, 0],
                    format_func=lambda x: "Yes (1) - Automated Payment" if x == 1 else "No (0) - Manual Payment",
                    index=0,
                    help="Strongest predictor in the model."
                )
                in_plan_days = st.number_input("Payment Plan Days:", min_value=0, max_value=450, value=30, step=30)
                in_tx_count = st.number_input("Prior Transaction Count:", min_value=1, max_value=100, value=int(medians.get("tx_count", 15)), step=1)
                in_tenure = st.number_input("Subscriber Tenure (Months Since Registration):", min_value=6, max_value=30, value=18, step=1)

            with col2:
                st.subheader("Listening Engagement Features")
                in_active_days = st.slider("Mean Monthly Active Days (0 to 31):", min_value=0.0, max_value=31.0, value=15.0, step=0.5)
                in_zero_months = st.slider("Months with Zero Listening Activity (0 to 6):", min_value=0, max_value=6, value=0, step=1)

            submit_btn = st.form_submit_button("Calculate Churn Risk")

        if submit_btn:
            # Construct feature vector using user inputs + training medians
            input_dict = {
                "mean_active_days": float(in_active_days),
                "slope_active_days": float(medians.get("slope_active_days", 0.0)),
                "zero_months": float(in_zero_months),
                "mean_songs_total": float(medians.get("mean_songs_total", 300.0)),
                "mean_num_unq": float(medians.get("mean_num_unq", 250.0)),
                "mean_total_secs": float(medians.get("mean_total_secs", 70000.0)),
                "active_days_diff": float(medians.get("active_days_diff", 0.0)),
                "is_auto_renew": float(in_auto_renew),
                "payment_plan_days": float(in_plan_days),
                "is_cancel": float(medians.get("is_cancel", 0.0)),
                "actual_amount_paid": float(medians.get("actual_amount_paid", 149.0)),
                "tx_count": float(in_tx_count),
                "cancel_count": float(medians.get("cancel_count", 0.0)),
                "has_transaction": 1.0,
                "months_since_registration": float(in_tenure)
            }

            input_df = pd.DataFrame([[input_dict[f] for f in feature_names]], columns=feature_names)
            prob_churn = float(model.predict_proba(input_df)[0, 1])

            st.markdown("---")
            st.subheader("Prediction Result")
            res_col1, res_col2 = st.columns([2, 3])

            with res_col1:
                st.metric(
                    label="Estimated Churn Probability",
                    value=f"{prob_churn * 100:.2f}%",
                    delta=f"{(prob_churn - 0.0876) * 100:+.2f} pp vs. Baseline (8.76%)",
                    delta_color="inverse"
                )
                if prob_churn > 0.35:
                    st.error("Risk Tier: HIGH RISK OF CHURN")
                elif prob_churn > 0.15:
                    st.warning("Risk Tier: MODERATE CHURN RISK")
                else:
                    st.success("Risk Tier: LOW CHURN RISK")

            with res_col2:
                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_churn * 100,
                    number={"suffix": "%"},
                    title={"text": "Churn Risk Gauge"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#EF553B" if prob_churn > 0.35 else ("#FFA15A" if prob_churn > 0.15 else "#00CC96")},
                        "steps": [
                            {"range": [0, 15], "color": "#e8f5e9"},
                            {"range": [15, 35], "color": "#fff3e0"},
                            {"range": [35, 100], "color": "#ffebee"}
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.75,
                            "value": 8.76
                        }
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with st.expander("Show Detailed Input Vector Passed to Estimator"):
                st.json(input_dict)


# -------------------------------------------------------------
# PAGE 6: Methodological Limitations
# -------------------------------------------------------------
elif selected_page == "6. Methodological Limitations":
    st.header("Methodological Limitations & Observational Constraints")

    st.markdown(
        """
        Transparent reporting of constraints is vital for evaluating the external validity and business applicability of this work:

        1. **Survivor & Selection Bias in Target Population:**
           - The KKBox `train_v2` dataset contains only users whose subscriptions were scheduled to expire around March 2017. Users who subscribed in early 2015 and churned later in 2015 or 2016 are excluded. Longitudinal curves therefore capture **activity presence among surviving subscribers**, rather than standard onboarding survival curves.
        
        2. **Registration Count Discontinuity (July 2015):**
           - A sharp spike in recorded registrations occurs between June 2015 (~4,700 users) and July 2015 (~15,900 users). This indicates an underlying shift in partner distribution channels, data logging protocols, or promotional onboarding, which affects cross-cohort comparability.

        3. **Listening Features are Weak Early-Warning Proxies:**
           - Unlike content completion or browsing behavior in video platforms, music streaming activity frequently occurs as background audio. Active listening history alone yields an ROC-AUC of only ~0.62–0.66, demonstrating that behavioral listening decay is secondary to transactional and billing mechanisms.

        4. **Single-Month Churn Window (March 2017):**
           - Churn is evaluated strictly over a single calendar month. Platform-wide pricing changes, billing cycle dates, or seasonality specific to Q1 2017 may exert disproportionate effects that do not generalize across all quarters.

        5. **Single Streaming Platform Dataset:**
           - All findings derive from KKBox. Whether these conclusions generalize to video OTT streaming (e.g., Netflix, Prime Video), where recurring billing is similarly automated but content consumption requires visual attention, remains an empirical question.
        """
    )
