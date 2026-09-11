"""
Medical Insurance Fraud Detection — Easy/Medium Version 2 (Streamlit dashboard)
Classification: Logistic Regression | Anomaly detection: Isolation Forest (ML)
Run with: streamlit run app.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"

st.set_page_config(page_title="Fraud Detection (Easy/Medium)", page_icon="🩺", layout="wide")

FEATURES = [
    "ClaimAmount", "DeductibleAmt", "ClaimDurationDays", "PatientAge",
    "ChronicConditionCount", "NumDiagnosisCodes", "IsInpatient", "ProviderAvgClaimAmt",
]


@st.cache_resource
def load_artifacts():
    clf = joblib.load(MODEL_DIR / "fraud_classifier.joblib")
    anomaly = joblib.load(MODEL_DIR / "anomaly_detector.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    with open(MODEL_DIR / "metrics.json") as f:
        metrics = json.load(f)
    with open(MODEL_DIR / "coefficients.json") as f:
        coefficients = json.load(f)
    return clf, anomaly, scaler, metrics, coefficients


@st.cache_data
def load_stream_sample():
    return pd.read_csv(DATA_DIR / "stream_sample.csv").sample(frac=1.0, random_state=3).reset_index(drop=True)


clf, anomaly_model, scaler, metrics, coefficients = load_artifacts()
stream_df = load_stream_sample()

st.title("🩺 Medical Insurance Fraud Detection — Easy/Medium Version")
st.caption(
    "Classification (Logistic Regression) + genuine unsupervised anomaly detection "
    "(Isolation Forest), trained on real Medicare claims data."
)

tab1, tab2, tab3 = st.tabs(["📊 Overview", "📡 Real-Time Claim Stream", "🧾 Score a Claim"])

# ==========================================================================
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Claims evaluated", f"{metrics['n_train'] + metrics['n_test']:,}")
    c2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c3.metric("Average Precision", f"{metrics['average_precision']:.3f}")

    st.info(
        "This version uses a **plain random train/test split** and only 8 basic features "
        "for the classifier — a deliberate simplicity trade-off, disclosed here rather than "
        "hidden. The anomaly detector, however, is genuine unsupervised machine learning "
        "(Isolation Forest), not a manual statistics rule.",
        icon="ℹ️",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("ROC Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics["roc_curve"]["fpr"], y=metrics["roc_curve"]["tpr"],
                                  mode="lines", line=dict(color="#c0392b", width=3), name="Model"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Confusion Matrix (threshold = 0.5)")
        cm = np.array(metrics["confusion_matrix"])
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Reds",
                            x=["Pred: Legit", "Pred: Fraud"], y=["Actual: Legit", "Actual: Fraud"])
        fig_cm.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("What drives the classifier? (Logistic Regression coefficients)")
    coef_df = pd.DataFrame(coefficients, columns=["Feature", "Coefficient"]).sort_values("Coefficient")
    fig_coef = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h",
                       color="Coefficient", color_continuous_scale="RdBu_r")
    fig_coef.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig_coef, use_container_width=True)

    st.subheader("Anomaly score distribution (Isolation Forest)")
    fig_anom = px.histogram(
        stream_df, x="anomaly_score", color=stream_df["label"].map({0: "Legitimate", 1: "Fraud-linked"}),
        nbins=40, opacity=0.7, barmode="overlay",
        labels={"anomaly_score": "Anomaly score (higher = more unusual)"},
    )
    fig_anom.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_anom, use_container_width=True)
    st.caption(
        "The Isolation Forest never sees the fraud label during training — it only learns "
        "what a 'typical' claim looks like statistically, then flags claims that are easy to "
        "isolate with random splits as more anomalous. It catches unusual claims independently "
        "of anything the classifier learned."
    )

# ==========================================================================
with tab2:
    st.subheader("Live Claim Intake Simulation")
    st.caption("Claims arrive one at a time and are scored instantly by both models.")

    if "ptr" not in st.session_state:
        st.session_state.ptr = 0
        st.session_state.processed = 0
        st.session_state.flagged = 0
        st.session_state.alerts = []

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        batch = st.slider("Claims per run", 5, 40, 12)
    with c2:
        speed = st.select_slider("Speed", ["Slow", "Normal", "Fast"], value="Normal")
    with c3:
        st.write("")
        if st.button("↺ Reset", use_container_width=True):
            for k in ["ptr", "processed", "flagged", "alerts"]:
                st.session_state.pop(k, None)
            st.rerun()

    delay = {"Slow": 0.5, "Normal": 0.25, "Fast": 0.08}[speed]
    start = st.button("▶ Start / Continue", type="primary")

    metric_ph, feed_ph = st.empty(), st.empty()

    def render():
        with metric_ph.container():
            m1, m2, m3 = st.columns(3)
            m1.metric("Processed", st.session_state.processed)
            m2.metric("Flagged", st.session_state.flagged)
            rate = (st.session_state.flagged / st.session_state.processed * 100) if st.session_state.processed else 0
            m3.metric("Flag rate", f"{rate:.1f}%")
        with feed_ph.container():
            st.markdown("**Latest claims (most recent first)**")
            if st.session_state.alerts:
                st.dataframe(pd.DataFrame(st.session_state.alerts[::-1][:12]), use_container_width=True, hide_index=True)
            else:
                st.caption("No claims processed yet.")

    if start:
        n = len(stream_df)
        for _ in range(batch):
            row = stream_df.iloc[st.session_state.ptr % n]
            prob = float(row["fraud_probability"])
            anom = float(row["anomaly_score"])
            flagged = prob >= 0.5
            st.session_state.processed += 1
            if flagged:
                st.session_state.flagged += 1
            st.session_state.alerts.append({
                "Claim ID": row["ClaimID"],
                "Amount": f"${row['ClaimAmount']:,.0f}",
                "Fraud Probability": f"{prob:.2f}",
                "Anomaly Score": f"{anom:.2f}" + (" ⚠️" if anom > 0.55 else ""),
                "Flagged": "🔴 Yes" if flagged else "🟢 No",
            })
            st.session_state.ptr += 1
            render()
            time.sleep(delay)
    else:
        render()

    st.caption(
        "**Anomaly Score** comes from the Isolation Forest — an unsupervised model that never "
        "saw fraud labels during training. A higher score (⚠️ above ~0.55, roughly the top 10% "
        "most unusual in this data) means the claim was "
        "unusually easy to separate from typical claims on its raw attributes alone, independent "
        "of what the classifier predicts."
    )

# ==========================================================================
with tab3:
    st.subheader("Score a New Claim")
    with st.form("score_form"):
        col1, col2 = st.columns(2)
        with col1:
            amt = st.number_input("Claim amount ($)", min_value=0, value=1000, step=50)
            deductible = st.number_input("Deductible paid ($)", min_value=0, value=100, step=10)
            duration = st.number_input("Claim duration (days)", min_value=0, value=3, step=1)
            age = st.number_input("Patient age", min_value=0, max_value=110, value=65, step=1)
        with col2:
            chronic = st.slider("Chronic condition count", 0, 11, 2)
            n_diag = st.slider("# diagnosis codes", 0, 10, 3)
            is_inpatient = st.selectbox("Claim type", ["Outpatient", "Inpatient"]) == "Inpatient"
            provider_avg = st.number_input("Provider's average claim amount ($)", min_value=0, value=1000, step=50)
        submitted = st.form_submit_button("Score this claim", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "ClaimAmount": amt, "DeductibleAmt": deductible, "ClaimDurationDays": duration,
            "PatientAge": age, "ChronicConditionCount": chronic, "NumDiagnosisCodes": n_diag,
            "IsInpatient": int(is_inpatient), "ProviderAvgClaimAmt": provider_avg,
        }])
        X = scaler.transform(row[FEATURES])
        prob = float(clf.predict_proba(X)[:, 1][0])
        anom = float(-anomaly_model.score_samples(X)[0])

        col1, col2 = st.columns([1, 2])
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100, title={"text": "Fraud probability (%)"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "black"},
                       "steps": [{"range": [0, 40], "color": "#2ca02c"},
                                 {"range": [40, 70], "color": "#e6b800"},
                                 {"range": [70, 100], "color": "#d62728"}]},
            ))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown(f"**Isolation Forest anomaly score:** {anom:.2f}" + (" ⚠️ statistically unusual" if anom > 0.55 else " — within normal range"))
            st.markdown("**Top contributing factors (Logistic Regression weights):**")
            for feat, coef in coefficients[:4]:
                direction = "increases" if coef > 0 else "decreases"
                st.markdown(f"- `{feat}` {direction} fraud probability (weight: {coef:.2f})")
