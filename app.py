from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
# Prefer model/ but fall back to root-level artifacts (repository contains copies at root)
MODEL_PATH = ROOT / "model" / "loan_default_pipeline.joblib"
ALT_MODEL_PATH = ROOT / "loan_default_pipeline.joblib"
META_PATH = ROOT / "model" / "model_metadata.json"
ALT_META_PATH = ROOT / "model_metadata.json"
METRICS_PATH = ROOT / "model" / "metrics.json"
ALT_METRICS_PATH = ROOT / "metrics.json"
FI_PATH = ROOT / "model" / "feature_importance.csv"
ALT_FI_PATH = ROOT / "feature_importance.csv"
DATA_PATH = ROOT / "data" / "loan_default_nigeria_synthetic.csv"

# Resolve which paths exist (support both layout styles)
if MODEL_PATH.exists():
    _MODEL_PATH = MODEL_PATH
    _META_PATH = META_PATH
    _METRICS_PATH = METRICS_PATH
    _FI_PATH = FI_PATH
elif ALT_MODEL_PATH.exists():
    _MODEL_PATH = ALT_MODEL_PATH
    _META_PATH = ALT_META_PATH
    _METRICS_PATH = ALT_METRICS_PATH
    _FI_PATH = ALT_FI_PATH
else:
    # Keep the older, clearer instruction for users who run the scripts from repo root
    st.error("Model artifact not found. Run: python generate_data.py && python train.py")
    st.stop()

st.set_page_config(
    page_title="AI-04 Loan Default Risk Predictor",
    page_icon="💳",
    layout="wide",
)

st.title("💳 AI-04 — Loan Default Risk Predictor")
st.caption("Educational Nigerian micro-lending risk-scoring MVP")

model = joblib.load(_MODEL_PATH)
metadata = json.loads(_META_PATH.read_text())
metrics = json.loads(_METRICS_PATH.read_text())
feature_importance = pd.read_csv(_FI_PATH)

st.warning(
    "This is an educational decision-support prototype using synthetic data. "
    "It must not be used as an automated real-world lending decision without "
    "validated representative data, fairness/compliance review and human oversight."
)

with st.sidebar:
    st.header("Model")
    st.write(f"Selected: **{metadata['selected_model']}**")
    st.metric("Test ROC-AUC", f"{metrics['test_roc_auc']:.3f}")
    st.metric("Test F1", f"{metrics['test_f1']:.3f}")
    st.write(f"Decision threshold: **{metadata['decision_threshold']:.3f}**")

tab1, tab2, tab3 = st.tabs(["Single Applicant", "CSV Batch", "Model Evaluation"])

with tab1:
    st.subheader("Applicant information")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", 18, 80, 32)
        gender = st.selectbox("Gender", ["Female", "Male"])
        employment_type = st.selectbox(
            "Employment type",
            ["Salaried", "Self-employed", "Trader", "Casual", "Unemployed"],
        )
        monthly_income = st.number_input("Monthly income (₦)", 0.0, 20_000_000.0, 180_000.0, step=10_000.0)
        loan_amount = st.number_input("Loan amount (₦)", 0.0, 20_000_000.0, 250_000.0, step=10_000.0)

    with c2:
        loan_term = st.selectbox("Loan term (months)", [1, 2, 3, 6, 9, 12], index=3)
        previous_loans = st.number_input("Previous loans", 0, 50, 2)
        previous_defaults = st.number_input("Previous defaults", 0, 50, 0)
        repayment_history = st.slider("Repayment history score", 0.0, 100.0, 85.0)
        savings_balance = st.number_input("Savings balance (₦)", 0.0, 20_000_000.0, 150_000.0, step=10_000.0)

    with c3:
        business_age = st.number_input("Business age (months)", 0, 600, 36)
        daily_sales = st.number_input("Daily sales (₦)", 0.0, 5_000_000.0, 25_000.0, step=1_000.0)
        dti = st.number_input("Debt-to-income ratio", 0.0, 5.0, 0.35, step=0.01)
        collateral = st.selectbox("Collateral available", ["No", "Yes"])
        region = st.selectbox(
            "Region",
            ["South South", "South West", "South East", "North Central", "North West", "North East"],
        )

    if st.button("Calculate risk", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "employment_type": employment_type,
            "monthly_income_ngn": monthly_income,
            "loan_amount_ngn": loan_amount,
            "loan_term_months": loan_term,
            "previous_loans": previous_loans,
            "previous_defaults": previous_defaults,
            "repayment_history_score": repayment_history,
            "savings_balance_ngn": savings_balance,
            "business_age_months": business_age,
            "daily_sales_ngn": daily_sales,
            "debt_to_income_ratio": dti,
            "collateral_available": collateral,
            "region": region,
        }])

        probability = float(model.predict_proba(row)[:, 1][0])
        score = probability * 100
        band = "Low" if score < 30 else ("Medium" if score <= 60 else "High")

        m1, m2, m3 = st.columns(3)
        m1.metric("Default probability", f"{probability:.1%}")
        m2.metric("Risk score", f"{score:.1f}/100")
        m3.metric("Risk band", band)

        st.progress(min(max(probability, 0.0), 1.0))

        st.subheader("Key model factors")
        top = feature_importance.head(8).copy()
        top["feature"] = (
            top["feature"]
            .str.replace("num__", "", regex=False)
            .str.replace("cat__", "", regex=False)
        )
        st.dataframe(top, use_container_width=True, hide_index=True)

        st.info(
            "Interpret the score as a model estimate, not a guaranteed outcome. "
            "A high score means the model estimates higher default likelihood relative "
            "to the synthetic examples used during training."
        )

with tab2:
    st.subheader("Batch CSV scoring")
    st.write("Upload a CSV containing the applicant feature columns. The target column `defaulted` is optional.")
    template = pd.DataFrame(columns=metadata["all_features"])
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        "ai04_loan_template.csv",
        "text/csv",
    )

    upload = st.file_uploader("Upload applicant CSV", type=["csv"])
    if upload:
        batch = pd.read_csv(upload)
        missing = [c for c in metadata["all_features"] if c not in batch.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            probs = model.predict_proba(batch[metadata["all_features"]])[:, 1]
            out = batch.copy()
            out["default_probability"] = probs
            out["risk_score"] = probs * 100
            out["risk_band"] = pd.cut(
                out["risk_score"],
                bins=[-np.inf, 30, 60, np.inf],
                labels=["Low", "Medium", "High"],
                right=True,
            )
            st.dataframe(out, use_container_width=True)
            st.download_button(
                "Download scored CSV",
                out.to_csv(index=False).encode("utf-8"),
                "ai04_scored_applicants.csv",
                "text/csv",
            )

with tab3:
    st.subheader("Evaluation")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("ROC-AUC", f"{metrics['test_roc_auc']:.3f}")
    e2.metric("Accuracy", f"{metrics['test_accuracy']:.3f}")
    e3.metric("Precision", f"{metrics['test_precision']:.3f}")
    e4.metric("Recall", f"{metrics['test_recall']:.3f}")

    cm_path = (ROOT / "model" / "confusion_matrix.png") if (ROOT / "model" / "confusion_matrix.png").exists() else (ROOT / "confusion_matrix.png")
    roc_path = (ROOT / "model" / "roc_curve.png") if (ROOT / "model" / "roc_curve.png").exists() else (ROOT / "roc_curve.png")
    a, b = st.columns(2)
    with a:
        st.image(str(cm_path), caption="Held-out test confusion matrix")
    with b:
        st.image(str(roc_path), caption="Held-out test ROC curve")

    st.subheader("Top global feature importance")
    st.bar_chart(feature_importance.head(12).set_index("feature")["importance"])

    st.caption(
        f"Training dataset: {metrics['data_rows']:,} synthetic records; "
        f"default rate: {metrics['default_rate']:.1%}."
    )
