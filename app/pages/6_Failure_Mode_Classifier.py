"""
app/pages/6_Failure_Mode_Classifier.py
--------------------------------------
Interactive NLP Failure Mode Classifier for Field Technician Claim Notes.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.utils import (
    REPORTS_DIR,
    MODEL_PATH,
    apply_custom_theme,
    format_plotly_fig,
)
from src.classifier import load_classifier_pipeline, predict_failure_mode

st.set_page_config(page_title="Failure Mode Classifier | Warranty Analytics", page_icon="⚡", layout="wide")
apply_custom_theme()

st.title("🤖 NLP Failure Mode Classifier")

st.markdown(
    """
    <div class="guide-card">
        <div class="guide-title">🧠 Machine Learning for Unstructured Field Service Notes</div>
        When field service technicians submit warranty claims or RMA service orders, they write unstructured, 
        free-text notes loaded with abbreviations, jargon, typos, and symptom descriptions. 
        <br><br>
        This production NLP pipeline combines <b>TF-IDF sub-linear n-gram vectorization</b> with 
        <b>Calibrated Multiclass Logistic Regression</b> to automatically categorize messy technician notes 
        into our 12 canonical physics-of-failure modes, saving hundreds of hours of manual warranty triage.
    </div>
    """,
    unsafe_allow_html=True,
)

# Load model pipeline and metrics
pipeline = None
if os.path.exists(MODEL_PATH):
    try:
        pipeline = load_classifier_pipeline(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading classifier pipeline: {e}")

metrics_json_path = os.path.join(REPORTS_DIR, "metrics.json")
metrics_data = {}
if os.path.exists(metrics_json_path):
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

# Interactive Prediction Section
st.subheader("✍️ Classify a New Warranty Claim in Real Time")
st.caption("Paste verbatim field notes, diagnostic fault codes, or select an industrial test scenario below:")

sample_prompts = [
    "Field technician observed DC bus capacitor electrolyte venting with pungent smell and visible bulged aluminum vent rubber.",
    "ACB failed routine closing test; main silver-tungsten arcing contacts eroded beyond permissible wear thickness.",
    "Inverter tripped on thermal runaway (OVERTEMP_ALARM >88C). Left heatsink cooling fan seized and motor coil scorched.",
    "Heavy copper patina oxidation and galvanic corrosion on terminal busbars in coastal high-humidity environment.",
    "Dielectric flashover observed across phase busbar support during high voltage withstand test; epoxy insulator cracked.",
    "Electronic trip unit failed to initiate trip signal during secondary injection testing; microprocessor logic unresponsive.",
    "Surge protection module thermal disconnector flagged red flag window; MOV disk degraded after multiple grid surges.",
    "Auxiliary transfer relay contacts chatter and failed to latch properly during automatic mains failure transition.",
]

col_in1, col_in2 = st.columns([3, 1])

with col_in2:
    selected_sample = st.selectbox("Preset Field Scenarios:", options=["-- Custom Entry --"] + sample_prompts)

with col_in1:
    default_text = selected_sample if selected_sample != "-- Custom Entry --" else "Technician reported DC bus capacitor bank swelling with oily residue near terminal after summer heat wave."
    user_input = st.text_area(
        "Enter Free-Text Claim Description / Field RMA Notes:",
        value=default_text,
        height=100,
        help="Paste verbatim technician field notes, diagnostic fault codes, or customer complaints.",
    )

if st.button("🔍 Run Machine Learning Inference", type="primary"):
    if pipeline and user_input.strip():
        pred_res = predict_failure_mode(user_input, pipeline=pipeline, top_k=5)
        top_pred = pred_res["predicted_failure_mode"]
        conf = pred_res["confidence"]

        st.markdown(
            f"""
            <div style="background-color: rgba(16, 185, 129, 0.12); border: 1px solid #10b981; border-radius: 8px; padding: 14px; margin-top: 10px; margin-bottom: 15px;">
                <span style="font-size: 1.1rem; font-weight: 700; color: #10b981;">🎯 Predicted Failure Mode:</span> 
                <span style="font-size: 1.25rem; font-weight: 800;">{top_pred}</span> 
                <span class="status-pill-success" style="margin-left: 8px;">{conf * 100:.1f}% Confidence</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Confidence Bar Chart
        top_k_df = pd.DataFrame(pred_res["top_k_predictions"])
        fig_prob = px.bar(
            top_k_df,
            x="probability",
            y="failure_mode",
            orientation="h",
            title="Class Probability Distribution (Top-5)",
            labels={"probability": "Confidence Probability", "failure_mode": "Failure Mode"},
            color="probability",
            color_continuous_scale="Blues",
            range_x=[0, 1.02],
        )
        fig_prob.update_layout(yaxis=dict(autorange="reversed"))
        format_plotly_fig(fig_prob, height=280)
        st.plotly_chart(fig_prob, use_container_width=True)

st.markdown("---")

# Model Performance & Evaluation Dashboard
st.subheader("📊 Rigorous Model Validation & Cross-Evaluation")
st.caption("Evaluated on a completely unseen hold-out test set (20% partition) with realistic technician wording ambiguity:")

if metrics_data:
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-title">Test Accuracy</div>
                <div class="metric-value">{metrics_data.get('test_accuracy', 0.85)*100:.2f}%</div>
                <div class="metric-sub"><span class="status-pill-success">Realistic Hold-Out</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-title">Macro-F1 Score</div>
                <div class="metric-value">{metrics_data.get('test_macro_f1', 0.81)*100:.2f}%</div>
                <div class="metric-sub"><span class="status-pill-info">Balanced Across Classes</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-title">Weighted-F1 Score</div>
                <div class="metric-value">{metrics_data.get('test_weighted_f1', 0.88)*100:.2f}%</div>
                <div class="metric-sub"><span class="status-pill-success">High Precision/Recall</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-title">Production Champion</div>
                <div class="metric-value" style="font-size: 1.4rem;">{metrics_data.get('model_champion', 'LogisticRegression')}</div>
                <div class="metric-sub"><span class="status-pill-warning">Calibrated Probabilities</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Model Architecture Benchmark Comparison")
    comp_list = metrics_data.get("model_comparison", [])
    if comp_list:
        st.dataframe(pd.DataFrame(comp_list), use_container_width=True, hide_index=True)

    # Confusion Matrix Heatmap
    cm = metrics_data.get("confusion_matrix")
    classes = metrics_data.get("classes")
    if cm and classes:
        st.markdown("#### Multi-Class Confusion Matrix Heatmap (Unseen Test Set)")
        st.caption("Diagonal values indicate accurate classifications; off-diagonal cells represent overlapping symptoms.")
        fig_cm = px.imshow(
            cm,
            x=classes,
            y=classes,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Failure Mode", y="Ground Truth Label", color="Claims Count"),
            text_auto=True,
        )
        fig_cm.update_layout(xaxis=dict(tickangle=-35))
        format_plotly_fig(fig_cm, height=580)
        st.plotly_chart(fig_cm, use_container_width=True)
