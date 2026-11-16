"""
app/pages/7_PPIT_CAR_Tracker.py
-------------------------------
Problem Priority Identification Table (PPIT) & 8D CAR Report Generator.
"""

import os
import sys
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.utils import (
    load_all_data,
    render_sidebar_filters,
    apply_custom_theme,
    format_plotly_fig,
    CAR_DIR,
)
from src.metrics import rank_ppit_car
from src.report import generate_car_pdf

st.set_page_config(page_title="PPIT / CAR Tracker | Warranty Analytics", page_icon="⚡", layout="wide")
apply_custom_theme()

products_df, sales_df, claims_df = load_all_data()
f_claims, f_sales, f_prods = render_sidebar_filters(claims_df, sales_df, products_df)

st.title("🛡️ Problem Priority Identification Table (PPIT) & 8D CAR Tracker")

st.markdown(
    """
    <div class="guide-card">
        <div class="guide-title">📘 What is PPIT and Why Does Reliability Engineering Use It?</div>
        Rather than simply chasing the highest raw claim count, the <b>Problem Priority Identification Table (PPIT)</b> 
        quantitatively ranks field quality risks using a weighted multi-attribute model:
        <ul>
            <li><b>Warranty Financial Expense (40% weight):</b> Direct dollar drain on warranty reserves.</li>
            <li><b>PPM Acceleration & Trend (35% weight):</b> Statistical rate velocity (how rapidly failures are accelerating in recent months).</li>
            <li><b>Fleet Volume Exposure (25% weight):</b> Total active units operating in customer facilities.</li>
        </ul>
        Issues scoring <b>≥ 60</b> or exhibiting <b>> 2.5σ rate acceleration</b> trigger mandatory 
        <b>8D Corrective Action Requests (CAR)</b> for multidisciplinary root cause elimination.
    </div>
    """,
    unsafe_allow_html=True,
)

# Compute PPIT Table
ppit_df = rank_ppit_car(f_claims, f_sales, f_prods)

# Summary Counters
n_qualified = len(ppit_df[ppit_df["car_status"].str.contains("QUALIFIED")])
n_monitoring = len(ppit_df[ppit_df["car_status"].str.contains("MONITORING")])

k1, k2, k3 = st.columns(3)
with k1:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Triggered for 8D / CAR</div>
            <div class="metric-value" style="color: #ef4444;">{n_qualified} Critical Issues</div>
            <div class="metric-sub"><span class="status-pill-danger">Immediate Action Required</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Statistical Watchlist</div>
            <div class="metric-value" style="color: #f59e0b;">{n_monitoring} Issues</div>
            <div class="metric-sub"><span class="status-pill-warning">Under SPC Surveillance</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Evaluated Defect Modes</div>
            <div class="metric-value">{len(ppit_df)} Combinations</div>
            <div class="metric-sub"><span class="status-pill-success">100% Risk Assessed</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Ranked Priority Table
st.subheader("📋 Ranked Problem Priority Identification Table")
st.caption("Issues are ranked from highest to lowest risk. Green status indicates normal variation; Red indicates formal CAR qualification.")

display_cols = [
    "ppit_rank", "car_id", "product_family", "failure_mode", "ppit_score",
    "car_status", "total_claims", "total_warranty_cost", "avg_claim_cost",
    "sigma_deviation", "baseline_ppm", "recent_ppm"
]

# Format dataframe nicely
styled_ppit = ppit_df[display_cols].copy()
styled_ppit["total_warranty_cost"] = styled_ppit["total_warranty_cost"].map(lambda x: f"${x:,.2f}")
styled_ppit["avg_claim_cost"] = styled_ppit["avg_claim_cost"].map(lambda x: f"${x:,.2f}")
styled_ppit["baseline_ppm"] = styled_ppit["baseline_ppm"].map(lambda x: f"{x:,.0f}")
styled_ppit["recent_ppm"] = styled_ppit["recent_ppm"].map(lambda x: f"{x:,.0f}")
styled_ppit["sigma_deviation"] = styled_ppit["sigma_deviation"].map(lambda x: f"{x:+.1f}σ")

st.dataframe(
    styled_ppit,
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# Interactive 8D PDF Generator
st.subheader("📄 Interactive 8D Corrective Action Request (CAR) PDF Generator")
st.markdown("Select any ranked defect issue below to review the root-cause summary and generate an official, publication-quality 8D Engineering Report.")

issue_choices = [
    f"Rank #{r['ppit_rank']}: {r['car_id']} - {r['product_family']} | {r['failure_mode']} (PPIT Score: {r['ppit_score']})"
    for _, r in ppit_df.iterrows()
]

selected_item = st.selectbox("Select Defect Investigation:", options=issue_choices)

# Extract selected row
sel_rank = int(selected_item.split(":")[0].replace("Rank #", ""))
sel_row = ppit_df[ppit_df["ppit_rank"] == sel_rank].iloc[0]
car_dict = sel_row.to_dict()

c_preview, c_action = st.columns([2, 1])

with c_preview:
    st.markdown(f"#### 🔍 8D Executive Summary: `{car_dict['car_id']}`")
    st.markdown(
        f"""
        - **Product Family:** **{car_dict['product_family']}**
        - **Failure Mode:** **{car_dict['failure_mode']}**
        - **Total Incidents:** **{car_dict['total_claims']:,} claims**
        - **Cumulative Warranty Expense:** **${car_dict['total_warranty_cost']:,.2f}**
        - **PPM Shift:** **{car_dict['baseline_ppm']:,.0f} → {car_dict['recent_ppm']:,.0f} PPM** (`{car_dict['sigma_deviation']:+.1f}σ acceleration`)
        - **Qualification Status:** `{car_dict['car_status']}`
        """
    )

with c_action:
    st.markdown("#### 🚀 Report Actions")
    pdf_filename = f"{car_dict['car_id']}_8D_Report.pdf"
    target_path = os.path.join(CAR_DIR, pdf_filename)
    os.makedirs(CAR_DIR, exist_ok=True)

    # Generate button
    if st.button("⚡ Generate / Update 8D Report", type="primary", use_container_width=True):
        with st.spinner("Compiling 8D PDF report with ReportLab..."):
            generate_car_pdf(car_dict, target_path)
            st.session_state[f"car_{car_dict['car_id']}"] = True
            st.success("8D Report successfully compiled!")

    # Ensure file exists or create on demand
    if not os.path.exists(target_path):
        generate_car_pdf(car_dict, target_path)

    if os.path.exists(target_path):
        with open(target_path, "rb") as f:
            pdf_data = f.read()
        st.download_button(
            label=f"📥 Download {pdf_filename}",
            data=pdf_data,
            file_name=pdf_filename,
            mime="application/pdf",
            use_container_width=True,
        )

# Preview 8D sections
with st.expander("📖 Expand to Preview Full 8-Discipline Engineering Details on Screen"):
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.markdown("**D1: Team Cross-Functional Leadership**")
        st.caption("Champion: VP Quality | Leader: Senior Reliability Engineer | Engineering, Sourcing, Field Service")
        st.markdown("**D2: Problem Description**")
        st.caption(f"Field failure acceleration observed on {car_dict['product_family']} with {car_dict['failure_mode']} defect symptoms.")
        st.markdown("**D3: Interim Containment Actions (ICA)**")
        st.caption("Quarantine warehouse inventory, issue urgent field service technical advisory, audit distributor stock.")
        st.markdown("**D4: Root Cause Analysis & 5-Why**")
        st.caption("Material degradation / process drift combined with accelerated field thermal-electrical cycling stress.")
    with d_col2:
        st.markdown("**D5: Permanent Corrective Actions (PCA)**")
        st.caption("Issue Engineering Change Order (ECO) with upgraded thermal/dielectric specs; re-qualify tier-1 supplier.")
        st.markdown("**D6: Verification & Validation Plan**")
        st.caption("Execute 1,000-hr HALT (Highly Accelerated Life Testing) and thermal chamber stress cycling.")
        st.markdown("**D7: Systemic Prevention of Recurrence**")
        st.caption("Revise Design FMEA (DFMEA), incoming inspection sampling plan (AQL 0.25), and standard operating procedures.")
        st.markdown("**D8: Team Recognition & Sign-Off**")
        st.caption("Sign-off by VP Quality & Chief Engineer upon 6 consecutive months within PPM baseline target.")
