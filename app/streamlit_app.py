"""
app/streamlit_app.py
--------------------
Main Landing Page for Warranty Analytics & Reliability Dashboard.
Industrial Electrical Equipment Reliability & Quality Engineering System.
"""

import os
import sys
import pandas as pd
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.utils import (
    load_all_data,
    apply_custom_theme,
    REPORTS_DIR,
    EXCEL_PATH,
    DATA_DIR,
)

st.set_page_config(
    page_title="Warranty Analytics & Reliability Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply sleek modern theme
apply_custom_theme()

products_df, sales_df, claims_df = load_all_data()

# Hero Header
st.title("⚡ Warranty Analytics & Reliability Dashboard")
st.markdown("### Industrial Electrical Equipment Reliability & Quality Engineering System")

st.markdown(
    """
    <div class="guide-card">
        <div class="guide-title">🎯 Mission & System Overview</div>
        This production-grade analytics platform replicates the mission-critical workflows of a 
        <b>Senior Reliability / Quality Engineer</b> at a commercial electrical equipment manufacturer 
        (Uninterruptible Power Supplies, LV Switchgear, Air Circuit Breakers, Molded Case Circuit Breakers, and Power Distribution Units).
        <br><br>
        It provides an end-to-end reliability analytics stack: <b>automated data quality governance</b>, 
        <b>financial liability accruals (CPU & reserves)</b>, <b>Statistical Process Control (Shewhart p-charts)</b>, 
        <b>Weibull parametric life data analysis with right-censoring</b>, <b>NLP automated failure mode classification</b>, 
        and <b>automated 8D Corrective Action Request (CAR) PDF publishing</b>.
    </div>
    """,
    unsafe_allow_html=True,
)

# High-Level Architecture Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Clean Warranty Claims</div>
            <div class="metric-value">{len(claims_df):,}</div>
            <div class="metric-sub"><span class="status-pill-success">100% Quality Certified</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Manufactured Fleet</div>
            <div class="metric-value">{int(sales_df["quantity"].sum()):,}</div>
            <div class="metric-sub"><span class="status-pill-success">36 Monthly Ship Cohorts</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Product Families</div>
            <div class="metric-value">{len(products_df)} Families</div>
            <div class="metric-sub"><span class="status-pill-warning">UPS & Power Equipment</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Failure Modes</div>
            <div class="metric-value">{claims_df['failure_mode'].nunique()} Modes</div>
            <div class="metric-sub"><span class="status-pill-danger">Physics-of-Failure</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Quick Navigation & Workflow Guide
st.subheader("🧭 Platform Navigation & Analytical Workflows")
st.caption("Use the sidebar on the left or the cards below to navigate between analytics modules:")

nav1, nav2 = st.columns(2)

with nav1:
    st.markdown(
        """
        #### 📊 Surveillance & Quality Metrics
        * **[1. Executive Overview](Executive_Overview):** Top-level KPI scorecards (Total Claims, Global PPM, Cost Per Unit, Accrual Reserves, Top PPIT Priority) and monthly warranty spend trajectory.
        * **[2. Top N & Pareto Analysis](Top_N_and_Pareto):** 80/20 Pareto principle applied across Failure Modes, Sub-assembly Components, and Geographical Service Territories with dual-axis cumulative curves.
        * **[3. PPM & CPU Trends](PPM_and_CPU_Trends):** Longitudinal shipment cohort PPM tracking with Shewhart p-chart Upper Control Limits ($\mu + 3\sigma$) and warning bands ($\mu + 2\sigma$).
        * **[4. Emerging Issues Early Warning](Emerging_Issues):** Automated anomaly detection feed capturing sudden statistical rate shifts ($>2\sigma$) with root-cause drill-downs to suspect manufacturing lot batches.
        """
    )

with nav2:
    st.markdown(
        """
        #### 🔬 Reliability Physics & Engineering Action
        * **[5. Reliability & Weibull Analysis](Reliability_Analysis):** 2-parameter Weibull life modeling ($\beta, \eta$) using `lifelines` accounting for right-censored operating units, $B_{10}$ warranty life, and Bernard rank probability plots.
        * **[6. NLP Failure Mode Classifier](Failure_Mode_Classifier):** Live machine learning inference (TF-IDF + Calibrated Logistic Regression) that reads unstructured technician claim notes and predicts canonical failure modes.
        * **[7. PPIT & 8D CAR Tracker](PPIT_CAR_Tracker):** Multi-attribute risk prioritization ranking issues by Cost (40%), PPM Acceleration (35%), and Volume (25%) with one-click **8D PDF Report** compilation.
        * **[8. Data Quality Governance](Data_Quality):** Comprehensive audit trail detailing duplicate deduplication, chronological date inversion fixes, and cost outlier winsorization.
        """
    )

st.markdown("---")

# Quick Exports & Downloads
st.subheader("📥 Executive Downloads & Downstream Exports")
st.caption("One-click exports for executive presentations, Power BI modeling, and offline statistical audits.")

exp_col1, exp_col2, exp_col3 = st.columns(3)

with exp_col1:
    if os.path.exists(EXCEL_PATH):
        with open(EXCEL_PATH, "rb") as f:
            st.download_button(
                label="📊 Download Excel Summary Workbook (.xlsx)",
                data=f.read(),
                file_name="warranty_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Comprehensive 6-sheet formatted workbook (Executive KPIs, Pareto, Cohort PPM, Weibull, PPIT).",
                use_container_width=True,
            )

with exp_col2:
    csv_bytes = claims_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📄 Export Cleaned Claims Dataset (.csv)",
        data=csv_bytes,
        file_name="claims_clean.csv",
        mime="text/csv",
        help="Export all 26,800 validated warranty records for external analysis.",
        use_container_width=True,
    )

with exp_col3:
    sales_bytes = sales_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="🚚 Export Sales Shipments Dataset (.csv)",
        data=sales_bytes,
        file_name="sales.csv",
        mime="text/csv",
        help="Export 36 months of shipment batch records (456k+ units).",
        use_container_width=True,
    )

st.markdown("---")
st.info("💡 **Notice:** This project utilizes an entirely synthetic, procedurally generated dataset constructed specifically for demonstration and engineering portfolio validation. No real company or proprietary data was used.")
