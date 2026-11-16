"""
app/pages/4_Emerging_Issues.py
------------------------------
Automated Early Warning System for Emerging Quality Anomalies.
"""

import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.utils import (
    load_all_data,
    render_sidebar_filters,
    apply_custom_theme,
    format_plotly_fig,
)
from src.spc import detect_emerging_anomalies, get_anomaly_drilldown

st.set_page_config(page_title="Emerging Issues | Warranty Analytics", page_icon="⚡", layout="wide")
apply_custom_theme()

products_df, sales_df, claims_df = load_all_data()
f_claims, f_sales, f_prods = render_sidebar_filters(claims_df, sales_df, products_df)

st.title("🚨 Emerging Quality Anomalies & Early Warning Feed")

st.markdown(
    """
    <div class="guide-card">
        <div class="guide-title">⚡ Automated Anomaly Detection Algorithm</div>
        Traditional warranty reviews wait 6–12 months for claims to mature, by which time thousands of defective units are already in the field. 
        This engine continuously compares <b>recent 3-month failure velocity</b> against <b>historical baseline rate</b> for every 
        <i>(Product Family × Failure Mode)</i> pair:
        <br><br>
        <div style="background-color: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; font-family: monospace; font-size: 0.9rem;">
            Rate Shift (Z-Score) = (Recent 3-Month PPM − Historical Baseline PPM) ÷ Baseline Standard Deviation (σ)
        </div>
        <br>
        Any defect accelerating at <b>&gt; +2.0σ</b> is immediately flagged with severity tags and recommended containment actions.
    </div>
    """,
    unsafe_allow_html=True,
)

# Run anomaly detection
anomalies_df = detect_emerging_anomalies(f_claims, f_sales, sigma_threshold=2.0)

if anomalies_df.empty:
    st.success("✅ No statistical process shifts or emerging quality spikes detected in the active filter window.")
else:
    # Highlight planted critical story if present
    planted_emerging = anomalies_df[anomalies_df["is_planted_emerging_issue"]]
    if not planted_emerging.empty:
        pe = planted_emerging.iloc[0]
        st.markdown(
            f"""
            <div style="background-color: rgba(239, 68, 68, 0.12); border: 2px solid #ef4444; border-radius: 10px; padding: 16px; margin-bottom: 20px;">
                <div style="font-weight: 700; font-size: 1.1rem; color: #ef4444; margin-bottom: 6px;">
                    ⚠️ CRITICAL QUALITY ALERT DETECTED (HIGH SEVERITY ANOMALY)
                </div>
                <div style="font-size: 0.95rem; line-height: 1.5;">
                    <b>{pe['product_family']}</b> exhibiting <b>{pe['failure_mode']}</b> has experienced a massive 
                    <span style="background-color: #ef4444; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 700;">+{pe['sigma_deviation']:.1f}σ statistical rate shift</span>!
                    <br>
                    • <b>Recent PPM:</b> {pe['recent_ppm']:,.0f} PPM (vs historical baseline of {pe['baseline_ppm']:,.0f} PPM, a jump of +{pe['delta_ppm']:,.0f} PPM).<br>
                    • <b>Financial Exposure:</b> {pe['recent_claims_count']:,} recent claims generating <b>${pe['recent_warranty_cost']:,.2f}</b> in immediate warranty expense.<br>
                    • <b>Containment Action:</b> {pe['recommended_action']}.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("📋 Active Anomaly Surveillance Feed")
    st.caption("All product defect modes exceeding statistical baseline thresholds:")

    # Format dataframe nicely
    display_df = anomalies_df[[
        "product_family", "failure_mode", "severity", "baseline_ppm",
        "recent_ppm", "delta_ppm", "sigma_deviation", "recent_claims_count",
        "recent_warranty_cost", "recommended_action"
    ]].copy()
    display_df["baseline_ppm"] = display_df["baseline_ppm"].map(lambda x: f"{x:,.0f}")
    display_df["recent_ppm"] = display_df["recent_ppm"].map(lambda x: f"{x:,.0f}")
    display_df["delta_ppm"] = display_df["delta_ppm"].map(lambda x: f"{x:+,.0f}")
    display_df["sigma_deviation"] = display_df["sigma_deviation"].map(lambda x: f"{x:+.1f}σ")
    display_df["recent_warranty_cost"] = display_df["recent_warranty_cost"].map(lambda x: f"${x:,.2f}")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.subheader("🔍 Deep-Dive Root Cause Drilldown")
    st.caption("Select any flagged anomaly to isolate suspect manufacturing batches, regional distribution, and failed sub-assemblies.")

    issue_options = [f"{r['product_family']} | {r['failure_mode']}" for _, r in anomalies_df.iterrows()]
    selected_issue = st.selectbox("Select Anomaly for Investigation:", options=issue_options)

    if selected_issue:
        sel_prod, sel_fm = selected_issue.split(" | ")
        drilldown = get_anomaly_drilldown(f_claims, sel_prod, sel_fm)

        d_col1, d_col2 = st.columns(2)

        with d_col1:
            st.markdown("#### 🏭 Manufacturing Batches & Lots")
            by_batch = drilldown["by_batch"]
            if not by_batch.empty:
                fig_batch = px.bar(
                    by_batch.head(8),
                    x="batch_id",
                    y="claims_count",
                    color="total_cost",
                    title=f"Claims Count by Manufacturing Lot: {sel_prod}",
                    labels={"batch_id": "Lot / Batch ID", "claims_count": "Claims", "total_cost": "Cost ($)"},
                    color_continuous_scale="Reds",
                )
                format_plotly_fig(fig_batch, height=380)
                st.plotly_chart(fig_batch, use_container_width=True)
                st.dataframe(by_batch.head(8), use_container_width=True, hide_index=True)

        with d_col2:
            st.markdown("#### 🗺️ Geographic Distribution of Failures")
            by_reg = drilldown["by_region"]
            if not by_reg.empty:
                fig_reg = px.pie(
                    by_reg,
                    names="region",
                    values="claims_count",
                    title=f"Regional Concentration: {sel_fm}",
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                )
                format_plotly_fig(fig_reg, height=380)
                st.plotly_chart(fig_reg, use_container_width=True)

            st.markdown("#### 🔧 Physical Sub-Assembly Breakdown")
            by_comp = drilldown["by_component"]
            if not by_comp.empty:
                fig_comp = px.bar(
                    by_comp,
                    x="component",
                    y="claims_count",
                    title="Defective Components Identified",
                    labels={"component": "Component", "claims_count": "Count"},
                    color_discrete_sequence=["#38bdf8"],
                )
                format_plotly_fig(fig_comp, height=300)
                st.plotly_chart(fig_comp, use_container_width=True)
