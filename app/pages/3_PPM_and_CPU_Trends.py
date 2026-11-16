"""
app/pages/3_PPM_and_CPU_Trends.py
---------------------------------
Longitudinal PPM & CPU Trends with Shewhart Statistical Control Limits.
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
from src.metrics import calculate_ppm, calculate_cpu
from src.spc import compute_monthly_spc_for_product

st.set_page_config(page_title="PPM & CPU Trends | Warranty Analytics", page_icon="⚡", layout="wide")
apply_custom_theme()

products_df, sales_df, claims_df = load_all_data()
f_claims, f_sales, f_prods = render_sidebar_filters(claims_df, sales_df, products_df)

st.title("📉 Longitudinal PPM & CPU Trends with Shewhart Control Limits")

st.markdown(
    """
    <div class="guide-card">
        <div class="guide-title">📘 Statistical Process Control (SPC) in Warranty Management</div>
        Tracking raw warranty claims without statistical normalization is deceptive because shipment volumes fluctuate. 
        <b>Shewhart p-charts</b> establish rigorous statistical control boundaries for cohort failure rates:
        <ul>
            <li><b>Centerline (p̄):</b> Baseline expected defect rate calculated across stable historical manufacturing cohorts.</li>
            <li><b>Upper Warning Limit (UWL = μ + 2σ):</b> Threshold indicating 95.4% confidence boundary. Points here warrant increased surveillance.</li>
            <li><b>Upper Control Limit (UCL = μ + 3σ):</b> 99.7% statistical threshold. Any cohort breaching UCL represents <i>Special Cause Variation</i> 
            (e.g., supplier component lot defect, assembly error, or firmware bug) requiring formal 8D root-cause investigation.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["SPC Shewhart Control Chart", "Longitudinal PPM by Family", "Cohort Maturation Heatmap"])

with tab1:
    st.subheader("Statistical Process Control (Shewhart p-Chart)")
    sel_prod = st.selectbox("Select Product Family for SPC Analysis:", options=sorted(f_prods["product_family"].unique()))

    spc_df = compute_monthly_spc_for_product(f_claims, f_sales, sel_prod)

    if not spc_df.empty:
        fig_spc = go.Figure()

        # Actual PPM line
        fig_spc.add_trace(go.Scatter(
            x=spc_df["ship_month"],
            y=spc_df["actual_ppm"],
            name="Actual Cohort PPM",
            mode="lines+markers",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(size=6),
        ))

        # Centerline
        fig_spc.add_trace(go.Scatter(
            x=spc_df["ship_month"],
            y=spc_df["centerline_ppm"],
            name="Centerline (p-bar)",
            mode="lines",
            line=dict(color="#10b981", width=2, dash="dash"),
        ))

        # Upper Warning Limit (+2 Sigma)
        fig_spc.add_trace(go.Scatter(
            x=spc_df["ship_month"],
            y=spc_df["uwl_2sigma"],
            name="UWL (+2σ Warning)",
            mode="lines",
            line=dict(color="#f59e0b", width=1.5, dash="dot"),
        ))

        # Upper Control Limit (+3 Sigma)
        fig_spc.add_trace(go.Scatter(
            x=spc_df["ship_month"],
            y=spc_df["ucl_3sigma"],
            name="UCL (+3σ Breach)",
            mode="lines",
            line=dict(color="#ef4444", width=2.2, dash="dashdot"),
        ))

        # Highlight out of control points
        ooc_pts = spc_df[spc_df["spc_status"].str.contains("OUT_OF_CONTROL")]
        if not ooc_pts.empty:
            fig_spc.add_trace(go.Scatter(
                x=ooc_pts["ship_month"],
                y=ooc_pts["actual_ppm"],
                name="Out-of-Control Special Cause",
                mode="markers",
                marker=dict(color="#ef4444", size=13, symbol="circle-open", line=dict(width=3, color="#ef4444")),
            ))

        fig_spc.update_layout(
            xaxis=dict(title="Shipment Cohort (Year-Month)", tickangle=-45),
            yaxis=dict(title="Cohort PPM Rate"),
        )
        format_plotly_fig(fig_spc, height=480)
        st.plotly_chart(fig_spc, use_container_width=True)

        st.caption("Detailed statistical bounds calculated per monthly cohort:")
        st.dataframe(spc_df[["ship_month", "units_shipped", "total_claims", "actual_ppm", "centerline_ppm", "uwl_2sigma", "ucl_3sigma", "spc_status"]], use_container_width=True, hide_index=True)


with tab2:
    st.subheader("Longitudinal Cohort PPM Across Portfolio")
    cohort_ppm = calculate_ppm(f_claims, f_sales, group_by=["product_family", "ship_month"])
    fig_all = px.line(
        cohort_ppm,
        x="ship_month",
        y="ppm",
        color="product_family",
        title="Monthly Shipment Cohort PPM Trajectory by Equipment Family",
        labels={"ship_month": "Shipment Month", "ppm": "Cohort PPM Rate", "product_family": "Family"},
    )
    fig_all.update_layout(xaxis=dict(tickangle=-45))
    format_plotly_fig(fig_all, height=500)
    st.plotly_chart(fig_all, use_container_width=True)


with tab3:
    st.subheader("Maturation Cohort Heatmap (Ship Month vs Claim Month)")
    st.markdown(
        """
        <div class="guide-card">
            <div class="guide-title">🔍 Reading the Maturation Triangle</div>
            This matrix tracks the lag time between product shipment and field claim filing:
            <ul>
                <li><b>Leading Diagonal:</b> Claims filed in the exact same month or month following shipment — signifies <i>Infant Mortality</i> (manufacturing assembly errors, loose terminations, dead-on-arrival).</li>
                <li><b>Vertical Color Bands:</b> If an entire vertical column lights up across multiple historical cohorts, it indicates an <i>external environmental shock</i> (such as a severe grid surge or extreme weather event).</li>
                <li><b>Horizontal Color Intensities:</b> Indicates a single defective manufacturing lot continuing to generate elevated claims over its operating lifecycle.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cohort_matrix = f_claims.pivot_table(
        index="ship_month",
        columns="claim_month",
        values="claim_id",
        aggfunc="count",
        fill_value=0,
    )

    fig_heat = px.imshow(
        cohort_matrix,
        color_continuous_scale="Viridis",
        labels=dict(x="Claim Filing Month", y="Shipment Cohort Month", color="Claims Count"),
        aspect="auto",
    )
    fig_heat.update_layout(xaxis=dict(tickangle=-45))
    format_plotly_fig(fig_heat, height=540)
    st.plotly_chart(fig_heat, use_container_width=True)
