"""
app/pages/2_Top_N_and_Pareto.py
-------------------------------
80/20 Pareto Analysis: Failure Modes, Components, and Geographic Regions.
"""

import os
import sys
import pandas as pd
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
from src.metrics import calculate_pareto

st.set_page_config(page_title="Top N & Pareto | Warranty Analytics", page_icon="⚡", layout="wide")
apply_custom_theme()

products_df, sales_df, claims_df = load_all_data()
f_claims, f_sales, f_prods = render_sidebar_filters(claims_df, sales_df, products_df)

st.title("📊 Top N & 80/20 Pareto Analysis")

st.markdown(
    """
    <div class="guide-card">
        <div class="guide-title">📘 The 80/20 Principle in Reliability Engineering</div>
        In electrical manufacturing, <b>roughly 80% of warranty expenditures and field RMA incidents stem from 20% of defect modes</b>. 
        Pareto analysis isolates the <i>vital few</i> failure causes from the <i>trivial many</i>, directing engineering change orders (ECOs) 
        and supplier quality audits to the highest-ROI areas.
        <br><br>
        <b>How to interpret:</b> The left vertical bars display individual defect magnitude, while the right orange line plots cumulative percentage. 
        The green dotted line marks the 80% cumulative threshold.
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Failure Modes (80/20)", "Component Sub-Assemblies", "Regional Concentration"])


def create_pareto_chart(df: pd.DataFrame, category_col: str, metric_mode: str = "count"):
    """Creates a dual-axis Plotly Pareto chart with 80% cutoff threshold."""
    fig = go.Figure()

    if metric_mode == "count":
        y_bars = df["claim_count"]
        y_line = df["cum_count_pct"]
        bar_name = "Claim Incidents"
        bar_color = "#3b82f6"
        y_axis_title = "Claim Incidents Count"
    else:
        y_bars = df["total_cost"]
        y_line = df["cum_cost_pct"]
        bar_name = "Warranty Expense ($)"
        bar_color = "#ef4444"
        y_axis_title = "Warranty Cost ($)"

    fig.add_trace(go.Bar(
        x=df[category_col],
        y=y_bars,
        name=bar_name,
        marker_color=bar_color,
        yaxis="y",
    ))

    fig.add_trace(go.Scatter(
        x=df[category_col],
        y=y_line,
        name="Cumulative %",
        mode="lines+markers",
        line=dict(color="#f59e0b", width=3),
        marker=dict(size=7),
        yaxis="y2",
    ))

    # Add 80% reference line
    fig.add_trace(go.Scatter(
        x=df[category_col],
        y=[80.0] * len(df),
        name="80% Vital Few Cutoff",
        mode="lines",
        line=dict(color="#10b981", width=2, dash="dash"),
        yaxis="y2",
    ))

    fig.update_layout(
        xaxis=dict(title=category_col.replace("_", " ").title(), tickangle=-30),
        yaxis=dict(title=y_axis_title),
        yaxis2=dict(
            title="Cumulative Percentage (%)",
            overlaying="y",
            side="right",
            range=[0, 105],
        ),
    )
    return format_plotly_fig(fig, height=460)


with tab1:
    st.subheader("Canonical Failure Mode Pareto Analysis")
    metric_choice = st.radio("Pareto Metric:", ["Claim Frequency (Count)", "Financial Impact (Warranty Cost)"], horizontal=True)
    metric_key = "count" if "Count" in metric_choice else "cost"

    pareto_fm = calculate_pareto(f_claims, column="failure_mode")
    fig_fm = create_pareto_chart(pareto_fm, "failure_mode", metric_key)
    st.plotly_chart(fig_fm, use_container_width=True)

    # Narrative callout
    top_fm_count = pareto_fm.iloc[0]["failure_mode"]
    top_cost_fm = pareto_fm.sort_values(by="total_cost", ascending=False).iloc[0]["failure_mode"]
    st.info(
        f"💡 **Reliability Engineering Insight:** The highest frequency defect is **{top_fm_count}** ({pareto_fm.iloc[0]['count_pct']:.1f}% of claims). "
        f"However, the highest dollar drain on warranty reserves is **{top_cost_fm}** "
        f"(${pareto_fm.sort_values(by='total_cost', ascending=False).iloc[0]['total_cost']:,.2f}), demonstrating why dual-perspective Pareto evaluation is essential."
    )

    st.dataframe(pareto_fm, use_container_width=True, hide_index=True)


with tab2:
    st.subheader("Component Sub-Assembly Pareto")
    pareto_comp = calculate_pareto(f_claims, column="component", top_n=12)
    fig_comp = create_pareto_chart(pareto_comp, "component", "count")
    st.plotly_chart(fig_comp, use_container_width=True)
    st.dataframe(pareto_comp, use_container_width=True, hide_index=True)


with tab3:
    st.subheader("Geographic & Regional Defect Concentration")
    pareto_reg = calculate_pareto(f_claims, column="region")
    fig_reg = create_pareto_chart(pareto_reg, "region", "cost")
    st.plotly_chart(fig_reg, use_container_width=True)

    st.markdown("#### 🌊 Environmental Stress Analysis (Corrosion in Coastal Regions)")
    corr_claims = f_claims[f_claims["failure_mode"] == "Corrosion"]
    corr_by_reg = corr_claims.groupby("region", as_index=False).agg(
        corrosion_claims=("claim_id", "count"),
        total_corrosion_cost=("repair_cost", "sum"),
    ).sort_values(by="corrosion_claims", ascending=False)
    
    st.dataframe(corr_by_reg, use_container_width=True, hide_index=True)
    st.caption("Notice the significant concentration of corrosion incidents in South India due to coastal marine humidity (>85% RH) and airborne saline aerosol exposure.")
