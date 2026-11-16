"""
app/pages/1_Executive_Overview.py
---------------------------------
Executive Overview: Key Reliability & Financial Warranty KPIs.
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
from src.metrics import calculate_ppm, calculate_cpu, calculate_accruals, rank_ppit_car

st.set_page_config(page_title="Executive Overview | Warranty Analytics", page_icon="⚡", layout="wide")
apply_custom_theme()

products_df, sales_df, claims_df = load_all_data()
f_claims, f_sales, f_prods = render_sidebar_filters(claims_df, sales_df, products_df)

st.title("📈 Executive Warranty & Reliability Overview")

st.markdown(
    """
    <div class="guide-card">
        <div class="guide-title">💡 How to Read the Executive Scorecard</div>
        This dashboard aggregates fleet-wide warranty claims and manufacturing shipments to track operational reliability and financial liability:
        <ul>
            <li><b>Global PPM (Parts Per Million):</b> Standardized failure rate metric: <code>(Claims ÷ Units Shipped) × 1,000,000</code>. A lower PPM indicates higher field reliability.</li>
            <li><b>Cost Per Unit (CPU):</b> Total warranty expense divided by total units shipped. Reflects the unit-level margin impact of warranty claims.</li>
            <li><b>Warranty Accrual Reserve:</b> Actuarial reserve required on the balance sheet: <code>Rolling CPU × Units under Active Warranty − Claims Incurred to Date</code>.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# Compute Core Metrics
total_shipped = int(f_sales["quantity"].sum())
total_claims = len(f_claims)
total_warranty_cost = float(f_claims["repair_cost"].sum())
global_ppm = round((total_claims / total_shipped) * 1_000_000.0, 1) if total_shipped > 0 else 0.0
global_cpu = round(total_warranty_cost / total_shipped, 2) if total_shipped > 0 else 0.0

accruals_res = calculate_accruals(f_sales, f_claims, f_prods)
total_reserve = accruals_res["total_warranty_reserve"]

ppit_df = rank_ppit_car(f_claims, f_sales, f_prods)
top_issue_str = f"{ppit_df.iloc[0]['product_family']} ({ppit_df.iloc[0]['failure_mode']})" if not ppit_df.empty else "N/A"

# KPI Cards with modern layout
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Valid Claims</div>
            <div class="metric-value">{total_claims:,}</div>
            <div class="metric-sub"><span class="status-pill-success">Certified Claims</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi2:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Fleet Shipped</div>
            <div class="metric-value">{total_shipped:,}</div>
            <div class="metric-sub"><span class="status-pill-info">Active Population</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi3:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Global Fleet PPM</div>
            <div class="metric-value">{global_ppm:,.0f}</div>
            <div class="metric-sub"><span class="status-pill-warning">Per Million Units</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi4:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Cost Per Unit (CPU)</div>
            <div class="metric-value">${global_cpu:,.2f}</div>
            <div class="metric-sub"><span class="status-pill-danger">Warranty Drain</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi5:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Accrual Reserve</div>
            <div class="metric-value">${total_reserve:,.2f}</div>
            <div class="metric-sub"><span class="status-pill-warning">Estimated Liability</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Visual Trajectory Charts
c_left, c_right = st.columns(2)

with c_left:
    st.subheader("Monthly Claims & Warranty Expense Trajectory")
    monthly_trend = f_claims.groupby("claim_month", as_index=False).agg(
        claims_count=("claim_id", "count"),
        total_cost=("repair_cost", "sum"),
    ).sort_values(by="claim_month")

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        x=monthly_trend["claim_month"],
        y=monthly_trend["claims_count"],
        name="Claims Count",
        marker_color="#3b82f6",
        yaxis="y",
    ))
    fig_trend.add_trace(go.Scatter(
        x=monthly_trend["claim_month"],
        y=monthly_trend["total_cost"],
        name="Warranty Expense ($)",
        mode="lines+markers",
        line=dict(color="#ef4444", width=3),
        yaxis="y2",
    ))
    fig_trend.update_layout(
        xaxis=dict(title="Claim Filing Month", tickangle=-45),
        yaxis=dict(title="Claims Count"),
        yaxis2=dict(title="Warranty Expense ($)", overlaying="y", side="right"),
    )
    format_plotly_fig(fig_trend, height=400)
    st.plotly_chart(fig_trend, use_container_width=True)

with c_right:
    st.subheader("Actual PPM vs Target Ceiling by Product Family")
    prod_ppm = calculate_ppm(f_claims, f_sales, group_by="product_family")
    target_map = f_prods.set_index("product_family")["target_ppm"].to_dict()
    prod_ppm["target_ppm"] = prod_ppm["product_family"].map(target_map).fillna(3000)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=prod_ppm["product_family"],
        y=prod_ppm["ppm"],
        name="Actual Fleet PPM",
        marker_color="#2563eb",
    ))
    fig_bar.add_trace(go.Scatter(
        x=prod_ppm["product_family"],
        y=prod_ppm["target_ppm"],
        name="PPM Target Ceiling",
        mode="lines+markers",
        line=dict(color="#f59e0b", width=2.5, dash="dash"),
    ))
    fig_bar.update_layout(
        xaxis=dict(title="Product Family", tickangle=-25),
        yaxis=dict(title="PPM Rate"),
    )
    format_plotly_fig(fig_bar, height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# Portfolio Health Summary Table
st.subheader("📋 Product Portfolio Health & Financial Liability Summary")
st.caption("Detailed breakdown across all 8 electrical equipment families with target compliance status.")

portfolio_summary = []
cohort_accruals = accruals_res.get("cohort_accruals_df", pd.DataFrame())

for prod in sorted(f_prods["product_family"].unique()):
    p_sales = int(f_sales[f_sales["product_family"] == prod]["quantity"].sum())
    p_claims = f_claims[f_claims["product_family"] == prod]
    p_claims_cnt = len(p_claims)
    p_cost = float(p_claims["repair_cost"].sum())
    p_ppm = round((p_claims_cnt / p_sales) * 1_000_000.0, 1) if p_sales > 0 else 0.0
    p_cpu = round(p_cost / p_sales, 2) if p_sales > 0 else 0.0
    
    p_accrual = 0.0
    if not cohort_accruals.empty and "product_family" in cohort_accruals.columns:
        p_accrual = float(cohort_accruals[cohort_accruals["product_family"] == prod]["remaining_accrual"].sum())

    target = target_map.get(prod, 3000)
    status = "🔴 Exceeds Target" if p_ppm > target else "🟢 Within Target"

    portfolio_summary.append({
        "Product Family": prod,
        "Units Shipped": f"{p_sales:,}",
        "Total Claims": f"{p_claims_cnt:,}",
        "PPM Rate": f"{p_ppm:,.0f}",
        "Target PPM": f"{target:,.0f}",
        "PPM Status": status,
        "Total Warranty Cost": f"${p_cost:,.2f}",
        "Cost Per Unit (CPU)": f"${p_cpu:,.2f}",
        "Active Accrual Reserve": f"${p_accrual:,.2f}",
    })

df_portfolio = pd.DataFrame(portfolio_summary)
st.dataframe(df_portfolio, use_container_width=True, hide_index=True)

# Export button
csv_p = df_portfolio.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Portfolio Health Summary Table (CSV)",
    data=csv_p,
    file_name="portfolio_health_summary.csv",
    mime="text/csv",
)
