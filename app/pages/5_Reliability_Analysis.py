"""
app/pages/5_Reliability_Analysis.py
-----------------------------------
Weibull Life Data Analysis & Reliability Engineering (Right-Censored).
"""

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
    load_all_data,
    render_sidebar_filters,
    apply_custom_theme,
    format_plotly_fig,
)
from src.reliability import (
    fit_all_products_weibull,
    compute_weibull_metrics_for_family,
    generate_weibull_curves,
    get_weibull_probability_plot_points,
)

st.set_page_config(page_title="Reliability Analysis | Warranty Analytics", page_icon="⚡", layout="wide")
apply_custom_theme()

products_df, sales_df, claims_df = load_all_data()
f_claims, f_sales, f_prods = render_sidebar_filters(claims_df, sales_df, products_df)

st.title("🔬 Reliability & Weibull Life Data Analysis")

st.markdown(
    """
    <div class="guide-card">
        <div class="guide-title">📘 Parametric Life Data Analysis with Right-Censoring</div>
        A critical error in naive warranty analytics is analyzing only failed units. In reality, <b>over 94% of shipped units are currently 
        operating in customer facilities without failing</b> (known as <i>right-censored</i> data). 
        Using <code>lifelines.WeibullFitter</code>, we fit 2-parameter Weibull distributions:
        <ul>
            <li><b>Shape Parameter (β):</b> Identifies the physics-of-failure regime:
                <ul>
                    <li><b>β &lt; 1.0: Infant Mortality</b> — Failures decrease with operating age (poor assembly, loose solder, dead-on-arrival).</li>
                    <li><b>β ≈ 1.0: Random Failures</b> — Exponential constant hazard rate (random grid transients, user error).</li>
                    <li><b>β &gt; 1.2: Wear-Out &amp; Aging</b> — Failure rate increases over time (capacitor drying, contact pitting, thermal fatigue).</li>
                </ul>
            </li>
            <li><b>Scale Parameter (η) Characteristic Life:</b> Operating age at which exactly 63.2% of the fleet will have failed.</li>
            <li><b>B10 Life:</b> Operating duration by which 10% of the fleet is expected to fail. Used by engineering to set warranty terms.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# Portfolio Weibull Table
st.subheader("📋 Portfolio Weibull Parameter Summary (Right-Censored Fleet)")
with st.spinner("Fitting parametric Weibull models on field population..."):
    weibull_summary = fit_all_products_weibull(f_claims, f_sales)

st.dataframe(weibull_summary, use_container_width=True, hide_index=True)

st.markdown("---")

# Single Product Deep-Dive
st.subheader("🎯 Single Product Family Weibull Deep-Dive")
sel_prod = st.selectbox("Select Product Family for Parametric Curve Plotting:", options=weibull_summary["product_family"].tolist())

p_meta = weibull_summary[weibull_summary["product_family"] == sel_prod].iloc[0]
beta = float(p_meta["beta_shape"])
eta = float(p_meta["eta_scale_days"])
b10 = float(p_meta["b10_days"])
mttf = float(p_meta["mttf_days"])
regime = str(p_meta["failure_regime"])

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Shape Parameter (β)</div>
            <div class="metric-value">{beta:.3f}</div>
            <div class="metric-sub"><span class="status-pill-info">{regime}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m2:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Characteristic Life (η)</div>
            <div class="metric-value">{eta:,.0f} d</div>
            <div class="metric-sub"><span class="status-pill-success">63.2% Failure Horizon</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m3:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">B10 Warranty Life</div>
            <div class="metric-value">{b10:,.0f} d</div>
            <div class="metric-sub"><span class="status-pill-warning">10% Cumulative Failure</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m4:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Mean Time To Failure</div>
            <div class="metric-value">{mttf:,.0f} d</div>
            <div class="metric-sub"><span class="status-pill-success">Expected Fleet MTTF</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Visualizations: Probability Plot & Hazard Rate Curve
curves = generate_weibull_curves(beta, eta)
t_pts, x_ln_t, y_weibull = get_weibull_probability_plot_points(f_claims, sel_prod)

w_col1, w_col2 = st.columns(2)

with w_col1:
    st.markdown("#### Weibull Probability Plot (Bernard Median Ranks)")
    st.caption("Linearized Weibull coordinates: points following a straight line confirm Weibull distribution fit.")
    fig_prob = go.Figure()

    # Empirical median ranks
    if len(x_ln_t) > 0:
        fig_prob.add_trace(go.Scatter(
            x=x_ln_t,
            y=y_weibull,
            mode="markers",
            name="Field RMA Data (Bernard Ranks)",
            marker=dict(size=4, color="#38bdf8", opacity=0.7),
        ))

    # Fitted linear Weibull line
    x_line = np.linspace(min(x_ln_t) if len(x_ln_t) > 0 else 2.0, max(x_ln_t) if len(x_ln_t) > 0 else 7.5, 50)
    y_line = beta * (x_line - np.log(eta))
    fig_prob.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name=f"Fitted Model (β={beta:.2f}, η={eta:.0f}d)",
        line=dict(color="#ef4444", width=2.5),
    ))

    fig_prob.update_layout(
        xaxis=dict(title="ln(Time to Failure in Days)"),
        yaxis=dict(title="ln(-ln(1 - F(t))) [Weibull Coordinate]"),
    )
    format_plotly_fig(fig_prob, height=420)
    st.plotly_chart(fig_prob, use_container_width=True)

with w_col2:
    st.markdown("#### Instantaneous Failure Rate / Hazard Function h(t)")
    st.caption("Indicates whether failure hazard accelerates, remains constant, or burns in over time.")
    fig_haz = go.Figure()
    fig_haz.add_trace(go.Scatter(
        x=curves["days"],
        y=curves["hazard_rate_per_1k_days"],
        mode="lines",
        name="Hazard Rate h(t)",
        line=dict(color="#10b981", width=3),
    ))
    fig_haz.update_layout(
        xaxis=dict(title="Operating Time (Days)"),
        yaxis=dict(title="Instantaneous Failure Rate (per 1,000 Days)"),
    )
    format_plotly_fig(fig_haz, height=420)
    st.plotly_chart(fig_haz, use_container_width=True)
