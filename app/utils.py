"""
app/utils.py
------------
Streamlit Shared Utility Module:
  - Directory constants and file paths
  - Cached data loaders
  - Automatic pipeline bootstrapping for Streamlit Community Cloud
  - Unified sidebar filter widget
  - Color palette & formatting helpers
  - Modern UI styling & Plotly chart theming
"""

import os
import subprocess
import sys
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Directory Paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
CAR_DIR = os.path.join(REPORTS_DIR, "car_reports")
EXCEL_PATH = os.path.join(REPORTS_DIR, "warranty_summary.xlsx")
POWERBI_DIR = os.path.join(DATA_DIR, "powerbi")

# File Paths
CLEAN_CLAIMS_PATH = os.path.join(DATA_DIR, "claims_clean.csv")
SALES_PATH = os.path.join(DATA_DIR, "sales.csv")
PRODUCTS_PATH = os.path.join(DATA_DIR, "products.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "failure_mode_pipeline.joblib")


def apply_custom_theme():
    """
    Applies custom modern CSS that adapts seamlessly to both Streamlit Dark and Light themes.
    Provides glassmorphic cards, clear metric boxes, and sleek badges.
    """
    st.markdown(
        """
        <style>
            /* Global font & smooth transitions */
            html, body, [class*="css"] {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }

            /* Metric cards adaptive to theme */
            .metric-box {
                background-color: var(--secondary-background-color);
                border: 1px solid rgba(128, 128, 128, 0.2);
                border-radius: 10px;
                padding: 14px 18px;
                margin-bottom: 12px;
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }
            .metric-box:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 14px rgba(0, 0, 0, 0.2);
            }
            .metric-title {
                font-size: 0.8rem;
                font-weight: 600;
                color: var(--text-color);
                opacity: 0.75;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-bottom: 4px;
            }
            .metric-value {
                font-size: 1.85rem;
                font-weight: 700;
                color: var(--text-color);
                line-height: 1.2;
            }
            .metric-sub {
                font-size: 0.82rem;
                margin-top: 4px;
                font-weight: 500;
            }

            /* Guide / Educational Callout Card */
            .guide-card {
                background-color: rgba(59, 130, 246, 0.08);
                border-left: 4px solid #3b82f6;
                border-radius: 0 8px 8px 0;
                padding: 12px 18px;
                margin-bottom: 16px;
                color: var(--text-color);
                line-height: 1.5;
            }
            .guide-title {
                font-weight: 700;
                font-size: 0.95rem;
                color: #3b82f6;
                margin-bottom: 4px;
            }

            /* Status Pills */
            .status-pill-success {
                background-color: rgba(16, 185, 129, 0.15);
                color: #10b981;
                border: 1px solid #10b981;
                border-radius: 9999px;
                padding: 2px 10px;
                font-size: 0.75rem;
                font-weight: 600;
            }
            .status-pill-danger {
                background-color: rgba(239, 68, 68, 0.15);
                color: #ef4444;
                border: 1px solid #ef4444;
                border-radius: 9999px;
                padding: 2px 10px;
                font-size: 0.75rem;
                font-weight: 600;
            }
            .status-pill-warning {
                background-color: rgba(245, 158, 11, 0.15);
                color: #f59e0b;
                border: 1px solid #f59e0b;
                border-radius: 9999px;
                padding: 2px 10px;
                font-size: 0.75rem;
                font-weight: 600;
            }

            /* Sidebar aesthetics */
            section[data-testid="stSidebar"] {
                border-right: 1px solid rgba(128, 128, 128, 0.15);
            }

            /* Dataframes styling */
            div[data-testid="stDataFrame"] {
                border-radius: 8px;
                overflow: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_plotly_fig(fig, height: int = 420):
    """
    Applies clean, theme-transparent layout and styling to Plotly figures.
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=30, b=40),
        height=height,
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor="rgba(128, 128, 128, 0.15)",
            zerolinecolor="rgba(128, 128, 128, 0.2)",
        ),
        yaxis=dict(
            gridcolor="rgba(128, 128, 128, 0.15)",
            zerolinecolor="rgba(128, 128, 128, 0.2)",
        ),
    )
    if "yaxis2" in fig.layout:
        fig.update_layout(
            yaxis2=dict(
                gridcolor="rgba(128, 128, 128, 0.1)",
                zerolinecolor="rgba(128, 128, 128, 0.2)",
            )
        )
    return fig


def bootstrap_if_needed():
    """Runs data generator, db cleaning, and model training if artifacts are missing."""
    if not os.path.exists(CLEAN_CLAIMS_PATH) or not os.path.exists(MODEL_PATH):
        with st.spinner("Initializing warranty analytics database & training ML model (first-time launch)..."):
            from src.generate_data import main as gen_main
            from src.db import clean_and_validate_data
            from src.classifier import train_and_evaluate_models
            from src.export import main as export_main

            gen_main()
            clean_and_validate_data()
            train_and_evaluate_models()
            export_main()


@st.cache_data(show_spinner=False)
def load_all_data():
    """Loads products, sales, and clean claims into memory."""
    bootstrap_if_needed()
    products_df = pd.read_csv(PRODUCTS_PATH)
    sales_df = pd.read_csv(SALES_PATH)
    claims_df = pd.read_csv(CLEAN_CLAIMS_PATH)
    return products_df, sales_df, claims_df


def render_sidebar_filters(claims_df: pd.DataFrame, sales_df: pd.DataFrame, products_df: pd.DataFrame):
    """
    Renders standardized interactive sidebar filters.
    Returns filtered copies of claims_df and sales_df.
    """
    st.sidebar.markdown("### 🎛️ Analytics Scope Filters")
    st.sidebar.caption("Filter by product family, sales territory, and ship dates.")

    # Product Family Filter
    all_products = sorted(products_df["product_family"].unique())
    selected_products = st.sidebar.multiselect(
        "Product Family",
        options=all_products,
        default=all_products,
        help="Select one or more electrical product families to analyze.",
    )

    # Region Filter
    all_regions = sorted(sales_df["region"].unique())
    selected_regions = st.sidebar.multiselect(
        "Geographic Region",
        options=all_regions,
        default=all_regions,
        help="Filter by sales/service territory.",
    )

    # Date Range Filter (by Ship Date)
    sales_df["ship_dt"] = pd.to_datetime(sales_df["ship_date"])
    min_date = sales_df["ship_dt"].min().date()
    max_date = sales_df["ship_dt"].max().date()

    date_range = st.sidebar.date_input(
        "Shipment Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Filter dataframes
    f_claims = claims_df.copy()
    f_sales = sales_df.copy()

    if selected_products:
        f_claims = f_claims[f_claims["product_family"].isin(selected_products)]
        f_sales = f_sales[f_sales["product_family"].isin(selected_products)]

    if selected_regions:
        f_claims = f_claims[f_claims["region"].isin(selected_regions)]
        f_sales = f_sales[f_sales["region"].isin(selected_regions)]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
        f_sales = f_sales[(f_sales["ship_dt"].dt.date >= start_d) & (f_sales["ship_dt"].dt.date <= end_d)]
        f_claims = f_claims[
            (pd.to_datetime(f_claims["ship_date"]).dt.date >= start_d)
            & (pd.to_datetime(f_claims["ship_date"]).dt.date <= end_d)
        ]

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📊 Filter Population")
    st.sidebar.caption(f"• **Active Claims:** `{len(f_claims):,}` / `{len(claims_df):,}`")
    st.sidebar.caption(f"• **Fleet Shipped:** `{int(f_sales['quantity'].sum()):,}` units")
    st.sidebar.caption(f"• **Coverage:** `{len(selected_products)}` families, `{len(selected_regions)}` regions")

    return f_claims, f_sales, products_df
