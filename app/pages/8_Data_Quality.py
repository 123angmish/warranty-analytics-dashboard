"""
app/pages/8_Data_Quality.py
---------------------------
Data Quality Governance Scorecard & Audit Trail.
"""

import os
import sys
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.utils import load_all_data, apply_custom_theme
from src.db import get_data_quality_summary

st.set_page_config(page_title="Data Quality | Warranty Analytics", page_icon="⚡", layout="wide")
apply_custom_theme()

st.title("🛡️ Data Quality Governance & Ingestion Audit")

st.markdown(
    """
    <div class="guide-card">
        <div class="guide-title">📘 Data Quality Framework in Warranty Analytics</div>
        Warranty datasets from ERP systems and third-party dealer service networks are notorious for data contamination. 
        Before computing PPM rates, Weibull life curves, or accrual reserves, this ingestion pipeline enforces an automated 
        <b>5-Dimensional Data Quality Framework</b>:
        <ul>
            <li><b>Uniqueness:</b> Elimination of duplicate warranty claims submitted across overlapping billing periods.</li>
            <li><b>Validity:</b> Enforcement of chronological ordering (preventing claims occurring prior to shipment dates).</li>
            <li><b>Accuracy:</b> Inversion of negative claim expense adjustments and statistical winsorization of astronomical data-entry typos.</li>
            <li><b>Consistency:</b> Normalization of non-standard technician casing, typos, and abbreviations to our canonical 12-item physics-of-failure taxonomy.</li>
            <li><b>Completeness:</b> Imputation of missing customer segments and surrogate generation for omitted serial identifiers.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# Fetch DQ summary from SQLite
dq_sum = get_data_quality_summary()

# Scorecard
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Cleanliness Score</div>
            <div class="metric-value" style="color: #10b981;">{dq_sum['cleanliness_score']}%</div>
            <div class="metric-sub"><span class="status-pill-success">Production Certified</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Raw Claims Ingested</div>
            <div class="metric-value">{dq_sum['raw_count']:,}</div>
            <div class="metric-sub"><span class="status-pill-info">ERP / Dealer Ingest</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Clean Claims Retained</div>
            <div class="metric-value">{dq_sum['clean_count']:,}</div>
            <div class="metric-sub"><span class="status-pill-success">Downstream Certified</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Duplicates Pruned</div>
            <div class="metric-value" style="color: #ef4444;">{dq_sum['records_dropped']:,}</div>
            <div class="metric-sub"><span class="status-pill-danger">Exact Duplicate Rows</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Audit Trail Log Table
st.subheader("📋 Automated Quality Remediation Audit Trail")
st.caption("Every remediation action executed during data ingestion is logged to SQLite table `data_quality_log`:")

log_df = dq_sum.get("log_df", pd.DataFrame())

if not log_df.empty:
    st.dataframe(
        log_df[["dimension", "severity", "rule_name", "count_affected", "action_taken", "details"]],
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")

# Data Quality Dimension Rules Explained
st.subheader("📐 Quality Assurance Rules & Remediation Protocols")

st.markdown("""
| Dimension | Rule Name | Defect Detected in Raw Feed | Automated Remediation Protocol |
| :--- | :--- | :--- | :--- |
| **Uniqueness** | `DUPLICATE_CLAIM_ID` | Exact duplicate claim records filed multiple times by dealers. | Dropped duplicate rows; retained initial occurrence only. |
| **Validity** | `CHRONOLOGY_CLAIM_BEFORE_SHIP` | Data entry error: `claim_date < ship_date`. | Projected chronological claim date (`ship_date + 14 days`). |
| **Accuracy** | `NEGATIVE_REPAIR_COST` | Accounting negative expense reversal entries. | Inverted absolute value or recalculated parts + labor cost. |
| **Accuracy** | `EXTREME_COST_OUTLIER` | Astronomical claim expense entry typos (> $50,000). | Winsorized cost capped at plausible maximum (1.2x product unit price). |
| **Consistency** | `INCONSISTENT_FAILURE_MODE_LABELS` | Synonyms, mixed casing (`CAPACITOR_AGING`, `cap aging`). | Remapped to canonical 12-item physics-of-failure ontology. |
| **Completeness** | `MISSING_SERIAL_NUMBER` | Missing serial number in field claim form. | Generated surrogate identifier (`SN-UNKNOWN-<ID>`). |
| **Completeness** | `MISSING_CUSTOMER_TYPE` | Unassigned customer segment. | Imputed modal segment (`Commercial Office Park`). |
""")
