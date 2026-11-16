"""
tests/test_spc.py
-----------------
Unit tests for SPC p-chart control limits and emerging anomaly detection.
"""

import numpy as np
import pandas as pd
import pytest
from src.spc import compute_p_chart_limits, detect_emerging_anomalies


def test_compute_p_chart_limits():
    # 10 months, 10 claims each, 1,000 units each -> p_bar = 0.01 (10,000 PPM)
    claims = np.array([10] * 10)
    units = np.array([1000] * 10)

    spc = compute_p_chart_limits(claims, units)
    assert np.allclose(spc["actual_ppm"], 10000.0)
    assert np.allclose(spc["centerline_ppm"], 10000.0)
    # UCL should be > centerline
    assert np.all(spc["ucl_3sigma"] > spc["centerline_ppm"])
    # UWL should be between centerline and UCL
    assert np.all(spc["uwl_2sigma"] > spc["centerline_ppm"])
    assert np.all(spc["uwl_2sigma"] < spc["ucl_3sigma"])


def test_detect_emerging_anomalies():
    # Construct 28 months dataset: 24 baseline months with 1 claim, last 4 months with 25 claims
    claims_rows, sales_rows = [], []
    for m in range(1, 29):
        y = 2023 + (m - 1) // 12
        month_num = ((m - 1) % 12) + 1
        m_str = f"{y}-{month_num:02d}"
        sales_rows.append({"product_family": "Online UPS 10kVA", "ship_month": m_str, "quantity": 1000})

        n_claims = 25 if m >= 25 else 1
        for c in range(n_claims):
            claims_rows.append({
                "claim_id": f"C_{m}_{c}",
                "product_family": "Online UPS 10kVA",
                "failure_mode": "Capacitor Aging",
                "claim_month": m_str,
                "repair_cost": 350.0,
            })

    c_df = pd.DataFrame(claims_rows)
    s_df = pd.DataFrame(sales_rows)

    anom = detect_emerging_anomalies(c_df, s_df, baseline_window_months=24, recent_window_months=4)
    assert not anom.empty
    top_issue = anom.iloc[0]
    assert top_issue["product_family"] == "Online UPS 10kVA"
    assert top_issue["failure_mode"] == "Capacitor Aging"
    assert top_issue["sigma_deviation"] > 3.0
    assert bool(top_issue["is_planted_emerging_issue"]) is True
