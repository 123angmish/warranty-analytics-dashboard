"""
tests/test_reliability.py
-------------------------
Unit tests for Weibull reliability analysis.
"""

import numpy as np
import pandas as pd
import pytest
from src.reliability import (
    classify_failure_regime,
    compute_weibull_metrics_for_family,
    generate_weibull_curves,
)


def test_classify_failure_regime():
    assert "Infant Mortality" in classify_failure_regime(0.75)
    assert "Random" in classify_failure_regime(1.05)
    assert "Wear-Out" in classify_failure_regime(2.35)


def test_generate_weibull_curves():
    curves = generate_weibull_curves(beta=2.0, eta=500.0, max_days=1000)
    assert len(curves["days"]) == 100
    assert curves["reliability"][0] > curves["reliability"][-1]
    assert curves["unreliability_cdf"][0] < curves["unreliability_cdf"][-1]
    # For wear out beta > 1, hazard rate should increase with time
    assert curves["hazard_rate_per_1k_days"][-1] > curves["hazard_rate_per_1k_days"][1]


def test_compute_weibull_metrics_for_family():
    # Small synthetic failure dataset
    claims = pd.DataFrame([
        {"product_family": "UPS-A", "ship_date": "2024-01-01", "claim_date": "2024-03-01"},
        {"product_family": "UPS-A", "ship_date": "2024-01-01", "claim_date": "2024-04-01"},
        {"product_family": "UPS-A", "ship_date": "2024-01-01", "claim_date": "2024-05-01"},
        {"product_family": "UPS-A", "ship_date": "2024-01-01", "claim_date": "2024-06-01"},
        {"product_family": "UPS-A", "ship_date": "2024-01-01", "claim_date": "2024-07-01"},
    ])
    sales = pd.DataFrame([
        {"product_family": "UPS-A", "ship_date": "2024-01-01", "quantity": 100},
    ])

    res = compute_weibull_metrics_for_family(claims, sales, "UPS-A", max_censored_sample=50)
    assert res["product_family"] == "UPS-A"
    assert res["beta_shape"] > 0
    assert res["eta_scale_days"] > 0
    assert res["b10_days"] > 0
    assert res["mttf_days"] > 0
    assert "failure_regime" in res
