"""
tests/test_pages_smoke.py
-------------------------
Smoke test verifying that all Streamlit app pages and utils import cleanly
and execute their core data processing logic without raising exceptions.
"""

import os
import sys
import pytest
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.utils import (
    load_all_data,
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    CAR_DIR,
    EXCEL_PATH,
    POWERBI_DIR,
    apply_custom_theme,
    format_plotly_fig,
)
from src.metrics import (
    calculate_ppm,
    calculate_cpu,
    calculate_accruals,
    calculate_pareto,
    rank_ppit_car,
)
from src.spc import compute_monthly_spc_for_product, detect_emerging_anomalies, get_anomaly_drilldown
from src.reliability import (
    fit_all_products_weibull,
    compute_weibull_metrics_for_family,
    generate_weibull_curves,
    get_weibull_probability_plot_points,
)
from src.classifier import load_classifier_pipeline, predict_failure_mode
from src.report import generate_car_pdf
from src.db import get_data_quality_summary


def test_app_utils_exports():
    """Verify all paths and directory constants exist and are valid."""
    assert os.path.exists(DATA_DIR)
    assert os.path.exists(MODELS_DIR)
    assert os.path.exists(REPORTS_DIR)
    assert CAR_DIR is not None
    assert EXCEL_PATH is not None
    assert POWERBI_DIR is not None

    prods, sales, claims = load_all_data()
    assert len(prods) == 8
    assert len(sales) > 0
    assert len(claims) >= 25000


def test_page1_executive_logic():
    """Verify Page 1 Executive Overview calculations."""
    prods, sales, claims = load_all_data()
    shipped = int(sales["quantity"].sum())
    total_claims = len(claims)
    ppm = (total_claims / shipped) * 1_000_000.0
    cpu = float(claims["repair_cost"].sum()) / shipped
    accruals = calculate_accruals(sales, claims, prods)
    ppit = rank_ppit_car(claims, sales, prods)

    assert ppm > 0
    assert cpu > 0
    assert accruals["total_warranty_reserve"] > 0
    assert len(ppit) > 0


def test_page2_pareto_logic():
    """Verify Page 2 Pareto calculations."""
    prods, sales, claims = load_all_data()
    pareto_fm = calculate_pareto(claims, column="failure_mode")
    pareto_comp = calculate_pareto(claims, column="component")
    pareto_reg = calculate_pareto(claims, column="region")

    assert len(pareto_fm) == 12
    assert "cum_count_pct" in pareto_fm.columns
    assert len(pareto_comp) > 0
    assert len(pareto_reg) > 0


def test_page3_spc_trends_logic():
    """Verify Page 3 SPC Trends calculations."""
    prods, sales, claims = load_all_data()
    sel_prod = prods["product_family"].iloc[0]
    spc_df = compute_monthly_spc_for_product(claims, sales, sel_prod)

    assert not spc_df.empty
    assert "centerline_ppm" in spc_df.columns
    assert "ucl_3sigma" in spc_df.columns


def test_page4_emerging_issues_logic():
    """Verify Page 4 Emerging Issues calculations."""
    prods, sales, claims = load_all_data()
    anomalies = detect_emerging_anomalies(claims, sales, sigma_threshold=2.0)

    assert not anomalies.empty
    top_issue = anomalies.iloc[0]
    drill = get_anomaly_drilldown(claims, top_issue["product_family"], top_issue["failure_mode"])
    assert "by_batch" in drill
    assert "by_region" in drill
    assert "by_component" in drill


def test_page5_reliability_logic():
    """Verify Page 5 Reliability analysis logic."""
    prods, sales, claims = load_all_data()
    weibull_summary = fit_all_products_weibull(claims, sales)
    assert len(weibull_summary) == 8

    sel_prod = weibull_summary["product_family"].iloc[0]
    t_pts, x_ln_t, y_weibull = get_weibull_probability_plot_points(claims, sel_prod)
    assert len(t_pts) > 0
    assert len(x_ln_t) == len(y_weibull)


def test_page6_classifier_logic():
    """Verify Page 6 Classifier prediction."""
    pipeline_path = os.path.join(MODELS_DIR, "failure_mode_pipeline.joblib")
    pipeline = load_classifier_pipeline(pipeline_path)
    res = predict_failure_mode("DC bus capacitor swelling with electrolyte leak", pipeline=pipeline)
    assert "predicted_failure_mode" in res
    assert res["predicted_failure_mode"] in [
        "Capacitor Aging", "Contact Wear", "Insulation Breakdown",
        "Overheating / Fan Failure", "Battery Degradation", "Firmware Fault",
        "Loose Connection / Hotspot", "Trip Unit Malfunction", "Corrosion",
        "PCB Solder Defect", "Surge / MOV Degradation", "Mechanism / Latch Failure",
    ]


def test_page7_ppit_car_pdf():
    """Verify Page 7 PPIT and PDF compilation."""
    prods, sales, claims = load_all_data()
    ppit_df = rank_ppit_car(claims, sales, prods)
    top_item = ppit_df.iloc[0].to_dict()

    test_pdf = os.path.join(CAR_DIR, f"TEST_{top_item['car_id']}.pdf")
    out_path = generate_car_pdf(top_item, test_pdf)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 1000
    if os.path.exists(out_path):
        os.remove(out_path)


def test_page8_data_quality():
    """Verify Page 8 Data Quality Summary."""
    dq_sum = get_data_quality_summary()
    assert dq_sum["cleanliness_score"] > 90
    assert dq_sum["clean_count"] >= 25000
