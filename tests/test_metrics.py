"""
tests/test_metrics.py
---------------------
Unit tests for reliability metrics using hand-computed deterministic fixtures.
Tests:
  - calculate_ppm (overall and grouped)
  - calculate_cpu (overall and grouped)
  - calculate_pareto (count, cost, cumulative %)
  - calculate_accruals (warranty reserve liability)
  - detect_ppm_trends (Shewhart control limits & 2-sigma/3-sigma spikes)
  - rank_ppit_car (PPIT priority ranking & CAR qualification)
"""

import pandas as pd
import pytest
from src.metrics import (
    calculate_ppm,
    calculate_cpu,
    calculate_pareto,
    calculate_accruals,
    detect_ppm_trends,
    rank_ppit_car,
)


@pytest.fixture
def sample_sales_and_claims():
    """Small hand-computed dataset for deterministic testing."""
    sales = pd.DataFrame([
        {"shipment_id": "S1", "product_family": "UPS-A", "region": "North", "ship_date": "2024-01-15", "ship_month": "2024-01", "quantity": 1000},
        {"shipment_id": "S2", "product_family": "UPS-A", "region": "South", "ship_date": "2024-01-20", "ship_month": "2024-01", "quantity": 4000},
        {"shipment_id": "S3", "product_family": "UPS-B", "region": "North", "ship_date": "2024-02-10", "ship_month": "2024-02", "quantity": 5000},
    ])

    claims = pd.DataFrame([
        # 5 claims for UPS-A in North
        {"claim_id": "C1", "product_family": "UPS-A", "region": "North", "ship_date": "2024-01-15", "ship_month": "2024-01", "claim_date": "2024-03-01", "failure_mode": "Capacitor Aging", "repair_cost": 200.0},
        {"claim_id": "C2", "product_family": "UPS-A", "region": "North", "ship_date": "2024-01-15", "ship_month": "2024-01", "claim_date": "2024-03-05", "failure_mode": "Capacitor Aging", "repair_cost": 300.0},
        {"claim_id": "C3", "product_family": "UPS-A", "region": "North", "ship_date": "2024-01-15", "ship_month": "2024-01", "claim_date": "2024-03-10", "failure_mode": "Capacitor Aging", "repair_cost": 150.0},
        {"claim_id": "C4", "product_family": "UPS-A", "region": "North", "ship_date": "2024-01-15", "ship_month": "2024-01", "claim_date": "2024-03-15", "failure_mode": "Fan Failure", "repair_cost": 150.0},
        {"claim_id": "C5", "product_family": "UPS-A", "region": "North", "ship_date": "2024-01-15", "ship_month": "2024-01", "claim_date": "2024-03-20", "failure_mode": "Fan Failure", "repair_cost": 200.0},
        # 10 claims for UPS-A in South
        *([{"claim_id": f"CS{i}", "product_family": "UPS-A", "region": "South", "ship_date": "2024-01-20", "ship_month": "2024-01", "claim_date": "2024-04-01", "failure_mode": "Contact Wear", "repair_cost": 500.0} for i in range(10)]),
        # 10 claims for UPS-B in North
        *([{"claim_id": f"CB{i}", "product_family": "UPS-B", "region": "North", "ship_date": "2024-02-10", "ship_month": "2024-02", "claim_date": "2024-04-10", "failure_mode": "Contact Wear", "repair_cost": 400.0} for i in range(10)]),
    ])

    products = pd.DataFrame([
        {"product_family": "UPS-A", "category": "UPS", "unit_price": 2000.0, "warranty_months": 24, "target_ppm": 3000},
        {"product_family": "UPS-B", "category": "UPS", "unit_price": 3000.0, "warranty_months": 24, "target_ppm": 2000},
    ])

    return sales, claims, products


def test_calculate_ppm_overall(sample_sales_and_claims):
    sales, claims, _ = sample_sales_and_claims
    # Total units = 1000 + 4000 + 5000 = 10,000 units
    # Total claims = 5 + 10 + 10 = 25 claims
    # Expected global PPM = (25 / 10000) * 1,000,000 = 2,500.0 PPM
    res = calculate_ppm(claims, sales)
    assert len(res) == 1
    assert res.iloc[0]["units_shipped"] == 10000
    assert res.iloc[0]["total_claims"] == 25
    assert res.iloc[0]["ppm"] == 2500.0


def test_calculate_ppm_by_product_and_region(sample_sales_and_claims):
    sales, claims, _ = sample_sales_and_claims
    res = calculate_ppm(claims, sales, group_by=["product_family", "region"])
    # UPS-A North: 5 claims / 1000 units -> 5,000.0 PPM
    row_an = res[(res["product_family"] == "UPS-A") & (res["region"] == "North")].iloc[0]
    assert row_an["units_shipped"] == 1000
    assert row_an["total_claims"] == 5
    assert row_an["ppm"] == 5000.0

    # UPS-A South: 10 claims / 4000 units -> 2,500.0 PPM
    row_as = res[(res["product_family"] == "UPS-A") & (res["region"] == "South")].iloc[0]
    assert row_as["units_shipped"] == 4000
    assert row_as["total_claims"] == 10
    assert row_as["ppm"] == 2500.0


def test_calculate_cpu(sample_sales_and_claims):
    sales, claims, _ = sample_sales_and_claims
    # Total cost = (200+300+150+150+200) + (10*500) + (10*400) = 1,000 + 5,000 + 4,000 = $10,000
    # Total units = 10,000
    # Overall CPU = 10000 / 10000 = $1.00
    res_overall = calculate_cpu(claims, sales)
    assert res_overall.iloc[0]["total_warranty_cost"] == 10000.0
    assert res_overall.iloc[0]["cpu"] == 1.0

    # Grouped by product
    res_prod = calculate_cpu(claims, sales, group_by="product_family")
    # UPS-A: units=5000, cost=6000 -> CPU = 6000/5000 = 1.20
    row_a = res_prod[res_prod["product_family"] == "UPS-A"].iloc[0]
    assert row_a["units_shipped"] == 5000
    assert row_a["total_warranty_cost"] == 6000.0
    assert row_a["cpu"] == 1.20


def test_calculate_pareto(sample_sales_and_claims):
    _, claims, _ = sample_sales_and_claims
    # Claims count per failure mode:
    # Contact Wear: 20 claims, cost = (10*500) + (10*400) = 9,000
    # Capacitor Aging: 3 claims, cost = 200+300+150 = 650
    # Fan Failure: 2 claims, cost = 150+200 = 350
    # Total claims = 25
    pareto = calculate_pareto(claims, column="failure_mode")
    assert pareto.iloc[0]["failure_mode"] == "Contact Wear"
    assert pareto.iloc[0]["claim_count"] == 20
    assert pareto.iloc[0]["count_pct"] == 80.0
    assert pareto.iloc[0]["cum_count_pct"] == 80.0
    assert pareto.iloc[0]["total_cost"] == 9000.0

    assert pareto.iloc[1]["failure_mode"] == "Capacitor Aging"
    assert pareto.iloc[1]["claim_count"] == 3
    assert pareto.iloc[1]["cum_count_pct"] == 92.0

    assert pareto.iloc[2]["cum_count_pct"] == 100.0


def test_calculate_accruals():
    sales = pd.DataFrame([
        {"shipment_id": "S1", "product_family": "UPS-A", "ship_date": "2025-01-01", "ship_month": "2025-01", "quantity": 1000},
    ])
    claims = pd.DataFrame([
        {"claim_id": "C1", "product_family": "UPS-A", "ship_date": "2025-01-01", "ship_month": "2025-01", "repair_cost": 2500.0},
    ])
    products = pd.DataFrame([
        {"product_family": "UPS-A", "warranty_months": 24, "unit_price": 1000.0},
    ])
    # Trailing CPU: 2500 / 1000 = $2.50 per unit
    # Expected liability: 1000 * 2.50 = $2500.0
    # Claims paid = $2500.0
    # Remaining accrual = $0.0
    res = calculate_accruals(sales, claims, products, as_of_date="2025-06-01")
    assert res["total_expected_liability"] == 2500.0
    assert res["total_claims_paid_active"] == 2500.0
    assert res["total_warranty_reserve"] == 0.0


def test_detect_ppm_trends():
    # Build 27 months: 24 baseline months with 1 claim per 1000 units (1000 PPM),
    # then 3 recent months with 10 claims per 1000 units (10,000 PPM).
    sales_rows, claims_rows = [], []
    for m in range(1, 28):
        y = 2023 + (m - 1) // 12
        month_num = ((m - 1) % 12) + 1
        m_str = f"{y}-{month_num:02d}"
        sales_rows.append({"product_family": "UPS-A", "ship_month": m_str, "quantity": 1000})

        n_claims = 10 if m >= 25 else 1
        for c in range(n_claims):
            claims_rows.append({
                "claim_id": f"C_{m}_{c}",
                "product_family": "UPS-A",
                "failure_mode": "Capacitor Aging",
                "ship_month": m_str,
                "repair_cost": 300.0,
            })

    s_df = pd.DataFrame(sales_rows)
    c_df = pd.DataFrame(claims_rows)

    trends = detect_ppm_trends(c_df, s_df, group_col="failure_mode", baseline_months_count=24, recent_months_count=3)
    assert not trends.empty
    top_issue = trends.iloc[0]
    assert top_issue["failure_mode"] == "Capacitor Aging"
    assert top_issue["recent_3m_ppm"] == 10000.0
    assert bool(top_issue["is_anomaly"]) is True
    assert "OUT_OF_CONTROL" in top_issue["status"] or "SPIKE_DETECTED" in top_issue["status"]


def test_rank_ppit_car(sample_sales_and_claims):
    sales, claims, products = sample_sales_and_claims
    ranking = rank_ppit_car(claims, sales, products)
    assert not ranking.empty
    assert "ppit_score" in ranking.columns
    assert "car_status" in ranking.columns
    assert "car_id" in ranking.columns
    assert ranking.iloc[0]["ppit_rank"] == 1
    # Check that highest ranking item has top score
    assert ranking.iloc[0]["ppit_score"] >= ranking.iloc[-1]["ppit_score"]
