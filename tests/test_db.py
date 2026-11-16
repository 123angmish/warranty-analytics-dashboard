"""
tests/test_db.py
----------------
Unit tests for data validation, cleaning, and SQLite logging pipeline.
"""

import os
import sqlite3
import pandas as pd
import pytest
from src.db import clean_and_validate_data, get_data_quality_summary, load_dataset


@pytest.fixture
def temp_test_db(tmp_path):
    """Creates a temporary SQLite database with raw test data."""
    db_file = os.path.join(tmp_path, "test_warranty.db")
    conn = sqlite3.connect(db_file)

    products = pd.DataFrame([
        {"product_family": "Online UPS 10kVA", "category": "UPS", "unit_price": 4200.0, "warranty_months": 24, "target_ppm": 3500, "typical_failure_modes": "Capacitor Aging"},
        {"product_family": "Air Circuit Breaker (ACB)", "category": "Switchgear", "unit_price": 4100.0, "warranty_months": 36, "target_ppm": 3800, "typical_failure_modes": "Contact Wear"},
    ])
    products.to_sql("products", conn, index=False)

    sales = pd.DataFrame([
        {"shipment_id": "S1", "product_family": "Online UPS 10kVA", "region": "North India", "ship_date": "2024-01-10", "ship_month": "2024-01", "quantity": 100, "batch_id": "B1", "unit_price": 4200.0},
    ])
    sales.to_sql("sales", conn, index=False)

    # Claims with anomalies:
    # 1 duplicate claim_id (C1 duplicated)
    # 1 invalid date (claim_date before ship_date)
    # 1 inconsistent label ('CAPACITOR_AGING' -> 'Capacitor Aging')
    # 1 missing serial_number
    # 1 extreme cost outlier ($90,000)
    claims = pd.DataFrame([
        {"claim_id": "C1", "product_family": "Online UPS 10kVA", "serial_number": "SN-001", "batch_id": "B1", "region": "North India", "ship_date": "2024-01-10", "claim_date": "2024-02-15", "failure_mode": "Capacitor Aging", "component": "Cap", "repair_cost": 450.0, "labor_cost": 150.0, "parts_cost": 300.0, "claim_status": "Approved", "claim_description": "Cap failed", "customer_type": "Data Center"},
        {"claim_id": "C1", "product_family": "Online UPS 10kVA", "serial_number": "SN-001", "batch_id": "B1", "region": "North India", "ship_date": "2024-01-10", "claim_date": "2024-02-15", "failure_mode": "Capacitor Aging", "component": "Cap", "repair_cost": 450.0, "labor_cost": 150.0, "parts_cost": 300.0, "claim_status": "Approved", "claim_description": "Cap failed", "customer_type": "Data Center"},  # Duplicate
        {"claim_id": "C2", "product_family": "Online UPS 10kVA", "serial_number": "SN-002", "batch_id": "B1", "region": "North India", "ship_date": "2024-01-10", "claim_date": "2023-12-01", "failure_mode": "CAPACITOR_AGING", "component": "Cap", "repair_cost": 450.0, "labor_cost": 150.0, "parts_cost": 300.0, "claim_status": "Approved", "claim_description": "Cap failed", "customer_type": "Data Center"},  # Bad date & bad label
        {"claim_id": "C3", "product_family": "Online UPS 10kVA", "serial_number": None, "batch_id": "B1", "region": "North India", "ship_date": "2024-01-10", "claim_date": "2024-03-01", "failure_mode": "Capacitor Aging", "component": "Cap", "repair_cost": 95000.0, "labor_cost": 25000.0, "parts_cost": 70000.0, "claim_status": "Approved", "claim_description": "Cap failed", "customer_type": None},  # Missing serial & extreme cost & missing cust_type
    ])
    claims.to_sql("claims_raw", conn, index=False)

    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS data_quality_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        logged_at TEXT,
        rule_name TEXT,
        dimension TEXT,
        severity TEXT,
        count_affected INTEGER,
        action_taken TEXT,
        details TEXT
    );
    """)
    conn.commit()
    conn.close()

    return db_file


def test_clean_and_validate_pipeline(temp_test_db):
    clean_df, summary = clean_and_validate_data(temp_test_db)

    # 4 raw records, 1 duplicate dropped -> 3 clean records
    assert summary["total_raw_records"] == 4
    assert summary["total_clean_records"] == 3
    assert summary["records_dropped"] == 1

    # Check deduplication
    assert len(clean_df[clean_df["claim_id"] == "C1"]) == 1

    # Check chronological date rectification (C2 was 2023-12-01 before 2024-01-10 ship_date)
    c2 = clean_df[clean_df["claim_id"] == "C2"].iloc[0]
    assert pd.to_datetime(c2["claim_date"]) >= pd.to_datetime(c2["ship_date"])

    # Check label standardization (C2 had 'CAPACITOR_AGING' -> 'Capacitor Aging')
    assert c2["failure_mode"] == "Capacitor Aging"

    # Check missing serial number imputation (C3 had None)
    c3 = clean_df[clean_df["claim_id"] == "C3"].iloc[0]
    assert c3["serial_number"] is not None
    assert "SN-IMP-C3" in c3["serial_number"]

    # Check customer_type imputation (C3 had None)
    assert c3["customer_type"] == "Commercial Office Park"

    # Check extreme cost outlier capping (C3 was $95,000, should be capped)
    assert c3["repair_cost"] < 10000.0

    # Check DQ summary function
    dq_sum = get_data_quality_summary(temp_test_db)
    assert dq_sum["raw_count"] == 4
    assert dq_sum["clean_count"] == 3
    assert not dq_sum["log_df"].empty
