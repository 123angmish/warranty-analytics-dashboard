"""
tests/test_export.py
--------------------
Unit tests for Excel workbook and Power BI star-schema export.
"""

import os
import pandas as pd
import pytest
from src.export import export_excel_workbook, export_powerbi_star_schema


def test_export_excel_workbook(tmp_path):
    target_xlsx = os.path.join(tmp_path, "test_summary.xlsx")
    products = pd.DataFrame([
        {"product_family": "UPS-A", "category": "UPS", "unit_price": 1000.0, "warranty_months": 24, "target_ppm": 3000},
    ])
    sales = pd.DataFrame([
        {"shipment_id": "S1", "product_family": "UPS-A", "region": "North", "ship_date": "2024-01-01", "ship_month": "2024-01", "quantity": 500, "unit_price": 1000.0},
    ])
    claims = pd.DataFrame([
        {"claim_id": "C1", "product_family": "UPS-A", "region": "North", "ship_date": "2024-01-01", "claim_date": "2024-03-01", "failure_mode": "Capacitor Aging", "component": "Cap", "repair_cost": 250.0, "labor_cost": 100.0, "parts_cost": 150.0, "claim_status": "Approved"},
    ])

    out = export_excel_workbook(claims, sales, products, output_path=target_xlsx)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 2000


def test_export_powerbi_star_schema(tmp_path, monkeypatch):
    import src.export
    monkeypatch.setattr(src.export, "POWERBI_DIR", str(tmp_path))

    products = pd.DataFrame([
        {"product_family": "UPS-A", "category": "UPS", "unit_price": 1000.0, "warranty_months": 24, "target_ppm": 3000},
    ])
    sales = pd.DataFrame([
        {"shipment_id": "S1", "product_family": "UPS-A", "region": "North", "ship_date": "2024-01-01", "ship_month": "2024-01", "quantity": 500, "unit_price": 1000.0},
    ])
    claims = pd.DataFrame([
        {"claim_id": "C1", "product_family": "UPS-A", "region": "North", "ship_date": "2024-01-01", "claim_date": "2024-03-01", "failure_mode": "Capacitor Aging", "component": "Cap", "repair_cost": 250.0, "labor_cost": 100.0, "parts_cost": 150.0, "claim_status": "Approved", "batch_id": "B1", "serial_number": "SN1", "customer_type": "Commercial"},
    ])

    paths = export_powerbi_star_schema(claims, sales, products)
    for name, p in paths.items():
        assert os.path.exists(p)
        assert os.path.getsize(p) > 0
