"""
tests/test_report.py
--------------------
Unit tests for 8D PDF report generation.
"""

import os
import pytest
from src.report import generate_car_pdf


def test_generate_car_pdf(tmp_path):
    target_pdf = os.path.join(tmp_path, "test_CAR_8D.pdf")
    car_data = {
        "car_id": "CAR-TEST-001",
        "product_family": "Online UPS 10kVA",
        "failure_mode": "Capacitor Aging",
        "ppit_score": 92.4,
        "total_warranty_cost": 45000.0,
        "total_claims": 120,
        "recent_ppm": 19500.0,
        "baseline_ppm": 3200.0,
        "sigma_deviation": 5.1,
    }

    out_path = generate_car_pdf(car_data, output_path=target_pdf)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 1000  # valid PDF size
