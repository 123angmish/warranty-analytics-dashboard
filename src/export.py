"""
src/export.py
-------------
Automated Data Export Engine for Excel and Power BI:
  1. Generates `reports/warranty_summary.xlsx` using `openpyxl` with styled sheets:
     - Executive Summary
     - Top N Pareto
     - PPM by Product Family
     - Weibull Reliability Results
     - PPIT / CAR Ranking
     - Data Quality Audit Log
  2. Generates Star-Schema CSVs in `data/powerbi/`:
     - fact_claims.csv
     - fact_sales.csv
     - dim_product.csv
     - dim_region.csv
     - dim_date.csv
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd
from src.db import load_dataset, get_data_quality_summary
from src.metrics import calculate_ppm, calculate_cpu, calculate_accruals, calculate_pareto, rank_ppit_car
from src.reliability import fit_all_products_weibull

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
POWERBI_DIR = os.path.join(DATA_DIR, "powerbi")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
EXCEL_PATH = os.path.join(REPORTS_DIR, "warranty_summary.xlsx")


def style_header_row(ws, col_count: int, title: str = ""):
    """Applies corporate styling to header row in Excel worksheet."""
    header_fill = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    thin_border = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1"),
    )

    for col in range(1, col_count + 1):
        cell = ws.cell(row=1, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border


def auto_fit_columns(ws):
    """Adjusts column widths based on maximum content length."""
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            val = cell.value
            if val is not None:
                max_len = max(max_len, len(str(val)))
        ws.column_dimensions[col_letter].width = max(12, min(max_len + 3, 40))


def export_excel_workbook(
    claims_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    products_df: pd.DataFrame,
    output_path: str = EXCEL_PATH,
) -> str:
    """
    Creates reports/warranty_summary.xlsx containing 6 styled sheets with complete analytics.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    wb = openpyxl.Workbook()

    # ---------------------------------------------------------
    # Sheet 1: Executive Summary
    # ---------------------------------------------------------
    ws_exec = wb.active
    ws_exec.title = "Executive Summary"

    total_shipped = int(sales_df["quantity"].sum())
    total_claims = len(claims_df)
    total_warranty_cost = float(claims_df["repair_cost"].sum())
    global_ppm = round((total_claims / total_shipped) * 1_000_000.0, 1)
    global_cpu = round(total_warranty_cost / total_shipped, 2)
    accruals = calculate_accruals(sales_df, claims_df, products_df)
    total_reserve = accruals["total_warranty_reserve"]

    exec_kpis = [
        ["KPI Metric", "Value", "Unit / Context", "Status Target"],
        ["Total Valid Claims", f"{total_claims:,}", "Warranty Incidents", "Monitored"],
        ["Total Units Shipped", f"{total_shipped:,}", "Units (3-Year Fleet)", "Active Population"],
        ["Global Claim Rate (PPM)", f"{global_ppm:,.1f}", "Parts Per Million", "< 60,000 PPM"],
        ["Global Cost Per Unit (CPU)", f"${global_cpu:,.2f}", "USD / Shipped Unit", "< $25.00"],
        ["Total Warranty Expense", f"${total_warranty_cost:,.2f}", "USD Paid to Date", "Budget Limit"],
        ["Active Warranty Reserve Accrual", f"${total_reserve:,.2f}", "USD Balance Sheet Liability", "Funded"],
        ["Active Product Families", f"{len(products_df)}", "Families", "Standard Portfolio"],
        ["Monitored Failure Modes", "12", "Physics-of-Failure Modes", "100% Tracked"],
    ]

    for r_idx, row in enumerate(exec_kpis, 1):
        for c_idx, val in enumerate(row, 1):
            cell = ws_exec.cell(row=r_idx, column=c_idx, value=val)
            if r_idx == 1:
                pass  # styled below
            else:
                cell.font = Font(name="Calibri", size=10)
                cell.alignment = Alignment(vertical="center")

    style_header_row(ws_exec, len(exec_kpis[0]))
    auto_fit_columns(ws_exec)

    # ---------------------------------------------------------
    # Sheet 2: Top N Pareto
    # ---------------------------------------------------------
    ws_pareto = wb.create_sheet(title="Top N Pareto")
    pareto_fm = calculate_pareto(claims_df, column="failure_mode")

    pareto_headers = [
        "Failure Mode", "Claim Count", "Total Warranty Cost ($)", "Avg Cost / Claim ($)",
        "Count %", "Cumulative Count %", "Cost %", "Cumulative Cost %"
    ]
    for c_idx, h in enumerate(pareto_headers, 1):
        ws_pareto.cell(row=1, column=c_idx, value=h)

    for r_idx, row in pareto_fm.iterrows():
        ws_pareto.cell(row=r_idx + 2, column=1, value=row["failure_mode"])
        ws_pareto.cell(row=r_idx + 2, column=2, value=int(row["claim_count"]))
        ws_pareto.cell(row=r_idx + 2, column=3, value=float(row["total_cost"]))
        ws_pareto.cell(row=r_idx + 2, column=4, value=float(row["avg_cost"]))
        ws_pareto.cell(row=r_idx + 2, column=5, value=float(row["count_pct"]))
        ws_pareto.cell(row=r_idx + 2, column=6, value=float(row["cum_count_pct"]))
        ws_pareto.cell(row=r_idx + 2, column=7, value=float(row["cost_pct"]))
        ws_pareto.cell(row=r_idx + 2, column=8, value=float(row["cum_cost_pct"]))

    style_header_row(ws_pareto, len(pareto_headers))
    auto_fit_columns(ws_pareto)

    # ---------------------------------------------------------
    # Sheet 3: PPM by Product Family
    # ---------------------------------------------------------
    ws_ppm = wb.create_sheet(title="PPM by Product")
    prod_ppm = calculate_ppm(claims_df, sales_df, group_by="product_family")

    ppm_headers = ["Product Family", "Units Shipped", "Total Claims", "PPM"]
    for c_idx, h in enumerate(ppm_headers, 1):
        ws_ppm.cell(row=1, column=c_idx, value=h)

    for r_idx, row in prod_ppm.iterrows():
        ws_ppm.cell(row=r_idx + 2, column=1, value=row["product_family"])
        ws_ppm.cell(row=r_idx + 2, column=2, value=int(row["units_shipped"]))
        ws_ppm.cell(row=r_idx + 2, column=3, value=int(row["total_claims"]))
        ws_ppm.cell(row=r_idx + 2, column=4, value=float(row["ppm"]))

    style_header_row(ws_ppm, len(ppm_headers))
    auto_fit_columns(ws_ppm)

    # ---------------------------------------------------------
    # Sheet 4: Weibull Reliability Results
    # ---------------------------------------------------------
    ws_weibull = wb.create_sheet(title="Weibull Reliability")
    weibull_df = fit_all_products_weibull(claims_df, sales_df)

    w_headers = [
        "Product Family", "Shape (Beta)", "Scale (Eta Days)", "B10 Life (Days)",
        "B50 Median Life (Days)", "MTTF (Days)", "Failure Regime", "Total Shipped", "Failures Observed"
    ]
    for c_idx, h in enumerate(w_headers, 1):
        ws_weibull.cell(row=1, column=c_idx, value=h)

    for r_idx, row in weibull_df.iterrows():
        ws_weibull.cell(row=r_idx + 2, column=1, value=row["product_family"])
        ws_weibull.cell(row=r_idx + 2, column=2, value=float(row["beta_shape"]))
        ws_weibull.cell(row=r_idx + 2, column=3, value=float(row["eta_scale_days"]))
        ws_weibull.cell(row=r_idx + 2, column=4, value=float(row["b10_days"]))
        ws_weibull.cell(row=r_idx + 2, column=5, value=float(row["b50_days"]))
        ws_weibull.cell(row=r_idx + 2, column=6, value=float(row["mttf_days"]))
        ws_weibull.cell(row=r_idx + 2, column=7, value=str(row["failure_regime"]))
        ws_weibull.cell(row=r_idx + 2, column=8, value=int(row["total_shipped"]))
        ws_weibull.cell(row=r_idx + 2, column=9, value=int(row["total_failures"]))

    style_header_row(ws_weibull, len(w_headers))
    auto_fit_columns(ws_weibull)

    # ---------------------------------------------------------
    # Sheet 5: PPIT CAR Ranking
    # ---------------------------------------------------------
    ws_ppit = wb.create_sheet(title="PPIT CAR Ranking")
    ppit_df = rank_ppit_car(claims_df, sales_df, products_df)

    ppit_headers = [
        "PPIT Rank", "CAR ID", "Product Family", "Failure Mode", "PPIT Score",
        "CAR Status", "Total Claims", "Total Cost ($)", "Avg Claim Cost ($)", "Sigma Shift"
    ]
    for c_idx, h in enumerate(ppit_headers, 1):
        ws_ppit.cell(row=1, column=c_idx, value=h)

    for r_idx, row in ppit_df.iterrows():
        ws_ppit.cell(row=r_idx + 2, column=1, value=int(row["ppit_rank"]))
        ws_ppit.cell(row=r_idx + 2, column=2, value=str(row["car_id"]))
        ws_ppit.cell(row=r_idx + 2, column=3, value=str(row["product_family"]))
        ws_ppit.cell(row=r_idx + 2, column=4, value=str(row["failure_mode"]))
        ws_ppit.cell(row=r_idx + 2, column=5, value=float(row["ppit_score"]))
        ws_ppit.cell(row=r_idx + 2, column=6, value=str(row["car_status"]))
        ws_ppit.cell(row=r_idx + 2, column=7, value=int(row["total_claims"]))
        ws_ppit.cell(row=r_idx + 2, column=8, value=float(row["total_warranty_cost"]))
        ws_ppit.cell(row=r_idx + 2, column=9, value=float(row["avg_claim_cost"]))
        ws_ppit.cell(row=r_idx + 2, column=10, value=float(row["sigma_deviation"]))

    style_header_row(ws_ppit, len(ppit_headers))
    auto_fit_columns(ws_ppit)

    # ---------------------------------------------------------
    # Sheet 6: Data Quality Audit
    # ---------------------------------------------------------
    ws_dq = wb.create_sheet(title="Data Quality Audit")
    dq_sum = get_data_quality_summary()
    dq_log_df = dq_sum.get("log_df", pd.DataFrame())

    dq_headers = ["Rule Name", "Dimension", "Severity", "Records Affected", "Action Taken", "Details"]
    for c_idx, h in enumerate(dq_headers, 1):
        ws_dq.cell(row=1, column=c_idx, value=h)

    if not dq_log_df.empty:
        for r_idx, row in dq_log_df.iterrows():
            ws_dq.cell(row=r_idx + 2, column=1, value=str(row["rule_name"]))
            ws_dq.cell(row=r_idx + 2, column=2, value=str(row["dimension"]))
            ws_dq.cell(row=r_idx + 2, column=3, value=str(row["severity"]))
            ws_dq.cell(row=r_idx + 2, column=4, value=int(row["count_affected"]))
            ws_dq.cell(row=r_idx + 2, column=5, value=str(row["action_taken"]))
            ws_dq.cell(row=r_idx + 2, column=6, value=str(row["details"]))

    style_header_row(ws_dq, len(dq_headers))
    auto_fit_columns(ws_dq)

    wb.save(output_path)
    print(f"Exported comprehensive Excel workbook -> {output_path}")
    return output_path


def export_powerbi_star_schema(
    claims_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> Dict[str, str]:
    """
    Exports clean star-schema CSVs in data/powerbi/ for Power BI Desktop:
      - dim_product.csv
      - dim_region.csv
      - dim_date.csv
      - fact_claims.csv
      - fact_sales.csv
    """
    os.makedirs(POWERBI_DIR, exist_ok=True)
    paths = {}

    # 1. dim_product
    dim_product = products_df.copy()
    dim_product["product_id"] = [f"PRD-{i+1:02d}" for i in range(len(dim_product))]
    prod_id_map = dim_product.set_index("product_family")["product_id"].to_dict()
    p_path = os.path.join(POWERBI_DIR, "dim_product.csv")
    dim_product[["product_id", "product_family", "category", "unit_price", "warranty_months", "target_ppm"]].to_csv(p_path, index=False)
    paths["dim_product"] = p_path

    # 2. dim_region
    regions = sorted(sales_df["region"].unique())
    dim_region = pd.DataFrame({
        "region_id": [f"REG-{i+1:02d}" for i in range(len(regions))],
        "region_name": regions,
        "territory_tier": ["Tier 1 Core" if r in ["West India", "North India"] else "Tier 2 Expansion" for r in regions],
    })
    region_id_map = dim_region.set_index("region_name")["region_id"].to_dict()
    r_path = os.path.join(POWERBI_DIR, "dim_region.csv")
    dim_region.to_csv(r_path, index=False)
    paths["dim_region"] = r_path

    # 3. dim_date
    start_dt = pd.to_datetime("2023-01-01")
    end_dt = pd.to_datetime("2026-02-01")
    date_range = pd.date_range(start_dt, end_dt, freq="D")
    dim_date = pd.DataFrame({
        "date_id": date_range.strftime("%Y%m%d").astype(int),
        "full_date": date_range.strftime("%Y-%m-%d"),
        "year": date_range.year,
        "quarter": [f"Q{q}" for q in date_range.quarter],
        "month_number": date_range.month,
        "month_name": date_range.strftime("%B"),
        "year_month": date_range.strftime("%Y-%m"),
    })
    d_path = os.path.join(POWERBI_DIR, "dim_date.csv")
    dim_date.to_csv(d_path, index=False)
    paths["dim_date"] = d_path

    # 4. fact_sales
    fact_sales = sales_df.copy()
    fact_sales["product_id"] = fact_sales["product_family"].map(prod_id_map)
    fact_sales["region_id"] = fact_sales["region"].map(region_id_map)
    fact_sales["date_id"] = pd.to_datetime(fact_sales["ship_date"]).dt.strftime("%Y%m%d").astype(int)
    if "batch_id" not in fact_sales.columns:
        fact_sales["batch_id"] = "BATCH-GEN"
    fact_sales["total_revenue"] = (fact_sales["quantity"] * fact_sales["unit_price"]).round(2)
    s_cols = [
        "shipment_id", "product_id", "region_id", "date_id", "batch_id",
        "quantity", "unit_price", "total_revenue"
    ]
    s_path = os.path.join(POWERBI_DIR, "fact_sales.csv")
    fact_sales[s_cols].to_csv(s_path, index=False)
    paths["fact_sales"] = s_path

    # 5. fact_claims
    fact_claims = claims_df.copy()
    fact_claims["product_id"] = fact_claims["product_family"].map(prod_id_map)
    fact_claims["region_id"] = fact_claims["region"].map(region_id_map)
    fact_claims["claim_date_id"] = pd.to_datetime(fact_claims["claim_date"]).dt.strftime("%Y%m%d").astype(int)
    fact_claims["ship_date_id"] = pd.to_datetime(fact_claims["ship_date"]).dt.strftime("%Y%m%d").astype(int)
    if "batch_id" not in fact_claims.columns:
        fact_claims["batch_id"] = "BATCH-GEN"
    if "serial_number" not in fact_claims.columns:
        fact_claims["serial_number"] = "SN-GEN"
    if "customer_type" not in fact_claims.columns:
        fact_claims["customer_type"] = "Commercial Office Park"
    c_cols = [
        "claim_id", "product_id", "region_id", "claim_date_id", "ship_date_id",
        "batch_id", "serial_number", "failure_mode", "component",
        "parts_cost", "labor_cost", "repair_cost", "claim_status", "customer_type"
    ]
    c_path = os.path.join(POWERBI_DIR, "fact_claims.csv")
    fact_claims[c_cols].to_csv(c_path, index=False)
    paths["fact_claims"] = c_path

    print(f"Exported Power BI Star-Schema tables to -> {POWERBI_DIR}")
    return paths


def main():
    print("=" * 60)
    print("RUNNING EXPORTS: EXCEL SUMMARY WORKBOOK & POWER BI STAR-SCHEMA")
    print("=" * 60)
    claims_df = load_dataset("claims_clean")
    sales_df = load_dataset("sales")
    products_df = load_dataset("products")

    export_excel_workbook(claims_df, sales_df, products_df)
    export_powerbi_star_schema(claims_df, sales_df, products_df)
    print("Exports complete.")


if __name__ == "__main__":
    main()
