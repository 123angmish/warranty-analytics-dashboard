"""
src/metrics.py
--------------
Reliability & Warranty Metrics Computation Engine for Electrical Equipment.
Calculates:
  1. Parts Per Million (PPM) = (Total Claims / Units Shipped) * 1,000,000
  2. Cost Per Unit (CPU) = Total Warranty Repair Cost / Units Shipped
  3. Warranty Reserve Accruals (Estimated Warranty Liability)
  4. Pareto Analysis (Count & Cost Pareto with Cumulative %)
  5. Statistical Process Control (Shewhart Trend Detection, UCL/LCL/2-Sigma Spikes)
  6. Problem Priority Identification Table (PPIT) & CAR Ranking
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 1. Parts Per Million (PPM)
# -----------------------------------------------------------------------------
def calculate_ppm(
    claims_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    group_by: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Computes Parts Per Million (PPM) warranty claim rate:
        PPM = (Total Valid Claims / Total Units Shipped) * 1,000,000

    Args:
        claims_df: DataFrame containing warranty claims (must contain 'claim_id' and grouping columns)
        sales_df: DataFrame containing shipments (must contain 'quantity' and grouping columns)
        group_by: Column or list of columns to aggregate by (e.g., 'product_family', 'region', 'ship_month')

    Returns:
        DataFrame with grouping column(s), units_shipped, total_claims, and ppm (rounded to 1 decimal).
    """
    if group_by is None:
        total_shipped = int(sales_df["quantity"].sum())
        total_claims = int(len(claims_df))
        ppm = round((total_claims / total_shipped) * 1_000_000, 1) if total_shipped > 0 else 0.0
        return pd.DataFrame([{
            "units_shipped": total_shipped,
            "total_claims": total_claims,
            "ppm": ppm,
        }])

    if isinstance(group_by, str):
        group_by = [group_by]

    # Aggregate sales
    sales_agg = sales_df.groupby(group_by, as_index=False)["quantity"].sum()
    sales_agg.rename(columns={"quantity": "units_shipped"}, inplace=True)

    # Aggregate claims
    claims_agg = claims_df.groupby(group_by, as_index=False)["claim_id"].count()
    claims_agg.rename(columns={"claim_id": "total_claims"}, inplace=True)

    # Outer join to capture cohorts with sales and 0 claims
    merged = pd.merge(sales_agg, claims_agg, on=group_by, how="left")
    merged["total_claims"] = merged["total_claims"].fillna(0).astype(int)
    merged["units_shipped"] = merged["units_shipped"].fillna(0).astype(int)

    merged["ppm"] = np.where(
        merged["units_shipped"] > 0,
        (merged["total_claims"] / merged["units_shipped"]) * 1_000_000.0,
        0.0,
    )
    merged["ppm"] = merged["ppm"].round(1)
    return merged.sort_values(by=group_by).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 2. Cost Per Unit (CPU)
# -----------------------------------------------------------------------------
def calculate_cpu(
    claims_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    group_by: Optional[Union[str, List[str]]] = None,
    cost_col: str = "repair_cost",
) -> pd.DataFrame:
    """
    Computes Cost Per Unit (CPU):
        CPU = Total Warranty Cost / Total Units Shipped

    Args:
        claims_df: DataFrame containing claims and cost column
        sales_df: DataFrame containing shipments with 'quantity'
        group_by: Column or list of columns to group by
        cost_col: Name of cost column (default 'repair_cost')

    Returns:
        DataFrame with grouping column(s), units_shipped, total_warranty_cost, and cpu (currency).
    """
    if group_by is None:
        total_shipped = int(sales_df["quantity"].sum())
        total_cost = round(float(claims_df[cost_col].sum()), 2)
        cpu = round(total_cost / total_shipped, 2) if total_shipped > 0 else 0.0
        return pd.DataFrame([{
            "units_shipped": total_shipped,
            "total_warranty_cost": total_cost,
            "cpu": cpu,
        }])

    if isinstance(group_by, str):
        group_by = [group_by]

    sales_agg = sales_df.groupby(group_by, as_index=False)["quantity"].sum()
    sales_agg.rename(columns={"quantity": "units_shipped"}, inplace=True)

    claims_agg = claims_df.groupby(group_by, as_index=False)[cost_col].sum()
    claims_agg.rename(columns={cost_col: "total_warranty_cost"}, inplace=True)

    merged = pd.merge(sales_agg, claims_agg, on=group_by, how="left")
    merged["total_warranty_cost"] = merged["total_warranty_cost"].fillna(0.0).round(2)
    merged["units_shipped"] = merged["units_shipped"].fillna(0).astype(int)

    merged["cpu"] = np.where(
        merged["units_shipped"] > 0,
        merged["total_warranty_cost"] / merged["units_shipped"],
        0.0,
    )
    merged["cpu"] = merged["cpu"].round(2)
    return merged.sort_values(by=group_by).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 3. Warranty Accruals (Estimated Warranty Liability)
# -----------------------------------------------------------------------------
def calculate_accruals(
    sales_df: pd.DataFrame,
    claims_df: pd.DataFrame,
    products_df: pd.DataFrame,
    as_of_date: Optional[str] = None,
    rolling_cpu_window_months: int = 12,
    cost_col: str = "repair_cost",
) -> Dict[str, Union[float, pd.DataFrame]]:
    """
    Computes estimated warranty liability (Accruals / Warranty Reserve) for active products.
    
    Standard Industry Methodology:
      1. For each product family, determine the rolling CPU over the trailing 12 months.
      2. Identify all shipment cohorts that are still currently within active warranty
         (e.g., ship_date + warranty_months >= as_of_date).
      3. For each active cohort:
         - Expected Total Warranty Liability = Units Shipped * Rolling CPU
         - Actual Warranty Claims Paid = Sum of claims paid for that cohort to date
         - Net Remaining Accrual Reserve = max(0.0, Expected Total Liability - Actual Paid)
      4. Total Corporate Warranty Reserve = Sum of Net Remaining Accrual Reserves across all active cohorts.

    Returns:
        Dictionary with:
          - 'total_warranty_reserve': Total dollars required in warranty reserve
          - 'total_expected_liability': Gross expected warranty expense for active cohorts
          - 'total_claims_paid_active': Total paid to date on currently active cohorts
          - 'cohort_accruals_df': Breakdown DataFrame per product and cohort
    """
    # Create copies
    s_df = sales_df.copy()
    c_df = claims_df.copy()

    s_df["ship_date_dt"] = pd.to_datetime(s_df["ship_date"])
    c_df["ship_date_dt"] = pd.to_datetime(c_df["ship_date"])

    if "ship_month" not in s_df.columns:
        s_df["ship_month"] = s_df["ship_date"].astype(str).str.slice(0, 7)
    if "ship_month" not in c_df.columns:
        c_df["ship_month"] = c_df["ship_date"].astype(str).str.slice(0, 7)

    if as_of_date is None:
        as_of_dt = s_df["ship_date_dt"].max()
    else:
        as_of_dt = pd.to_datetime(as_of_date)

    # Product warranty lookup
    warranty_map = products_df.set_index("product_family")["warranty_months"].to_dict()

    # Calculate rolling baseline CPU per product family over trailing 12 months
    baseline_cutoff = as_of_dt - pd.DateOffset(months=rolling_cpu_window_months)
    recent_sales = s_df[s_df["ship_date_dt"] >= baseline_cutoff]
    recent_claims = c_df[c_df["ship_date_dt"] >= baseline_cutoff]

    prod_cpu = {}
    for prod in products_df["product_family"].unique():
        p_sales = recent_sales[recent_sales["product_family"] == prod]["quantity"].sum()
        p_costs = recent_claims[recent_claims["product_family"] == prod][cost_col].sum()
        if p_sales > 0:
            prod_cpu[prod] = p_costs / p_sales
        else:
            # Fallback to overall CPU for product
            all_p_sales = s_df[s_df["product_family"] == prod]["quantity"].sum()
            all_p_costs = c_df[c_df["product_family"] == prod][cost_col].sum()
            prod_cpu[prod] = (all_p_costs / all_p_sales) if all_p_sales > 0 else 25.0

    # Evaluate active cohorts
    cohort_rows = []
    # Group sales by product_family and ship_month
    sales_cohorts = s_df.groupby(["product_family", "ship_month"], as_index=False).agg({
        "quantity": "sum",
        "ship_date_dt": "min",
    })

    claims_paid_cohorts = c_df.groupby(["product_family", "ship_month"], as_index=False)[cost_col].sum()
    claims_paid_cohorts.rename(columns={cost_col: "claims_paid"}, inplace=True)

    merged_cohorts = pd.merge(sales_cohorts, claims_paid_cohorts, on=["product_family", "ship_month"], how="left")
    merged_cohorts["claims_paid"] = merged_cohorts["claims_paid"].fillna(0.0)

    for _, row in merged_cohorts.iterrows():
        prod = row["product_family"]
        ship_month = row["ship_month"]
        units = row["quantity"]
        claims_paid = row["claims_paid"]
        ship_dt = row["ship_date_dt"]

        w_months = warranty_map.get(prod, 24)
        warranty_end_dt = ship_dt + pd.DateOffset(months=w_months)

        # Is cohort still within warranty?
        is_active = warranty_end_dt >= as_of_dt
        if is_active:
            unit_cpu = prod_cpu.get(prod, 30.0)
            expected_liability = round(units * unit_cpu, 2)
            # Net accrual cannot be negative (if claims already exceed expected, accrual is exhausted)
            remaining_accrual = max(0.0, round(expected_liability - claims_paid, 2))

            cohort_rows.append({
                "product_family": prod,
                "ship_month": ship_month,
                "units_shipped": units,
                "warranty_months": w_months,
                "warranty_end_date": warranty_end_dt.strftime("%Y-%m-%d"),
                "rolling_cpu": round(unit_cpu, 2),
                "expected_liability": expected_liability,
                "claims_paid": round(claims_paid, 2),
                "remaining_accrual": remaining_accrual,
            })

    cohort_df = pd.DataFrame(cohort_rows)
    if cohort_df.empty:
        return {
            "total_warranty_reserve": 0.0,
            "total_expected_liability": 0.0,
            "total_claims_paid_active": 0.0,
            "cohort_accruals_df": pd.DataFrame(),
        }

    total_reserve = round(float(cohort_df["remaining_accrual"].sum()), 2)
    total_expected = round(float(cohort_df["expected_liability"].sum()), 2)
    total_paid = round(float(cohort_df["claims_paid"].sum()), 2)

    return {
        "total_warranty_reserve": total_reserve,
        "total_expected_liability": total_expected,
        "total_claims_paid_active": total_paid,
        "cohort_accruals_df": cohort_df.sort_values(by=["product_family", "ship_month"]).reset_index(drop=True),
    }


# -----------------------------------------------------------------------------
# 4. Pareto Analysis (Count & Cost)
# -----------------------------------------------------------------------------
def calculate_pareto(
    claims_df: pd.DataFrame,
    column: str = "failure_mode",
    cost_col: str = "repair_cost",
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Computes 80/20 Pareto breakdown by any attribute (failure_mode, component, region, etc.).
    Calculates frequency, cost, percentages, and cumulative percentages.

    Args:
        claims_df: Clean claims DataFrame
        column: Categorical column to analyze (e.g. 'failure_mode', 'component', 'region')
        cost_col: Financial cost column name
        top_n: Optional maximum number of categories to return

    Returns:
        DataFrame sorted descending by claim count, with cumulative % columns.
    """
    if claims_df.empty:
        return pd.DataFrame(columns=[
            column, "claim_count", "total_cost", "avg_cost",
            "count_pct", "cum_count_pct", "cost_pct", "cum_cost_pct"
        ])

    grouped = claims_df.groupby(column, as_index=False).agg(
        claim_count=("claim_id", "count"),
        total_cost=(cost_col, "sum"),
        avg_cost=(cost_col, "mean"),
    )

    total_claims = grouped["claim_count"].sum()
    total_cost_all = grouped["total_cost"].sum()

    grouped.sort_values(by="claim_count", ascending=False, inplace=True)
    grouped["count_pct"] = (grouped["claim_count"] / total_claims) * 100.0 if total_claims > 0 else 0.0
    grouped["cum_count_pct"] = grouped["count_pct"].cumsum().round(2)

    grouped["cost_pct"] = (grouped["total_cost"] / total_cost_all) * 100.0 if total_cost_all > 0 else 0.0
    grouped["cum_cost_pct"] = grouped["cost_pct"].cumsum().round(2)

    grouped["total_cost"] = grouped["total_cost"].round(2)
    grouped["avg_cost"] = grouped["avg_cost"].round(2)
    grouped["count_pct"] = grouped["count_pct"].round(2)
    grouped["cost_pct"] = grouped["cost_pct"].round(2)

    if top_n is not None and top_n > 0:
        grouped = grouped.head(top_n)

    return grouped.reset_index(drop=True)


# -----------------------------------------------------------------------------
# 5. Trend Detection & Shewhart Control Charts (2-Sigma Spikes)
# -----------------------------------------------------------------------------
def detect_ppm_trends(
    claims_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    group_col: str = "failure_mode",
    baseline_months_count: int = 24,
    recent_months_count: int = 3,
    sigma_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Statistical Process Control (SPC) Shewhart Trend Detection:
      1. Computes monthly PPM across ship_month cohorts for each group_col category.
         (If group_col is failure_mode or component, denominator is total monthly shipments).
      2. Uses the first `baseline_months_count` cohorts as the in-control baseline
         to compute baseline mean (mu) and standard deviation (sigma).
      3. Calculates Upper Control Limit (UCL = mu + 3*sigma) and Upper Warning Limit (UWL = mu + 2*sigma).
      4. Evaluates the trailing `recent_months_count` cohorts.
      5. Flags any group whose recent average PPM exceeds baseline by > sigma_threshold * sigma.

    Returns:
        DataFrame ranking groups by recent sigma deviation with status flags:
          - 'SPIKE_DETECTED' (Recent PPM > Baseline + 2*sigma)
          - 'OUT_OF_CONTROL' (Recent PPM > Baseline + 3*sigma)
          - 'NORMAL' (Within normal statistical limits)
    """
    if claims_df.empty or sales_df.empty:
        return pd.DataFrame()

    # Determine how to calculate cohort PPM
    if group_col in sales_df.columns:
        cohort_ppm = calculate_ppm(claims_df, sales_df, group_by=[group_col, "ship_month"])
    else:
        # Group sales by ship_month only
        sales_by_month = sales_df.groupby("ship_month", as_index=False)["quantity"].sum()
        sales_by_month.rename(columns={"quantity": "units_shipped"}, inplace=True)

        claims_by_grp_month = claims_df.groupby([group_col, "ship_month"], as_index=False)["claim_id"].count()
        claims_by_grp_month.rename(columns={"claim_id": "total_claims"}, inplace=True)

        # Merge
        cohort_ppm = pd.merge(claims_by_grp_month, sales_by_month, on="ship_month", how="left")
        cohort_ppm["ppm"] = np.where(
            cohort_ppm["units_shipped"] > 0,
            (cohort_ppm["total_claims"] / cohort_ppm["units_shipped"]) * 1_000_000.0,
            0.0,
        ).round(1)

    unique_months = sorted(sales_df["ship_month"].unique())

    if len(unique_months) < (recent_months_count + 3):
        return pd.DataFrame()

    baseline_months = unique_months[:baseline_months_count]
    recent_months = unique_months[-recent_months_count:]

    results = []
    for grp_val, grp_data in cohort_ppm.groupby(group_col):
        grp_data = grp_data.set_index("ship_month")

        # Baseline stats
        b_data = grp_data.reindex(baseline_months)["ppm"].fillna(0.0)
        mu = float(b_data.mean())
        sigma = float(b_data.std(ddof=1)) if len(b_data) > 1 else 0.0
        if sigma <= 0.001:
            sigma = max(10.0, mu * 0.15)

        ucl_3sigma = mu + 3.0 * sigma
        uwl_2sigma = mu + 2.0 * sigma

        # Recent stats
        r_data = grp_data.reindex(recent_months)["ppm"].fillna(0.0)
        recent_mean_ppm = float(r_data.mean())
        recent_delta_ppm = recent_mean_ppm - mu
        sigma_dev = round(recent_delta_ppm / sigma, 2)

        if recent_mean_ppm >= ucl_3sigma:
            status = "OUT_OF_CONTROL (3-Sigma)"
            is_anomaly = True
        elif recent_mean_ppm >= uwl_2sigma:
            status = "SPIKE_DETECTED (2-Sigma)"
            is_anomaly = True
        elif sigma_dev >= sigma_threshold:
            status = "WARNING_TREND"
            is_anomaly = True
        else:
            status = "IN_CONTROL (Normal)"
            is_anomaly = False

        results.append({
            group_col: grp_val,
            "baseline_mean_ppm": round(mu, 1),
            "baseline_std_sigma": round(sigma, 1),
            "upper_control_limit_ucl": round(ucl_3sigma, 1),
            "upper_warning_limit_uwl": round(uwl_2sigma, 1),
            "recent_3m_ppm": round(recent_mean_ppm, 1),
            "delta_ppm": round(recent_delta_ppm, 1),
            "sigma_deviation": sigma_dev,
            "status": status,
            "is_anomaly": bool(is_anomaly),
        })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.sort_values(by="sigma_deviation", ascending=False, inplace=True)
        results_df.reset_index(drop=True, inplace=True)

    return results_df


# -----------------------------------------------------------------------------
# 6. PPIT & CAR (Corrective Action Request) Ranking
# -----------------------------------------------------------------------------
def rank_ppit_car(
    claims_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
    cost_col: str = "repair_cost",
    car_threshold_score: float = 70.0,
) -> pd.DataFrame:
    """
    Computes Problem Priority Identification Table (PPIT) ranking and Corrective Action Request (CAR) qualification.

    Reliability Engineering Scoring Algorithm:
      PPIT Score (0 to 100) incorporates:
        1. Financial Severity (Total warranty expense incurred) - Weight: 40%
        2. PPM Velocity & Trend (Recent sigma deviation / upward slope) - Weight: 35%
        3. Customer & Volume Exposure (Total affected units / claims) - Weight: 25%

    Issues scoring >= car_threshold_score (e.g. 70.0) qualify for formal 8D Corrective Action Request (CAR).
    """
    # 1. Total claims and cost per failure mode + product family combo
    c_df = claims_df.copy()
    s_df = sales_df.copy()
    if "ship_month" not in c_df.columns and "ship_date" in c_df.columns:
        c_df["ship_month"] = c_df["ship_date"].astype(str).str.slice(0, 7)
    if "ship_month" not in s_df.columns and "ship_date" in s_df.columns:
        s_df["ship_month"] = s_df["ship_date"].astype(str).str.slice(0, 7)

    issue_agg = c_df.groupby(["product_family", "failure_mode"], as_index=False).agg(
        total_claims=("claim_id", "count"),
        total_warranty_cost=(cost_col, "sum"),
        avg_claim_cost=(cost_col, "mean"),
        earliest_claim=("claim_date", "min") if "claim_date" in c_df.columns else ("ship_date", "min"),
        latest_claim=("claim_date", "max") if "claim_date" in c_df.columns else ("ship_date", "max"),
    )

    if issue_agg.empty:
        return pd.DataFrame()

    # 2. Get trend sigma deviation for each issue
    cohort_ppm = c_df.groupby(["product_family", "failure_mode", "ship_month"], as_index=False)["claim_id"].count()
    sales_month = s_df.groupby(["product_family", "ship_month"], as_index=False)["quantity"].sum()
    merged = pd.merge(sales_month, cohort_ppm, on=["product_family", "ship_month"], how="left")
    merged["claim_id"] = merged["claim_id"].fillna(0)
    merged["ppm"] = (merged["claim_id"] / merged["quantity"]) * 1_000_000.0

    all_months = sorted(sales_month["ship_month"].unique())
    recent_m = all_months[-3:] if len(all_months) >= 3 else all_months
    base_m = all_months[:-3] if len(all_months) >= 6 else all_months

    trend_factors = {}
    for (p, fm), grp in merged.groupby(["product_family", "failure_mode"]):
        grp_idx = grp.set_index("ship_month")
        b_vals = grp_idx.reindex(base_m)["ppm"].fillna(0.0)
        r_vals = grp_idx.reindex(recent_m)["ppm"].fillna(0.0)
        b_mean = float(b_vals.mean())
        b_std = float(b_vals.std()) if len(b_vals) > 1 else max(10.0, b_mean * 0.2)
        r_mean = float(r_vals.mean())
        dev = (r_mean - b_mean) / (b_std if b_std > 0 else 1.0)
        trend_factors[(p, fm)] = (dev, b_mean, r_mean)

    # Attach trend stats
    devs, b_means, r_means = [], [], []
    for _, r in issue_agg.iterrows():
        tf = trend_factors.get((r["product_family"], r["failure_mode"]), (0.0, 0.0, 0.0))
        devs.append(round(tf[0], 2))
        b_means.append(round(tf[1], 1))
        r_means.append(round(tf[2], 1))

    issue_agg["sigma_deviation"] = devs
    issue_agg["baseline_ppm"] = b_means
    issue_agg["recent_ppm"] = r_means

    # 3. Normalize components to 0-100 scale
    max_cost = issue_agg["total_warranty_cost"].max()
    max_claims = issue_agg["total_claims"].max()
    max_sigma = max(1.0, issue_agg["sigma_deviation"].max())

    norm_cost = (issue_agg["total_warranty_cost"] / max_cost) * 100.0 if max_cost > 0 else 0.0
    norm_claims = (issue_agg["total_claims"] / max_claims) * 100.0 if max_claims > 0 else 0.0
    norm_trend = (issue_agg["sigma_deviation"].clip(lower=0.0) / max_sigma) * 100.0 if max_sigma > 0 else 0.0

    # Composite PPIT Score: 40% cost + 35% trend + 25% claims
    ppit_score = (0.40 * norm_cost) + (0.35 * norm_trend) + (0.25 * norm_claims)
    issue_agg["ppit_score"] = ppit_score.round(1)

    # Qualification Status
    issue_agg["car_status"] = np.where(
        issue_agg["ppit_score"] >= car_threshold_score,
        "QUALIFIED FOR 8D / CAR",
        np.where(issue_agg["ppit_score"] >= 45.0, "MONITORING (Warning)", "LOW PRIORITY"),
    )

    issue_agg["car_id"] = [f"CAR-2026-{i+101:03d}" for i in range(len(issue_agg))]
    issue_agg.sort_values(by="ppit_score", ascending=False, inplace=True)
    issue_agg.reset_index(drop=True, inplace=True)
    issue_agg["ppit_rank"] = issue_agg.index + 1

    return issue_agg
