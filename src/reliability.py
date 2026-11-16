"""
src/reliability.py
------------------
Weibull Life Data Analysis & Reliability Engineering Engine using `lifelines`.
Fits 2-parameter Weibull distribution (shape beta, scale eta) per product family,
handling right-censored units (units shipped without warranty claims).
Computes:
  - Shape parameter (beta) and Scale parameter (eta, in days)
  - B10 Life: Time at which 10% of field population is expected to fail
  - B50 Life (Median life) and MTTF (Mean Time To Failure)
  - Reliability regime classification: Infant Mortality vs Random vs Wear-out
  - Weibull probability plot plotting data (linearized log-log coordinates)
  - Hazard rate / failure rate curve h(t)
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from lifelines import WeibullFitter
from scipy.special import gamma

CUTOFF_DATE = "2026-01-31"


def classify_failure_regime(beta: float) -> str:
    """Classifies Weibull failure mechanism based on shape parameter beta."""
    if beta < 0.95:
        return "Infant Mortality (Early-life / Quality Defect)"
    elif 0.95 <= beta <= 1.25:
        return "Random / Constant Hazard (Operational Stress)"
    else:
        return "Wear-Out (Aging / Contact Erosion / Fatigue)"


def compute_weibull_metrics_for_family(
    claims_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    product_family: str,
    cutoff_date: str = CUTOFF_DATE,
    max_censored_sample: int = 15000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Fits 2-parameter Weibull distribution on a single product family taking into account
    both failed units (claims) and right-censored units (surviving units shipped).
    """
    p_claims = claims_df[claims_df["product_family"] == product_family].copy()
    p_sales = sales_df[sales_df["product_family"] == product_family].copy()

    if p_claims.empty or p_sales.empty:
        return {
            "product_family": product_family,
            "beta_shape": 1.0,
            "eta_scale_days": 1000.0,
            "b10_days": 105.0,
            "b50_days": 693.0,
            "mttf_days": 1000.0,
            "failure_regime": "Random / Constant Hazard (Operational Stress)",
            "total_shipped": int(p_sales["quantity"].sum()) if not p_sales.empty else 0,
            "total_failures": len(p_claims),
            "censored_count": 0,
        }

    cutoff_dt = pd.to_datetime(cutoff_date)

    # 1. Failed units (event = 1)
    p_claims["ship_dt"] = pd.to_datetime(p_claims["ship_date"])
    p_claims["claim_dt"] = pd.to_datetime(p_claims["claim_date"])
    # Time to failure in days (positive)
    ttf_days = (p_claims["claim_dt"] - p_claims["ship_dt"]).dt.days.clip(lower=1.0).values
    events_failed = np.ones(len(ttf_days), dtype=int)

    # 2. Censored units (event = 0)
    # Total units shipped
    total_shipped = int(p_sales["quantity"].sum())
    total_failures = len(p_claims)
    total_surviving = max(0, total_shipped - total_failures)

    # Calculate survival durations from ship_date to cutoff_date
    p_sales["ship_dt"] = pd.to_datetime(p_sales["ship_date"])
    p_sales["survival_days"] = (cutoff_dt - p_sales["ship_dt"]).dt.days.clip(lower=1.0)

    # Sample representative censored survival times proportional to shipment volume
    if total_surviving > 0:
        sample_size = min(total_surviving, max_censored_sample)
        probabilities = p_sales["quantity"] / p_sales["quantity"].sum()
        np.random.seed(random_state)
        sampled_indices = np.random.choice(p_sales.index, size=sample_size, p=probabilities)
        censored_durations = p_sales.loc[sampled_indices, "survival_days"].values
        events_censored = np.zeros(sample_size, dtype=int)
    else:
        censored_durations = np.array([])
        events_censored = np.array([], dtype=int)

    durations = np.concatenate([ttf_days, censored_durations])
    events = np.concatenate([events_failed, events_censored])

    # Fit Weibull using lifelines
    wf = WeibullFitter()
    wf.fit(durations=durations, event_observed=events)

    beta = round(float(wf.rho_), 3)  # In lifelines, rho_ is the shape parameter beta
    eta = round(float(wf.lambda_), 1)  # In lifelines, lambda_ is the scale parameter eta

    # B10 calculation: t where F(t) = 0.10 -> t = eta * (-ln(0.90))^(1/beta)
    # ln(0.90) = -0.1053605
    b10_days = round(eta * ((-math.log(0.90)) ** (1.0 / beta)), 1)
    # B50 calculation: t where F(t) = 0.50 -> t = eta * (ln(2))^(1/beta)
    b50_days = round(eta * ((math.log(2.0)) ** (1.0 / beta)), 1)
    # MTTF = eta * Gamma(1 + 1/beta)
    mttf_days = round(eta * float(gamma(1.0 + (1.0 / beta))), 1)

    regime = classify_failure_regime(beta)

    return {
        "product_family": product_family,
        "beta_shape": beta,
        "eta_scale_days": eta,
        "b10_days": b10_days,
        "b50_days": b50_days,
        "mttf_days": mttf_days,
        "failure_regime": regime,
        "total_shipped": total_shipped,
        "total_failures": total_failures,
        "censored_count": total_surviving,
    }


def fit_all_products_weibull(
    claims_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    cutoff_date: str = CUTOFF_DATE,
) -> pd.DataFrame:
    """
    Fits Weibull distribution across all distinct product families and returns
    a comprehensive reliability summary table.
    """
    families = sorted(sales_df["product_family"].unique())
    results = []
    for fam in families:
        metrics = compute_weibull_metrics_for_family(claims_df, sales_df, fam, cutoff_date=cutoff_date)
        results.append(metrics)

    res_df = pd.DataFrame(results)
    return res_df.sort_values(by="beta_shape", ascending=True).reset_index(drop=True)


def generate_weibull_curves(
    beta: float,
    eta: float,
    max_days: int = 1500,
    num_points: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Generates theoretical Weibull probability curves for plotting:
      - Time grid t
      - Reliability / Survival function R(t) = exp(-(t/eta)^beta)
      - Unreliability / Cumulative Failure CDF F(t) = 1 - R(t)
      - Failure Rate / Hazard function h(t) = (beta/eta) * (t/eta)^(beta-1)
    """
    t = np.linspace(1.0, max_days, num_points)
    # Reliability function
    r_t = np.exp(-((t / eta) ** beta))
    f_t = 1.0 - r_t
    # Hazard rate per 1,000 operating days
    h_t = (beta / eta) * ((t / eta) ** (beta - 1.0)) * 1000.0

    return {
        "days": t,
        "reliability": r_t,
        "unreliability_cdf": f_t,
        "hazard_rate_per_1k_days": h_t,
    }


def get_weibull_probability_plot_points(
    claims_df: pd.DataFrame,
    product_family: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates Bernard's median ranks for empirical Weibull probability plotting:
      x = ln(t)
      y = ln(-ln(1 - F_i))
      where F_i = (i - 0.3) / (N + 0.4)
    """
    p_claims = claims_df[claims_df["product_family"] == product_family].copy()
    if p_claims.empty:
        return np.array([]), np.array([]), np.array([])

    p_claims["ttf"] = (pd.to_datetime(p_claims["claim_date"]) - pd.to_datetime(p_claims["ship_date"])).dt.days.clip(lower=1.0)
    ttfs = np.sort(p_claims["ttf"].values)
    n = len(ttfs)

    ranks = np.arange(1, n + 1)
    median_ranks = (ranks - 0.3) / (n + 0.4)

    # Filter bounds
    valid = (median_ranks > 0) & (median_ranks < 1.0)
    t_valid = ttfs[valid]
    mr_valid = median_ranks[valid]

    x_ln_t = np.log(t_valid)
    y_weibull = np.log(-np.log(1.0 - mr_valid))

    return t_valid, x_ln_t, y_weibull
