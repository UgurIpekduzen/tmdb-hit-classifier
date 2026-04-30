"""Feature drift monitoring — KS-test and PSI."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PSI_BINS = 10
KS_ALPHA = 0.05

# PSI thresholds (descending order for first-match lookup)
_PSI_LEVELS: list[tuple[float, str]] = [(0.25, "high"), (0.1, "medium"), (0.0, "low")]


def _psi_continuous(ref: np.ndarray, prod: np.ndarray, bins: int) -> float:
    breakpoints = np.linspace(ref.min(), ref.max(), bins + 1)
    breakpoints[0]  -= 1e-9
    breakpoints[-1] += 1e-9
    ref_pct  = np.histogram(ref,  bins=breakpoints)[0] / len(ref)
    prod_pct = np.histogram(prod, bins=breakpoints)[0] / len(prod)
    ref_pct  = np.clip(ref_pct,  1e-6, None)
    prod_pct = np.clip(prod_pct, 1e-6, None)
    return float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))


def _psi_binary(ref: np.ndarray, prod: np.ndarray) -> float:
    p_ref  = float(np.nanmean(ref))
    p_prod = float(np.nanmean(prod))
    if not (0 < p_ref < 1) or not (0 < p_prod < 1):
        return np.nan
    return (
        (p_prod - p_ref) * np.log(p_prod / p_ref)
        + ((1 - p_prod) - (1 - p_ref)) * np.log((1 - p_prod) / (1 - p_ref))
    )


def _psi_level(psi: float) -> str:
    if np.isnan(psi):
        return "unknown"
    for threshold, level in _PSI_LEVELS:
        if psi >= threshold:
            return level
    return "low"


def drift_report(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    features: list[str],
    ks_alpha: float = KS_ALPHA,
    psi_bins: int = PSI_BINS,
) -> pd.DataFrame:
    """
    Compute per-feature drift between reference and production DataFrames.

    Parameters
    ----------
    reference  : reference distribution (e.g. training split)
    production : incoming data to compare (e.g. recent API requests)
    features   : feature columns to evaluate
    ks_alpha   : significance level for KS drift flag (default 0.05)
    psi_bins   : number of bins for continuous PSI (default 10)

    Returns
    -------
    DataFrame sorted by PSI descending with columns:
        feature, n_ref, n_prod, ks_stat, ks_pvalue, ks_drift, psi, psi_level, drift
    """
    rows = []
    for feat in features:
        if feat not in reference.columns or feat not in production.columns:
            continue

        ref_vals  = reference[feat].dropna().values
        prod_vals = production[feat].dropna().values

        if len(ref_vals) == 0 or len(prod_vals) == 0:
            rows.append({"feature": feat, "n_ref": len(ref_vals), "n_prod": len(prod_vals),
                         "ks_stat": None, "ks_pvalue": None, "ks_drift": None,
                         "psi": None, "psi_level": "unknown", "drift": False})
            continue

        ks_stat, ks_p = stats.ks_2samp(ref_vals, prod_vals)

        is_binary = set(np.unique(ref_vals)).issubset({0, 1, 0.0, 1.0})
        psi = _psi_binary(ref_vals, prod_vals) if is_binary else _psi_continuous(ref_vals, prod_vals, psi_bins)
        level = _psi_level(psi)

        rows.append({
            "feature":   feat,
            "n_ref":     int(len(ref_vals)),
            "n_prod":    int(len(prod_vals)),
            "ks_stat":   round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_p), 4),
            "ks_drift":  bool(ks_p < ks_alpha),
            "psi":       round(float(psi), 4) if not np.isnan(psi) else None,
            "psi_level": level,
            "drift":     level in ("medium", "high") or bool(ks_p < ks_alpha),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("psi", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def load_reference(
    csv_path: str,
    features: list[str],
    year_col: str = "release_year",
    train_cutoff: int = 2009,
) -> pd.DataFrame:
    """
    Load the training split from CSV as the reference distribution.

    Parameters
    ----------
    csv_path     : path to tmdb_model.csv
    features     : feature columns to retain
    year_col     : temporal split column (default 'release_year')
    train_cutoff : inclusive upper bound for training years (default 2009)
    """
    df = pd.read_csv(csv_path)
    if year_col in df.columns:
        df = df[df[year_col] <= train_cutoff]
    available = [f for f in features if f in df.columns]
    return df[available].reset_index(drop=True)
