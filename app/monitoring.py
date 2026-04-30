"""Feature drift monitoring — KS-test, PSI, drift direction, and prediction drift."""
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


def compute_feature_importance(
    predict_fn,
    reference: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    """
    Compute feature importance as |Spearman correlation| between each feature
    and the model's predicted probabilities on the reference data.

    Returns a dict of {feature: importance} normalized to [0, 1].
    Model-agnostic — works with any callable that takes a DataFrame.
    """
    proba = np.array(predict_fn(reference[features]))
    importance: dict[str, float] = {}
    for feat in features:
        corr, _ = stats.spearmanr(reference[feat], proba)
        importance[feat] = abs(float(corr)) if not np.isnan(corr) else 0.0
    max_val = max(importance.values()) or 1.0
    return {k: round(v / max_val, 4) for k, v in importance.items()}


def drift_report(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    features: list[str],
    ks_alpha: float = KS_ALPHA,
    psi_bins: int = PSI_BINS,
    feature_importance: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Compute per-feature drift between reference and production DataFrames.

    Parameters
    ----------
    reference          : reference distribution (e.g. training split)
    production         : incoming data to compare (e.g. recent API requests)
    features           : feature columns to evaluate
    ks_alpha           : significance level for KS drift flag (default 0.05)
    psi_bins           : number of bins for continuous PSI (default 10)
    feature_importance : {feature: 0–1 normalized importance}; when provided,
                         adds `importance` and `risk_score` (psi × importance)
                         columns and sorts by risk_score instead of psi

    Returns
    -------
    DataFrame with columns:
        feature, n_ref, n_prod,
        mean_ref, mean_prod, mean_shift,
        ks_stat, ks_pvalue, ks_drift,
        psi, psi_level,
        [importance, risk_score — if feature_importance provided],
        drift
    Sorted by risk_score (if importance given) or psi descending.
    """
    rows = []
    for feat in features:
        if feat not in reference.columns or feat not in production.columns:
            continue

        ref_vals  = reference[feat].dropna().values
        prod_vals = production[feat].dropna().values

        if len(ref_vals) == 0 or len(prod_vals) == 0:
            rows.append({"feature": feat, "n_ref": len(ref_vals), "n_prod": len(prod_vals),
                         "mean_ref": None, "mean_prod": None, "mean_shift": None,
                         "ks_stat": None, "ks_pvalue": None, "ks_drift": None,
                         "psi": None, "psi_level": "unknown", "drift": False})
            continue

        ks_stat, ks_p = stats.ks_2samp(ref_vals, prod_vals)

        is_binary = set(np.unique(ref_vals)).issubset({0, 1, 0.0, 1.0})
        psi   = _psi_binary(ref_vals, prod_vals) if is_binary else _psi_continuous(ref_vals, prod_vals, psi_bins)
        level = _psi_level(psi)

        mean_ref   = round(float(np.mean(ref_vals)),  4)
        mean_prod  = round(float(np.mean(prod_vals)), 4)
        mean_shift = round(mean_prod - mean_ref,       4)

        rows.append({
            "feature":    feat,
            "n_ref":      int(len(ref_vals)),
            "n_prod":     int(len(prod_vals)),
            "mean_ref":   mean_ref,
            "mean_prod":  mean_prod,
            "mean_shift": mean_shift,
            "ks_stat":    round(float(ks_stat), 4),
            "ks_pvalue":  round(float(ks_p), 4),
            "ks_drift":   bool(ks_p < ks_alpha),
            "psi":        round(float(psi), 4) if not np.isnan(psi) else None,
            "psi_level":  level,
            "drift":      level in ("medium", "high") or bool(ks_p < ks_alpha),
        })

    df = pd.DataFrame(rows)

    if feature_importance is not None:
        df["importance"]  = df["feature"].map(feature_importance).fillna(0.0)
        df["risk_score"]  = (df["psi"].fillna(0.0) * df["importance"]).round(4)
        sort_col = "risk_score"
    else:
        sort_col = "psi"

    return df.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)


def prediction_drift(
    ref_proba: np.ndarray,
    prod_proba: np.ndarray,
    ks_alpha: float = KS_ALPHA,
    psi_bins: int = PSI_BINS,
) -> dict:
    """
    Measure drift in model output probabilities between reference and production.

    Parameters
    ----------
    ref_proba  : predicted probabilities on reference data (train split)
    prod_proba : predicted probabilities on incoming production batch
    ks_alpha   : significance level for KS drift flag
    psi_bins   : bins for PSI computation

    Returns
    -------
    dict with keys:
        mean_ref, mean_prod, mean_shift,
        pct_hit_ref, pct_hit_prod,
        ks_stat, ks_pvalue, ks_drift,
        psi, psi_level, drift
    """
    ref_proba  = np.asarray(ref_proba,  dtype=float)
    prod_proba = np.asarray(prod_proba, dtype=float)

    ks_stat, ks_p = stats.ks_2samp(ref_proba, prod_proba)
    psi   = _psi_continuous(ref_proba, prod_proba, psi_bins)
    level = _psi_level(psi)

    threshold = 0.5
    return {
        "mean_ref":     round(float(np.mean(ref_proba)),  4),
        "mean_prod":    round(float(np.mean(prod_proba)), 4),
        "mean_shift":   round(float(np.mean(prod_proba) - np.mean(ref_proba)), 4),
        "pct_hit_ref":  round(float(np.mean(ref_proba  >= threshold)), 4),
        "pct_hit_prod": round(float(np.mean(prod_proba >= threshold)), 4),
        "ks_stat":      round(float(ks_stat), 4),
        "ks_pvalue":    round(float(ks_p), 4),
        "ks_drift":     bool(ks_p < ks_alpha),
        "psi":          round(float(psi), 4) if not np.isnan(psi) else None,
        "psi_level":    level,
        "drift":        level in ("medium", "high") or bool(ks_p < ks_alpha),
    }


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
