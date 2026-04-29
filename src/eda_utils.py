"""
Generic EDA utilities for feature screening and quality assessment.
Designed to be reusable across ML projects.
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency


def coverage_report(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """
    For each feature: non-null %, non-zero % (numeric), unique count.

    Returns
    -------
    DataFrame sorted by non_null_pct ascending (worst coverage first).
    """
    if features is None:
        features = df.columns.tolist()

    n = len(df)
    rows = []
    for col in features:
        if col not in df.columns:
            continue
        s = df[col]
        non_null = s.notna().sum() / n * 100
        non_zero = (s != 0).sum() / n * 100 if pd.api.types.is_numeric_dtype(s) else None
        rows.append({
            "feature":      col,
            "non_null_pct": round(non_null, 1),
            "non_zero_pct": round(non_zero, 1) if non_zero is not None else None,
            "n_unique":     int(s.nunique()),
            "dtype":        str(s.dtype),
        })

    return pd.DataFrame(rows).set_index("feature").sort_values("non_null_pct")


def find_constant_features(
    df: pd.DataFrame,
    features: list[str] | None = None,
    threshold: float = 0.97,
) -> pd.DataFrame:
    """
    Detect features where a single value dominates >= threshold of rows.

    Parameters
    ----------
    threshold : dominant value frequency to flag as constant (default 0.97 = 97%)

    Returns
    -------
    DataFrame with all features, sorted by dominant_pct descending.
    Flagged features have constant=True.
    """
    if features is None:
        features = df.columns.tolist()

    rows = []
    for col in features:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(normalize=True)
        dom_pct = counts.iloc[0]
        dom_val = counts.index[0]
        rows.append({
            "feature":       col,
            "dominant_value": dom_val,
            "dominant_pct":  round(dom_pct * 100, 1),
            "constant":      dom_pct >= threshold,
        })

    result = pd.DataFrame(rows).set_index("feature").sort_values(
        "dominant_pct", ascending=False
    )

    n_flagged = result["constant"].sum()
    print(f"Sabit feature (>= %{threshold*100:.0f}): {n_flagged} / {len(result)}")
    if n_flagged:
        flagged = result[result["constant"]].index.tolist()
        print(f"  → {flagged}")

    return result


def univariate_screening(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    alpha: float = 0.05,
    binary_max_unique: int = 10,
) -> pd.DataFrame:
    """
    Mann-Whitney U (continuous) or chi-square (binary/categorical) for each
    feature vs target. Applies Bonferroni correction.

    Parameters
    ----------
    binary_max_unique : features with <= this many unique int values use chi-square.

    Returns
    -------
    DataFrame sorted by p ascending (strongest signal first).
    Columns: test, statistic, p, p_bonferroni, significant.
    """
    rows = []
    for col in features:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        t = df.loc[s.index, target]

        is_binary = (
            s.nunique() <= 2
            or (pd.api.types.is_integer_dtype(s) and s.nunique() <= binary_max_unique)
        )

        if is_binary:
            ct = pd.crosstab(s, t)
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                rows.append({"feature": col, "test": "chi2", "statistic": None, "p": None})
                continue
            chi2, p, _, _ = chi2_contingency(ct)
            rows.append({"feature": col, "test": "chi2", "statistic": round(chi2, 3), "p": p})
        else:
            pos = s[t == 1]
            neg = s[t == 0]
            if len(pos) == 0 or len(neg) == 0:
                rows.append({"feature": col, "test": "mwu", "statistic": None, "p": None})
                continue
            stat, p = mannwhitneyu(pos, neg, alternative="two-sided")
            rows.append({"feature": col, "test": "mwu", "statistic": round(stat, 1), "p": p})

    result = pd.DataFrame(rows).set_index("feature")
    n_tests = int(result["p"].notna().sum())
    result["p_bonferroni"] = (result["p"] * n_tests).clip(upper=1.0)
    result["significant"] = result["p_bonferroni"] < alpha

    result["p"]            = result["p"].round(4)
    result["p_bonferroni"] = result["p_bonferroni"].round(4)

    bonf_threshold = alpha / n_tests if n_tests else alpha
    n_sig = int(result["significant"].sum())
    print(f"Bonferroni eşiği: p < {bonf_threshold:.4f}  ({n_tests} test)")
    print(f"Anlamlı feature: {n_sig} / {len(result)}")

    return result.sort_values("p")


def temporal_entity_agg(
    df: pd.DataFrame,
    entity_col: str,
    value_col: str,
    date_col: str = None,
    agg: str = "mean",
    smooth_k: float = 0,
    global_prior: float = None,
) -> pd.Series:
    """
    Leakage-safe expanding window aggregation per entity.

    For each row, aggregates value_col for that entity using ONLY rows that
    appear before it. If the DataFrame is not pre-sorted, pass date_col.

    Aggregation modes (agg parameter):
        'mean'  — expanding mean; supports hierarchical smoothing via smooth_k
        'count' — number of prior occurrences of the entity (value_col ignored)
        'sum'   — expanding sum

    Hierarchical smoothing (agg='mean', smooth_k > 0):
        smoothed = (n * raw_mean + K * global_prior) / (n + K)

    Parameters
    ----------
    entity_col   : column identifying the entity (e.g. director, user_id)
    value_col    : column to aggregate (e.g. roi, revenue, is_default)
    date_col     : if provided, df is sorted by this column before aggregation
    agg          : 'mean' | 'count' | 'sum'
    smooth_k     : smoothing strength for agg='mean'; 0 = no smoothing
    global_prior : prior mean for smoothing; computed from value_col if None

    Returns
    -------
    pd.Series aligned to df.index.

    Examples
    --------
    # Director film count
    df["director_film_count"] = temporal_entity_agg(
        df, "director", "id", date_col="release_date", agg="count"
    )

    # Actor smoothed ROI (shrink sparse entities towards global mean)
    df["actor_hist_roi"] = temporal_entity_agg(
        df, "actor_id", "roi_capped", date_col="release_date",
        agg="mean", smooth_k=5,
    )
    """
    if agg not in ("mean", "count", "sum"):
        raise ValueError(f"agg must be 'mean', 'count', or 'sum', got '{agg}'")

    if date_col is not None:
        df = df.sort_values(date_col).reset_index(drop=False)
        restore_index = True
    else:
        restore_index = False

    if agg == "mean" and smooth_k > 0 and global_prior is None:
        global_prior = float(df[value_col].mean())

    history: dict = {}
    results: list = []

    for _, row in df.iterrows():
        entity = row[entity_col]
        hist   = [v for v in history.get(entity, []) if not (isinstance(v, float) and np.isnan(v))]
        n      = len(hist)

        if agg == "count":
            val = float(n)
        elif agg == "sum":
            val = float(sum(hist)) if n > 0 else 0.0
        else:  # mean
            if n == 0:
                val = global_prior if smooth_k > 0 else 0.0
            else:
                raw = float(np.mean(hist))
                val = (n * raw + smooth_k * global_prior) / (n + smooth_k) if smooth_k > 0 else raw

        results.append(val)
        if agg != "count":
            history.setdefault(entity, []).append(row[value_col])
        else:
            history[entity] = hist + [1]

    out = pd.Series(results, name=f"{entity_col}_hist_{value_col}")

    if restore_index:
        out.index = df["index"]
        out = out.sort_index()

    return out
