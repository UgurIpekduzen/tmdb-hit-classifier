"""
Generic temporal validation utilities for ML pipelines.
Designed to be reusable across ML projects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import fbeta_score


def walk_forward_cv(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    year_col: str,
    model,
    train_window: int = 8,
    val_window: int = 2,
    step: int = 1,
    start_year: int | None = None,
    min_train_n: int = 150,
    min_val_n: int = 30,
    beta: int = 2,
) -> pd.DataFrame:
    """
    Sliding-window temporal cross-validation for year-indexed datasets.

    For each fold:
      - train on [start, start + train_window - 1]
      - validate on [train_end + 1, train_end + val_window]
      - slide by step years

    Folds where either split has fewer than min_train_n / min_val_n rows
    or a single class are skipped.

    Parameters
    ----------
    df           : DataFrame with year_col and target columns
    features     : feature column names to use
    target       : binary target column
    year_col     : integer year column (e.g. 'release_year')
    model        : sklearn-compatible estimator (cloned per fold — not mutated)
    train_window : number of years in the training window (default 8)
    val_window   : number of years in the validation window (default 2)
    step         : sliding step in years (default 1)
    start_year   : first training year; uses dataset min if None
    min_train_n  : minimum training rows to run a fold (default 150)
    min_val_n    : minimum validation rows to run a fold (default 30)
    beta         : beta for F-beta score (default 2)

    Returns
    -------
    DataFrame with columns:
        train_period, val_period, n_train, n_val,
        hit_rate_val, f{beta}_train, f{beta}_val, gap

    Examples
    --------
    df_cv = walk_forward_cv(
        df, FEATURES, TARGET, year_col="release_year",
        model=tuned_lgbm, train_window=8, val_window=2,
    )
    print(f"Mean gap: {df_cv['gap'].mean():.4f}")
    """
    metric_col = f"f{beta}"
    train_start = start_year if start_year is not None else int(df[year_col].min())
    max_year = int(df[year_col].max())
    records = []

    while True:
        train_end = train_start + train_window - 1
        val_start = train_end + 1
        val_end   = val_start + val_window - 1

        if val_end > max_year:
            break

        mask_tr = (df[year_col] >= train_start) & (df[year_col] <= train_end)
        mask_vl = (df[year_col] >= val_start)   & (df[year_col] <= val_end)

        X_tr, y_tr = df.loc[mask_tr, features], df.loc[mask_tr, target]
        X_vl, y_vl = df.loc[mask_vl, features], df.loc[mask_vl, target]

        if len(X_tr) < min_train_n or len(X_vl) < min_val_n:
            train_start += step
            continue
        if y_tr.nunique() < 2 or y_vl.nunique() < 2:
            train_start += step
            continue

        clf = clone(model)
        clf.fit(X_tr, y_tr)

        score_tr = fbeta_score(y_tr, clf.predict(X_tr), beta=beta)
        score_vl = fbeta_score(y_vl, clf.predict(X_vl), beta=beta)

        records.append({
            "train_period":      f"{train_start}–{train_end}",
            "val_period":        f"{val_start}–{val_end}",
            "val_end":           val_end,
            "n_train":           len(X_tr),
            "n_val":             len(X_vl),
            "hit_rate_val":      round(float(y_vl.mean()), 3),
            f"{metric_col}_train": round(score_tr, 4),
            f"{metric_col}_val":   round(score_vl, 4),
            "gap":               round(score_tr - score_vl, 4),
        })
        train_start += step

    result = pd.DataFrame(records)
    if result.empty:
        return result

    print(f"Walk-forward CV: {len(result)} fold  |  "
          f"gap mean={result['gap'].mean():.4f}  std={result['gap'].std():.4f}  "
          f"min={result['gap'].min():.4f}  max={result['gap'].max():.4f}")
    return result
