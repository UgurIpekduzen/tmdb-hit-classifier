"""
ML pipeline'ları için genel amaçlı temporal validasyon yardımcıları.
Farklı ML projelerinde yeniden kullanılabilir olacak şekilde tasarlanmıştır.
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
    Yıl indeksli veri setleri için sliding-window temporal cross-validation.

    Her fold için:
      - [start, start + train_window - 1] üzerinde eğitim
      - [train_end + 1, train_end + val_window] üzerinde validasyon
      - step yıl kadar kaydır

    Herhangi bir split'in min_train_n / min_val_n'den az satırı veya tek sınıfı
    varsa o fold atlanır.

    Parameters
    ----------
    df           : year_col ve target sütunlarını içeren DataFrame
    features     : kullanılacak feature sütun adları
    target       : binary target sütunu
    year_col     : tamsayı yıl sütunu (örn. 'release_year')
    model        : sklearn-uyumlu estimator (her fold'da klonlanır — değiştirilmez)
    train_window : eğitim penceresindeki yıl sayısı (varsayılan 8)
    val_window   : validasyon penceresindeki yıl sayısı (varsayılan 2)
    step         : yıl bazında kaydırma adımı (varsayılan 1)
    start_year   : ilk eğitim yılı; None ise veri setinin minimumu kullanılır
    min_train_n  : bir fold'u çalıştırmak için minimum eğitim satırı (varsayılan 150)
    min_val_n    : bir fold'u çalıştırmak için minimum validasyon satırı (varsayılan 30)
    beta         : F-beta skoru için beta (varsayılan 2)

    Returns
    -------
    Şu sütunlara sahip DataFrame:
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
