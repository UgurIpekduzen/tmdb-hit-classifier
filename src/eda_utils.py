"""
Feature tarama ve kalite değerlendirmesi için genel amaçlı EDA yardımcıları.
Farklı ML projelerinde yeniden kullanılabilir olacak şekilde tasarlanmıştır.
"""

import numpy as np
import pandas as pd


def coverage_report(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """
    Her feature için: non-null %, non-zero % (numerik), benzersiz değer sayısı.

    Returns
    -------
    non_null_pct'e göre artan sıralı DataFrame (en kötü kapsama önce).
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
    Tek bir değerin satırların >= threshold kadarına hakim olduğu feature'ları tespit eder.

    Parameters
    ----------
    threshold : sabit olarak işaretlenecek baskın değer frekansı (varsayılan 0.97 = %97)

    Returns
    -------
    dominant_pct'e göre azalan sıralı, tüm feature'ları içeren DataFrame.
    İşaretlenen feature'larda constant=True olur.
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
    Entity bazında leakage'a karşı güvenli expanding window agregasyonu.

    Her satır için, value_col'u o entity'nin SADECE ondan önce görünen satırlarını
    kullanarak agrega eder. DataFrame önceden sıralı değilse date_col verin.

    Agregasyon modları (agg parametresi):
        'mean'  — expanding ortalama; smooth_k ile hiyerarşik düzeltmeyi destekler
        'count' — entity'nin önceki görülme sayısı (value_col dikkate alınmaz)
        'sum'   — expanding toplam

    Hiyerarşik düzeltme (agg='mean', smooth_k > 0):
        smoothed = (n * raw_mean + K * global_prior) / (n + K)

    Parameters
    ----------
    entity_col   : entity'yi belirleyen sütun (örn. yönetmen, user_id)
    value_col    : agrega edilecek sütun (örn. roi, revenue, is_default)
    date_col     : verilirse, agregasyondan önce df bu sütuna göre sıralanır
    agg          : 'mean' | 'count' | 'sum'
    smooth_k     : agg='mean' için düzeltme gücü; 0 = düzeltme yok
    global_prior : düzeltme için prior ortalama; None ise value_col'dan hesaplanır

    Returns
    -------
    df.index ile hizalı pd.Series.

    Examples
    --------
    # Yönetmen film sayısı
    df["director_film_count"] = temporal_entity_agg(
        df, "director", "id", date_col="release_date", agg="count"
    )

    # Oyuncu düzeltilmiş ROI (seyrek entity'leri global ortalamaya doğru büzer)
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
        else:  # ortalama
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
