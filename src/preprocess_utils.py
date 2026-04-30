"""
Generic preprocessing utilities for structured ML datasets.
Designed to be reusable across ML projects.
"""

import ast
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MultiLabelBinarizer


# ── JSON parsing ─────────────────────────────────────────────────────────────

def parse_json_list(json_str, key: str = "name") -> list:
    """
    Extract a list of string values from a JSON array stored as a string.

    Parameters
    ----------
    json_str : string representation of a JSON array of dicts
    key      : dict key to extract from each item (default 'name')

    Examples
    --------
    parse_json_list('[{"id":1,"name":"Action"},{"id":2,"name":"Drama"}]')
    # → ['Action', 'Drama']
    """
    try:
        items = ast.literal_eval(json_str)
        if isinstance(items, list):
            return [item.get(key, "") for item in items if isinstance(item, dict)]
    except Exception:
        pass
    return []


def parse_json_iso_codes(json_str, key: str = "iso_3166_1") -> list:
    """
    Extract ISO codes from a JSON array stored as a string.

    Parameters
    ----------
    json_str : string representation of a JSON array of dicts
    key      : ISO code key (e.g. 'iso_3166_1' for countries, 'iso_639_1' for languages)

    Examples
    --------
    parse_json_iso_codes('[{"iso_3166_1":"US","name":"United States"}]')
    # → ['US']
    """
    return parse_json_list(json_str, key=key)


def json_denormalize(
    df: pd.DataFrame,
    id_col: str,
    json_col: str,
    field_map: dict,
) -> pd.DataFrame:
    """
    Denormalize a JSON array column into a long-format DataFrame.

    Parameters
    ----------
    id_col    : parent table identifier column (e.g. 'movie_id')
    json_col  : column containing JSON array strings
    field_map : {output_col: json_key} mapping for fields to extract

    Returns
    -------
    Long-format DataFrame with id_col + field_map keys as columns.

    Examples
    --------
    json_denormalize(df, 'id', 'cast',
        field_map={'person_id': 'id', 'name': 'name', 'character': 'character'})
    """
    rows = []
    for _, row in df.iterrows():
        try:
            items = ast.literal_eval(row[json_col])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                entry = {id_col: row[id_col]}
                entry.update({out: item.get(src, None) for out, src in field_map.items()})
                rows.append(entry)
        except Exception:
            pass
    return pd.DataFrame(rows)


# ── Categorical encoding ──────────────────────────────────────────────────────

def multilabel_encode(
    df: pd.DataFrame,
    col: str,
    prefix: str,
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    Multi-label binary encoding for a column containing lists of labels.

    Applies sklearn MultiLabelBinarizer and adds one binary column per label.

    Parameters
    ----------
    col            : column with list values (e.g. ['Action', 'Drama'])
    prefix         : column name prefix (e.g. 'genre' → 'genre_Action')
    drop_original  : drop the source column after encoding

    Returns
    -------
    df with new binary columns appended (and original dropped if requested).

    Examples
    --------
    df['genres_list'] = df['genres'].apply(lambda x: parse_json_list(x))
    df = multilabel_encode(df, 'genres_list', prefix='genre')
    """
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(df[col])
    encoded_df = pd.DataFrame(
        encoded,
        columns=[f"{prefix}_{c}" for c in mlb.classes_],
        index=df.index,
    )
    result = pd.concat([df, encoded_df], axis=1)
    if drop_original:
        result = result.drop(columns=[col])
    n = len(mlb.classes_)
    print(f"multilabel_encode: {n} sütun oluşturuldu ({prefix}_*)")
    return result


def topk_binary_encode(
    df: pd.DataFrame,
    col: str,
    k: int,
    prefix: str,
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    Create binary (0/1) columns for the top-K most frequent labels in a
    list-valued column.

    Parameters
    ----------
    col           : column with list values (e.g. ['sequel', 'based on novel'])
    k             : number of top labels to keep
    prefix        : column name prefix (e.g. 'kw' → 'kw_sequel')
    drop_original : drop the source column after encoding

    Returns
    -------
    df with k new binary columns appended.

    Examples
    --------
    df['keywords_list'] = df['keywords'].apply(lambda x: parse_json_list(x))
    df = topk_binary_encode(df, 'keywords_list', k=20, prefix='kw')
    """
    all_labels: list = []
    for items in df[col]:
        if isinstance(items, list):
            all_labels.extend(items)

    top_k = [label for label, _ in Counter(all_labels).most_common(k)]

    for label in top_k:
        slug = label.lower().replace(" ", "_").replace("-", "_")
        df[f"{prefix}_{slug}"] = df[col].apply(
            lambda x: 1 if isinstance(x, list) and label in x else 0
        )

    print(f"topk_binary_encode: top {k} label → {k} sütun ({prefix}_*)")
    print(f"  Seçilen: {', '.join(top_k[:5])}{'...' if k > 5 else ''}")

    if drop_original:
        df = df.drop(columns=[col])
    return df


# ── Schema inference ─────────────────────────────────────────────────────────

def infer_schema(
    df: pd.DataFrame,
    overrides: dict | None = None,
    description_fn=None,
) -> pd.DataFrame:
    """
    Infer measurement-level schema (Tür / Alt Tür) for each column from
    dtype and basic statistics.

    Inference rules
    ---------------
    object / category → Kategorik / Nominal
    bool              → Kategorik / Nominal
    datetime          → Sayısal   / Aralık (Interval)
    numeric:
      - name ends with '_id' or equals 'id' → Kategorik / Nominal
      - binary (only 0 and 1)               → Kategorik / Nominal
      - min >= 0                             → Sayısal   / Oran (Ratio)
      - otherwise                            → Sayısal   / Aralık (Interval)

    Parameters
    ----------
    df             : DataFrame to inspect
    overrides      : {col: (tür, alt_tür, açıklama)}
                     Use None for any field to keep the inferred value.
                     Override descriptions always take priority over description_fn.
                     Example: {"status": (None, "Ordinal", "Rumored < ... < Released")}
    description_fn : Callable(col, dtype, stats) -> str | None
                     Called for every column whose description is not set via overrides.
                     Receives a stats dict with keys:
                       col, dtype, tür, alt_tür, n_unique, null_pct, min, max, sample
                     Return a non-empty string to use as the description, or None / ""
                     to leave it blank.

                     Example with Claude API::

                         import anthropic
                         client = anthropic.Anthropic()

                         def llm_desc(col, dtype, stats):
                             prompt = (
                                 f"Column name: {col}\\n"
                                 f"dtype: {dtype}, tür: {stats['tür']}, alt_tür: {stats['alt_tür']}\\n"
                                 f"n_unique: {stats['n_unique']}, null%: {stats['null_pct']:.1f}\\n"
                                 f"sample values: {stats['sample']}\\n"
                                 "Write a concise one-line description for this dataset column."
                             )
                             msg = client.messages.create(
                                 model="claude-haiku-4-5-20251001",
                                 max_tokens=80,
                                 messages=[{"role": "user", "content": prompt}],
                             )
                             return msg.content[0].text.strip()

                         display(infer_schema(df, description_fn=llm_desc))

    Returns
    -------
    DataFrame with columns: Sütun, Tür, Alt Tür, Açıklama
    Indexed from 1.

    Examples
    --------
    OVERRIDES = {
        "status":       (None, "Ordinal",            "Rumored < Planned < ... < Released"),
        "release_date": (None, "Aralık (Interval)",  "Takvim yılı — gerçek sıfır yok"),
        "vote_average": (None, "Aralık (Interval)",  "Ortalama puan (0–10) — oransal değil"),
        "budget":       (None, None,                 "Bütçe ($) — 0 = bütçe yok"),
    }
    display(infer_schema(df, overrides=OVERRIDES))
    """
    overrides = overrides or {}
    rows = []

    for col in df.columns:
        dtype = df[col].dtype
        kind  = dtype.kind  # b=bool, i/u/f=numeric, M=datetime, O=object, etc.

        # ── type inference ────────────────────────────────────────────────
        col_lower = col.lower()
        if kind == "b":
            tur, alt = "Kategorik", "Nominal"
        elif hasattr(dtype, "categories") or kind == "O":
            tur, alt = "Kategorik", "Nominal"
        elif kind == "M":
            tur, alt = "Sayısal", "Aralık (Interval)"
        elif kind in ("i", "u", "f"):
            if col_lower == "id" or col_lower.endswith("_id"):
                tur, alt = "Kategorik", "Nominal"
            else:
                uniq = df[col].dropna().unique()
                if set(uniq).issubset({0, 1, 0.0, 1.0}):
                    tur, alt = "Kategorik", "Nominal"
                elif float(df[col].min()) >= 0:
                    tur, alt = "Sayısal", "Oran (Ratio)"
                else:
                    tur, alt = "Sayısal", "Aralık (Interval)"
        else:
            tur, alt = "?", "?"

        # ── apply overrides ───────────────────────────────────────────────
        desc = ""
        if col in overrides:
            ov_tur, ov_alt, ov_desc = overrides[col]
            if ov_tur  is not None: tur  = ov_tur
            if ov_alt  is not None: alt  = ov_alt
            if ov_desc is not None: desc = ov_desc

        # ── description_fn (only when no override description) ────────────
        if not desc and description_fn is not None:
            s = df[col]
            stats = {
                "col":      col,
                "dtype":    str(dtype),
                "tür":      tur,
                "alt_tür":  alt,
                "n_unique": int(s.nunique()),
                "null_pct": float(s.isna().mean() * 100),
                "min":      float(s.min()) if kind in ("i", "u", "f") else None,
                "max":      float(s.max()) if kind in ("i", "u", "f") else None,
                "sample":   s.dropna().head(3).tolist(),
            }
            result_desc = description_fn(col, dtype, stats)
            if result_desc:
                desc = str(result_desc)

        rows.append({"Sütun": col, "Tür": tur, "Alt Tür": alt, "Açıklama": desc})

    result = pd.DataFrame(rows)
    result.index = result.index + 1
    return result


# ── Feature selection ────────────────────────────────────────────────────────

def univariate_screening(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Univariate significance screening with Bonferroni correction.

    Uses chi-squared for binary/categorical features (≤ 10 unique values)
    and Mann-Whitney U for continuous features.

    Parameters
    ----------
    df       : DataFrame containing features and target
    features : list of feature column names to test
    target   : binary target column name
    alpha    : family-wise alpha before Bonferroni correction (default 0.05)

    Returns
    -------
    DataFrame sorted by p-value with columns:
        feature, test, p_value, significant, pos_hit_rate

    Examples
    --------
    screen = univariate_screening(df, FEATURES, 'hit')
    candidates = screen[~screen['significant']]
    """
    bonferroni = alpha / len(features)
    rows = []
    for col in features:
        hit  = df.loc[df[target] == 1, col].dropna()
        miss = df.loc[df[target] == 0, col].dropna()

        if df[col].nunique() <= 10:
            ct = pd.crosstab(df[col], df[target])
            _, p, _, _ = stats.chi2_contingency(ct)
            test = 'chi2'
        else:
            _, p = stats.mannwhitneyu(hit, miss, alternative='two-sided')
            test = 'mwu'

        pos_rate = df.loc[df[col] > 0, target].mean() if df[col].nunique() > 10 \
                   else df.loc[df[col] == 1, target].mean()

        rows.append({
            'feature':      col,
            'test':         test,
            'p_value':      p,
            'significant':  p < bonferroni,
            'pos_hit_rate': round(float(pos_rate), 3),
        })

    result = pd.DataFrame(rows).sort_values('p_value').reset_index(drop=True)
    print(f"univariate_screening: {len(features)} feature, Bonferroni α={bonferroni:.5f}")
    print(f"  Significant: {result['significant'].sum()} / {len(features)}")
    n_fail = (~result['significant']).sum()
    if n_fail:
        fails = result.loc[~result['significant'], 'feature'].tolist()
        print(f"  Bonferroni dışı: {', '.join(fails)}")
    return result


# ── Outlier handling ──────────────────────────────────────────────────────────

def winsorize(
    df: pd.DataFrame,
    cols: list[str],
    lo: float = 0.01,
    hi: float = 0.99,
) -> pd.DataFrame:
    """
    Clip each column at the given percentile bounds (winsorization).

    Parameters
    ----------
    cols : columns to winsorize
    lo   : lower percentile (default 0.01 = 1st percentile)
    hi   : upper percentile (default 0.99 = 99th percentile)

    Returns
    -------
    df with clipped columns (in-place modification on copies).
    """
    report = []
    for col in cols:
        if col not in df.columns:
            continue
        lo_val = df[col].quantile(lo)
        hi_val = df[col].quantile(hi)
        n_clipped = ((df[col] < lo_val) | (df[col] > hi_val)).sum()
        df[col] = df[col].clip(lower=lo_val, upper=hi_val)
        report.append({"col": col, "lo_bound": round(lo_val, 4),
                        "hi_bound": round(hi_val, 4), "n_clipped": int(n_clipped)})

    result = pd.DataFrame(report).set_index("col")
    print(f"winsorize ({lo*100:.0f}–{hi*100:.0f} percentile):")
    print(result.to_string())
    return df
