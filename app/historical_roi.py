"""
Pre-computed historical ROI snapshot tables for inference.

At inference time we don't have access to the full temporal expanding window,
so we use the latest smoothed ROI snapshot for each actor / director / company
derived from the training dataset (tmdb_model.csv + cast/crew CSV files).

Smoothing matches training: additive smoothing K=5, global prior = mean ROI.
"""

import numpy as np
import pandas as pd
from pathlib import Path

_SMOOTH_K   = 5
_ROI_CAP    = 72.0
_DATA_DIR   = Path(__file__).parent.parent / "data"

_actor_table:   dict[str, float] | None = None
_director_table: dict[str, float] | None = None
_company_table: dict[str, float] | None = None
_global_prior:  float | None = None


def _load_tables() -> None:
    global _actor_table, _director_table, _company_table, _global_prior

    df = pd.read_csv(_DATA_DIR / "tmdb_movies_clean.csv")
    cast_df = pd.read_csv(_DATA_DIR / "tmdb_cast_clean.csv")
    crew_df = pd.read_csv(_DATA_DIR / "tmdb_crew_clean.csv")
    companies_df = pd.read_csv(_DATA_DIR / "tmdb_companies_clean.csv")

    df["roi"] = np.where(df["budget"] > 0, df["revenue"] / df["budget"], np.nan)
    roi_map = df.set_index("id")["roi"].to_dict()

    roi_capped = {k: min(v, _ROI_CAP) for k, v in roi_map.items() if pd.notna(v)}
    _global_prior = float(np.mean(list(roi_capped.values()))) if roi_capped else 4.7

    def _smooth(values: list[float]) -> float:
        n = len(values)
        if n == 0:
            return _global_prior
        raw = float(np.mean(values))
        return (n * raw + _SMOOTH_K * _global_prior) / (n + _SMOOTH_K)

    # Actor snapshot
    cast_top3 = cast_df[cast_df["cast_order"] < 3].copy()
    cast_top3["roi_capped"] = cast_top3["movie_id"].map(roi_capped)
    cast_top3 = cast_top3.dropna(subset=["roi_capped"])
    actor_rois: dict[str, list[float]] = {}
    for _, row in cast_top3.iterrows():
        actor_rois.setdefault(row["name"], []).append(row["roi_capped"])
    _actor_table = {name: _smooth(vals) for name, vals in actor_rois.items()}

    # Director snapshot
    directors = crew_df[crew_df["job"] == "Director"].copy()
    directors["roi_capped"] = directors["movie_id"].map(roi_capped)
    directors = directors.dropna(subset=["roi_capped"])
    dir_rois: dict[str, list[float]] = {}
    for _, row in directors.iterrows():
        dir_rois.setdefault(row["name"], []).append(row["roi_capped"])
    _director_table = {name: _smooth(vals) for name, vals in dir_rois.items()}

    # Company snapshot
    companies_df["roi_capped"] = companies_df["movie_id"].map(roi_capped)
    companies_df = companies_df.dropna(subset=["roi_capped"])
    comp_rois: dict[str, list[float]] = {}
    for _, row in companies_df.iterrows():
        comp_rois.setdefault(row["company_name"], []).append(row["roi_capped"])
    _company_table = {name: _smooth(vals) for name, vals in comp_rois.items()}


def get_actor_roi(names: list[str]) -> float:
    if _actor_table is None:
        _load_tables()
    vals = [_actor_table[n] for n in names if n in _actor_table]
    return float(np.log1p(np.mean(vals))) if vals else float(np.log1p(_global_prior))


def get_director_roi(name: str) -> float:
    if _director_table is None:
        _load_tables()
    roi = _director_table.get(name, _global_prior)
    return float(np.log1p(roi))


def get_director_film_count(name: str) -> int:
    if _director_table is None:
        _load_tables()
    crew_df = pd.read_csv(_DATA_DIR / "tmdb_crew_clean.csv")
    count = len(crew_df[(crew_df["job"] == "Director") & (crew_df["name"] == name)])
    return max(0, count - 1)


def get_director_collab_count(name: str) -> int:
    if _director_table is None:
        _load_tables()
    crew_df = pd.read_csv(_DATA_DIR / "tmdb_crew_clean.csv")
    cast_df = pd.read_csv(_DATA_DIR / "tmdb_cast_clean.csv")
    dir_movies = crew_df[(crew_df["job"] == "Director") & (crew_df["name"] == name)]["movie_id"].unique()
    if len(dir_movies) == 0:
        return 0
    collabs = cast_df[cast_df["movie_id"].isin(dir_movies)]["name"].nunique()
    return int(collabs)


def get_company_roi(names: list[str]) -> float:
    if _company_table is None:
        _load_tables()
    vals = [_company_table[n] for n in names if n in _company_table]
    if not vals:
        # Partial match fallback
        for n in names:
            n_lower = n.lower()
            for key, val in _company_table.items():
                if n_lower in key.lower() or key.lower() in n_lower:
                    vals.append(val)
                    break
    return float(np.log1p(np.mean(vals))) if vals else float(np.log1p(_global_prior))


def get_global_prior() -> float:
    if _global_prior is None:
        _load_tables()
    return float(np.log1p(_global_prior))
