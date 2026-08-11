"""
Ham TMDB film dict'i → 26 model feature'ı.

/predict/raw'ın kullanıcı dostu girdiler kabul edebilmesi için
preprocess.ipynb / eda.ipynb'deki ön işleme pipeline'ını yeniden üretir.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app.historical_roi import (
    get_actor_roi,
    get_company_roi,
    get_director_collab_count,
    get_director_film_count,
    get_director_roi,
    get_global_prior,
)

_DATA_DIR = Path(__file__).parent.parent / "data"
_CPI_BASE_YEAR = 2010

_MAJOR_STUDIOS = {
    "Warner Bros": ["Warner Bros", "Warner Bros. Pictures", "WB"],
    "Universal":   ["Universal", "Universal Pictures", "Universal Studios"],
    "Disney":      ["Disney", "Walt Disney Pictures", "Walt Disney"],
    "Paramount":   ["Paramount", "Paramount Pictures"],
    "Sony":        ["Sony", "Sony Pictures"],
    "Columbia":    ["Columbia", "Columbia Pictures"],
    "20th Century":["20th Century Fox", "20th Century", "Twentieth Century Fox"],
    "Lionsgate":   ["Lionsgate", "Lions Gate", "Lionsgate Films"],
}

_GENRE_COLS = [
    "genre_Action", "genre_Adventure", "genre_Animation",
    "genre_Documentary", "genre_Drama", "genre_Family",
    "genre_Fantasy", "genre_Horror", "genre_Science Fiction",
]

_FEATURES = [
    "budget_adjusted", "runtime", "num_companies",
    "is_major_studio", "is_us_production", "has_homepage",
    "num_genres", "is_franchise", "is_summer", "is_holiday",
    "director_film_count", "director_collab_count",
    "actor_hist_roi", "company_hist_roi", "monthly_franchise_density",
    "genre_Action", "genre_Adventure", "genre_Animation",
    "genre_Documentary", "genre_Drama", "genre_Family",
    "genre_Fantasy", "genre_Horror", "genre_Science Fiction",
    "kw_independent_film", "kw_sequel",
]

_cpi_cache: dict[int, float] | None = None


def _load_cpi() -> dict[int, float]:
    global _cpi_cache
    if _cpi_cache is None:
        df = pd.read_csv(_DATA_DIR / "cpi_us.csv")
        _cpi_cache = dict(zip(df["year"].astype(int), df["cpi"].astype(float)))
    return _cpi_cache


def _cpi_adjust(budget: float, year: int) -> float:
    cpi = _load_cpi()
    cpi_base = cpi.get(_CPI_BASE_YEAR, 218.1)
    cpi_year = cpi.get(year, cpi_base)
    return budget * (cpi_base / cpi_year)


def _is_major_studio(companies: list[str]) -> int:
    companies_str = " ".join(companies).lower()
    for aliases in _MAJOR_STUDIOS.values():
        for alias in aliases:
            if alias.lower() in companies_str:
                return 1
    return 0


def _monthly_franchise_density(release_date: date, is_franchise: int) -> float:
    """Eğitim verisinden yaklaşık değer: aynı aydaki franchise filmlerinin oranı."""
    try:
        df = pd.read_csv(_DATA_DIR / "tmdb_model.csv")
        same_month = df[
            (df["release_month"] == release_date.month) &
            (df["release_year"] == release_date.year)
        ]
        if len(same_month) == 0:
            same_month = df[df["release_month"] == release_date.month]
        return float(same_month["is_franchise"].mean()) if len(same_month) > 0 else 0.3
    except Exception:
        return 0.3


def transform_raw_movie(raw: dict) -> pd.DataFrame:
    """
    Ham bir film dict'ini 26 model feature'ına sahip tek satırlık DataFrame'e dönüştürür.

    Beklenen anahtarlar:
        budget          : float  — prodüksiyon bütçesi, USD (nominal)
        runtime         : float  — süre, dakika
        release_date    : str    — "YYYY-MM-DD" veya "YYYY"
        genres          : list[str]  — örn. ["Action", "Adventure"]
        production_companies : list[str]
        production_countries : list[str]  — örn. ["US"] veya ["United States of America"]
        belongs_to_collection: bool | int
        homepage        : str | None
        director        : str    — yönetmen adı
        lead_actors     : list[str]  — en fazla 3 başrol oyuncusu adı
        keywords        : list[str]  — örn. ["sequel", "independent film"]
    """
    # ── Vizyon tarihi ─────────────────────────────────────────────────────
    rd_raw = raw.get("release_date", "2010")
    if isinstance(rd_raw, str) and len(rd_raw) >= 10:
        rd = datetime.strptime(rd_raw[:10], "%Y-%m-%d").date()
    elif isinstance(rd_raw, str):
        rd = date(int(rd_raw), 1, 1)
    else:
        rd = rd_raw

    year  = rd.year
    month = rd.month

    # ── Bütçe ─────────────────────────────────────────────────────────────
    budget = float(raw.get("budget", 0) or 0)
    budget_adjusted = _cpi_adjust(budget, year)

    # ── Süre ──────────────────────────────────────────────────────────────
    runtime = float(raw.get("runtime", 0) or 0)

    # ── Yapım şirketleri ──────────────────────────────────────────────────
    companies = [str(c) for c in raw.get("production_companies", [])]
    num_companies  = len(companies)
    is_major_studio = _is_major_studio(companies)
    company_hist_roi = get_company_roi(companies)

    # ── Ülkeler ───────────────────────────────────────────────────────────
    countries = [str(c).upper() for c in raw.get("production_countries", [])]
    is_us_production = int(any(c in ("US", "USA", "UNITED STATES OF AMERICA") for c in countries))

    # ── Pazarlama ─────────────────────────────────────────────────────────
    hp = raw.get("homepage")
    has_homepage = int(bool(hp and str(hp).strip()))

    # ── Türler ────────────────────────────────────────────────────────────
    genres = [str(g) for g in raw.get("genres", [])]
    genre_features = {col: 0 for col in _GENRE_COLS}
    for g in genres:
        key = f"genre_{g}"
        if key in genre_features:
            genre_features[key] = 1
    num_genres = len(genres)

    # ── Franchise / Seri ──────────────────────────────────────────────────
    btc = raw.get("belongs_to_collection", False)
    keywords = [str(k).lower() for k in raw.get("keywords", [])]
    kw_sequel = int("sequel" in keywords)
    kw_independent_film = int("independent film" in keywords)
    is_franchise = int(bool(btc) or kw_sequel or ":" in str(raw.get("title", "")))

    # ── Sezon ─────────────────────────────────────────────────────────────
    is_summer  = int(month in (6, 7, 8))
    is_holiday = int(month in (11, 12))

    # ── Aylık franchise yoğunluğu ─────────────────────────────────────────
    monthly_franchise_density = _monthly_franchise_density(rd, is_franchise)

    # ── Yönetmen ──────────────────────────────────────────────────────────
    director = str(raw.get("director", "") or "")
    director_film_count   = get_director_film_count(director) if director else 0
    director_collab_count = get_director_collab_count(director) if director else 0

    # ── Oyuncular ─────────────────────────────────────────────────────────
    lead_actors = [str(a) for a in raw.get("lead_actors", [])]
    actor_hist_roi = get_actor_roi(lead_actors) if lead_actors else get_global_prior()

    row = {
        "budget_adjusted":           budget_adjusted,
        "runtime":                   runtime,
        "num_companies":             num_companies,
        "is_major_studio":           is_major_studio,
        "is_us_production":          is_us_production,
        "has_homepage":              has_homepage,
        "num_genres":                num_genres,
        "is_franchise":              is_franchise,
        "is_summer":                 is_summer,
        "is_holiday":                is_holiday,
        "director_film_count":       director_film_count,
        "director_collab_count":     director_collab_count,
        "actor_hist_roi":            actor_hist_roi,
        "company_hist_roi":          company_hist_roi,
        "monthly_franchise_density": monthly_franchise_density,
        **genre_features,
        "kw_independent_film":       kw_independent_film,
        "kw_sequel":                 kw_sequel,
    }

    return pd.DataFrame([row])[_FEATURES]
