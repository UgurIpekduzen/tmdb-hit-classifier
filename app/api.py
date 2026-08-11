"""TMDB hit sınıflandırıcısı için FastAPI inference endpoint'i."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.monitoring import (
    compute_feature_importance,
    drift_report,
    load_reference,
    prediction_drift,
)
from app.inference import transform_raw_movie

logger = logging.getLogger(__name__)

REGISTRY_NAME = "tmdb-hit-classifier"
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_STAGE   = os.getenv("MODEL_STAGE", "Production")
THRESHOLD       = float(os.getenv("PREDICT_THRESHOLD", "0.5"))
MAX_BATCH_SIZE  = int(os.getenv("MAX_BATCH_SIZE", "500"))
DATASET_PATH    = os.getenv("DATASET_PATH", "data/tmdb_model.csv")

# "genre_Science Fiction" içinde boşluk var — DataFrame sütun hizalaması için olduğu gibi bırakıldı.
FEATURES: list[str] = [
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

def _predict_proba_raw(df: pd.DataFrame) -> list[float]:
    """`app` tam başlatılmadan önce de kullanılabilen bağımsız predict yardımcı fonksiyonu."""
    raw = _model.predict(df)
    if hasattr(raw, "ndim") and raw.ndim == 2:
        return raw[:, 1].tolist()
    return [float(v) for v in raw]


_model = None
_model_version: str = "unknown"
_reference: pd.DataFrame | None = None
_ref_proba: list[float] | None = None
_feature_importance: dict[str, float] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _reference, _ref_proba, _feature_importance
    mlflow.set_tracking_uri(MLFLOW_URI)
    _model = mlflow.pyfunc.load_model(f"models:/{REGISTRY_NAME}/{MODEL_STAGE}")
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(REGISTRY_NAME, stages=[MODEL_STAGE])
    _model_version = versions[0].version if versions else "unknown"
    logger.info("Model loaded: %s v%s (%s)", REGISTRY_NAME, _model_version, MODEL_STAGE)
    try:
        _reference = load_reference(DATASET_PATH, FEATURES)
        _ref_proba = _predict_proba_raw(_reference)
        _feature_importance = compute_feature_importance(_predict_proba_raw, _reference, FEATURES)
        logger.info(
            "Reference loaded: %d rows | top feature: %s",
            len(_reference),
            max(_feature_importance, key=_feature_importance.get),
        )
    except Exception as exc:
        logger.warning("Could not load reference distribution: %s", exc)
    yield


app = FastAPI(
    title="TMDB Hit Classifier",
    description="Movie box-office hit prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Şemalar ──────────────────────────────────────────────────────────────────

class MovieInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    budget_adjusted: float = Field(
        default=0.0, ge=0,
        description="CPI-adjusted production budget (USD, 2010 base year)")
    runtime: float = Field(
        default=0.0, ge=0, le=600,
        description="Film runtime in minutes (0–600)")
    num_companies: int = Field(
        default=1, ge=0,
        description="Number of production companies")
    is_major_studio: int = Field(
        default=0, ge=0, le=1,
        description="1 if produced by a major studio (Warner, Universal, Disney, Sony, Paramount, Fox)")
    is_us_production: int = Field(
        default=0, ge=0, le=1,
        description="1 if US production")
    has_homepage: int = Field(
        default=0, ge=0, le=1,
        description="1 if the film has a dedicated website")
    num_genres: int = Field(
        default=1, ge=0, le=20,
        description="Number of genres assigned to the film")
    is_franchise: int = Field(
        default=0, ge=0, le=1,
        description="1 if part of a known franchise or series")
    is_summer: int = Field(
        default=0, ge=0, le=1,
        description="1 if released in summer window (June–August)")
    is_holiday: int = Field(
        default=0, ge=0, le=1,
        description="1 if released in holiday window (November–December)")
    director_film_count: int = Field(
        default=1, ge=0,
        description="Number of feature films directed before this one")
    director_collab_count: int = Field(
        default=0, ge=0,
        description="Number of unique cast/crew collaborations by the director")
    actor_hist_roi: float = Field(
        default=0.0, ge=0,
        description="Lead actor historical ROI — log1p-transformed; 0 = no history")
    company_hist_roi: float = Field(
        default=0.0, ge=0,
        description="Production company historical ROI — log1p-transformed; 0 = no history")
    monthly_franchise_density: float = Field(
        default=0.0, ge=0, le=1,
        description="Fraction of franchise films among all releases in the same month (0–1)")
    genre_Action: int = Field(default=0, ge=0, le=1, description="Genre: Action")
    genre_Adventure: int = Field(default=0, ge=0, le=1, description="Genre: Adventure")
    genre_Animation: int = Field(default=0, ge=0, le=1, description="Genre: Animation")
    genre_Documentary: int = Field(default=0, ge=0, le=1, description="Genre: Documentary")
    genre_Drama: int = Field(default=0, ge=0, le=1, description="Genre: Drama")
    genre_Family: int = Field(default=0, ge=0, le=1, description="Genre: Family")
    genre_Fantasy: int = Field(default=0, ge=0, le=1, description="Genre: Fantasy")
    genre_Horror: int = Field(default=0, ge=0, le=1, description="Genre: Horror")
    genre_science_fiction: int = Field(
        default=0, ge=0, le=1,
        alias="genre_Science Fiction", description="Genre: Science Fiction")
    kw_independent_film: int = Field(default=0, ge=0, le=1, description="Keyword: independent film")
    kw_sequel: int = Field(default=0, ge=0, le=1, description="Keyword: sequel")

    def to_dataframe(self) -> pd.DataFrame:
        d = self.model_dump(by_alias=False)
        # Pydantic alanı boşluk içeremediği için gerçek sütun adına geri çeviriyoruz
        d["genre_Science Fiction"] = d.pop("genre_science_fiction")
        return pd.DataFrame([d])[FEATURES]


class PredictResponse(BaseModel):
    probability: float
    hit: bool
    threshold: float
    model_version: str


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


class FeatureDriftRow(BaseModel):
    feature: str
    n_ref: int
    n_prod: int
    mean_ref: float | None
    mean_prod: float | None
    mean_shift: float | None
    ks_stat: float | None
    ks_pvalue: float | None
    ks_drift: bool | None
    psi: float | None
    psi_level: str
    importance: float | None
    risk_score: float | None
    drift: bool


class PredictionDrift(BaseModel):
    mean_ref: float
    mean_prod: float
    mean_shift: float
    pct_hit_ref: float
    pct_hit_prod: float
    ks_stat: float
    ks_pvalue: float
    ks_drift: bool
    psi: float | None
    psi_level: str
    drift: bool


class DriftResponse(BaseModel):
    n_features: int
    n_drifted: int
    n_high_risk: int
    features: list[FeatureDriftRow]
    prediction_drift: PredictionDrift | None


# ── Endpoint'ler ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/info")
def info():
    return {
        "registry":   REGISTRY_NAME,
        "stage":      MODEL_STAGE,
        "version":    _model_version,
        "threshold":  THRESHOLD,
        "n_features": len(FEATURES),
        "features":   FEATURES,
    }


@app.post("/drift", response_model=DriftResponse)
def drift(movies: list[MovieInput]):
    """
    Yakın zamandaki bir istek grubunu eğitim referans dağılımıyla karşılaştırır.

    Feature bazlı drift (yön ve risk skoru ile) ve tahmin drift'ini döndürür.
    """
    if _reference is None:
        raise HTTPException(status_code=503, detail="Reference distribution not available")
    if not movies:
        raise HTTPException(status_code=422, detail="At least one sample required")
    if len(movies) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(movies)} exceeds limit of {MAX_BATCH_SIZE}",
        )
    import numpy as np
    prod_df    = pd.concat([m.to_dataframe() for m in movies], ignore_index=True)
    report     = drift_report(_reference, prod_df, FEATURES, feature_importance=_feature_importance)
    rows       = [FeatureDriftRow(**r) for r in report.to_dict(orient="records")]
    pred_drift = None
    if _ref_proba is not None:
        prod_proba = _predict_proba_raw(prod_df)
        pred_drift = PredictionDrift(**prediction_drift(
            np.array(_ref_proba), np.array(prod_proba)
        ))
    return DriftResponse(
        n_features=len(rows),
        n_drifted=sum(r.drift for r in rows),
        # risk_score = psi × importance; 0.1 üstü "yüksek risk" olarak kabul edilir
        n_high_risk=sum(1 for r in rows if r.risk_score is not None and r.risk_score > 0.1),
        features=rows,
        prediction_drift=pred_drift,
    )


def _predict_proba(df: pd.DataFrame) -> list[float]:
    return _predict_proba_raw(df)


@app.post("/predict", response_model=PredictResponse)
def predict(movie: MovieInput):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    proba = _predict_proba(movie.to_dataframe())[0]
    return PredictResponse(
        probability=round(proba, 4),
        hit=proba >= THRESHOLD,
        threshold=THRESHOLD,
        model_version=_model_version,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(movies: list[MovieInput]):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not movies:
        return BatchPredictResponse(predictions=[])
    if len(movies) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(movies)} exceeds limit of {MAX_BATCH_SIZE}",
        )
    df = pd.concat([m.to_dataframe() for m in movies], ignore_index=True)
    probas = _predict_proba(df)
    return BatchPredictResponse(predictions=[
        PredictResponse(
            probability=round(p, 4),
            hit=p >= THRESHOLD,
            threshold=THRESHOLD,
            model_version=_model_version,
        )
        for p in probas
    ])


# ── Ham girdi şeması ─────────────────────────────────────────────────────────

class RawMovieInput(BaseModel):
    """Kullanıcı dostu ham film girdisi. Feature'lar otomatik olarak türetilir."""
    title:                  str              = Field(default="", description="Film adı")
    budget:                 float            = Field(default=0.0, ge=0, description="Prodüksiyon bütçesi (nominal USD)")
    runtime:                float            = Field(default=0.0, ge=0, le=600, description="Süre (dakika)")
    release_date:           str              = Field(default="2010-01-01", description="Vizyon tarihi (YYYY-MM-DD)")
    genres:                 list[str]        = Field(default=[], description="Türler — örn. ['Action', 'Adventure']")
    production_companies:   list[str]        = Field(default=[], description="Yapım şirketleri")
    production_countries:   list[str]        = Field(default=[], description="Yapım ülkeleri — örn. ['US']")
    belongs_to_collection:  bool             = Field(default=False, description="Franchise / seri parçası mı?")
    homepage:               Optional[str]    = Field(default=None, description="Resmi web sitesi URL (yoksa boş bırakın)")
    director:               str              = Field(default="", description="Yönetmen adı")
    lead_actors:            list[str]        = Field(default=[], description="Başrol oyuncuları (en fazla 3 isim)")
    keywords:               list[str]        = Field(default=[], description="Anahtar kelimeler — örn. ['sequel', 'independent film']")


@app.post("/predict/raw", response_model=PredictResponse)
def predict_raw(movie: RawMovieInput):
    """
    Ham, kullanıcı dostu film verisini kabul eder ve 26 model feature'ını otomatik türetir.

    CPI bütçe düzeltmesini, tür/keyword encoding'ini, majör stüdyo tespitini,
    sezon feature'larını ve eğitim verisinden tarihsel ROI sorgusunu üstlenir.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        df = transform_raw_movie(movie.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Feature extraction failed: {exc}")
    proba = _predict_proba(df)[0]
    return PredictResponse(
        probability=round(proba, 4),
        hit=proba >= THRESHOLD,
        threshold=THRESHOLD,
        model_version=_model_version,
    )
