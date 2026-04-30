"""FastAPI inference endpoint for TMDB hit classifier."""

import logging
import os
from contextlib import asynccontextmanager

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

REGISTRY_NAME = "tmdb-hit-classifier"
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_STAGE   = os.getenv("MODEL_STAGE", "Production")
THRESHOLD       = float(os.getenv("PREDICT_THRESHOLD", "0.5"))
MAX_BATCH_SIZE  = int(os.getenv("MAX_BATCH_SIZE", "500"))

# "genre_Science Fiction" contains a space — kept as-is for DataFrame column alignment.
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

_model = None
_model_version: str = "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version
    mlflow.set_tracking_uri(MLFLOW_URI)
    _model = mlflow.pyfunc.load_model(f"models:/{REGISTRY_NAME}/{MODEL_STAGE}")
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(REGISTRY_NAME, stages=[MODEL_STAGE])
    _model_version = versions[0].version if versions else "unknown"
    logger.info("Model loaded: %s v%s (%s)", REGISTRY_NAME, _model_version, MODEL_STAGE)
    yield


app = FastAPI(
    title="TMDB Hit Classifier",
    description="Movie box-office hit prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ──────────────────────────────────────────────────────────────────

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
        d["genre_Science Fiction"] = d.pop("genre_science_fiction")
        return pd.DataFrame([d])[FEATURES]


class PredictResponse(BaseModel):
    probability: float
    hit: bool
    threshold: float
    model_version: str


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


# ── Endpoints ─────────────────────────────────────────────────────────────────

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


def _predict_proba(df: pd.DataFrame) -> list[float]:
    raw = _model.predict(df)
    # LightGBM pyfunc binary: 1D probability array
    # sklearn pyfunc: may return 2D (n_samples, 2)
    if hasattr(raw, "ndim") and raw.ndim == 2:
        return raw[:, 1].tolist()
    return [float(v) for v in raw]


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
