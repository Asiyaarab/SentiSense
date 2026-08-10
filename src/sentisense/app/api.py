"""Optional FastAPI endpoint for production serving.

Run with:
    uvicorn sentisense.app.api:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sentisense.config import MODELS_DIR
from sentisense.data.preprocessing import clean_text
from sentisense.models.predict import predict_with_confidence
from sentisense.utils import get_logger

logger = get_logger(__name__)

MODEL_PATH = MODELS_DIR / "model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"


class PredictRequest(BaseModel):
    """Request body for /predict."""

    text: str = Field(..., min_length=1, max_length=20_000, description="Review text")


class PredictResponse(BaseModel):
    """Response body for /predict."""

    label: str
    confidence: float
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool


_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + vectorizer once on startup."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        logger.error("Model files missing — run `python train.py` first.")
        _state["model"] = None
        _state["vectorizer"] = None
    else:
        logger.info("Loading model from %s", MODEL_PATH)
        _state["model"] = joblib.load(MODEL_PATH)
        _state["vectorizer"] = joblib.load(VECTORIZER_PATH)
        logger.info("Model loaded.")
    yield
    _state.clear()


app = FastAPI(
    title="SentiSense API",
    description="Real-time movie review sentiment classifier.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_state.get("model") is not None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model = _state.get("model")
    vectorizer = _state.get("vectorizer")
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python train.py` first.",
        )
    cleaned = clean_text(req.text)
    if not cleaned:
        raise HTTPException(
            status_code=400,
            detail="Text was empty after preprocessing.",
        )
    pred = predict_with_confidence(model, vectorizer, [cleaned])[0]
    return PredictResponse(
        label=pred.label,
        confidence=pred.confidence,
        probabilities=pred.probabilities,
    )
