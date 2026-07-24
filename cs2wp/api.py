"""api.py — model.pkl -> FastAPI POST /predict (Phase 6).

One endpoint. Takes a game-state snapshot, returns the calibrated probability
that the T side wins the round. Load-test it and record p50/p99 latency +
sustained req/s (see notebooks/phase6_loadtest.py).

Serving optimization: the deployed model is a StandardScaler + LogisticRegression
pipeline, i.e. a linear model. Rather than build a pandas DataFrame and call
sklearn per request (slow, and a sync handler serializes on the GIL), we fold
the scaler and classifier into a single weight vector once at startup —
P(T win) = sigmoid(w . x + b) — and serve it from an async handler as a plain
NumPy dot product. This cut p99 latency ~20x and lifted throughput ~10x (see
models/serving_metrics.json). A non-linear model falls back to predict_proba.

Run:  .venv/bin/uvicorn cs2wp.api:app --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/model.pkl")

_STATE: dict = {}


def _build_linear_kernel(model):
    """Fold StandardScaler + LogisticRegression into (w, b) so that
    P(T win) = sigmoid(w . x + b). Returns None if the model isn't that shape."""
    try:
        scaler = model.named_steps["standardscaler"]
        clf = model.named_steps["logisticregression"]
    except (AttributeError, KeyError):
        return None
    coef = clf.coef_[0]
    w = coef / scaler.scale_
    b = float(clf.intercept_[0] - np.dot(coef, scaler.mean_ / scaler.scale_))
    return w.astype(np.float64), b


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(MODEL_PATH, "rb") as fh:
        bundle = pickle.load(fh)
    _STATE["model"] = bundle["model"]
    _STATE["features"] = bundle["features"]
    _STATE["name"] = bundle.get("model_name", "model")
    _STATE["kernel"] = _build_linear_kernel(bundle["model"])
    yield
    _STATE.clear()


app = FastAPI(title="CS2 Round Win-Probability", version="1.0.0", lifespan=lifespan)


class GameState(BaseModel):
    """A single round snapshot — must match features.CORE_FEATURES."""

    players_alive_t: int = Field(ge=0, le=5, examples=[2])
    players_alive_ct: int = Field(ge=0, le=5, examples=[1])
    total_health_t: int = Field(ge=0, le=500, examples=[200])
    total_health_ct: int = Field(ge=0, le=500, examples=[81])
    bomb_planted: bool = Field(examples=[True])
    time_remaining: float = Field(ge=0, examples=[32.0])
    round_num: int = Field(ge=1, examples=[13])


class Prediction(BaseModel):
    t_win_prob: float = Field(description="Calibrated P(T side wins the round), 0-1")


def _proba(state: GameState) -> float:
    feats = _STATE["features"]
    row = state.model_dump()
    row["bomb_planted"] = int(row["bomb_planted"])
    x = np.fromiter((row[f] for f in feats), dtype=np.float64, count=len(feats))
    kernel = _STATE["kernel"]
    if kernel is not None:                      # fast linear path
        w, b = kernel
        return float(1.0 / (1.0 + np.exp(-(np.dot(w, x) + b))))
    return float(_STATE["model"].predict_proba(x.reshape(1, -1))[0, 1])  # fallback


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": _STATE.get("name"),
            "fast_kernel": _STATE.get("kernel") is not None,
            "features": _STATE.get("features")}


@app.post("/predict", response_model=Prediction)
async def predict(state: GameState) -> Prediction:
    """Return the calibrated probability that the T side wins this round."""
    return Prediction(t_win_prob=round(_proba(state), 4))
