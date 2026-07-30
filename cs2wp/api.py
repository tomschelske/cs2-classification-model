"""api.py — FastAPI backend for the model + dashboard.

Endpoints:
  POST /predict  — calibrated P(T win) for a game state (drives the calculator)
  GET  /round    — the precomputed featured-round replay + per-kill leverage
  GET  /health   — status

Takes a game-state snapshot, returns the calibrated probability that the T side
wins the round. Load-tested for p50/p99 latency (see notebooks/phase6_loadtest.py).

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

import json
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/model.pkl")
ROUND_PATH = Path("data/round_navi_falcons.json")

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
    _STATE["round"] = json.loads(ROUND_PATH.read_text()) if ROUND_PATH.exists() else None
    yield
    _STATE.clear()


app = FastAPI(title="CS2 Round Win-Probability", version="1.0.0", lifespan=lifespan)


class GameState(BaseModel):
    """A single round snapshot. Every field has a sensible default so the
    calculator can send just the controls it exposes and let the rest ride."""

    # core (the calculator's primary controls)
    players_alive_t: int = Field(5, ge=0, le=5)
    players_alive_ct: int = Field(5, ge=0, le=5)
    total_health_t: int = Field(500, ge=0, le=500)
    total_health_ct: int = Field(500, ge=0, le=500)
    bomb_planted: bool = False
    time_remaining: float = Field(60.0, ge=0)
    equip_value_t: int = Field(4000, ge=0)
    equip_value_ct: int = Field(4000, ge=0)
    # context (defaulted; behind the calculator's "advanced" toggle)
    utility_t: int = Field(0, ge=0)
    utility_ct: int = Field(0, ge=0)
    defuse_kits_ct: int = Field(0, ge=0, le=5)
    score_diff: int = 0
    round_num: int = Field(12, ge=1)
    map: str = Field("de_dust2", description="e.g. de_dust2, de_mirage, de_nuke")


class Prediction(BaseModel):
    t_win_prob: float = Field(description="Calibrated P(T side wins the round), 0-1")


def _proba(state: GameState) -> float:
    d = state.model_dump()
    d["bomb_planted"] = int(d["bomb_planted"])
    map_col = f"map_{d.pop('map')}"
    # map_* one-hot columns (from train's build_X) are set from the map string;
    # everything else is read straight off the request.
    x = np.fromiter(
        ((1.0 if f == map_col else 0.0) if f.startswith("map_") else float(d[f])
         for f in _STATE["features"]),
        dtype=np.float64, count=len(_STATE["features"]))
    kernel = _STATE["kernel"]
    if kernel is not None:                      # fast linear path
        w, b = kernel
        return float(1.0 / (1.0 + np.exp(-(np.dot(w, x) + b))))
    return float(_STATE["model"].predict_proba(x.reshape(1, -1))[0, 1])  # fallback


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": _STATE.get("name"),
            "fast_kernel": _STATE.get("kernel") is not None,
            "round_loaded": _STATE.get("round") is not None}


@app.get("/round")
def featured_round() -> dict:
    """The precomputed featured-round replay (curve + events with leverage)."""
    return _STATE.get("round") or {}


@app.post("/predict", response_model=Prediction)
async def predict(state: GameState) -> Prediction:
    """Return the calibrated probability that the T side wins this round."""
    return Prediction(t_win_prob=round(_proba(state), 4))
