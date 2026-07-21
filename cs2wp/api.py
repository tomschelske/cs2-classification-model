"""api.py — model.pkl -> FastAPI POST /predict (Phase 6).

One endpoint. Takes a game-state snapshot, returns a calibrated T win
probability. Load-test it and record p50/p99 latency + sustained req/s.

Run:  uvicorn cs2wp.api:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/model.pkl")

app = FastAPI(title="CS2 Round Win-Probability", version="0.1.0")


class GameState(BaseModel):
    """Core features — must match features.CORE_FEATURES."""

    players_alive_t: int = Field(ge=0, le=5)
    players_alive_ct: int = Field(ge=0, le=5)
    total_health_t: int = Field(ge=0)
    total_health_ct: int = Field(ge=0)
    bomb_planted: bool
    time_remaining: float
    round_num: int = Field(ge=1)


class Prediction(BaseModel):
    t_win_prob: float


_model = None  # lazy-loaded pickled model


def _load_model():
    global _model
    if _model is None:
        import pickle

        with open(MODEL_PATH, "rb") as fh:
            _model = pickle.load(fh)
    return _model


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=Prediction)
def predict(state: GameState) -> Prediction:
    """TODO (Phase 6): featurize `state`, run the model, return calibrated prob."""
    raise NotImplementedError("Phase 6: wire model inference")
