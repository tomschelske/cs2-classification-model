"""Model-level checks: the served fast kernel is mathematically equivalent to the
sklearn pipeline (the whole basis of the 419x serving optimization), and
predictions stay well-behaved."""
import pickle

import numpy as np

from cs2wp.api import GameState, MODEL_PATH, _build_linear_kernel

_BUNDLE = pickle.load(open(MODEL_PATH, "rb"))
_MODEL, _FEATS = _BUNDLE["model"], _BUNDLE["features"]


def _vector(**overrides) -> np.ndarray:
    """Build the model's feature vector the same way the API does."""
    d = GameState(**overrides).model_dump()
    d["bomb_planted"] = int(d["bomb_planted"])
    map_col = f"map_{d.pop('map')}"
    return np.array([(1.0 if f == map_col else 0.0) if f.startswith("map_") else float(d[f])
                     for f in _FEATS], dtype=np.float64)


def test_kernel_matches_pipeline():
    """sigmoid(w.x + b) must equal the sklearn pipeline's predict_proba."""
    w, b = _build_linear_kernel(_MODEL)
    states = [
        {}, {"equip_value_t": 800, "equip_value_ct": 20000},
        {"players_alive_t": 2, "players_alive_ct": 4, "bomb_planted": True},
        {"map": "de_nuke", "total_health_t": 250}, {"players_alive_ct": 0},
    ]
    for st in states:
        x = _vector(**st)
        kernel_p = 1.0 / (1.0 + np.exp(-(w @ x + b)))
        pipeline_p = _MODEL.predict_proba(x.reshape(1, -1))[0, 1]
        assert abs(kernel_p - pipeline_p) < 1e-6, f"kernel != pipeline for {st}"


def test_predictions_in_unit_interval():
    w, b = _build_linear_kernel(_MODEL)
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = _vector(players_alive_t=int(rng.integers(0, 6)),
                    players_alive_ct=int(rng.integers(0, 6)),
                    equip_value_t=int(rng.integers(0, 25000)),
                    equip_value_ct=int(rng.integers(0, 25000)))
        p = 1.0 / (1.0 + np.exp(-(w @ x + b)))
        assert 0.0 <= p <= 1.0


def test_kernel_is_available():
    assert _build_linear_kernel(_MODEL) is not None   # deployed model must be linear
