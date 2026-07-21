# CS2 Round Win-Probability Model

A machine-learning pipeline that predicts round outcomes in Counter-Strike 2
from live game state. Given a snapshot of a round in progress — players alive
per side, collective health, equipment, bomb state, time remaining — it outputs
a calibrated probability that the Terrorist side wins the round.

Full plan: [`cs2-win-probability-project.md`](cs2-win-probability-project.md).

## Pipeline

```
.dem demos --> parse.py --> features.py --> snapshots.parquet
                                                  |
                                                  v
                          train.py (baseline / logreg / LightGBM)
                                                  |
                                       model.pkl + metrics.json
                                                  |
                                                  v
                                    api.py (FastAPI POST /predict)
                                                  |
                                                  v
                                    frontend (live win-prob curve)
```

## Layout

| Path | Purpose |
|---|---|
| `cs2wp/parse.py` | `.dem` -> parsed demo data (awpy) |
| `cs2wp/features.py` | parsed data -> labeled snapshot rows |
| `cs2wp/baseline.py` | man-advantage rule — the bar to beat |
| `cs2wp/train.py` | snapshots -> `model.pkl` + `metrics.json` |
| `cs2wp/api.py` | FastAPI `POST /predict` |
| `data/demos/` | raw `.dem` files (gitignored) |
| `data/snapshots.parquet` | cached dataset (gitignored) |
| `models/` | `model.pkl` (gitignored) + `metrics.json` (versioned) |
| `notebooks/` | Phase 1 schema exploration |
| `frontend/` | round-replay visualization |

## Setup

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build phases

1. **Parse one demo** end-to-end; understand `ticks`/`rounds` schemas.
2. **Feature extraction** for one demo; hand-validate three snapshots.
3. **Scale** to 50–100 demos -> `snapshots.parquet` (parallel, log failures).
4. **Baseline + models** with match-level splits -> `metrics.json`.
5. **Interpretation** — SHAP / feature importance; three defensible findings.
6. **Serve** — FastAPI `/predict`; record p50/p99 latency.
7. **Visualize** — replay a held-out round with a live probability curve.

## Two things that make or break the project

- **Match-level splits.** Snapshots from one round share a label and are highly
  correlated; a random row split leaks them across train/test and inflates
  accuracy. Split whole matches.
- **A documented baseline.** The man-advantage rule (~mid-to-high 60s accuracy)
  is the bar. The headline metric is how far the trained model beats it.
