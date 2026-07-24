# CS2 Round Win-Probability Model

A machine-learning pipeline that predicts round outcomes in Counter-Strike 2
from live game state. Given a snapshot of a round in progress — players alive
per side, collective health, equipment, bomb state, time remaining — it outputs
a calibrated probability that the Terrorist side wins the round.

Full plan: [`cs2-win-probability-project.md`](cs2-win-probability-project.md).

## Results

Built from **22,706 labeled snapshots** parsed out of **44 pro demo maps** across
**16 series** (BLAST Bounty S2 + IEM Cologne Major 2026), split at the **series
level** so no team appears in both training and test. **21 features**: 7 core
(alive/HP per side, bomb, time, round) + 6 secondary (equipment value, utility,
defuse kits, score differential) + a map one-hot.

| | Accuracy | Log-loss |
|---|---|---|
| Man-advantage baseline | 71.4% | 1.974 |
| **Logistic regression** (deployed) | **80.3%** | **0.415** |
| LightGBM | 76.9% | 0.467 |

The trained model beats the baseline by **+8.9 pp accuracy** and cuts **log-loss
79%** — the headline being *calibration*: it turns hard 0/1 guesses into
probabilities that hold up (see `models/calibration.png`). **Equipment value is
the single strongest feature** — the eco/full-buy signal a bodies-and-HP model is
blind to. Served through FastAPI at **11,709 req/s, p99 9 ms** after folding the
pipeline into a single weight vector (**419× faster** inference than the naive
sklearn+DataFrame path).

**▶ Live demo — animated win-probability replay of a held-out round:**
<https://claude.ai/code/artifact/6eba7aea-3a9b-457d-af11-535a67a42390>

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
| `cs2wp/parse.py` | corpus runner: `data/demos/**/*.dem` -> `snapshots.parquet` |
| `cs2wp/features.py` | one demo -> labeled snapshot rows (windowed `demoparser2`) |
| `cs2wp/baseline.py` | man-advantage rule — the bar to beat |
| `cs2wp/train.py` | snapshots -> `model.pkl` + `metrics.json` (series-level split) |
| `cs2wp/api.py` | FastAPI `POST /predict` (folded linear kernel) |
| `notebooks/` | phase 1 exploration, interpretation, load test, replay data |
| `data/demos/` | raw `.dem` files (gitignored) |
| `data/snapshots.parquet` | cached dataset (gitignored) |
| `models/` | `model.pkl` (gitignored) + `metrics.json`, figures (versioned) |
| `frontend/replay.html` | animated round-replay page (self-contained) |

## Setup

Requires Python **3.11–3.13** (`awpy` caps below 3.14; demos are parsed via
`demoparser2`, pinned to 0.41.2 — see `requirements.txt`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # also needs libomp for LightGBM: brew install libomp
```

## Usage

```bash
# 1. Drop .dem files (or extract HLTV .rar) into data/demos/, then build the dataset:
python -m cs2wp.parse --skip-existing        # add --prune to delete demos after parsing
# 2. Train + evaluate (baseline / logreg / LightGBM):
python -m cs2wp.train                         # -> models/model.pkl, metrics.json
# 3. Serve:
uvicorn cs2wp.api:app --port 8000 --workers 4  # POST /predict ; docs at /docs
# Interpretation figures + round-replay data:
python notebooks/phase5_interpret.py
```

## Build phases — all complete ✅

1. **Parse** — one demo end-to-end; per-tick schema mapped (memory-safe windowed parse).
2. **Features** — `features.py` -> labeled snapshots; three hand-validated.
3. **Corpus** — 16 series -> `snapshots.parquet` (accumulating, prune-as-you-go).
4. **Models** — baseline / logreg / LightGBM, series-level splits -> `metrics.json`.
5. **Interpretation** — 3 findings (plant ≈ +20 pp; man-advantage compounds late;
   HP matters at equal bodies) + figures.
6. **Serve** — FastAPI `/predict`, load-tested (11.7k rps, p99 9 ms).
7. **Visualize** — animated replay of a held-out round (linked above).

## Two things that make or break the project

- **Match-level splits.** Snapshots from one round share a label and are highly
  correlated; a random row split leaks them across train/test and inflates
  accuracy. Split whole matches.
- **A documented baseline.** The man-advantage rule (~mid-to-high 60s accuracy)
  is the bar. The headline metric is how far the trained model beats it.
