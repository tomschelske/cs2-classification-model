# CS2 Round Win-Probability Model — Project Summary

An end-to-end machine-learning pipeline that predicts, from a live snapshot of a
Counter-Strike 2 round, the probability that the Terrorist side wins it — parsed
from professional match demos, trained with leakage-safe splits, served over an
API, and visualized as an animated replay.

**Live demo:** https://claude.ai/code/artifact/6eba7aea-3a9b-457d-af11-535a67a42390

---

## At a glance

| Metric | Value |
|---|---|
| Labeled snapshots | **22,706** (from 44 pro demo maps, 16 series, 2 events) |
| Model accuracy | **80.3%** (held-out, series-level split) |
| Log-loss | **0.415** |
| vs. man-advantage baseline | **+8.9 pp** accuracy, **−79%** log-loss |
| Serving throughput | **11,709 req/s** at **9 ms** p99 |
| Inference optimization | **419×** faster (381 µs → 0.9 µs) |

---

## The problem

Given a mid-round game state — players alive and total HP per side, equipment
value, utility and defuse kits, whether the bomb is planted, time remaining — the
model outputs a single calibrated probability that the T side wins. It's the same
class of model behind win-probability graphics in sports broadcasts. The value
isn't just *who's likely to win* but *how confident we should be*, updated on
every kill and the plant.

## Pipeline

```
HLTV .dem demos
  → parse.py      windowed demoparser2 → data/snapshots.parquet (accumulating)
  → features.py   round + tick data → labeled snapshots (21 features)
  → train.py      baseline / logistic regression / LightGBM → model.pkl + metrics.json
  → api.py        FastAPI POST /predict (folded linear kernel)
  → frontend      animated round-replay win-probability curve
```

## Data & feature engineering

- Parsed **44 maps** across **16 series** (BLAST Bounty S2 + IEM Cologne Major
  2026; ~14 teams, 8 maps) into **22,706 labeled snapshots** — each round sampled
  every 5 s of live play plus every kill and the bomb plant.
- **21 features:** 7 core (alive / total HP per side, bomb planted, time
  remaining, round) + 6 secondary (equipment value, utility, defuse kits, score
  differential per side) + a map one-hot.
- **Memory-bounded parsing.** On an 8 GB machine, the high-level parser OOMs on
  ~450 MB pro demos, so feature extraction requests only the exact ticks it
  samples (a few hundred per map, not millions) — a whole demo becomes a few
  thousand rows in ~5 s.

## Modeling & evaluation

| Model | Accuracy | Log-loss |
|---|---|---|
| Man-advantage baseline | 71.4% | 1.974 |
| **Logistic regression** (deployed) | **80.3%** | **0.415** |
| LightGBM | 76.9% | 0.467 |

- **Leakage-safe splits.** Snapshots from one round share a label and are highly
  correlated; maps within a Bo3 share the same two teams. The split is done at the
  **series level** (`GroupKFold` on `series_id`), so no team appears in both
  training and test.
- **Calibration is the headline.** The baseline emits hard 0/1 guesses (log-loss
  1.97); the trained model emits probabilities that hold up — when it says 70%, T
  wins ~70% of the time.
- **The simpler model won,** chosen by cross-validation: with mostly monotonic
  features, logistic regression generalized better than gradient boosting, which
  overfit series-specific quirks.

## Engineering highlights

- **Serving optimization (419×).** The deployed model is linear, so the
  `StandardScaler + LogisticRegression` pipeline is folded into a single weight
  vector — `P = σ(w·x + b)` — served from an async FastAPI handler as one NumPy
  dot product, instead of building a pandas DataFrame per request. Predictions are
  byte-identical; latency dropped 381 µs → 0.9 µs and the endpoint sustains
  **11,709 req/s at 9 ms p99** (ApacheBench, 4 workers, 0 failures over 20k).
- **Parser reliability.** Diagnosed and pinned around a parser regression whose
  second parse call per process crashed the interpreter, and validated round /
  event schemas against real data (recording artifacts, warmup filtering).

## What the model learned

- **Team economy is the single strongest signal** — equipment value outranks both
  bodies and HP. At an even 5v5 full-HP, being on an **eco drops win probability to
  ~9%**: equal men with no guns is a near-lost round.
- **A bomb plant is worth ~+20 pp** to the T side, holding men and HP fixed.
- **A man advantage compounds late** — a 3v2 (≈85%) is a far bigger edge than a
  5v4 (≈75%); the nonlinear model captures this endgame curvature.
- **Health is an edge even at equal bodies** — a 5v5 where CT has taken heavy
  damage swings ~+27 pp toward T.

## Tech stack

Python 3.13 · demoparser2 / awpy · pandas · scikit-learn · LightGBM · FastAPI /
uvicorn · matplotlib · HTML Canvas (frontend) · Parquet
