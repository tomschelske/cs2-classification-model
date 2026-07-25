# CS2 Round Win-Probability Model

**A machine learning pipeline that predicts round outcomes in Counter-Strike 2 from live game state.**

---

## 1. Project Summary

Given a snapshot of a Counter-Strike 2 round in progress — how many players are alive on each side, their collective health, equipment value, whether the bomb has been planted, how much time remains — the model outputs a probability that the Terrorist side wins the round.

This is the same class of model behind win-probability graphs in traditional sports broadcasts. The output is a single calibrated number between 0 and 1, updated continuously as the round evolves.

The project spans three distinct engineering surfaces:

- **Data engineering** — parsing thousands of binary demo files into a clean, labeled tabular dataset
- **Machine learning** — training and calibrating a classifier, benchmarked against a meaningful baseline
- **Deployment** — exposing the model through an API and a small visualization frontend

---

## 2. Why This Project

Most portfolio projects fail the "so what?" test on a resume because they produce no measurable outcome. A CRUD application has no numbers attached to it. This project generates hard metrics as a natural byproduct of building it correctly:

| Metric type | Where it comes from |
|---|---|
| Dataset scale | Number of demos parsed, rounds extracted, snapshots generated |
| Model quality | Accuracy and log-loss vs. a documented naive baseline |
| Engineering performance | Parsing throughput before and after optimization |
| Serving latency | p50/p99 inference time under load test |

None of these require exaggeration. They fall out of doing the work.

---

## 3. Data Source

### Demo files

Counter-Strike 2 records matches as `.dem` files containing full tick-level game state. Professional match demos are freely available from:

- **HLTV** — the primary archive of professional Counter-Strike matches, downloadable per match
- **FACEIT** — high-level matchmaking demos
- **CS2Stats** — aggregated demo hosting

Use **GOTV demos** (the broadcast/server-side recordings), not POV demos. POV demos are recorded from a single player's client and have significantly higher parse error rates.

A single professional match contains roughly 20-40 rounds. Fifty to one hundred matches yields a dataset in the low tens of thousands of rounds — more than sufficient for a well-performing tabular model.

### Parsing

The `awpy` library (Python ≥ 3.11) handles demo parsing, backed by a Rust parser under the hood:

```bash
pip install awpy
```

```python
from awpy import Demo

dem = Demo("g2-vs-navi.dem")
dem.parse()

dem.header    # map name, server info
dem.rounds    # round boundaries, winner, score
dem.ticks     # per-tick player state: health, armor, position, equipment
dem.kills     # kill events
dem.damages   # damage events
dem.bomb      # plant / defuse events
dem.grenades  # utility usage
```

Output arrives as Polars DataFrames, convertible to Pandas via `.to_pandas()`. Parsing a single demo takes roughly 4-5 seconds.

---

## 4. Problem Formulation

**Task:** Binary classification.

**Unit of observation:** A single *snapshot* — the state of one round at one moment in time.

**Label:** Which side eventually won that round (known from `dem.rounds`).

**Sampling strategy:** Extract snapshots at fixed intervals within each round — for example every 5 seconds of round time, plus event-triggered snapshots immediately after each kill and after a bomb plant. A 90-second round therefore produces ~18 training rows rather than one.

### Candidate features

**Core (implement first):**
- Players alive, T side
- Players alive, CT side
- Total health, T side
- Total health, CT side
- Bomb planted (boolean)
- Time remaining in round (or time since plant, if planted)
- Round number / half

**Secondary (add in iteration two):**
- Equipment value per side
- Utility remaining per side (smokes, flashes, molotovs, HE)
- Defuse kits present on CT side
- Map (one-hot encoded)
- Bomb site, if planted
- Score differential in the match

**Deliberately excluded:** player positions and raw coordinates. These are powerful but require spatial encoding that will balloon scope. Note them as future work.

---

## 5. The Baseline

**This section is the single most important part of the project.** A model reported without a baseline is uninterpretable.

Implement a trivial rule-based predictor first:

> Whichever side has more players alive wins. Ties resolve to CT.

Measure its accuracy on the test set. This typically lands somewhere in the mid-to-high 60% range — man advantage is genuinely the dominant signal in Counter-Strike. That number is the bar. The trained model must beat it by a margin large enough to be meaningful, and the delta between the two is the headline metric of the entire project.

Also record log-loss for both, since accuracy alone hides calibration quality. A model that says "80% T win" should see T win about 80% of the time.

---

## 6. Critical Methodology Note: Data Leakage

Snapshots taken from the same round are highly correlated with each other and share a label. A naive random row-level train/test split will place snapshots from the same round on both sides of the split, and the model will appear far more accurate than it is.

**Split at the match level.** Assign whole matches to train / validation / test. Never allow rounds — let alone snapshots — from one match to appear in more than one split.

Being able to explain this decision in an interview is worth more than several percentage points of accuracy.

---

## 7. Architecture

```
HLTV demos (.dem)
      |
      v
[ parse.py ]  awpy -> Polars DataFrames
      |
      v
[ features.py ]  round + tick data -> labeled snapshots
      |
      v
snapshots.parquet   <- cached dataset, parse once
      |
      v
[ train.py ]  baseline vs. logistic regression vs. gradient boosting
      |
      v
model.pkl + metrics.json
      |
      v
[ api.py ]  FastAPI: POST /predict -> {"t_win_prob": 0.73}
      |
      v
[ frontend ]  round timeline with live probability curve
```

**Stack:**

| Layer | Choice |
|---|---|
| Parsing | `awpy` (Python 3.11+) |
| Data handling | Polars / Pandas, Parquet for storage |
| Modeling | scikit-learn (baseline, logistic regression), LightGBM or XGBoost |
| Serving | FastAPI + Uvicorn |
| Frontend | React + Recharts, or plain HTML with a charting library |
| Compute | Local machine is sufficient; Google Colab free tier as an alternative |

No GPU is required. This is a tabular problem — gradient boosting on tens of thousands of rows trains in seconds to minutes on CPU. The only meaningfully time-consuming step is the initial parse across many demo files, which is I/O and CPU bound and trivially parallelizable across processes.

---

## 8. Build Phases

### Phase 1 — Parse one demo end to end
Download a single HLTV demo. Parse it with `awpy`. Print and inspect every DataFrame. Understand the schema of `ticks` and `rounds` before writing any pipeline code. Confirm you can correctly identify the winner of round 7.

*Exit criterion:* You can explain what a tick is and how rounds are delimited in the data.

### Phase 2 — Feature extraction for one demo
Write `features.py`: a function taking parsed demo data and returning a DataFrame of labeled snapshots. Validate by hand — pick three snapshots and manually confirm the alive counts and label are correct.

*Exit criterion:* One demo produces a correct snapshot table.

### Phase 3 — Scale to a corpus
Download 50-100 demos. Run the pipeline across all of them, in parallel. Write results to a single Parquet file. Log parse failures rather than crashing on them; some demos will be corrupt or truncated.

*Exit criterion:* A single `snapshots.parquet` with tens of thousands of labeled rows, and a recorded count of demos parsed vs. failed.

**Optimization opportunity:** Time this step naively first, then optimize (multiprocessing, selective property parsing, skipping unneeded tick fields). The before/after difference is a legitimate engineering metric.

### Phase 4 — Baseline and models
Implement the man-advantage baseline. Then logistic regression. Then gradient boosting. Match-level splits throughout. Record accuracy, log-loss, and a calibration curve for each.

*Exit criterion:* A `metrics.json` comparing all three, with the gradient-boosted model beating baseline on both metrics.

### Phase 5 — Interpretation
Produce SHAP or feature-importance plots. Answer concrete questions: how much is a bomb plant worth in win probability? How does a 5v4 differ from a 3v2? This is what makes the project interesting to talk about rather than just a scikit-learn tutorial.

*Exit criterion:* Three specific, defensible findings about Counter-Strike, expressed in probability terms.

### Phase 6 — Serve it
Wrap the model in FastAPI. One endpoint, `POST /predict`, taking a game state and returning a probability. Load test it and record p50/p99 latency and sustained requests per second.

*Exit criterion:* A running API with measured latency figures.

### Phase 7 — Visualize
A small frontend that replays a held-out round, showing the win-probability curve moving in real time against the round timeline, annotated with kills and the bomb plant. This is the artifact that makes the project legible to someone glancing at it for fifteen seconds.

*Exit criterion:* A shareable demo, ideally deployed.

---

## 9. Scope Discipline

Things that will tempt you and should be deferred to a "future work" section rather than attempted now:

- Player position and spatial features (requires map-aware encoding)
- Sequence models over full round trajectories (LSTM/transformer)
- Player-identity or skill-rating features
- Live in-game overlay integration
- Economy modeling across rounds

A finished project with seven features and honest metrics is worth considerably more than an abandoned one with forty.

---

## 10. Open Questions to Resolve During Phase 1

- What exactly does one row of `dem.ticks` represent, and at what frequency are ticks recorded in the demo?
- How is round time represented — absolute tick, or seconds since freeze-time end?
- How are warmup rounds and knife rounds marked, and how will they be filtered out?
- Does the bomb DataFrame distinguish plant, defuse, and explosion cleanly?
- How are overtime halves handled in the round numbering?

Answer these against real data before writing feature code.
