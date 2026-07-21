"""train.py — snapshots.parquet -> model.pkl + metrics.json (Phase 4).

Trains three predictors and compares them on a held-out set:
  1. man-advantage baseline (from baseline.py) — the bar
  2. logistic regression
  3. gradient boosting (LightGBM)

Critical: split at the MATCH level (GroupShuffleSplit / GroupKFold on
match_id), never row-level. See project doc, section 6.

Records for each model: accuracy, log-loss, and a calibration curve, into
metrics.json. The gradient-boosted model beating baseline on both metrics is
the Phase 4 exit criterion.
"""

from __future__ import annotations

from pathlib import Path

SNAPSHOTS_PATH = Path("data/snapshots.parquet")
MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("models/metrics.json")


def main() -> None:
    """TODO (Phase 4):
    1. Load snapshots.parquet.
    2. GroupShuffleSplit on match_id -> train / (val) / test.
    3. Evaluate baseline on test: accuracy + log-loss.
    4. Fit logistic regression; evaluate.
    5. Fit LightGBM; evaluate. Consider CalibratedClassifierCV.
    6. Write metrics.json (all three) and pickle the best model.
    """
    raise NotImplementedError("Phase 4: implement training + evaluation")


if __name__ == "__main__":
    main()
