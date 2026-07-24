"""train.py — snapshots.parquet -> model.pkl + metrics.json (Phase 4).

Trains three predictors and compares them on a held-out set:
  1. man-advantage baseline (baseline.py) — the bar
  2. logistic regression (scaled)
  3. gradient boosting (LightGBM)

CRITICAL: split at the SERIES level (series_id), never row- or map-level. Maps
within a Bo3/Bo5 share the same two teams, so grouping by series stops those
teams leaking across train/test (doc §6). We hold out ~25% of series as a
final test set and use GroupKFold on the rest to gauge fold-to-fold variance
(which also tells us whether the corpus is big enough yet).

Writes models/metrics.json (accuracy, log-loss, Brier, CV mean±std, and the
headline model-minus-baseline delta), models/model.pkl, and a calibration plot.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cs2wp.baseline import predict_t_win
from cs2wp.features import CATEGORICAL, CORE_FEATURES, LABEL, SECONDARY_FEATURES

SNAPSHOTS_PATH = Path("data/snapshots.parquet")
MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("models/metrics.json")
CALIB_PATH = Path("models/calibration.png")

SPLIT_KEY = "series_id"   # group by series, NOT match_id (maps share teams)
NUMERIC = CORE_FEATURES + SECONDARY_FEATURES   # 7 core + 6 secondary
RANDOM_STATE = 42
PROBA_CLIP = 1e-3         # keep baseline's hard 0/1 out of log(0)


def build_X(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric feature matrix: core + secondary, plus one-hot categoricals
    (map). One-hot columns are folded into the numeric matrix so the deployed
    StandardScaler+LogReg still reduces to a single linear kernel at serve time."""
    X = df[NUMERIC].copy()
    X["bomb_planted"] = X["bomb_planted"].astype(int)
    for cat in CATEGORICAL:
        X = pd.concat([X, pd.get_dummies(df[cat], prefix=cat).astype(int)], axis=1)
    return X


def _make_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def _make_lgbm():
    # Heavily regularized: with only 7 smooth features the win-prob surface is
    # nearly monotonic, so shallow, broad-leaf trees generalize far better than
    # a deep GBM (which overfits series-specific quirks).
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15, max_depth=4,
        min_child_samples=200, reg_lambda=10.0, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.9, random_state=RANDOM_STATE, verbose=-1,
    )


def _scores(y_true, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    p = np.clip(proba, PROBA_CLIP, 1 - PROBA_CLIP)
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "log_loss": round(float(log_loss(y_true, p, labels=[0, 1])), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
    }


def _baseline_proba(X: pd.DataFrame) -> np.ndarray:
    """Man-advantage rule as a probability (hard 0/1, clipped for log-loss)."""
    hard = np.array([predict_t_win(t, c) for t, c in
                     zip(X["players_alive_t"], X["players_alive_ct"])], dtype=float)
    return np.clip(hard, PROBA_CLIP, 1 - PROBA_CLIP)


def _cv(make_model, X, y, groups, n_splits=5) -> dict:
    """GroupKFold CV -> mean/std of accuracy & log-loss across held-out folds."""
    gkf = GroupKFold(n_splits=n_splits)
    accs, lls = [], []
    for tr, va in gkf.split(X, y, groups):
        m = make_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        proba = m.predict_proba(X.iloc[va])[:, 1]
        accs.append(accuracy_score(y.iloc[va], (proba >= 0.5).astype(int)))
        lls.append(log_loss(y.iloc[va], np.clip(proba, PROBA_CLIP, 1 - PROBA_CLIP),
                            labels=[0, 1]))
    return {
        "cv_accuracy_mean": round(float(np.mean(accs)), 4),
        "cv_accuracy_std": round(float(np.std(accs)), 4),
        "cv_log_loss_mean": round(float(np.mean(lls)), 4),
        "cv_log_loss_std": round(float(np.std(lls)), 4),
    }


def _save_calibration_plot(curves: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, (frac_pos, mean_pred) in curves.items():
        ax.plot(mean_pred, frac_pos, "o-", label=name)
    ax.set_xlabel("mean predicted P(T win)")
    ax.set_ylabel("observed T win rate")
    ax.set_title("Calibration (held-out test)")
    ax.legend(loc="upper left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_parquet(SNAPSHOTS_PATH)
    X_all = build_X(df)
    features = list(X_all.columns)   # full ordered list incl. map_* one-hots
    y_all = df[LABEL].astype(int)
    groups_all = df[SPLIT_KEY]

    n_series = groups_all.nunique()
    print(f"Data: {len(df)} snapshots, {df['match_id'].nunique()} maps, "
          f"{n_series} series | {len(features)} features | overall T-win {y_all.mean():.3f}")

    # --- series-level held-out test split ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(gss.split(X_all, y_all, groups_all))
    Xtr, Xte = X_all.iloc[tr_idx], X_all.iloc[te_idx]
    ytr, yte = y_all.iloc[tr_idx], y_all.iloc[te_idx]
    gtr = groups_all.iloc[tr_idx]
    print(f"Train: {len(Xtr)} rows / {gtr.nunique()} series  |  "
          f"Test: {len(Xte)} rows / {groups_all.iloc[te_idx].nunique()} series "
          f"(held-out: {sorted(groups_all.iloc[te_idx].unique())})\n")

    results: dict[str, dict] = {}
    calib_curves: dict[str, tuple] = {}

    # 1) baseline
    base_proba = _baseline_proba(Xte)
    results["baseline_man_advantage"] = _scores(yte, base_proba)
    calib_curves["baseline"] = calibration_curve(yte, base_proba, n_bins=10)

    # 2) logistic regression
    logreg = _make_logreg().fit(Xtr, ytr)
    lr_proba = logreg.predict_proba(Xte)[:, 1]
    results["logistic_regression"] = {**_scores(yte, lr_proba),
                                      **_cv(_make_logreg, Xtr, ytr, gtr)}
    calib_curves["logreg"] = calibration_curve(yte, lr_proba, n_bins=10)

    # 3) LightGBM
    lgbm = _make_lgbm().fit(Xtr, ytr)
    gb_proba = lgbm.predict_proba(Xte)[:, 1]
    results["lightgbm"] = {**_scores(yte, gb_proba),
                          **_cv(_make_lgbm, Xtr, ytr, gtr)}
    calib_curves["lightgbm"] = calibration_curve(yte, gb_proba, n_bins=10)

    # pick the best trained model by CV log-loss (chosen without touching test)
    trained = {"logistic_regression": logreg, "lightgbm": lgbm}
    best_name = min(trained, key=lambda n: results[n]["cv_log_loss_mean"])
    best_model = trained[best_name]

    # headline delta: best model vs baseline
    base_acc = results["baseline_man_advantage"]["accuracy"]
    base_ll = results["baseline_man_advantage"]["log_loss"]
    headline = {
        "best_model": best_name,
        "accuracy_gain_pp": round((results[best_name]["accuracy"] - base_acc) * 100, 2),
        "log_loss_reduction": round(base_ll - results[best_name]["log_loss"], 4),
    }

    # feature importances (gain) for the LightGBM model
    importances = dict(sorted(
        zip(features, (lgb.Booster(model_str=lgbm.booster_.model_to_string())
                       .feature_importance(importance_type="gain")).tolist()),
        key=lambda kv: -kv[1]))

    # --- report ---
    print(f"{'model':26s} {'acc':>7s} {'logloss':>9s} {'brier':>7s} "
          f"{'cv_acc':>14s} {'cv_logloss':>16s}")
    for name, r in results.items():
        cva = (f"{r['cv_accuracy_mean']:.3f}±{r['cv_accuracy_std']:.3f}"
               if "cv_accuracy_mean" in r else "—")
        cvl = (f"{r['cv_log_loss_mean']:.3f}±{r['cv_log_loss_std']:.3f}"
               if "cv_log_loss_mean" in r else "—")
        print(f"{name:26s} {r['accuracy']:7.3f} {r['log_loss']:9.3f} "
              f"{r['brier']:7.3f} {cva:>14s} {cvl:>16s}")
    print(f"\nHEADLINE  best model = {best_name} vs baseline: "
          f"{headline['accuracy_gain_pp']:+} pp accuracy, "
          f"{headline['log_loss_reduction']:+} log-loss (lower is better)")
    print("Top feature importance (gain):",
          {k: round(v) for k, v in list(importances.items())[:12]})

    # --- persist ---
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump({"model": best_model, "model_name": best_name,
                     "features": features, "split_key": SPLIT_KEY}, fh)
    METRICS_PATH.write_text(json.dumps({
        "dataset": {"snapshots": len(df), "maps": int(df["match_id"].nunique()),
                    "series": int(n_series), "t_win_rate": round(float(y_all.mean()), 4)},
        "test_series": sorted(groups_all.iloc[te_idx].unique()),
        "features": features,
        "models": results,
        "headline_vs_baseline": headline,
        "lightgbm_feature_importance_gain": {k: round(v) for k, v in importances.items()},
    }, indent=2))
    _save_calibration_plot(calib_curves, CALIB_PATH)
    print(f"\nSaved {MODEL_PATH}, {METRICS_PATH}, {CALIB_PATH}")


if __name__ == "__main__":
    main()
