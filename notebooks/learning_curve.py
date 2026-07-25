"""Learning curve — is the corpus big enough?

Trains the deployed model on an increasing number of series and measures
held-out performance. If the curve is still climbing, more data would help; if
it has plateaued, the model is feature-limited (more of the same demos mainly
shrinks variance rather than raising the ceiling).

The independent unit here is the *series* (~897 rounds / 16 series), not the
22.7k correlated snapshots — so the x-axis is number of training series.

Run:  PYTHONPATH=. .venv/bin/python notebooks/learning_curve.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupShuffleSplit

from cs2wp.features import LABEL
from cs2wp.train import RANDOM_STATE, SPLIT_KEY, build_X, _make_logreg

DATA = Path("data/snapshots.parquet")
OUT = Path("models/learning_curve.png")
SIZES = [2, 3, 4, 6, 8, 10, 12]
REPS = 12          # random series subsets averaged per size (variance estimate)
CLIP = 1e-3


def main() -> None:
    df = pd.read_parquet(DATA)
    X, y, g = build_X(df), df[LABEL].astype(int), df[SPLIT_KEY].values

    # fixed held-out test = same 4 series as train.py; vary the training pool
    tr, te = next(GroupShuffleSplit(1, test_size=0.25,
                                    random_state=RANDOM_STATE).split(X, y, g))
    pool = list(pd.unique(g[tr]))                 # 12 training series
    Xte, yte = X.iloc[te], y.iloc[te]
    rng = np.random.default_rng(0)

    rows = []
    print(f"{'train_series':>12} {'accuracy':>16} {'log_loss':>16}")
    for n in SIZES:
        accs, lls = [], []
        for _ in range(REPS if n < len(pool) else 1):
            sel = set(rng.choice(pool, size=n, replace=False))
            m = np.isin(g, list(sel)); m[te] = False
            idx = np.where(m)[0]
            mdl = _make_logreg().fit(X.iloc[idx], y.iloc[idx])
            p = mdl.predict_proba(Xte)[:, 1]
            accs.append(accuracy_score(yte, (p >= 0.5).astype(int)))
            lls.append(log_loss(yte, np.clip(p, CLIP, 1 - CLIP), labels=[0, 1]))
        rows.append((n, np.mean(accs), np.std(accs), np.mean(lls), np.std(lls)))
        print(f"{n:>12} {np.mean(accs):>9.3f}±{np.std(accs):.3f} "
              f"{np.mean(lls):>9.3f}±{np.std(lls):.3f}")

    _plot(np.array(rows))
    print(f"\nSaved {OUT}")


def _plot(r) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].errorbar(r[:, 0], r[:, 1], yerr=r[:, 2], marker="o", color="#b1740f",
                   capsize=3, lw=2)
    ax[0].set(xlabel="# training series", ylabel="held-out accuracy",
              title="Accuracy vs training data")
    ax[1].errorbar(r[:, 0], r[:, 3], yerr=r[:, 4], marker="o", color="#2b74b0",
                   capsize=3, lw=2)
    ax[1].set(xlabel="# training series", ylabel="held-out log-loss",
              title="Log-loss vs training data")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("Learning curve — both metrics plateau; error bars shrink with more series",
                 fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
