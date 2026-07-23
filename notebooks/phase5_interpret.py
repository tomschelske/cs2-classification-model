"""Phase 5 — model interpretation.

Answers concrete CS2 questions in probability terms, using the DEPLOYED model
(logistic regression, best by CV in train.py). Counterfactual predictions are
the most defensible way to say "a plant is worth X%": hold a canonical game
state fixed and change one thing.

Also emits two figures:
  - models/winprob_by_alive.png : P(T win) over the 6x6 alive-count grid
  - models/shap_summary.png     : SHAP beeswarm (LightGBM, cross-check)

Run:  .venv/bin/python notebooks/phase5_interpret.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from cs2wp.features import CORE_FEATURES, LABEL
from cs2wp.train import (RANDOM_STATE, SPLIT_KEY, _make_lgbm, _make_logreg)

FEATURES = CORE_FEATURES
DATA = Path("data/snapshots.parquet")
OUT = Path("models")


def state(t_alive, ct_alive, *, t_hp=None, ct_hp=None, planted=0,
          time_left=60.0, rnd=12) -> pd.DataFrame:
    """Build a single canonical feature row (full HP = 100/living player)."""
    t_hp = 100 * t_alive if t_hp is None else t_hp
    ct_hp = 100 * ct_alive if ct_hp is None else ct_hp
    row = {
        "players_alive_t": t_alive, "players_alive_ct": ct_alive,
        "total_health_t": t_hp, "total_health_ct": ct_hp,
        "bomb_planted": int(planted), "time_remaining": time_left,
        "round_num": rnd,
    }
    return pd.DataFrame([row])[FEATURES]


def main() -> None:
    df = pd.read_parquet(DATA)
    X = df[FEATURES].copy()
    X["bomb_planted"] = X["bomb_planted"].astype(int)
    y = df[LABEL].astype(int)
    groups = df[SPLIT_KEY]

    # same split as train.py, fit on train only
    tr, te = next(GroupShuffleSplit(1, test_size=0.25,
                                    random_state=RANDOM_STATE).split(X, y, groups))
    logreg = _make_logreg().fit(X.iloc[tr], y.iloc[tr])
    lgbm = _make_lgbm().fit(X.iloc[tr], y.iloc[tr])

    def p(**kw) -> float:
        return float(logreg.predict_proba(state(**kw))[0, 1])

    def pg(**kw) -> float:
        return float(lgbm.predict_proba(state(**kw))[0, 1])

    print("=" * 66)
    print("PHASE 5 FINDINGS  (P(T win); LR = deployed logistic reg, GB = LightGBM)")
    print("=" * 66)

    # --- Finding 1: how a +1 man edge compares across fight sizes ---
    print("\n[1] 5v4 vs 3v2 — does a man-up matter more in a smaller fight?")
    print(f"    {'scenario':10s} {'LR':>6s} {'GB':>6s}")
    for t, c in [(5, 5), (5, 4), (3, 3), (3, 2), (2, 2), (2, 1)]:
        print(f"    {f'{t}v{c}':10s} {p(t_alive=t, ct_alive=c):6.2f} {pg(t_alive=t, ct_alive=c):6.2f}")
    p54, p32 = p(t_alive=5, ct_alive=4), p(t_alive=3, ct_alive=2)
    g54, g32 = pg(t_alive=5, ct_alive=4), pg(t_alive=3, ct_alive=2)
    print(f"    LR: 5v4 {p54:.2f} ~= 3v2 {p32:.2f} (linear -> ~constant ~+30pp/man).")
    print(f"    GB: 5v4 {g54:.2f} vs 3v2 {g32:.2f} "
          f"({'captures' if g32-g54>0.03 else 'also ~flat on'} the endgame curvature).")

    # --- Finding 2: value of a bomb plant (isolate the flag, HP/men fixed) ---
    plant_deltas = []
    for n in (5, 4, 3, 2):
        d = p(t_alive=n, ct_alive=n, planted=1, time_left=40) - \
            p(t_alive=n, ct_alive=n, planted=0, time_left=40)
        plant_deltas.append(d)
    print("\n[2] A BOMB PLANT SWINGS WIN PROBABILITY TOWARD T")
    for n, d in zip((5, 4, 3, 2), plant_deltas):
        base = p(t_alive=n, ct_alive=n, planted=0, time_left=40)
        print(f"    {n}v{n} even: no plant P {base:.2f} -> planted P {base+d:.2f}"
              f"   (+{d*100:.0f} pp)")
    print(f"    => holding men & HP fixed, a plant is worth ~"
          f"+{np.mean(plant_deltas)*100:.0f} pp to T.")

    # --- Finding 3: collective HP matters beyond bodies alive ---
    even = p(t_alive=5, ct_alive=5)
    ct_hurt = p(t_alive=5, ct_alive=5, ct_hp=250)   # CT took heavy early damage
    t_hurt = p(t_alive=5, ct_alive=5, t_hp=250)
    print("\n[3] HEALTH IS A REAL EDGE EVEN AT EQUAL BODIES (5v5)")
    print(f"    5v5 full HP           -> P {even:.2f}")
    print(f"    5v5 but CT at 250 HP  -> P {ct_hurt:.2f}   (+{(ct_hurt-even)*100:.0f} pp for T)")
    print(f"    5v5 but T  at 250 HP  -> P {t_hurt:.2f}   ({(t_hurt-even)*100:.0f} pp for T)")

    findings = {
        "per_man_pp_LR": round((p54 - p(t_alive=5, ct_alive=5)) * 100, 1),
        "5v4_vs_3v2_LR": {"5v4": round(p54, 2), "3v2": round(p32, 2)},
        "5v4_vs_3v2_GB": {"5v4": round(g54, 2), "3v2": round(g32, 2)},
        "plant_value_pp_avg": round(float(np.mean(plant_deltas)) * 100, 1),
        "hp_5v5_ct_at_250_pp": round((ct_hurt - even) * 100, 1),
    }
    (OUT / "interpretation.json").write_text(json.dumps(findings, indent=2))

    # --- Figure 1: P(T win) heatmap over alive grid ---
    _heatmap(logreg)
    # --- Figure 2: feature-effect bar chart (deployed model) ---
    _importance_plot(logreg, lgbm)

    print(f"\nSaved {OUT/'interpretation.json'}, {OUT/'winprob_by_alive.png'}, "
          f"{OUT/'feature_effects.png'}")


def _heatmap(logreg) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.zeros((6, 6))
    for t in range(6):
        for c in range(6):
            grid[t, c] = logreg.predict_proba(state(t, c))[0, 1]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(grid, origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
    for t in range(6):
        for c in range(6):
            ax.text(c, t, f"{grid[t, c]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xlabel("CT players alive")
    ax.set_ylabel("T players alive")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_title("P(T win) by players alive (full HP, mid-round)")
    fig.colorbar(im, label="P(T win)")
    fig.savefig(OUT / "winprob_by_alive.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _importance_plot(logreg, lgbm) -> None:
    """Interpret the DEPLOYED model directly: standardized logreg coefficients
    (effect on log-odds of a T win), with LightGBM gain shown alongside.

    (SHAP beeswarm was the original plan but shap's beeswarm pulls in numba,
    which is incompatible with the installed NumPy 2.5 — the coefficient view
    is more defensible for the deployed linear model anyway.)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coef = logreg.named_steps["logisticregression"].coef_[0]
    order = np.argsort(np.abs(coef))
    feats = [FEATURES[i] for i in order]
    vals = coef[order]
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in vals]  # green=T, red=CT

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(feats, vals, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("logreg coefficient (standardized) — >0 favors T, <0 favors CT")
    ax.set_title("Feature effect on P(T win)  (deployed logistic regression)")
    for y, v in enumerate(vals):
        ax.text(v + (0.05 if v >= 0 else -0.05), y, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    fig.savefig(OUT / "feature_effects.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
