"""Phase 5 — model interpretation.

Answers concrete CS2 questions in probability terms, using the DEPLOYED model
(logistic regression, best by CV in train.py). Counterfactual predictions are
the most defensible way to say "a plant is worth X%": hold a canonical game
state fixed and change one thing.

Emits: models/winprob_by_alive.png (P(T win) over the alive grid),
models/feature_effects.png (deployed logreg coefficients), interpretation.json.

Run:  PYTHONPATH=. .venv/bin/python notebooks/phase5_interpret.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from cs2wp.features import LABEL
from cs2wp.train import RANDOM_STATE, SPLIT_KEY, build_X, _make_lgbm, _make_logreg

DATA = Path("data/snapshots.parquet")
OUT = Path("models")
FULL_BUY = 4300   # $ equip per living player on a full buy
ECO = 500         # $ equip per living player on an eco
_COLS: list[str] = []   # full model feature order (set in main)


def state(t_alive, ct_alive, *, t_hp=None, ct_hp=None, planted=0, time_left=60.0,
          rnd=12, equip_t=None, equip_ct=None, util_t=None, util_ct=None,
          kits=None, score_diff=0, map="de_mirage") -> pd.DataFrame:
    """One canonical snapshot. Defaults: full HP and a full buy for both sides."""
    a = lambda v, per, n: per * n if v is None else v
    row = {
        "players_alive_t": t_alive, "players_alive_ct": ct_alive,
        "total_health_t": a(t_hp, 100, t_alive), "total_health_ct": a(ct_hp, 100, ct_alive),
        "bomb_planted": int(planted), "time_remaining": time_left, "round_num": rnd,
        "equip_value_t": a(equip_t, FULL_BUY, t_alive),
        "equip_value_ct": a(equip_ct, FULL_BUY, ct_alive),
        "utility_t": a(util_t, 1, t_alive), "utility_ct": a(util_ct, 1, ct_alive),
        "defuse_kits_ct": kits if kits is not None else ct_alive,
        "score_diff": score_diff, "map": map,
    }
    return build_X(pd.DataFrame([row])).reindex(columns=_COLS, fill_value=0)


def main() -> None:
    global _COLS
    df = pd.read_parquet(DATA)
    X = build_X(df)
    _COLS = list(X.columns)
    y = df[LABEL].astype(int)
    groups = df[SPLIT_KEY]
    tr, _ = next(GroupShuffleSplit(1, test_size=0.25,
                                   random_state=RANDOM_STATE).split(X, y, groups))
    logreg = _make_logreg().fit(X.iloc[tr], y.iloc[tr])
    lgbm = _make_lgbm().fit(X.iloc[tr], y.iloc[tr])
    p = lambda **kw: float(logreg.predict_proba(state(**kw))[0, 1])
    pg = lambda **kw: float(lgbm.predict_proba(state(**kw))[0, 1])

    print("=" * 66)
    print("PHASE 5 FINDINGS  (P(T win); LR = deployed logistic reg, GB = LightGBM)")
    print("=" * 66)

    # [1] man advantage across fight sizes (economy held equal at full buy)
    print("\n[1] 5v4 vs 3v2 — does a man-up matter more in a smaller fight?")
    print(f"    {'scenario':10s} {'LR':>6s} {'GB':>6s}")
    for t, c in [(5, 5), (5, 4), (3, 3), (3, 2), (2, 1)]:
        print(f"    {f'{t}v{c}':10s} {p(t_alive=t, ct_alive=c):6.2f} {pg(t_alive=t, ct_alive=c):6.2f}")
    print(f"    LR 5v4 {p(t_alive=5, ct_alive=4):.2f} ~= 3v2 {p(t_alive=3, ct_alive=2):.2f} (linear); "
          f"GB 3v2 {pg(t_alive=3, ct_alive=2):.2f} > 5v4 {pg(t_alive=5, ct_alive=4):.2f} (endgame curve)")

    # [2] the value of a bomb plant
    print("\n[2] A BOMB PLANT SWINGS WIN PROBABILITY TOWARD T")
    plant_d = []
    for n in (5, 4, 3, 2):
        base = p(t_alive=n, ct_alive=n, planted=0, time_left=40)
        d = p(t_alive=n, ct_alive=n, planted=1, time_left=40) - base
        plant_d.append(d)
        print(f"    {n}v{n} even: {base:.2f} -> {base + d:.2f}  (+{d * 100:.0f} pp)")
    print(f"    => a plant is worth ~+{np.mean(plant_d) * 100:.0f} pp, men & HP fixed.")

    # [3] NEW — equipment (eco vs full buy), the feature added in iteration two
    even = p(t_alive=5, ct_alive=5)
    t_eco = p(t_alive=5, ct_alive=5, equip_t=ECO * 5)
    ct_eco = p(t_alive=5, ct_alive=5, equip_ct=ECO * 5)
    print("\n[3] EQUIPMENT DECIDES EVEN-NUMBERS FIGHTS (5v5, full HP)")
    print(f"    both full-buy      -> {even:.2f}")
    print(f"    T on eco (CT buy)  -> {t_eco:.2f}   ({(t_eco - even) * 100:+.0f} pp)")
    print(f"    CT on eco (T buy)  -> {ct_eco:.2f}   ({(ct_eco - even) * 100:+.0f} pp)")

    # [4] collective HP at equal bodies
    ct_hurt = p(t_alive=5, ct_alive=5, ct_hp=250)
    print("\n[4] HEALTH IS AN EDGE EVEN AT EQUAL BODIES (5v5, full buy)")
    print(f"    5v5 CT at 250 HP -> {ct_hurt:.2f}   (+{(ct_hurt - even) * 100:.0f} pp for T)")

    findings = {
        "plant_value_pp_avg": round(float(np.mean(plant_d)) * 100, 1),
        "equipment_5v5": {"t_eco_pp": round((t_eco - even) * 100, 1),
                          "ct_eco_pp": round((ct_eco - even) * 100, 1)},
        "hp_5v5_ct_at_250_pp": round((ct_hurt - even) * 100, 1),
        "man_up_5v4_vs_3v2_GB": {"5v4": round(pg(t_alive=5, ct_alive=4), 2),
                                 "3v2": round(pg(t_alive=3, ct_alive=2), 2)},
    }
    (OUT / "interpretation.json").write_text(json.dumps(findings, indent=2))
    _heatmap(logreg)
    _feature_effects(logreg)
    print(f"\nSaved {OUT/'interpretation.json'}, {OUT/'winprob_by_alive.png'}, "
          f"{OUT/'feature_effects.png'}")


def _heatmap(logreg) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.array([[logreg.predict_proba(state(t, c))[0, 1] for c in range(6)]
                     for t in range(6)])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(grid, origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
    for t in range(6):
        for c in range(6):
            ax.text(c, t, f"{grid[t, c]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xlabel("CT players alive"); ax.set_ylabel("T players alive")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_title("P(T win) by players alive (full HP + buy, mid-round)")
    fig.colorbar(im, label="P(T win)")
    fig.savefig(OUT / "winprob_by_alive.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _feature_effects(logreg, top=12) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    coef = logreg.named_steps["logisticregression"].coef_[0]
    order = np.argsort(np.abs(coef))[::-1][:top][::-1]   # top-N, ascending for barh
    feats = [_COLS[i] for i in order]
    vals = coef[order]
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in vals]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(feats, vals, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("logreg coefficient (standardized) — >0 favors T, <0 favors CT")
    ax.set_title(f"Top {top} feature effects on P(T win)  (deployed logistic regression)")
    for y, v in enumerate(vals):
        ax.text(v + (0.03 if v >= 0 else -0.03), y, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    fig.savefig(OUT / "feature_effects.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
