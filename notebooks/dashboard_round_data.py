"""Generate the dashboard's featured-round JSON: dense win-probability replay
PLUS a per-kill "leverage" value.

Leverage answers, at each kill, "how much did this duel's outcome swing the win
probability?" It's an INSTANTANEOUS comparison (not a simulation): from the
pre-duel roster, we score the state if the victim dies (what happened) vs. if the
attacker had died instead, and take the difference. The model reasons over
aggregate state, so this is "which side lost a body," not player-specific skill.

Run:  PYTHONPATH=. .venv/bin/python notebooks/dashboard_round_data.py \
        <rar> <map_substr> <round_num> <out.json>
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from demoparser2 import DemoParser

from cs2wp.features import (BOMB_TIME_S, GRENADES, ROUND_TIME_S, TEAM_CT, TEAM_T,
                            TICK_PROPS, _detect_tick_rate, _round_windows)
from cs2wp.train import build_X

DENSE_S = 1.0     # replay curve resolution (seconds)
PRE_TICKS = 24    # snapshot the pre-duel roster ~0.4s before each kill

_BUNDLE = pickle.load(open("models/model.pkl", "rb"))
_MODEL, _FEATS = _BUNDLE["model"], _BUNDLE["features"]


def _extract(rar: str) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="dash_"))
    subprocess.run(["unar", "-q", "-o", str(tmp), rar], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp


def _nades(inv) -> int:
    return sum(i in GRENADES for i in inv) if inv is not None else 0


def _aggregate(rows: pd.DataFrame) -> dict:
    """Sum a set of per-player rows into the model's aggregate features."""
    is_t, is_ct = rows.team_num == TEAM_T, rows.team_num == TEAM_CT
    alive_t, alive_ct = rows.is_alive & is_t, rows.is_alive & is_ct
    nd = rows.inventory.apply(_nades)
    return {
        "players_alive_t": int(alive_t.sum()), "players_alive_ct": int(alive_ct.sum()),
        "total_health_t": int(rows.health.where(is_t, 0).sum()),
        "total_health_ct": int(rows.health.where(is_ct, 0).sum()),
        "equip_value_t": int(rows.current_equip_value.where(alive_t, 0).sum()),
        "equip_value_ct": int(rows.current_equip_value.where(alive_ct, 0).sum()),
        "utility_t": int(nd.where(alive_t, 0).sum()),
        "utility_ct": int(nd.where(alive_ct, 0).sum()),
        "defuse_kits_ct": int((rows.has_defuser & alive_ct).sum()),
        "score_diff": int(rows.team_rounds_total.where(is_t, 0).max()
                          - rows.team_rounds_total.where(is_ct, 0).max()),
    }


def _score(agg: dict, *, planted: bool, time_left: float, round_num: int,
           map_name: str) -> float:
    row = {**agg, "bomb_planted": int(planted), "time_remaining": time_left,
           "round_num": round_num, "map": map_name}
    X = build_X(pd.DataFrame([row])).reindex(columns=_FEATS, fill_value=0)
    return float(_MODEL.predict_proba(X)[0, 1])


def _time_left(tick, start, plant_tick, tr) -> tuple[bool, float]:
    planted = plant_tick is not None and tick >= plant_tick
    if planted:
        return True, max(0.0, BOMB_TIME_S - (tick - plant_tick) / tr)
    return False, max(0.0, ROUND_TIME_S - (tick - start) / tr)


def main(rar: str, map_substr: str, round_num: int, out: str) -> None:
    tmp = _extract(rar)
    dem = next(d for d in tmp.rglob("*.dem") if map_substr in d.name)
    p = DemoParser(str(dem))
    map_name = p.parse_header().get("map_name")
    win = _round_windows(p)[round_num - 1]
    tr = _detect_tick_rate(p, win["start_tick"] + 100)
    start, end = win["start_tick"], win["end_tick"]

    kills = p.parse_event("player_death")
    kills = kills[(kills.tick >= start) & (kills.tick <= end)]
    plants = [int(t) for t in p.parse_event("bomb_planted")["tick"] if start < t < end]
    defuses = [int(t) for t in p.parse_event("bomb_defused")["tick"] if start < t < end]
    plant_tick = min(plants) if plants else None
    sides = p.parse_ticks(["team_num"], ticks=[start + 200])
    side_of = dict(zip(sides["name"], sides["team_num"]))
    side_lbl = lambda n: "T" if side_of.get(n) == TEAM_T else "CT"

    # ---- dense replay curve ----
    step = int(DENSE_S * tr)
    dense = sorted(set(range(start, end, step)) | {int(t) for t in kills.tick} | set(plants) | set(defuses))
    td = p.parse_ticks(TICK_PROPS, ticks=dense)
    series = []
    for t, g in td.groupby("tick"):
        planted, tl = _time_left(int(t), start, plant_tick, tr)
        agg = _aggregate(g)
        pr = _score(agg, planted=planted, time_left=tl, round_num=round_num, map_name=map_name)
        series.append({"t": round((int(t) - start) / tr, 1), "p": round(pr, 4),
                       "alive_t": agg["players_alive_t"], "alive_ct": agg["players_alive_ct"],
                       "hp_t": agg["total_health_t"], "hp_ct": agg["total_health_ct"],
                       "planted": planted})

    # ---- per-kill leverage (pre-duel roster, victim-dies vs attacker-dies) ----
    pre_ticks = sorted({max(start, int(k.tick) - PRE_TICKS) for k in kills.itertuples()})
    pre = p.parse_ticks(TICK_PROPS, ticks=pre_ticks)
    events = []
    for k in kills.itertuples(index=False):
        kt = int(k.tick)
        roster = pre[pre.tick == max(start, kt - PRE_TICKS)]
        planted, tl = _time_left(kt, start, plant_tick, tr)
        lev = None
        if k.attacker_name in set(roster.name) and k.user_name in set(roster.name):
            p_actual = _score(_aggregate(roster[roster.name != k.user_name]),
                              planted=planted, time_left=tl, round_num=round_num, map_name=map_name)
            p_cf = _score(_aggregate(roster[roster.name != k.attacker_name]),
                          planted=planted, time_left=tl, round_num=round_num, map_name=map_name)
            lev = {"p_actual": round(p_actual, 3), "p_counterfactual": round(p_cf, 3),
                   "swing_pp": round((p_actual - p_cf) * 100, 1)}
        events.append({"t": round((kt - start) / tr, 1), "type": "kill",
                       "side": side_lbl(k.attacker_name),
                       "attacker": k.attacker_name, "victim": k.user_name,
                       "text": f"{k.attacker_name} ⌖ {k.user_name}", "leverage": lev})
    for t in plants:
        events.append({"t": round((t - start) / tr, 1), "type": "plant", "side": "T",
                       "text": "Bomb planted", "leverage": None})
    for t in defuses:
        events.append({"t": round((t - start) / tr, 1), "type": "defuse", "side": "CT",
                       "text": "Bomb defused", "leverage": None})
    if win["reason"] == "bomb_exploded":
        events.append({"t": series[-1]["t"], "type": "explode", "side": "T",
                       "text": "Bomb exploded", "leverage": None})
    events.sort(key=lambda e: e["t"])

    winner = "T" if str(win["winner"]).upper().startswith("T") else "CT"
    out_obj = {
        "meta": {"map": map_name, "round": round_num, "winner": winner,
                 "reason": win["reason"], "duration": series[-1]["t"], "tickrate": tr},
        "series": series, "events": events,
    }
    Path(out).write_text(json.dumps(out_obj))
    subprocess.run(["rm", "-rf", str(tmp)])
    ks = [e for e in events if e["type"] == "kill" and e["leverage"]]
    print(f"round {round_num} on {map_name}: {len(series)} points, {len(events)} events, "
          f"winner={winner}, dur={series[-1]['t']}s")
    print("kill leverage (swing pp, actual -> counterfactual):")
    for e in ks:
        print(f"  {e['t']:>5}s  {e['text']:<22} {e['leverage']['swing_pp']:+6.1f} pp  "
              f"({e['leverage']['p_actual']:.2f} vs {e['leverage']['p_counterfactual']:.2f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
