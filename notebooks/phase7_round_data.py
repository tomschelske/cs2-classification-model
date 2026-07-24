"""Phase 7 — dense per-tick win-probability reconstruction for one round.

Re-extracts a demo, samples one round finely (every DENSE_S seconds + every
kill + bomb events), runs the deployed model, and writes a self-contained JSON
(curve + annotated events) for the replay artifact.

Run:  PYTHONPATH=. .venv/bin/python notebooks/phase7_round_data.py \
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

from cs2wp.features import (BOMB_TIME_S, ROUND_TIME_S, TEAM_CT, TEAM_T,
                            TICK_PROPS, _detect_tick_rate, _round_windows)

DENSE_S = 1.0  # curve resolution in seconds


def _extract(rar: str) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="phase7_"))
    subprocess.run(["unar", "-q", "-o", str(tmp), rar], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp


def main(rar: str, map_substr: str, round_num: int, out: str) -> None:
    tmp = _extract(rar)
    dem = next(d for d in tmp.rglob("*.dem") if map_substr in d.name)
    p = DemoParser(str(dem))
    map_name = p.parse_header().get("map_name")
    windows = _round_windows(p)
    win = windows[round_num - 1]
    tr = _detect_tick_rate(p, win["start_tick"] + 100)
    start, end = win["start_tick"], win["end_tick"]

    # events in-round
    kills = p.parse_event("player_death")
    kills = kills[(kills.tick >= start) & (kills.tick <= end)]
    plants = [int(t) for t in p.parse_event("bomb_planted")["tick"] if start < t < end]
    defuses = [int(t) for t in p.parse_event("bomb_defused")["tick"] if start < t < end]
    explodes = [int(t) for t in p.parse_event("bomb_exploded")["tick"] if start < t < end]

    # name -> side (constant within a round)
    sides = p.parse_ticks(["team_num"], ticks=[start + 200])
    side_of = dict(zip(sides["name"], sides["team_num"]))

    # dense tick grid + event ticks
    step = int(DENSE_S * tr)
    ticks = set(range(start, end, step)) | {int(t) for t in kills.tick} | set(plants) | set(defuses)
    ticks = sorted(t for t in ticks if start <= t <= end)

    td = p.parse_ticks(TICK_PROPS, ticks=ticks)
    is_t, is_ct = td.team_num == TEAM_T, td.team_num == TEAM_CT
    td["at"] = (td.is_alive & is_t).astype(int)
    td["ac"] = (td.is_alive & is_ct).astype(int)
    td["ht"] = td.health.where(is_t, 0)
    td["hc"] = td.health.where(is_ct, 0)
    agg = td.groupby("tick").agg(alive_t=("at", "sum"), alive_ct=("ac", "sum"),
                                 hp_t=("ht", "sum"), hp_ct=("hc", "sum"),
                                 planted=("is_bomb_planted", "max")).reset_index()

    m = pickle.load(open("models/model.pkl", "rb"))
    model, feats = m["model"], m["features"]
    plant_tick = min(plants) if plants else None

    rows = []
    for r in agg.itertuples(index=False):
        t = int(r.tick)
        since_start = (t - start) / tr
        planted = bool(r.planted)
        if planted and plant_tick:
            time_left = max(0.0, BOMB_TIME_S - (t - plant_tick) / tr)
        else:
            time_left = max(0.0, ROUND_TIME_S - since_start)
        rows.append({"players_alive_t": r.alive_t, "players_alive_ct": r.alive_ct,
                     "total_health_t": r.hp_t, "total_health_ct": r.hp_ct,
                     "bomb_planted": int(planted), "time_remaining": time_left,
                     "round_num": round_num, "_t": round(since_start, 1)})
    X = pd.DataFrame(rows)[feats].copy()
    probs = model.predict_proba(X)[:, 1]

    series = [{"t": rows[i]["_t"], "p": round(float(probs[i]), 4),
               "alive_t": int(rows[i]["players_alive_t"]),
               "alive_ct": int(rows[i]["players_alive_ct"]),
               "hp_t": int(rows[i]["total_health_t"]),
               "hp_ct": int(rows[i]["total_health_ct"]),
               "planted": bool(rows[i]["bomb_planted"])} for i in range(len(rows))]

    def side_label(name):
        return "T" if side_of.get(name) == TEAM_T else "CT"

    events = []
    for k in kills.itertuples(index=False):
        events.append({"t": round((int(k.tick) - start) / tr, 1), "type": "kill",
                       "side": side_label(k.attacker_name),
                       "text": f"{k.attacker_name} ⌖ {k.user_name}"})
    for t in plants:
        events.append({"t": round((t - start) / tr, 1), "type": "plant", "side": "T",
                       "text": "Bomb planted"})
    for t in defuses:
        events.append({"t": round((t - start) / tr, 1), "type": "defuse", "side": "CT",
                       "text": "Bomb defused"})
    for t in explodes:
        events.append({"t": round((t - start) / tr, 1), "type": "explode", "side": "T",
                       "text": "Bomb exploded"})
    events.sort(key=lambda e: e["t"])

    teams = map_substr  # fallback
    parts = Path(dem).stem.split("-m")[0]
    winner = "T" if str(win["winner"]).upper().startswith("T") else "CT"
    out_obj = {
        "meta": {"match": parts, "map": map_name, "round": round_num,
                 "winner": winner, "reason": win["reason"],
                 "duration": series[-1]["t"], "tickrate": tr},
        "series": series, "events": events,
    }
    Path(out).write_text(json.dumps(out_obj))
    print(f"round {round_num} on {map_name}: {len(series)} points, {len(events)} events, "
          f"winner={winner} ({win['reason']}), dur={series[-1]['t']}s")
    print("P(T win) curve:", [s["p"] for s in series])
    print("events:", [(e["t"], e["type"], e["text"]) for e in events])
    subprocess.run(["rm", "-rf", str(tmp)])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
