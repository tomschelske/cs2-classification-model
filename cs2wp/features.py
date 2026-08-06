"""features.py — parsed demo -> labeled snapshot rows (Phase 2).

A *snapshot* is the state of one round at one moment in time. Each round is
sampled at fixed intervals (every SNAPSHOT_INTERVAL_S of live play) plus
event-triggered points (each kill, the bomb plant), so a ~90s round yields
~20 rows, not one. The label is which side eventually won that round.

Memory-safe by construction (see [[env-and-parsing-constraints]]): we never
materialize the full tick table. We compute the exact set of snapshot ticks
first, then pull only those frames in a SINGLE parse_ticks call. A whole
~450MB demo becomes a few thousand rows and runs in seconds on 8GB RAM.

Every row carries match_id + round_num so the train/test split can be done at
the MATCH level — never let snapshots from one round cross the split (doc §6).

Requires demoparser2==0.41.2 (0.41.4 panics on the 2nd parse call).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from demoparser2 import DemoParser

# --- Sampling / game constants ------------------------------------------------
SNAPSHOT_INTERVAL_S = 5      # fixed-interval sample cadence within a round
ROUND_TIME_S = 115          # CS2 competitive round time (1:55) of live play
BOMB_TIME_S = 40            # C4 timer after plant
DEFAULT_TICK_RATE = 64      # CS2 GOTV; verified per-demo below

# Per-tick properties we pull (all validated against demoparser2 0.41.2).
# team_num: 2 = TERRORIST side, 3 = CT side (current side at that tick).
TICK_PROPS = [
    "health", "is_alive", "team_num", "is_bomb_planted",
    # secondary (iteration two):
    "current_equip_value",  # $ value of a player's current gear -> equipment/side
    "has_defuser",          # CT defuse kit
    "inventory",            # item-name list -> utility remaining
    "team_rounds_total",    # this team's round score -> score differential
]

# Grenade item names in `inventory` (used to count utility remaining per side).
GRENADES = frozenset({
    "Flashbang", "Smoke Grenade", "High Explosive Grenade",
    "Incendiary Grenade", "Molotov", "Decoy Grenade",
})

# --- Output schema ------------------------------------------------------------
CORE_FEATURES = [
    "players_alive_t",
    "players_alive_ct",
    "total_health_t",
    "total_health_ct",
    "bomb_planted",       # bool
    "time_remaining",     # seconds on the active clock (round, or bomb if planted)
    "round_num",
]
# Iteration-two features (see project doc §4). equip/utility/kits count only
# living players (dead players hold no firepower); score_diff = T score - CT.
SECONDARY_FEATURES = [
    "equip_value_t",
    "equip_value_ct",
    "utility_t",
    "utility_ct",
    "defuse_kits_ct",
    "score_diff",
]
# Categorical context, one-hot encoded downstream (in train.py). (bomb_site was
# tried but demoparser2 returns map-specific entity IDs, not A/B — dropped.)
CATEGORICAL = ["map"]

# NOTE — minor unenforced side-asymmetry (not a real-state failure).
# The per-side features here are ABSOLUTE (equip_value_t AND equip_value_ct, etc.),
# and nothing forces the model to weight the two sides as mirror images: it learns
# e.g. +1.15 for equip_value_t but only -0.70 for equip_value_ct. So a *balanced*
# state isn't guaranteed neutral, and the model extrapolates a slight lean on
# balanced-but-off-distribution synthetic inputs (e.g. a full economy with kits=0
# and no utility). On REAL states this is negligible — the model is well-calibrated
# at round openings by buy level (checked: dust2 openings 0.42 pred vs 0.43 actual;
# equal full-buy openings 0.51 vs 0.54). (Caution: dust2 is CT-sided *overall*
# (0.43) but T-favored on equal *full-buy* rounds (~0.60) — don't compare a
# conditional prediction to the marginal base rate.) Optional tidy-up: reframe the
# symmetric per-side features as T-minus-CT DIFFERENTIALS so a balanced state -> 0.

LABEL = "t_win"           # 1 if the T side won the round, else 0
GROUP_KEY = "match_id"    # split on this — never let a match cross the split

TEAM_T = 2
TEAM_CT = 3


def _detect_tick_rate(p: DemoParser, probe_tick: int) -> int:
    """Infer tick rate from game_time over a 64-tick span (falls back to 64)."""
    try:
        df = p.parse_ticks(["game_time"], ticks=[probe_tick, probe_tick + 64])
        gt = df.groupby("tick")["game_time"].first()
        dt = float(gt.loc[probe_tick + 64] - gt.loc[probe_tick])
        if dt > 0:
            return int(round(64 / dt))
    except Exception:  # noqa: BLE001 — any hiccup -> default
        pass
    return DEFAULT_TICK_RATE


def _round_windows(p: DemoParser) -> list[dict]:
    """Pair each round_end (round #, winner) with its preceding live-play start.

    round_freeze_end marks the end of freeze time = start of live play. We pair
    by tick order (robust to the round-number offset where round_end.round
    starts at 2), requiring each window's start to fall after the prior round.
    """
    fe_ticks = sorted(int(t) for t in p.parse_event("round_freeze_end")["tick"])
    re = p.parse_event("round_end").sort_values("tick").reset_index(drop=True)

    windows: list[dict] = []
    prev_end = -1
    for i, row in enumerate(re.itertuples(index=False)):
        end_tick = int(row.tick)
        # largest freeze_end strictly inside (prev_end, end_tick)
        cands = [t for t in fe_ticks if prev_end < t < end_tick]
        if not cands:
            continue  # malformed round (no live start found) — skip
        windows.append(
            {
                # contiguous 1..N over KEPT rounds — demoparser emits a junk
                # tick-1 round_end (round 0, no winner) plus warmup/knife ends
                # that have no freeze_end and get skipped above; numbering by
                # len(windows) makes round_num=1 the real pistol round.
                "round_num": len(windows) + 1,
                "raw_round": int(row.round), # demoparser's own numbering
                "start_tick": max(cands),
                "end_tick": end_tick,
                "winner": str(row.winner),
                "reason": str(row.reason),
            }
        )
        prev_end = end_tick
    return windows


def _snapshot_ticks(win: dict, kill_ticks: list[int], plant_ticks: list[int],
                    tick_rate: int) -> list[int]:
    """Ticks to sample for one round: fixed interval + kills + plant, in-window."""
    start, end = win["start_tick"], win["end_tick"]
    step = SNAPSHOT_INTERVAL_S * tick_rate
    ticks = set(range(start, end, step))                       # every 5s of live play
    ticks.update(t for t in kill_ticks if start < t < end)     # after each kill
    ticks.update(t for t in plant_ticks if start < t < end)    # after the plant
    return sorted(t for t in ticks if start <= t < end)


def build_snapshots(demo_path: str | Path, match_id: str | None = None) -> pd.DataFrame:
    """Turn one demo into a DataFrame of labeled snapshot rows."""
    demo_path = Path(demo_path)
    match_id = match_id or demo_path.stem
    p = DemoParser(str(demo_path))
    map_name = p.parse_header().get("map_name")

    windows = _round_windows(p)
    if not windows:
        return pd.DataFrame(columns=[GROUP_KEY, "map", *CORE_FEATURES, LABEL])

    tick_rate = _detect_tick_rate(p, windows[0]["start_tick"] + 100)
    kill_ticks = [int(t) for t in p.parse_event("player_death")["tick"]]
    plant_ticks = [int(t) for t in p.parse_event("bomb_planted")["tick"]]

    # tick -> round window, and first plant tick per round
    tick_to_win: dict[int, dict] = {}
    plant_by_round: dict[int, int] = {}
    all_ticks: set[int] = set()
    for win in windows:
        for t in _snapshot_ticks(win, kill_ticks, plant_ticks, tick_rate):
            tick_to_win[t] = win
            all_ticks.add(t)
        in_round = [t for t in plant_ticks if win["start_tick"] < t < win["end_tick"]]
        if in_round:
            plant_by_round[win["round_num"]] = min(in_round)

    # ---- ONE windowed tick parse for the whole demo ----
    td = p.parse_ticks(TICK_PROPS, ticks=sorted(all_ticks))

    is_t = td["team_num"] == TEAM_T
    is_ct = td["team_num"] == TEAM_CT
    alive_t = td["is_alive"] & is_t
    alive_ct = td["is_alive"] & is_ct
    td["alive_t"] = alive_t.astype(int)
    td["alive_ct"] = alive_ct.astype(int)
    td["hp_t"] = td["health"].where(is_t, 0)
    td["hp_ct"] = td["health"].where(is_ct, 0)
    # --- secondary: equipment, utility, kits, score (living players only) ---
    td["nades"] = td["inventory"].apply(
        lambda inv: sum(i in GRENADES for i in inv) if inv is not None else 0)
    td["equip_t"] = td["current_equip_value"].where(alive_t, 0)
    td["equip_ct"] = td["current_equip_value"].where(alive_ct, 0)
    td["util_t"] = td["nades"].where(alive_t, 0)
    td["util_ct"] = td["nades"].where(alive_ct, 0)
    td["kit_ct"] = (td["has_defuser"] & alive_ct).astype(int)
    td["score_t"] = td["team_rounds_total"].where(is_t, 0)
    td["score_ct"] = td["team_rounds_total"].where(is_ct, 0)

    agg = td.groupby("tick").agg(
        players_alive_t=("alive_t", "sum"),
        players_alive_ct=("alive_ct", "sum"),
        total_health_t=("hp_t", "sum"),
        total_health_ct=("hp_ct", "sum"),
        bomb_planted=("is_bomb_planted", "max"),
        equip_value_t=("equip_t", "sum"),
        equip_value_ct=("equip_ct", "sum"),
        utility_t=("util_t", "sum"),
        utility_ct=("util_ct", "sum"),
        defuse_kits_ct=("kit_ct", "sum"),
        score_t=("score_t", "max"),
        score_ct=("score_ct", "max"),
    ).reset_index()

    # attach round meta + time features + label
    rows = []
    for r in agg.itertuples(index=False):
        win = tick_to_win[int(r.tick)]
        since_start = (int(r.tick) - win["start_tick"]) / tick_rate
        planted = bool(r.bomb_planted)
        if planted and win["round_num"] in plant_by_round:
            since_plant = (int(r.tick) - plant_by_round[win["round_num"]]) / tick_rate
            time_remaining = max(0.0, BOMB_TIME_S - since_plant)
        else:
            time_remaining = max(0.0, ROUND_TIME_S - since_start)
        rows.append(
            {
                GROUP_KEY: match_id,
                "map": map_name,
                "round_num": win["round_num"],
                "tick": int(r.tick),
                "seconds_since_start": round(since_start, 3),
                "players_alive_t": int(r.players_alive_t),
                "players_alive_ct": int(r.players_alive_ct),
                "total_health_t": int(r.total_health_t),
                "total_health_ct": int(r.total_health_ct),
                "bomb_planted": planted,
                "time_remaining": round(time_remaining, 2),
                # secondary features
                "equip_value_t": int(r.equip_value_t),
                "equip_value_ct": int(r.equip_value_ct),
                "utility_t": int(r.utility_t),
                "utility_ct": int(r.utility_ct),
                "defuse_kits_ct": int(r.defuse_kits_ct),
                "score_diff": int(r.score_t) - int(r.score_ct),
                LABEL: int(win["winner"].upper().startswith("T")),  # 'T'/'TERRORIST'
            }
        )

    return pd.DataFrame(rows).sort_values(["round_num", "tick"]).reset_index(drop=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m cs2wp.features <path-to.dem>")
    df = build_snapshots(sys.argv[1])
    print(f"snapshots: {len(df)} rows across {df['round_num'].nunique()} rounds")
    print(df.to_string(max_rows=40))
