"""Phase 1 schema exploration — memory-safe, event + per-tick-state based.

Answers the project doc's Phase 1 open questions against a real demo without
ever materializing the full tick table (which OOMs on 8 GB). Uses demoparser2
directly: events are cheap; ticks are only ever requested at specific frames.

REQUIRES demoparser2==0.41.2. In 0.41.4 the 2nd parse call per process panics
(garbage allocation) on py3.13/arm64 — a global-state regression. 0.41.2 is
clean: multiple parse_event/parse_ticks calls per process all work.

Validated per-tick props: health, armor_value, is_alive, life_state,
team_num (2=T, 3=CT), team_name, X, Y, Z, total_rounds_played,
is_freeze_period, game_time, is_bomb_planted.

Run:  .venv/bin/python notebooks/phase1_explore.py <path-to.dem>
"""

from __future__ import annotations

import sys

from demoparser2 import DemoParser

STATE_PROPS = [
    "health", "armor_value", "is_alive", "team_num", "team_name",
    "total_rounds_played", "is_freeze_period", "is_bomb_planted", "game_time",
]


def hr(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def main(path: str) -> None:
    p = DemoParser(path)

    hr("HEADER")
    print("  map:", p.parse_header().get("map_name"), flush=True)

    hr("ROUND BOUNDARIES & OUTCOMES")
    rs = p.parse_event("round_start")[["round", "tick"]].rename(columns={"tick": "start_tick"})
    fe = p.parse_event("round_freeze_end").rename(columns={"tick": "freeze_end_tick"})
    re = p.parse_event("round_end")[["round", "tick", "winner", "reason"]].rename(columns={"tick": "end_tick"})
    print(f"round_start: {len(rs)} | round_freeze_end: {len(fe)} | round_end: {len(re)}", flush=True)
    print(re.to_string(), flush=True)

    hr("BOMB & KILL EVENTS")
    for ev in ("bomb_planted", "bomb_defused", "bomb_exploded", "player_death"):
        df = p.parse_event(ev)
        print(f"  {ev}: {len(df)} rows", flush=True)

    hr("TICK RATE (from game_time delta)")
    two = p.parse_ticks(["game_time"], ticks=[6000, 6064])
    gt = two.groupby("tick")["game_time"].first()
    dt = gt.loc[6064] - gt.loc[6000]
    print(f"  64 ticks spanned {dt:.4f}s  ->  ~{64 / dt:.1f} ticks/sec", flush=True)

    hr("PER-PLAYER STATE @ one live tick (windowed, all 10 players)")
    snap = p.parse_ticks(STATE_PROPS, ticks=[6000])
    front = [c for c in ("name", "tick") if c in snap.columns]
    cols = front + [c for c in STATE_PROPS if c in snap.columns]
    print(snap[cols].to_string(), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python notebooks/phase1_explore.py <path-to.dem>")
    main(sys.argv[1])
