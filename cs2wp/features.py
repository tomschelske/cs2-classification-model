"""features.py — parsed demo data -> labeled snapshot rows (Phase 2).

A *snapshot* is the state of one round at one moment in time. Each round is
sampled at fixed intervals (e.g. every 5s) plus event-triggered points (after
each kill, after a bomb plant), so a ~90s round yields ~18 rows, not one.

The label is which side eventually won that round (from ``dem.rounds``).

IMPORTANT: every row must carry a ``match_id`` (and ``round_num``) so the
train/test split can be done at the match level. Leaking snapshots from one
round across the split is the classic way to get a falsely high accuracy — see
project doc, section 6.
"""

from __future__ import annotations

# Core feature columns to implement first (project doc, section 4).
CORE_FEATURES = [
    "players_alive_t",
    "players_alive_ct",
    "total_health_t",
    "total_health_ct",
    "bomb_planted",       # bool
    "time_remaining",     # seconds (or time since plant, if planted)
    "round_num",
]

LABEL = "t_win"           # 1 if the T side won the round, else 0
GROUP_KEY = "match_id"    # split on this — never let a match cross the split

SNAPSHOT_INTERVAL_S = 5   # sample cadence within a round


def build_snapshots(dem, match_id: str):
    """Turn one parsed demo into a DataFrame of labeled snapshot rows.

    TODO (Phase 2):
      1. For each real round (drop warmup/knife), get its winner from dem.rounds.
      2. Walk the round's ticks at SNAPSHOT_INTERVAL_S, plus kill/plant events.
      3. At each sampled tick, aggregate per-side alive counts, total health,
         bomb state, and time remaining.
      4. Emit one row per snapshot with CORE_FEATURES + LABEL + match_id.

    Validate by hand: pick three snapshots and confirm alive counts + label.
    """
    raise NotImplementedError("Phase 2: implement snapshot extraction")
