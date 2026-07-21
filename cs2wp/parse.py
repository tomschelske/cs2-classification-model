"""parse.py — .dem files -> parsed demo data (Phase 1 & 3).

Thin wrapper around ``awpy.Demo`` so the rest of the pipeline never touches the
parser directly. Phase 1 is about *understanding* the schemas that come out of
here before any feature code is written.

Open questions to answer against real data (see project doc, section 11):
  - What does one row of ``dem.ticks`` represent, and at what tick rate?
  - Is round time absolute-tick or seconds-since-freeze-end?
  - How are warmup / knife rounds marked so they can be filtered?
  - Does the bomb table cleanly separate plant / defuse / explosion?
  - How is overtime handled in round numbering?
"""

from __future__ import annotations

from pathlib import Path


def parse_demo(path: str | Path):
    """Parse a single .dem file and return the awpy Demo object.

    Parsing is ~4-5s per demo and CPU/IO bound. Returns the parsed ``Demo`` so
    callers can reach ``.header``, ``.rounds``, ``.ticks``, ``.kills``,
    ``.damages``, ``.bomb``, ``.grenades``.
    """
    from awpy import Demo

    dem = Demo(str(path))
    dem.parse()
    return dem


if __name__ == "__main__":
    # Phase 1 smoke test: parse one demo and print every schema.
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m cs2wp.parse <path-to.dem>")

    dem = parse_demo(sys.argv[1])
    print("HEADER:", dem.header)
    for name in ("rounds", "ticks", "kills", "damages", "bomb", "grenades"):
        df = getattr(dem, name, None)
        print(f"\n=== {name} ===")
        if df is None:
            print("  (none)")
            continue
        print("  shape:", getattr(df, "shape", "?"))
        print("  columns:", list(getattr(df, "columns", [])))
