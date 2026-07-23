"""parse.py — corpus runner: data/demos/**/*.dem -> data/snapshots.parquet (Phase 3).

Runs features.build_snapshots over every demo found under data/demos/, merges
the results into a single accumulating Parquet file (keyed by match_id), logs
failures instead of crashing, and reports parse throughput.

Why serial by default: demoparser2 is already internally multithreaded (one
demo saturates several cores), and these pro demos are ~450MB each. On an 8GB
machine, running demos one at a time — letting each use all cores — is faster
and safer than holding several ~450MB parses in RAM at once. Use --workers N
on machines with more memory.

Storage: with --prune, each .dem is deleted once its snapshots are safely in
the Parquet, so you never hold more than a batch of demos on disk. The Parquet
accumulates across runs, so you can collect demos in batches.

Usage:
    python -m cs2wp.parse                      # parse all demos -> parquet
    python -m cs2wp.parse --skip-existing      # only new demos (incremental)
    python -m cs2wp.parse --workers 3          # parallel (needs RAM)
    python -m cs2wp.parse --prune              # delete each .dem after parsing

Requires demoparser2==0.41.2 (see features.py).
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cs2wp.features import GROUP_KEY, build_snapshots

DEMOS_DIR = Path("data/demos")
OUT_PATH = Path("data/snapshots.parquet")
REPORT_PATH = Path("data/parse_report.json")


@dataclass
class Result:
    match_id: str
    path: str
    rows: int
    rounds: int
    seconds: float
    error: str | None = None
    df: pd.DataFrame | None = None  # dropped before JSON report


def find_demos(demos_dir: Path) -> list[Path]:
    return sorted(demos_dir.rglob("*.dem"))


def process_one(path: Path) -> Result:
    """Parse a single demo into snapshots. Never raises — errors are captured."""
    match_id = path.stem
    t0 = time.perf_counter()
    try:
        df = build_snapshots(path, match_id=match_id)
        df["series_id"] = path.parent.name  # the Bo3/Bo5 series folder
        return Result(match_id, str(path), len(df), int(df["round_num"].nunique()),
                      time.perf_counter() - t0, None, df)
    except Exception:  # noqa: BLE001 — log and continue, never crash the corpus run
        return Result(match_id, str(path), 0, 0, time.perf_counter() - t0,
                      traceback.format_exc(), None)


def _existing_match_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    return set(pd.read_parquet(out_path, columns=[GROUP_KEY])[GROUP_KEY].unique())


def _merge_into_parquet(new_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Merge new rows into the Parquet, replacing any rows with the same match_id."""
    if out_path.exists():
        old = pd.read_parquet(out_path)
        old = old[~old[GROUP_KEY].isin(new_df[GROUP_KEY].unique())]
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    return combined


def run(demos_dir: Path, out_path: Path, workers: int, skip_existing: bool,
        prune: bool) -> int:
    demos = find_demos(demos_dir)
    if not demos:
        print(f"No .dem files found under {demos_dir}/")
        return 1

    if skip_existing:
        have = _existing_match_ids(out_path)
        demos = [d for d in demos if d.stem not in have]
        if not demos:
            print("Nothing new to parse (all demos already in parquet).")
            return 0

    print(f"Parsing {len(demos)} demo(s) with workers={workers}"
          f"{' [--prune]' if prune else ''}\n")

    results: list[Result] = []
    t_start = time.perf_counter()

    def handle(res: Result) -> None:
        results.append(res)
        if res.error:
            print(f"  FAIL  {res.match_id}  ({res.seconds:.1f}s)  "
                  f"{res.error.strip().splitlines()[-1]}")
            return
        # Merge immediately so a crash mid-run never loses completed work,
        # and so --prune only deletes demos already saved.
        _merge_into_parquet(res.df, out_path)
        print(f"  ok    {res.match_id}  {res.rows} rows / {res.rounds} rounds  "
              f"({res.seconds:.1f}s)")
        if prune:
            Path(res.path).unlink(missing_ok=True)
            print(f"        pruned {Path(res.path).name}")
        res.df = None  # free memory / keep out of JSON report

    if workers <= 1:
        for d in demos:
            handle(process_one(d))
    else:
        # Fresh process per demo (max_tasks_per_child=1) bounds memory and
        # isolates any parser global state.
        with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as ex:
            for res in ex.map(process_one, demos):
                handle(res)

    total_s = time.perf_counter() - t_start
    ok = [r for r in results if not r.error]
    failed = [r for r in results if r.error]
    total_rows = sum(r.rows for r in ok)

    print("\n" + "=" * 60)
    print(f"Parsed {len(ok)}/{len(results)} demos  |  {len(failed)} failed")
    print(f"Snapshots this run: {total_rows}")
    if ok:
        print(f"Throughput: {total_s / len(ok):.1f}s/demo  "
              f"({total_rows / total_s:.0f} snapshots/s)  over {total_s:.1f}s")
    if out_path.exists():
        full = pd.read_parquet(out_path)
        print(f"Parquet now: {len(full)} rows, {full[GROUP_KEY].nunique()} matches "
              f"-> {out_path}")

    # Persist a machine-readable report (doubles as the parse-throughput metric).
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({
        "parsed": len(ok), "failed": len(failed), "rows_this_run": total_rows,
        "seconds": round(total_s, 2),
        "seconds_per_demo": round(total_s / len(ok), 2) if ok else None,
        "failures": [{"match_id": r.match_id, "path": r.path,
                      "error": r.error.strip().splitlines()[-1]} for r in failed],
    }, indent=2))
    print(f"Report: {REPORT_PATH}")
    return 0 if not failed else 2


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse demos into snapshots.parquet")
    ap.add_argument("--demos-dir", type=Path, default=DEMOS_DIR)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel demos (default 1; raise only with spare RAM)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip demos whose match_id is already in the parquet")
    ap.add_argument("--prune", action="store_true",
                    help="delete each .dem after its rows are saved to parquet")
    args = ap.parse_args()
    raise SystemExit(run(args.demos_dir, args.out, args.workers,
                         args.skip_existing, args.prune))


if __name__ == "__main__":
    main()
