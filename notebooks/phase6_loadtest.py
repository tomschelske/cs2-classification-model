"""Phase 6 load test — p50/p99 latency and sustained req/s for /predict.

Fires N requests at fixed concurrency against a running API and reports the
latency distribution and throughput. Writes models/serving_metrics.json.

Run (with the API already up on :8000):
    PYTHONPATH=. .venv/bin/python notebooks/phase6_loadtest.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

URL = "http://localhost:8000/predict"
N = 5000
CONCURRENCY = 50
PAYLOAD = {"players_alive_t": 2, "players_alive_ct": 2, "total_health_t": 200,
           "total_health_ct": 186, "bomb_planted": False, "time_remaining": 40.0,
           "round_num": 13}


async def worker(client, sem, lat):
    async with sem:
        t0 = time.perf_counter()
        r = await client.post(URL, json=PAYLOAD)
        r.raise_for_status()
        lat.append((time.perf_counter() - t0) * 1000)  # ms


async def main() -> None:
    lat: list[float] = []
    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    async with httpx.AsyncClient(limits=limits, timeout=30) as client:
        # warm up
        for _ in range(50):
            await client.post(URL, json=PAYLOAD)
        sem = asyncio.Semaphore(CONCURRENCY)
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(client, sem, lat) for _ in range(N)))
        wall = time.perf_counter() - t0

    lat.sort()
    pct = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]
    out = {
        "requests": N, "concurrency": CONCURRENCY,
        "throughput_rps": round(N / wall, 1),
        "latency_ms": {
            "p50": round(pct(0.50), 2), "p90": round(pct(0.90), 2),
            "p95": round(pct(0.95), 2), "p99": round(pct(0.99), 2),
            "mean": round(statistics.mean(lat), 2), "max": round(max(lat), 2),
        },
        "wall_seconds": round(wall, 2),
    }
    Path("models/serving_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
