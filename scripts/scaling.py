"""Scaling sweep for the features endpoint: throughput and latency vs concurrency.

Run once per server configuration (for example 1, 2 and 4 workers). Each run appends
one line per concurrency level to a JSON-lines file:

    uv run python scripts/scaling.py --url http://127.0.0.1:8765 --label "2 workers"
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx


async def client(
    http: httpx.AsyncClient,
    hadm_ids: list[int],
    as_of: str,
    stop: float,
    latencies: list[float],
    errors: list[str],
) -> None:
    while time.perf_counter() < stop:
        hadm = random.choice(hadm_ids)
        start = time.perf_counter()
        try:
            r = await http.get(f"/admissions/{hadm}/features", params={"as_of": as_of})
            if r.status_code == 200:
                latencies.append(time.perf_counter() - start)
            else:
                errors.append(str(r.status_code))
        except httpx.HTTPError as exc:
            errors.append(type(exc).__name__)


async def level(
    url: str, n_clients: int, seconds: float, hadm_ids: list[int], as_of: str
) -> dict[str, object]:
    latencies: list[float] = []
    errors: list[str] = []
    limits = httpx.Limits(max_connections=n_clients, max_keepalive_connections=n_clients)
    async with httpx.AsyncClient(base_url=url, limits=limits, timeout=30) as http:
        stop = time.perf_counter() + seconds
        began = time.perf_counter()
        await asyncio.gather(
            *(client(http, hadm_ids, as_of, stop, latencies, errors) for _ in range(n_clients))
        )
        elapsed = time.perf_counter() - began
    q = statistics.quantiles(latencies, n=100) if len(latencies) > 1 else [float("nan")] * 99
    throughput = len(latencies) / elapsed
    mean_s = statistics.fmean(latencies) if latencies else float("nan")
    return {
        "clients": n_clients,
        "requests": len(latencies),
        "errors": len(errors),
        "throughput_rps": round(throughput, 1),
        "p50_ms": round(q[49] * 1000, 1),
        "p95_ms": round(q[94] * 1000, 1),
        "p99_ms": round(q[98] * 1000, 1),
        "littles_law_in_flight": round(throughput * mean_s, 1),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--label", required=True, help='server configuration, e.g. "2 workers"')
    ap.add_argument("--clients", default="10,25,50,100,200")
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--hadm-first", type=int, default=20000001)
    ap.add_argument("--hadm-count", type=int, default=300)
    ap.add_argument("--as-of", default="2151-06-01T00:00:00")
    ap.add_argument("--out", default="results/scaling.jsonl")
    args = ap.parse_args()

    random.seed(0)
    hadm_ids = list(range(args.hadm_first, args.hadm_first + args.hadm_count))
    async with httpx.AsyncClient(base_url=args.url, timeout=30) as http:
        await http.get(f"/admissions/{hadm_ids[0]}/features", params={"as_of": args.as_of})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for n in (int(c) for c in args.clients.split(",")):
        row = {"label": args.label, **await level(args.url, n, args.seconds, hadm_ids, args.as_of)}
        print(json.dumps(row))
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
