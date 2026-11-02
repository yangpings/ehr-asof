"""Scaling sweep for the features endpoint: throughput and latency vs concurrency.

Run once per server configuration (for example 1, 2 and 4 workers). Each run appends
one line per concurrency level to a JSON-lines file:

    uv run python scripts/scaling.py --url http://127.0.0.1:8765 --label "2 workers"

One Python process can only generate so much load (about 140 req/s on the first
sweep). ``--procs 4`` splits the clients across 4 processes so the generator is no
longer the bottleneck. ``--batch 20`` sends POST /features/batch with 20 admissions
per request instead of one GET per admission.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import httpx


async def client(
    http: httpx.AsyncClient,
    hadm_ids: list[int],
    as_of: str,
    batch: int,
    stop: float,
    latencies: list[float],
    errors: list[str],
) -> None:
    while time.perf_counter() < stop:
        start = time.perf_counter()
        try:
            if batch > 1:
                items = [{"hadm_id": h, "as_of": as_of} for h in random.sample(hadm_ids, batch)]
                r = await http.post("/features/batch", json={"items": items})
            else:
                hadm = random.choice(hadm_ids)
                r = await http.get(f"/admissions/{hadm}/features", params={"as_of": as_of})
            if r.status_code == 200:
                latencies.append(time.perf_counter() - start)
            else:
                errors.append(str(r.status_code))
        except httpx.HTTPError as exc:
            errors.append(type(exc).__name__)


async def _run_clients(
    url: str, n_clients: int, seconds: float, hadm_ids: list[int], as_of: str, batch: int
) -> tuple[list[float], list[str], float]:
    latencies: list[float] = []
    errors: list[str] = []
    limits = httpx.Limits(max_connections=n_clients, max_keepalive_connections=n_clients)
    async with httpx.AsyncClient(base_url=url, limits=limits, timeout=30) as http:
        stop = time.perf_counter() + seconds
        began = time.perf_counter()
        await asyncio.gather(
            *(
                client(http, hadm_ids, as_of, batch, stop, latencies, errors)
                for _ in range(n_clients)
            )
        )
        elapsed = time.perf_counter() - began
    return latencies, errors, elapsed


def _worker(
    url: str, n_clients: int, seconds: float, hadm_ids: list[int], as_of: str, batch: int, seed: int
) -> tuple[list[float], list[str], float]:
    random.seed(seed)
    return asyncio.run(_run_clients(url, n_clients, seconds, hadm_ids, as_of, batch))


def level(
    url: str,
    n_clients: int,
    seconds: float,
    hadm_ids: list[int],
    as_of: str,
    procs: int,
    batch: int,
) -> dict[str, object]:
    procs = max(1, min(procs, n_clients))
    shares = [n_clients // procs + (1 if i < n_clients % procs else 0) for i in range(procs)]
    if procs == 1:
        parts = [_worker(url, n_clients, seconds, hadm_ids, as_of, batch, 0)]
    else:
        with ProcessPoolExecutor(max_workers=procs) as pool:
            futures = [
                pool.submit(_worker, url, share, seconds, hadm_ids, as_of, batch, i)
                for i, share in enumerate(shares)
            ]
            parts = [f.result() for f in futures]
    latencies = [x for p in parts for x in p[0]]
    errors = [x for p in parts for x in p[1]]
    elapsed = max(p[2] for p in parts)
    q = statistics.quantiles(latencies, n=100) if len(latencies) > 1 else [float("nan")] * 99
    throughput = len(latencies) / elapsed
    mean_s = statistics.fmean(latencies) if latencies else float("nan")
    return {
        "clients": n_clients,
        "procs": procs,
        "batch": batch,
        "requests": len(latencies),
        "errors": len(errors),
        "throughput_rps": round(throughput, 1),
        "admissions_per_s": round(throughput * batch, 1),
        "p50_ms": round(q[49] * 1000, 1),
        "p95_ms": round(q[94] * 1000, 1),
        "p99_ms": round(q[98] * 1000, 1),
        "littles_law_in_flight": round(throughput * mean_s, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--label", required=True, help='server configuration, e.g. "2 workers"')
    ap.add_argument("--clients", default="10,25,50,100,200")
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--procs", type=int, default=1, help="load-generator processes")
    ap.add_argument("--batch", type=int, default=1, help="admissions per request (POST batch)")
    ap.add_argument("--hadm-first", type=int, default=20000001)
    ap.add_argument("--hadm-count", type=int, default=300)
    ap.add_argument("--as-of", default="2151-06-01T00:00:00")
    ap.add_argument("--out", default="results/scaling.jsonl")
    args = ap.parse_args()

    hadm_ids = list(range(args.hadm_first, args.hadm_first + args.hadm_count))
    httpx.get(
        f"{args.url}/admissions/{hadm_ids[0]}/features", params={"as_of": args.as_of}, timeout=30
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for n in (int(c) for c in args.clients.split(",")):
        stats = level(args.url, n, args.seconds, hadm_ids, args.as_of, args.procs, args.batch)
        row = {"label": args.label, **stats}
        print(json.dumps(row))
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()