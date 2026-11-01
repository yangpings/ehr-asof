"""Closed-loop load test for the features endpoint (Part 6).

N clients each send one request, wait for the answer, and send the next, for a fixed
duration. Usage:
    uv run python scripts/loadtest.py --hadm-id 20000001 --clients 20 --seconds 15
"""

import argparse
import asyncio
import statistics
import time

import httpx


async def worker(
    http: httpx.AsyncClient,
    path: str,
    params: dict[str, str],
    stop: float,
    latencies: list[float],
    errors: list[int],
) -> None:
    while time.perf_counter() < stop:
        start = time.perf_counter()
        try:
            r = await http.get(path, params=params)
            r.raise_for_status()
            latencies.append(time.perf_counter() - start)
        except httpx.HTTPError:
            errors.append(1)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--hadm-id", type=int, default=20000001)
    ap.add_argument("--as-of", default="2150-06-01T00:00:00")
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=15)
    args = ap.parse_args()

    path = f"/admissions/{args.hadm_id}/features"
    params = {"as_of": args.as_of, "mode": "asof"}
    latencies: list[float] = []
    errors: list[int] = []
    limits = httpx.Limits(max_connections=args.clients)
    async with httpx.AsyncClient(base_url=args.url, limits=limits, timeout=30) as http:
        await http.get(path, params=params)  # warm-up
        began = time.perf_counter()
        stop = began + args.seconds
        await asyncio.gather(
            *(worker(http, path, params, stop, latencies, errors) for _ in range(args.clients))
        )
        elapsed = time.perf_counter() - began

    throughput = len(latencies) / elapsed
    mean_s = statistics.fmean(latencies)
    q = statistics.quantiles(latencies, n=100)
    print(f"requests {len(latencies)}  errors {len(errors)}  clients {args.clients}")
    print(f"throughput {throughput:.1f} req/s")
    print(
        f"latency p50 {q[49] * 1000:.0f} ms  p95 {q[94] * 1000:.0f} ms  mean {mean_s * 1000:.0f} ms"
    )
    # Little's law: requests in flight = throughput x time in system.
    print(
        f"Little's law: L = lambda x W = {throughput:.1f} x {mean_s:.3f} s = "
        f"{throughput * mean_s:.1f} (should be close to {args.clients} clients)"
    )


if __name__ == "__main__":
    asyncio.run(main())
