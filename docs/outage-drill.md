# Outage drill: Postgres stopped under a running API

**Date:** 2026-11-01 · **Environment:** `docker compose` (Postgres 16, one-shot loader, API with 2 Uvicorn workers) on Windows 11 with Docker Desktop · **API image:** 740 MB on disk (168 MB compressed) · **Host port:** 8765

## Timeline

| Step (UTC, 2026-11-01) | Action | `/health/live` | `/health/ready` |
| --- | --- | --- | --- |
| Before 08:07 | Stack up: db healthy, loader exited 0, api healthy | 200 | 200 |
| ~08:07 | `docker compose stop db` | 200 (0.55 ms) | **503** `{"detail":"database unavailable"}`, `Retry-After: 5`, after 4.0 s |
| ~08:07 | `docker compose start db`, wait 5 s | 200 | **200** (6.3 ms), no API restart |

## Impact

While the database was down, every request that needed it failed fast and cleanly with 503 and a `Retry-After` header. No request returned 500, no process crashed, and the API container stayed healthy. Service recovered on its own within seconds of the database coming back.

## What each response proves

- **Liveness stayed 200.** `/health/live` never touches the database, so an orchestrator would not restart a healthy process just because a dependency was down. Restarting would not have helped and would have added load at recovery.
- **Readiness went to 503.** A load balancer polling `/health/ready` would stop routing traffic to this instance, and clients are told when to retry.
- **Readiness recovered without a restart.** `pool_pre_ping=True` discards dead connections and opens new ones, so no manual intervention was needed.
- **Every response is traceable.** The 503 carried `x-request-id: 8ff3779f0b0b4da8a6e0498a1000cf5f`, and the same ID appears in the API's JSON log line with status 503, so a failed client request can be matched to the server log.

## What I'd change

- **Fail readiness faster.** The 503 took 4.0 s because the probe waited for the 3-second connect timeout (plus the pool's pre-ping). A health probe should answer well within its own timeout; a dedicated short timeout (for example 1 s) for the readiness check would make outages visible sooner.
- **Alert on readiness.** In production, a sustained `ready = 503` should page someone; liveness should not.
- **Client retries.** Clients should honour `Retry-After` with backoff rather than retrying immediately.