"""Prometheus metrics for the service.

With several Uvicorn workers each worker is its own process with its own counters.
Set ``PROMETHEUS_MULTIPROC_DIR`` (as the Docker image does) and every worker writes
its values there; ``/metrics`` then reports the sum across workers.
"""

from __future__ import annotations

import os

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    multiprocess,
)

REQUEST_SECONDS = Histogram(
    "ehr_asof_request_duration_seconds",
    "Request latency by route and status.",
    ["method", "route", "status"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
POOL_EXHAUSTED = Counter(
    "ehr_asof_pool_exhausted_total", "Requests rejected with 503 because the pool was full."
)
DB_UNAVAILABLE = Counter(
    "ehr_asof_db_unavailable_total", "Requests answered 503 because the database was down."
)
BATCH_SIZE = Histogram(
    "ehr_asof_batch_size",
    "Admissions per batch request.",
    buckets=(1, 5, 10, 25, 50, 100, 200),
)


def render() -> tuple[bytes, str]:
    """The current metrics in Prometheus text format, and its content type."""
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)  # type: ignore[no-untyped-call]
        return generate_latest(registry), CONTENT_TYPE_LATEST
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
