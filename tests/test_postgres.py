"""Runs only when EHR_ASOF_DATABASE_URL points at Postgres (as in CI)."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from ehr_asof.api import Settings, create_app

URL = os.environ.get("EHR_ASOF_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not URL.startswith("postgresql"), reason="needs Postgres")


def test_ready_and_features_on_postgres() -> None:
    with TestClient(create_app(Settings(database_url=URL))) as client:
        assert client.get("/health/ready").status_code == 200
        r = client.get("/admissions/20000001/features", params={"as_of": "2200-01-01T00:00:00"})
        assert r.status_code == 200
        assert r.json()["n_events_used"] > 0


def test_timeline_and_batch_on_postgres() -> None:
    with TestClient(create_app(Settings(database_url=URL))) as client:
        t = client.get("/admissions/20000001/timeline", params={"as_of": "2200-01-01T00:00:00"})
        assert t.status_code == 200
        assert t.json()["counts"]["known"] > 0
        items = [
            {"hadm_id": 20000001, "as_of": "2200-01-01T00:00:00"},
            {"hadm_id": 20000002, "as_of": "2200-01-01T00:00:00"},
        ]
        b = client.post("/features/batch", json={"items": items})
        assert b.status_code == 200
        assert [r["hadm_id"] for r in b.json()["results"]] == [20000001, 20000002]
        assert client.get("/delays").status_code == 200
