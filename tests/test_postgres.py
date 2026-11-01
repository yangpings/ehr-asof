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
