from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from ehr_asof.api import Settings, create_app
from ehr_asof.db import init_db, load, make_engine
from ehr_asof.events import build_events


@pytest.fixture()
def client(tmp_path: Path, tables: dict[str, pd.DataFrame]) -> Iterator[TestClient]:
    url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = make_engine(url)
    init_db(engine)
    load(engine, tables["admissions"], build_events(tables))
    engine.dispose()
    with TestClient(create_app(Settings(database_url=url))) as c:
        yield c


def _lab_with_delay(tables: dict[str, pd.DataFrame]) -> pd.Series:
    labs = tables["labevents"].dropna(subset=["storetime"])
    labs = labs[(labs["storetime"] - labs["charttime"]) > pd.Timedelta(hours=2)]
    return labs.iloc[0]


def test_health(client: TestClient) -> None:
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").status_code == 200


def test_asof_excludes_unstored_lab(client: TestClient, tables: dict[str, pd.DataFrame]) -> None:
    lab = _lab_with_delay(tables)
    between = (lab["charttime"] + pd.Timedelta(minutes=30)).isoformat()
    params = {"as_of": between}
    asof = client.get(
        f"/admissions/{lab['hadm_id']}/features", params={**params, "mode": "asof"}
    ).json()
    naive = client.get(
        f"/admissions/{lab['hadm_id']}/features", params={**params, "mode": "naive"}
    ).json()
    assert naive["n_events_used"] > asof["n_events_used"]
    assert "x-request-id" in client.get("/health/live").headers


def test_unknown_admission_404(client: TestClient) -> None:
    r = client.get("/admissions/1/features", params={"as_of": "2150-01-01T00:00:00"})
    assert r.status_code == 404


def test_bad_mode_422(client: TestClient) -> None:
    r = client.get(
        "/admissions/1/features", params={"as_of": "2150-01-01T00:00:00", "mode": "future"}
    )
    assert r.status_code == 422


def test_database_down_returns_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    monkeypatch.setattr("sqlalchemy.orm.Session.execute", boom)
    r = client.get("/health/ready")
    assert r.status_code == 503
    assert r.headers["retry-after"] == "5"
    assert client.get("/health/live").status_code == 200


def test_ready_times_out_on_a_slow_database(
    tmp_path: Path, tables: dict[str, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    url = f"sqlite:///{tmp_path / 'slow.db'}"
    engine = make_engine(url)
    init_db(engine)
    engine.dispose()

    def slow(*args: object, **kwargs: object) -> None:
        time.sleep(1.0)

    with TestClient(create_app(Settings(database_url=url, ready_timeout_s=0.2))) as c:
        monkeypatch.setattr("sqlalchemy.orm.Session.execute", slow)
        start = time.perf_counter()
        r = c.get("/health/ready")
        elapsed = time.perf_counter() - start
    assert r.status_code == 503
    assert r.headers["retry-after"] == "5"
    assert elapsed < 0.9


def test_pool_exhaustion_returns_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from sqlalchemy.exc import TimeoutError as PoolTimeoutError

    def busy(*args: object, **kwargs: object) -> None:
        raise PoolTimeoutError("QueuePool limit reached")

    monkeypatch.setattr("sqlalchemy.orm.Session.execute", busy)
    r = client.get("/admissions/20000001/features", params={"as_of": "2150-06-01T00:00:00"})
    assert r.status_code == 503
    assert r.json() == {"detail": "server busy"}
    assert r.headers["retry-after"] == "1"
