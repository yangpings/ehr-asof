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


def test_timeline_labels_a_late_lab_as_leaked(
    client: TestClient, tables: dict[str, pd.DataFrame]
) -> None:
    lab = _lab_with_delay(tables)
    as_of = (lab["charttime"] + pd.Timedelta(minutes=30)).isoformat()
    url = f"/admissions/{lab['hadm_id']}"
    body = client.get(f"{url}/timeline", params={"as_of": as_of}).json()
    leaked = [
        e
        for e in body["events"]
        if e["code"] == f"lab:{lab['itemid']}"
        and abs(pd.Timestamp(e["event_time"]) - lab["charttime"]) < pd.Timedelta(seconds=1)
    ]
    assert leaked and leaked[0]["status"] == "leaked"
    assert leaked[0]["delay_h"] > 2

    # The counts agree with the features endpoint: as-of uses the known events,
    # a naive pipeline uses known + leaked.
    asof = client.get(f"{url}/features", params={"as_of": as_of, "mode": "asof"}).json()
    naive = client.get(f"{url}/features", params={"as_of": as_of, "mode": "naive"}).json()
    counts = body["counts"]
    assert counts["known"] == asof["n_events_used"]
    assert counts["known"] + counts["leaked"] == naive["n_events_used"]
    assert sum(counts.values()) == len(body["events"])


def test_timeline_status_filter(client: TestClient, tables: dict[str, pd.DataFrame]) -> None:
    lab = _lab_with_delay(tables)
    as_of = (lab["charttime"] + pd.Timedelta(minutes=30)).isoformat()
    params = {"as_of": as_of, "status": "leaked"}
    body = client.get(f"/admissions/{lab['hadm_id']}/timeline", params=params).json()
    assert body["events"]
    assert {e["status"] for e in body["events"]} == {"leaked"}
    assert len(body["events"]) == body["counts"]["leaked"]


def test_timeline_unknown_admission_404(client: TestClient) -> None:
    r = client.get("/admissions/1/timeline", params={"as_of": "2150-01-01T00:00:00"})
    assert r.status_code == 404


def _three_items(tables: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    adm = tables["admissions"].head(3)
    return [
        {"hadm_id": int(h), "as_of": (t + pd.Timedelta(hours=h_off)).isoformat()}
        for h, t, h_off in zip(adm["hadm_id"], adm["admittime"], (6, 24, 72), strict=True)
    ]


@pytest.mark.parametrize("mode", ["asof", "naive"])
def test_batch_matches_single_requests(
    client: TestClient, tables: dict[str, pd.DataFrame], mode: str
) -> None:
    items = _three_items(tables)
    body = client.post("/features/batch", json={"items": items, "mode": mode}).json()
    assert body["missing"] == []
    for item, row in zip(items, body["results"], strict=True):
        single = client.get(
            f"/admissions/{item['hadm_id']}/features",
            params={"as_of": item["as_of"], "mode": mode},
        ).json()
        assert row["n_events_used"] == single["n_events_used"]
        assert row["features"] == pytest.approx(single["features"])


def test_batch_missing_and_validation(client: TestClient, tables: dict[str, pd.DataFrame]) -> None:
    items = [*_three_items(tables), {"hadm_id": 1, "as_of": "2150-01-01T00:00:00"}]
    body = client.post("/features/batch", json={"items": items}).json()
    assert body["missing"] == [1]
    assert len(body["results"]) == 3
    dup = [items[0], items[0]]
    assert client.post("/features/batch", json={"items": dup}).status_code == 422
    assert client.post("/features/batch", json={"items": []}).status_code == 422


def test_batch_size_limit(tmp_path: Path, tables: dict[str, pd.DataFrame]) -> None:
    url = f"sqlite:///{tmp_path / 'b.db'}"
    engine = make_engine(url)
    init_db(engine)
    engine.dispose()
    with TestClient(create_app(Settings(database_url=url, batch_max=2))) as c:
        r = c.post("/features/batch", json={"items": _three_items(tables)})
    assert r.status_code == 422


def test_risk_without_model_is_503(client: TestClient) -> None:
    r = client.get("/admissions/20000001/risk", params={"as_of": "2150-06-01T00:00:00"})
    assert r.status_code == 503
    assert client.get("/model").status_code == 503


@pytest.fixture()
def model_client(tmp_path: Path, tables: dict[str, pd.DataFrame]) -> Iterator[TestClient]:
    from ehr_asof.model import save, train_risk_model

    url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = make_engine(url)
    init_db(engine)
    load(engine, tables["admissions"], build_events(tables))
    engine.dispose()
    path = save(train_risk_model(tables), tmp_path / "risk.pkl")
    with TestClient(create_app(Settings(database_url=url, model_path=str(path)))) as c:
        yield c


def test_risk_reports_three_scores(
    model_client: TestClient, tables: dict[str, pd.DataFrame]
) -> None:
    adm = tables["admissions"].iloc[0]
    as_of = (adm["admittime"] + pd.Timedelta(hours=24)).isoformat()
    url = f"/admissions/{adm['hadm_id']}"
    body = model_client.get(f"{url}/risk", params={"as_of": as_of}).json()
    assert set(body["risk"]) == {"reported", "honest", "deployed"}
    assert all(0 <= v <= 1 for v in body["risk"].values())
    assert body["hours_since_admission"] == pytest.approx(24)
    timeline = model_client.get(f"{url}/timeline", params={"as_of": as_of}).json()
    assert body["leaked_events"] == timeline["counts"]["leaked"]
    assert model_client.get("/model").json()["horizon_hours"] == 24
    r = model_client.get("/admissions/1/risk", params={"as_of": as_of})
    assert r.status_code == 404


def test_delays_slowest_first_and_cached(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = client.get("/delays", params={"limit": 500}).json()
    medians = [r["median_h"] for r in rows]
    assert medians == sorted(medians, reverse=True)
    # Diagnosis codes become available at discharge, days after admission.
    assert rows[0]["source"] == "dx"
    labs = client.get("/delays", params={"source": "lab"}).json()
    assert labs and {r["source"] for r in labs} == {"lab"}

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("the report should come from the cache")

    monkeypatch.setattr("ehr_asof.api.all_event_times", boom)
    assert client.get("/delays").status_code == 200


def test_metrics_counts_requests(client: TestClient) -> None:
    client.get("/health/live")
    text = client.get("/metrics").text
    assert "ehr_asof_request_duration_seconds_count" in text
    assert 'route="/health/live"' in text
    assert "ehr_asof_pool_exhausted_total" in text
