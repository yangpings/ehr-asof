"""HTTP service: features for an admission, as of any timestamp.

Run with ``ehr-asof serve`` or ``uvicorn ehr_asof.api:app``. Configuration comes
from environment variables (see ``Settings``).
"""

import asyncio
import json
import logging
import math
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Literal, Self

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.orm import Session, sessionmaker

from ehr_asof import __version__, metrics
from ehr_asof.db import (
    admission_events,
    all_event_times,
    events_as_of,
    events_for_many,
    existing_admissions,
    get_admission,
    make_engine,
)
from ehr_asof.events import delay_report
from ehr_asof.features import availability_status, build_features, visible_events
from ehr_asof.model import RiskModel
from ehr_asof.model import load as load_model


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EHR_ASOF_")

    database_url: str = "sqlite:///./ehr_asof.db"
    log_level: str = "INFO"
    # Readiness must answer well within a load balancer's probe timeout, so it
    # gives up sooner than the database connect timeout.
    ready_timeout_s: float = 1.0
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout_s: float = 2.0
    # Risk model file written by ``ehr-asof train`` (synthetic data only). Unset: /risk is 503.
    model_path: str | None = None
    batch_max: int = 200
    delays_cache_s: float = 300.0


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, default=str)


log = logging.getLogger("ehr_asof.api")


def _configure_logging(level: str) -> None:
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        log.addHandler(handler)
    log.setLevel(level)
    log.propagate = False


def get_session(request: Request) -> Iterator[Session]:
    with request.app.state.sessions() as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


class FeatureResponse(BaseModel):
    hadm_id: int
    as_of: datetime
    mode: Literal["naive", "asof"]
    n_events_used: int
    features: dict[str, float]
    model_version: str


class TimelineEvent(BaseModel):
    source: str
    code: str
    value: float
    event_time: datetime
    available_time: datetime
    delay_h: float
    status: Literal["known", "leaked", "future"]


class TimelineCounts(BaseModel):
    known: int
    leaked: int
    future: int


class TimelineResponse(BaseModel):
    hadm_id: int
    as_of: datetime
    admittime: datetime
    dischtime: datetime
    counts: TimelineCounts
    events: list[TimelineEvent]


class BatchItem(BaseModel):
    hadm_id: int
    as_of: datetime


class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(min_length=1)
    mode: Literal["naive", "asof"] = "asof"

    @model_validator(mode="after")
    def unique_admissions(self) -> Self:
        ids = [item.hadm_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("each hadm_id may appear only once per batch")
        return self


class BatchRow(BaseModel):
    hadm_id: int
    as_of: datetime
    n_events_used: int
    features: dict[str, float]


class BatchResponse(BaseModel):
    mode: Literal["naive", "asof"]
    results: list[BatchRow]
    missing: list[int]
    model_version: str


class RiskScores(BaseModel):
    reported: float
    honest: float
    deployed: float


class RiskResponse(BaseModel):
    hadm_id: int
    as_of: datetime
    hours_since_admission: float
    risk: RiskScores
    leaked_events: int
    model: dict[str, object]


class DelayRow(BaseModel):
    source: str
    code: str
    count: int
    median_h: float
    p90_h: float
    max_h: float


def _feature_dict(row: pd.Series) -> dict[str, float]:
    """Present values only: drop missing labs and absent diagnosis flags."""
    out: dict[str, float] = {}
    for key, value in row.items():
        v = float(value)
        if math.isnan(v) or (str(key).startswith("flag:") and v == 0.0):
            continue
        out[str(key)] = v
    return out


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    _configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(
            settings.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout_s,
        )
        app.state.sessions = sessionmaker(engine, expire_on_commit=False)
        app.state.risk_model = None
        if settings.model_path:
            try:
                app.state.risk_model = load_model(settings.model_path)
            except (OSError, TypeError) as exc:
                log.warning("risk model not loaded", extra={"fields": {"error": str(exc)}})
        app.state.delays = None
        yield
        engine.dispose()

    app = FastAPI(title="ehr-asof", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id", uuid.uuid4().hex)
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        route = request.scope.get("route")
        metrics.REQUEST_SECONDS.labels(
            request.method, getattr(route, "path", "unmatched"), str(response.status_code)
        ).observe(time.perf_counter() - start)
        log.info(
            "request",
            extra={
                "fields": {
                    "request_id": rid,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                }
            },
        )
        return response

    @app.exception_handler(OperationalError)
    @app.exception_handler(DBAPIError)
    async def database_unavailable(request: Request, exc: Exception) -> JSONResponse:
        metrics.DB_UNAVAILABLE.inc()
        log.error("database unavailable", extra={"fields": {"error": type(exc).__name__}})
        return JSONResponse(
            {"detail": "database unavailable"}, status_code=503, headers={"Retry-After": "5"}
        )

    @app.exception_handler(PoolTimeoutError)
    async def server_busy(request: Request, exc: Exception) -> JSONResponse:
        metrics.POOL_EXHAUSTED.inc()
        log.warning("connection pool exhausted", extra={"fields": {"error": type(exc).__name__}})
        return JSONResponse(
            {"detail": "server busy"}, status_code=503, headers={"Retry-After": "1"}
        )

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        """Is the process up? Never touches the database."""
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready(request: Request) -> dict[str, str]:
        """Can we serve traffic? 503 when the database is unreachable or too slow."""

        def check() -> None:
            with request.app.state.sessions() as session:
                session.execute(text("SELECT 1"))

        try:
            await asyncio.wait_for(run_in_threadpool(check), timeout=settings.ready_timeout_s)
        except TimeoutError:
            log.error("database unavailable", extra={"fields": {"error": "ready timeout"}})
            raise HTTPException(
                status_code=503, detail="database unavailable", headers={"Retry-After": "5"}
            ) from None
        return {"status": "ready"}

    @app.get("/admissions/{hadm_id}/features", response_model=FeatureResponse)
    def features(
        hadm_id: int,
        session: SessionDep,
        as_of: Annotated[
            datetime, Query(description="ISO timestamp; only events known by then are used")
        ],
        mode: Annotated[Literal["naive", "asof"], Query()] = "asof",
    ) -> FeatureResponse:
        admission = get_admission(session, hadm_id)
        if admission is None:
            raise HTTPException(status_code=404, detail=f"admission {hadm_id} not found")
        as_of_naive = as_of.replace(tzinfo=None)
        clock = "available_time" if mode == "asof" else "event_time"
        events = events_as_of(session, hadm_id, as_of_naive, clock)
        # Hand the connection back before the CPU-bound feature build, so the pool
        # isn't exhausted by requests that no longer need the database.
        session.close()
        cutoffs = pd.DataFrame({"hadm_id": [hadm_id], "cutoff": [pd.Timestamp(as_of_naive)]})
        matrix = build_features(events, cutoffs, mode, codes=sorted(events["code"].unique()))
        row = matrix.iloc[0]
        values = {
            k: float(v) for k, v in row.items() if not (isinstance(v, float) and math.isnan(v))
        }
        return FeatureResponse(
            hadm_id=hadm_id,
            as_of=as_of_naive,
            mode=mode,
            n_events_used=len(events),
            features=values,
            model_version=__version__,
        )

    @app.get("/admissions/{hadm_id}/timeline", response_model=TimelineResponse)
    def timeline(
        hadm_id: int,
        session: SessionDep,
        as_of: Annotated[datetime, Query(description="ISO timestamp to judge each event against")],
        status: Annotated[
            Literal["known", "leaked", "future"] | None,
            Query(description="Return only events with this status"),
        ] = None,
    ) -> TimelineResponse:
        """Every event of an admission on both clocks, labelled known / leaked / future."""
        admission = get_admission(session, hadm_id)
        if admission is None:
            raise HTTPException(status_code=404, detail=f"admission {hadm_id} not found")
        admittime, dischtime = admission.admittime, admission.dischtime
        as_of_naive = as_of.replace(tzinfo=None)
        events = admission_events(session, hadm_id)
        session.close()
        events["status"] = availability_status(events, pd.Timestamp(as_of_naive))
        events["delay_h"] = (
            (events["available_time"] - events["event_time"]).dt.total_seconds() / 3600
        ).round(2)
        counts = events["status"].value_counts()
        if status is not None:
            events = events[events["status"] == status]
        cols = ["source", "code", "value", "event_time", "available_time", "delay_h", "status"]
        return TimelineResponse(
            hadm_id=hadm_id,
            as_of=as_of_naive,
            admittime=admittime,
            dischtime=dischtime,
            counts=TimelineCounts(
                known=int(counts.get("known", 0)),
                leaked=int(counts.get("leaked", 0)),
                future=int(counts.get("future", 0)),
            ),
            events=[TimelineEvent(**row) for row in events[cols].to_dict("records")],
        )

    @app.post("/features/batch", response_model=BatchResponse)
    def features_batch(body: BatchRequest, session: SessionDep) -> BatchResponse:
        """Features for many admissions, each at its own ``as_of``, in one round trip."""
        if len(body.items) > settings.batch_max:
            raise HTTPException(
                status_code=422, detail=f"at most {settings.batch_max} items per batch"
            )
        metrics.BATCH_SIZE.observe(len(body.items))
        ids = [item.hadm_id for item in body.items]
        found = existing_admissions(session, ids)
        items = [item for item in body.items if item.hadm_id in found]
        cutoffs = pd.DataFrame(
            {
                "hadm_id": [item.hadm_id for item in items],
                "cutoff": [pd.Timestamp(item.as_of.replace(tzinfo=None)) for item in items],
            }
        )
        clock = "available_time" if body.mode == "asof" else "event_time"
        until = cutoffs["cutoff"].max().to_pydatetime() if items else datetime.min
        events = events_for_many(session, list(found), until, clock) if items else None
        session.close()
        results: list[BatchRow] = []
        if events is not None:
            codes = sorted(events["code"].unique())
            matrix = build_features(events, cutoffs, body.mode, codes)
            used = visible_events(events, cutoffs, body.mode).groupby("hadm_id").size()
            for item, cutoff in zip(items, cutoffs["cutoff"], strict=True):
                results.append(
                    BatchRow(
                        hadm_id=item.hadm_id,
                        as_of=cutoff.to_pydatetime(),
                        n_events_used=int(used.get(item.hadm_id, 0)),
                        features=_feature_dict(matrix.loc[item.hadm_id]),
                    )
                )
        return BatchResponse(
            mode=body.mode,
            results=results,
            missing=[i for i in ids if i not in found],
            model_version=__version__,
        )

    def _risk_model(request: Request) -> RiskModel:
        model: RiskModel | None = request.app.state.risk_model
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="risk model not loaded; run `ehr-asof train` and set EHR_ASOF_MODEL_PATH",
            )
        return model

    @app.get("/model")
    def model_info(request: Request) -> dict[str, object]:
        """What the served risk model is: outcome, horizon, size, when it was trained."""
        return _risk_model(request).info()

    @app.get("/admissions/{hadm_id}/risk", response_model=RiskResponse)
    def risk(
        hadm_id: int,
        request: Request,
        session: SessionDep,
        as_of: Annotated[datetime, Query(description="ISO timestamp of the prediction")],
    ) -> RiskResponse:
        """Mortality risk at ``as_of``: reported vs honest vs deployed, side by side."""
        model = _risk_model(request)
        admission = get_admission(session, hadm_id)
        if admission is None:
            raise HTTPException(status_code=404, detail=f"admission {hadm_id} not found")
        admittime = admission.admittime
        as_of_naive = as_of.replace(tzinfo=None)
        events = admission_events(session, hadm_id)
        session.close()
        cutoff = pd.Timestamp(as_of_naive)
        cutoffs = pd.DataFrame({"hadm_id": [hadm_id], "cutoff": [cutoff]})
        scores = model.score(events, cutoffs).iloc[0]
        leaked = int((availability_status(events, cutoff) == "leaked").sum())
        return RiskResponse(
            hadm_id=hadm_id,
            as_of=as_of_naive,
            hours_since_admission=round(
                (cutoff - pd.Timestamp(admittime)).total_seconds() / 3600, 2
            ),
            risk=RiskScores(
                reported=round(float(scores["reported"]), 4),
                honest=round(float(scores["honest"]), 4),
                deployed=round(float(scores["deployed"]), 4),
            ),
            leaked_events=leaked,
            model=model.info(),
        )

    @app.get("/delays", response_model=list[DelayRow])
    def delays(
        request: Request,
        session: SessionDep,
        source: Annotated[
            Literal["lab", "lab_nostore", "dx"] | None, Query(description="Only this source")
        ] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 20,
    ) -> list[DelayRow]:
        """Which signals arrive late, and by how much: slowest codes first."""
        cached = request.app.state.delays
        if cached is None or time.monotonic() - cached[0] > settings.delays_cache_s:
            frame = all_event_times(session)
            session.close()
            report = delay_report(frame).reset_index()
            cached = (time.monotonic(), report)
            request.app.state.delays = cached
        report = cached[1]
        if source is not None:
            report = report[report["source"] == source]
        report = report.sort_values("median_h", ascending=False).head(limit)
        return [
            DelayRow(
                source=str(r["source"]),
                code=str(r["code"]),
                count=int(r["count"]),
                median_h=round(float(r["median_h"]), 2),
                p90_h=round(float(r["p90_h"]), 2),
                max_h=round(float(r["max_h"]), 2),
            )
            for r in report.to_dict("records")
        ]

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        """Prometheus scrape endpoint."""
        body, content_type = metrics.render()
        return Response(content=body, media_type=content_type)

    return app


app = create_app()
