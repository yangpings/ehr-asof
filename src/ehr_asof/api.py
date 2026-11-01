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
from typing import Annotated, Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.orm import Session, sessionmaker

from ehr_asof import __version__
from ehr_asof.db import events_as_of, get_admission, make_engine
from ehr_asof.features import build_features


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
        log.error("database unavailable", extra={"fields": {"error": type(exc).__name__}})
        return JSONResponse(
            {"detail": "database unavailable"}, status_code=503, headers={"Retry-After": "5"}
        )

    @app.exception_handler(PoolTimeoutError)
    async def server_busy(request: Request, exc: Exception) -> JSONResponse:
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

    return app


app = create_app()
