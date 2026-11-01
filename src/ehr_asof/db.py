"""Database layer: events stored with both clocks, queried as of a timestamp."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    create_engine,
    delete,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class AdmissionRow(Base):
    __tablename__ = "admissions"

    hadm_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    subject_id: Mapped[int] = mapped_column(BigInteger, index=True)
    admittime: Mapped[datetime] = mapped_column(DateTime)
    dischtime: Mapped[datetime] = mapped_column(DateTime)


class EventRow(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_id: Mapped[int] = mapped_column(BigInteger)
    hadm_id: Mapped[int] = mapped_column(BigInteger)
    source: Mapped[str] = mapped_column(String(16))
    code: Mapped[str] = mapped_column(String(64))
    value: Mapped[float] = mapped_column(Float)
    event_time: Mapped[datetime] = mapped_column(DateTime)
    available_time: Mapped[datetime] = mapped_column(DateTime)

    # The as-of query filters one admission by available_time: index exactly that.
    __table_args__ = (
        Index("ix_events_hadm_available", "hadm_id", "available_time"),
        Index("ix_events_hadm_event", "hadm_id", "event_time"),
    )


def make_engine(
    url: str, pool_size: int = 5, max_overflow: int = 10, pool_timeout: float = 2.0
) -> Engine:
    if not url.startswith("postgresql"):
        return create_engine(url, pool_pre_ping=True)
    # Under overload, waiting for a free connection should fail fast (and become a
    # 503) rather than hold the request for SQLAlchemy's default 30 seconds.
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        connect_args={"connect_timeout": 3, "options": "-c statement_timeout=5000"},
    )


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def load(
    engine: Engine, admissions: pd.DataFrame, events: pd.DataFrame, chunksize: int = 50_000
) -> int:
    """Replace the contents of both tables. Returns the number of events written."""
    with engine.begin() as conn:
        conn.execute(delete(EventRow))
        conn.execute(delete(AdmissionRow))
    adm = admissions[["hadm_id", "subject_id", "admittime", "dischtime"]]
    adm.to_sql("admissions", engine, if_exists="append", index=False, chunksize=chunksize)
    cols = ["subject_id", "hadm_id", "source", "code", "value", "event_time", "available_time"]
    events[cols].to_sql("events", engine, if_exists="append", index=False, chunksize=chunksize)
    return len(events)


def get_admission(session: Session, hadm_id: int) -> AdmissionRow | None:
    return session.get(AdmissionRow, hadm_id)


def events_as_of(session: Session, hadm_id: int, as_of: datetime, clock: str) -> pd.DataFrame:
    """Events for one admission whose ``clock`` column is at or before ``as_of``.

    The filter runs in SQL, so a later code path cannot accidentally see the future.
    """
    column = EventRow.available_time if clock == "available_time" else EventRow.event_time
    stmt = (
        select(
            EventRow.subject_id,
            EventRow.hadm_id,
            EventRow.source,
            EventRow.code,
            EventRow.value,
            EventRow.event_time,
            EventRow.available_time,
        )
        .where(EventRow.hadm_id == hadm_id, column <= as_of)
        .order_by(EventRow.event_time)
    )
    rows = session.execute(stmt).all()
    frame = pd.DataFrame(
        rows,
        columns=[
            "subject_id",
            "hadm_id",
            "source",
            "code",
            "value",
            "event_time",
            "available_time",
        ],
    )
    for col in ("event_time", "available_time"):
        frame[col] = pd.to_datetime(frame[col])
    return frame
