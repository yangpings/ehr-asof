"""Point-in-time feature construction.

``build_features`` is the whole project in one function: the same code builds a
feature matrix in two modes that differ only in which clock they trust.

- ``mode="naive"``: keep events whose ``event_time <= cutoff``.
- ``mode="asof"``:  keep events whose ``available_time <= cutoff``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Mode = Literal["naive", "asof"]


def vocabulary(events: pd.DataFrame, min_count: int = 1) -> list[str]:
    """Codes to use as feature columns, fitted on training events only."""
    counts = events["code"].value_counts()
    return sorted(counts[counts >= min_count].index.tolist())


def visible_events(events: pd.DataFrame, cutoffs: pd.DataFrame, mode: Mode) -> pd.DataFrame:
    """Events each admission could see at its cutoff, under the chosen clock."""
    if mode not in ("naive", "asof"):
        raise ValueError(f"mode must be 'naive' or 'asof', got {mode!r}")
    clock = "event_time" if mode == "naive" else "available_time"
    merged = events.merge(cutoffs[["hadm_id", "cutoff"]], on="hadm_id", how="inner")
    return merged[merged[clock] <= merged["cutoff"]]


Status = Literal["known", "leaked", "future"]


def availability_status(events: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Label each event by what a system could know about it at ``as_of``.

    - ``known``: already available (``available_time <= as_of``); an as-of model uses it.
    - ``leaked``: it happened (``event_time <= as_of``) but wasn't available yet; only a
      naive pipeline uses it.
    - ``future``: it hadn't happened yet.
    """
    known = events["available_time"] <= as_of
    happened = events["event_time"] <= as_of
    status = np.where(known, "known", np.where(happened, "leaked", "future"))
    return pd.Series(status, index=events.index, name="status")


def build_features(
    events: pd.DataFrame,
    cutoffs: pd.DataFrame,
    mode: Mode,
    codes: list[str] | None = None,
) -> pd.DataFrame:
    """One row per admission in ``cutoffs`` (columns ``hadm_id``, ``cutoff``).

    Lab codes become ``last:<code>`` (latest visible value, NaN if none) and
    ``count:labs``. Diagnosis codes become 0/1 flags ``flag:<code>``.
    """
    codes = codes if codes is not None else vocabulary(events)
    lab_codes = [c for c in codes if c.startswith("lab:")]
    dx_codes = [c for c in codes if c.startswith("dx:")]
    seen = visible_events(events, cutoffs, mode)
    index = pd.Index(cutoffs["hadm_id"].to_numpy(), name="hadm_id")

    labs = seen[seen["source"].str.startswith("lab")].sort_values("event_time", kind="stable")
    last = labs.groupby(["hadm_id", "code"])["value"].last().unstack()
    last = last.reindex(index=index, columns=lab_codes)
    last.columns = [f"last:{c}" for c in lab_codes]

    n_labs = labs.groupby("hadm_id").size().reindex(index, fill_value=0).rename("count:labs")

    dx = seen[seen["source"] == "dx"]
    flags = (
        dx.assign(one=1)
        .pivot_table(index="hadm_id", columns="code", values="one", aggfunc="max")
        .reindex(index=index, columns=dx_codes)
        .fillna(0)
        .astype(np.int8)
    )
    flags.columns = [f"flag:{c}" for c in dx_codes]

    return pd.concat([last, n_labs.astype(float), flags], axis=1)


def leakage_report(naive: pd.DataFrame, asof: pd.DataFrame) -> dict[str, float]:
    """How much of the naive matrix could not have been known at the cutoff.

    A cell leaks when its naive value differs from its as-of value (two NaNs
    count as equal). Returns overall and per-family fractions.
    """
    if not naive.index.equals(asof.index) or list(naive.columns) != list(asof.columns):
        raise ValueError("naive and asof matrices must share index and columns")
    a, b = naive.to_numpy(dtype=float), asof.to_numpy(dtype=float)
    differs = ~((a == b) | (np.isnan(a) & np.isnan(b)))
    out = {
        "cells_leaked": float(differs.mean()) if differs.size else 0.0,
        "admissions_with_any_leak": float(differs.any(axis=1).mean()) if len(a) else 0.0,
    }
    for family in ("last:", "count:", "flag:"):
        cols = [i for i, c in enumerate(naive.columns) if c.startswith(family)]
        if cols:
            out[f"{family.rstrip(':')}_cells_leaked"] = float(differs[:, cols].mean())
    return out
