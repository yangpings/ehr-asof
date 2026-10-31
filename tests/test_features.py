from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ehr_asof.features import build_features, leakage_report, visible_events

BASE = datetime(2150, 1, 1)


@st.composite
def event_sets(draw: st.DrawFn) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_adm = draw(st.integers(1, 4))
    rows = []
    for hadm in range(n_adm):
        for _ in range(draw(st.integers(0, 12))):
            t = BASE + timedelta(minutes=draw(st.integers(0, 3000)))
            delay = timedelta(minutes=draw(st.integers(0, 2000)))
            is_dx = draw(st.booleans())
            rows.append(
                {
                    "subject_id": hadm,
                    "hadm_id": hadm,
                    "source": "dx" if is_dx else "lab",
                    "code": draw(
                        st.sampled_from(["dx:A", "dx:B"] if is_dx else ["lab:1", "lab:2"])
                    ),
                    "value": 1.0 if is_dx else float(draw(st.integers(0, 100))),
                    "event_time": t,
                    "available_time": t + delay,
                }
            )
    events = pd.DataFrame(
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
        events[col] = pd.to_datetime(events[col])
    cutoffs = pd.DataFrame(
        {
            "hadm_id": list(range(n_adm)),
            "cutoff": [BASE + timedelta(minutes=draw(st.integers(0, 4000))) for _ in range(n_adm)],
        }
    )
    return events, cutoffs


CODES = ["dx:A", "dx:B", "lab:1", "lab:2"]


@settings(max_examples=200, deadline=None)
@given(event_sets())
def test_asof_equals_features_of_available_events_only(
    data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The as-of matrix is unchanged if we delete every event not yet available."""
    events, cutoffs = data
    full = build_features(events, cutoffs, "asof", CODES)
    merged = events.merge(cutoffs, on="hadm_id")
    available = merged[merged["available_time"] <= merged["cutoff"]].drop(columns="cutoff")
    only_available = build_features(available, cutoffs, "asof", CODES)
    pd.testing.assert_frame_equal(full, only_available)


@settings(max_examples=200, deadline=None)
@given(event_sets(), st.integers(1, 500))
def test_planted_future_value_never_used(
    data: tuple[pd.DataFrame, pd.DataFrame], minutes: int
) -> None:
    """A lab drawn before the cutoff but stored after it must not appear as-of,
    and does appear in the naive matrix: that's the leak."""
    events, cutoffs = data
    hadm = int(cutoffs["hadm_id"].iloc[0])
    cutoff = cutoffs["cutoff"].iloc[0]
    planted = pd.DataFrame(
        [
            {
                "subject_id": hadm,
                "hadm_id": hadm,
                "source": "lab",
                "code": "lab:9",
                "value": -999.0,
                "event_time": cutoff - timedelta(minutes=1),
                "available_time": cutoff + timedelta(minutes=minutes),
            }
        ]
    )
    with_plant = pd.concat([events, planted], ignore_index=True)
    codes = [*CODES, "lab:9"]
    asof = build_features(with_plant, cutoffs, "asof", codes)
    naive = build_features(with_plant, cutoffs, "naive", codes)
    assert not (asof["last:lab:9"] == -999.0).any()
    assert naive.loc[hadm, "last:lab:9"] == -999.0


def test_vocabulary_columns_stable() -> None:
    events = pd.DataFrame(
        columns=["subject_id", "hadm_id", "source", "code", "value", "event_time", "available_time"]
    )
    events = events.astype(
        {"event_time": "datetime64[ns]", "available_time": "datetime64[ns]", "source": str}
    )
    cutoffs = pd.DataFrame({"hadm_id": [1], "cutoff": [pd.Timestamp(BASE)]})
    feats = build_features(events, cutoffs, "asof", CODES)
    assert list(feats.columns) == [
        "last:lab:1",
        "last:lab:2",
        "count:labs",
        "flag:dx:A",
        "flag:dx:B",
    ]
    assert np.isnan(feats.loc[1, "last:lab:1"])
    assert feats.loc[1, "flag:dx:A"] == 0


def test_bad_mode_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        visible_events(
            pd.DataFrame({"hadm_id": []}), pd.DataFrame({"hadm_id": [], "cutoff": []}), "future"
        )  # type: ignore[arg-type]


def test_leakage_report_detects_difference() -> None:
    idx = pd.Index([1, 2], name="hadm_id")
    naive = pd.DataFrame({"last:lab:1": [1.0, np.nan], "flag:dx:A": [1, 0]}, index=idx)
    asof = pd.DataFrame({"last:lab:1": [1.0, np.nan], "flag:dx:A": [0, 0]}, index=idx)
    rep = leakage_report(naive, asof)
    assert rep["cells_leaked"] == 0.25
    assert rep["admissions_with_any_leak"] == 0.5
    assert rep["flag_cells_leaked"] == 0.5
    assert rep["last_cells_leaked"] == 0.0


def test_leakage_report_requires_alignment() -> None:
    a = pd.DataFrame({"x": [1.0]}, index=[1])
    b = pd.DataFrame({"y": [1.0]}, index=[1])
    with pytest.raises(ValueError):
        leakage_report(a, b)
