"""Turn MIMIC-IV tables into one long event table with two clocks.

Every event carries:

- ``event_time``: when the thing happened clinically (a sample was drawn).
- ``available_time``: when a system at the bedside could first have known it.

Naive pipelines filter on ``event_time``. A deployed model only ever sees events
whose ``available_time`` has passed. The gap between the two is the leakage.
"""

from __future__ import annotations

import pandas as pd

EVENT_COLUMNS = ["subject_id", "hadm_id", "source", "code", "value", "event_time", "available_time"]


def lab_events(labevents: pd.DataFrame) -> pd.DataFrame:
    """Labs happen at ``charttime`` and become available at ``storetime``.

    When ``storetime`` is missing we fall back to ``charttime`` and say so in
    ``source``, because that row's availability is an assumption, not a fact.
    """
    labs = labevents.dropna(subset=["valuenum", "charttime"])
    missing_store = labs["storetime"].isna()
    return pd.DataFrame(
        {
            "subject_id": labs["subject_id"].astype("int64"),
            "hadm_id": labs["hadm_id"].astype("int64"),
            "source": missing_store.map({True: "lab_nostore", False: "lab"}),
            "code": "lab:" + labs["itemid"].astype(str),
            "value": labs["valuenum"].astype(float),
            "event_time": labs["charttime"],
            "available_time": labs["storetime"].fillna(labs["charttime"]),
        }
    )


def diagnosis_events(diagnoses_icd: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    """ICD codes have no timestamp in MIMIC-IV; they are coded after discharge.

    A common shortcut treats them as known at admission (``event_time`` =
    ``admittime``). Their honest ``available_time`` is ``dischtime``.
    """
    dx = diagnoses_icd.merge(
        admissions[["hadm_id", "admittime", "dischtime"]], on="hadm_id", how="inner"
    )
    return pd.DataFrame(
        {
            "subject_id": dx["subject_id"].astype("int64"),
            "hadm_id": dx["hadm_id"].astype("int64"),
            "source": "dx",
            "code": "dx:" + dx["icd_code"].astype(str).str.strip(),
            "value": 1.0,
            "event_time": dx["admittime"],
            "available_time": dx["dischtime"],
        }
    )


def build_events(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate lab and diagnosis events, sorted by admission and time."""
    parts = [
        lab_events(tables["labevents"]),
        diagnosis_events(tables["diagnoses_icd"], tables["admissions"]),
    ]
    events = pd.concat(parts, ignore_index=True)[EVENT_COLUMNS]
    return events.sort_values(["hadm_id", "event_time"], kind="stable").reset_index(drop=True)


def delay_report(events: pd.DataFrame) -> pd.DataFrame:
    """Hours between ``event_time`` and ``available_time``, per code.

    This is the first thing to look at on a new dataset: which signals arrive
    late, and by how much.
    """
    delay_h = (events["available_time"] - events["event_time"]).dt.total_seconds() / 3600
    frame = events.assign(delay_h=delay_h)
    report = frame.groupby(["source", "code"])["delay_h"].describe(percentiles=[0.5, 0.9])
    return report[["count", "50%", "90%", "max"]].rename(
        columns={"50%": "median_h", "90%": "p90_h", "max": "max_h"}
    )
