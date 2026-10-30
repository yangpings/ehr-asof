from __future__ import annotations

import pandas as pd

from ehr_asof.events import build_events, delay_report, diagnosis_events, lab_events


def test_lab_available_at_storetime() -> None:
    labs = pd.DataFrame(
        {
            "subject_id": [1, 1],
            "hadm_id": [10, 10],
            "itemid": [50912, 50912],
            "charttime": pd.to_datetime(["2150-01-01 08:00", "2150-01-01 09:00"]),
            "storetime": pd.to_datetime(["2150-01-01 10:30", None]),
            "valuenum": [1.1, 1.3],
        }
    )
    ev = lab_events(labs)
    assert ev["available_time"].iloc[0] == pd.Timestamp("2150-01-01 10:30")
    # missing storetime falls back to charttime and is flagged in `source`
    assert ev["available_time"].iloc[1] == pd.Timestamp("2150-01-01 09:00")
    assert ev["source"].tolist() == ["lab", "lab_nostore"]


def test_diagnoses_available_at_discharge() -> None:
    adm = pd.DataFrame(
        {
            "hadm_id": [10],
            "admittime": pd.to_datetime(["2150-01-01"]),
            "dischtime": pd.to_datetime(["2150-01-05"]),
        }
    )
    dx = pd.DataFrame({"subject_id": [1], "hadm_id": [10], "icd_code": ["I469 "]})
    ev = diagnosis_events(dx, adm)
    assert ev["code"].iloc[0] == "dx:I469"
    assert ev["event_time"].iloc[0] == pd.Timestamp("2150-01-01")
    assert ev["available_time"].iloc[0] == pd.Timestamp("2150-01-05")


def test_available_never_before_event(tables: dict[str, pd.DataFrame]) -> None:
    ev = build_events(tables)
    assert (ev["available_time"] >= ev["event_time"]).all()


def test_delay_report_shape(tables: dict[str, pd.DataFrame]) -> None:
    report = delay_report(build_events(tables))
    assert {"count", "median_h", "p90_h", "max_h"} <= set(report.columns)
    assert (report["median_h"] >= 0).all()
