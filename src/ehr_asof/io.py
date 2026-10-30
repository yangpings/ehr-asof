"""Read and write MIMIC-IV-shaped tables.

``read_tables`` accepts the layout of the MIMIC-IV demo download
(``<root>/hosp/admissions.csv.gz`` ...) or a flat directory of CSV files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TABLES = ("patients", "admissions", "labevents", "d_labitems", "diagnoses_icd", "prescriptions")

DATETIME_COLUMNS: dict[str, list[str]] = {
    "patients": ["dod"],
    "admissions": ["admittime", "dischtime", "deathtime"],
    "labevents": ["charttime", "storetime"],
    "prescriptions": ["starttime", "stoptime"],
}

USECOLS: dict[str, list[str]] = {
    "labevents": [
        "labevent_id",
        "subject_id",
        "hadm_id",
        "itemid",
        "charttime",
        "storetime",
        "valuenum",
    ],
    "prescriptions": ["subject_id", "hadm_id", "starttime", "stoptime", "drug"],
}


def _find(root: Path, table: str) -> Path | None:
    for candidate in (
        root / "hosp" / f"{table}.csv.gz",
        root / "hosp" / f"{table}.csv",
        root / f"{table}.csv.gz",
        root / f"{table}.csv",
    ):
        if candidate.exists():
            return candidate
    return None


def read_tables(root: str | Path, tables: tuple[str, ...] = TABLES) -> dict[str, pd.DataFrame]:
    """Load the named tables, parsing their timestamp columns.

    Missing optional tables (prescriptions) are returned empty; missing required
    tables raise ``FileNotFoundError`` naming the file that was expected.
    """
    root = Path(root)
    out: dict[str, pd.DataFrame] = {}
    for table in tables:
        path = _find(root, table)
        if path is None:
            if table == "prescriptions":
                out[table] = pd.DataFrame(columns=USECOLS["prescriptions"])
                continue
            raise FileNotFoundError(
                f"{table}: expected {root}/hosp/{table}.csv.gz or {root}/{table}.csv"
            )
        frame = pd.read_csv(path, usecols=USECOLS.get(table), low_memory=False)
        for col in DATETIME_COLUMNS.get(table, []):
            if col in frame:
                frame[col] = pd.to_datetime(frame[col], errors="coerce")
        out[table] = frame
    if "labevents" in out:
        # Lab rows without an admission cannot be placed on an admission timeline.
        out["labevents"] = out["labevents"].dropna(subset=["hadm_id"])
        out["labevents"]["hadm_id"] = out["labevents"]["hadm_id"].astype("int64")
    return out


def write_tables(tables: dict[str, pd.DataFrame], root: str | Path) -> Path:
    """Write tables in the MIMIC-IV demo layout: ``<root>/hosp/<table>.csv.gz``."""
    hosp = Path(root) / "hosp"
    hosp.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(hosp / f"{name}.csv.gz", index=False)
    return hosp
