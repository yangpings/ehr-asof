"""Load MIMIC-IV hosp tables into Postgres for SQL exploration (Part 2b).

Usage: uv run --extra postgres python notes/load_postgres.py ~/data/mimic-iv-demo [URL]
"""

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

TABLES = ("patients", "admissions", "labevents", "d_labitems", "diagnoses_icd")
TIME_COLUMNS = {
    "dod",
    "admittime",
    "dischtime",
    "deathtime",
    "edregtime",
    "edouttime",
    "charttime",
    "storetime",
}

root = Path(sys.argv[1]).expanduser()
url = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "postgresql+psycopg://postgres:dev@localhost:5432/postgres"
)
engine = create_engine(url)
for table in TABLES:
    frame = pd.read_csv(root / "hosp" / f"{table}.csv.gz", low_memory=False)
    for col in TIME_COLUMNS & set(frame.columns):
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
    frame.to_sql(table, engine, if_exists="replace", index=False, chunksize=10_000)
    print(f"{table}: {len(frame):,} rows")