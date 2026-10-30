# Exploration: MIMIC-IV Clinical Database Demo v2.2

Loaded into Postgres 16 (Docker) with `notes/load_postgres.py`; queries in `notes/explore.sql`.

## Data
- 100 patients, 275 admissions, 107,727 lab results (79,307 linked to an admission; 28,420, or 26%, are outpatient and are dropped by `read_tables`).

## How late do lab results arrive? (`storetime - charttime`)
- The raw ranking of all lab types is dominated by tests with only one or two results (e.g. Hemoglobin C, 227 h, n = 1) and by outpatient tests, so it is not informative.
- Restricted to inpatient labs with at least 50 results, the slowest are manual blood-smear reads:

| Lab | n | Median (h) | p90 (h) |
| --- | --- | --- | --- |
| Blasts | 57 | 2.80 | 19.03 |
| Schistocytes | 70 | 2.66 | 5.10 |
| Microcytes | 244 | 2.47 | 4.79 |

- Implication: a model predicting at 24 h from `charttime` would use smear results drawn in the previous ~2–5 h that were not yet available.

## Missing storetime
- 0.92% of all lab rows have no `storetime` (inpatient: X%). The fallback to `charttime` (flagged `lab_nostore`) affects very few rows.

## Admissions and readmission gaps
- Dates are shifted into the 2100s for de-identification; intervals within a patient are real.
- Some "readmissions" start 0–1 minute after discharge (e.g. one patient with gaps of 00:01 and 00:00). These are transfers or back-to-back records, not true readmissions, so a naive 30-day readmission rule would count them.
- Several patients return within days (1–11 days), others after years.

## Index (EXPLAIN ANALYZE)
- Without index: Seq Scan, execution time ___ ms.
- With index on `hadm_id`: Index Scan, execution time ___ ms.