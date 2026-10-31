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
- 0.92% of all lab rows have no `storetime` (inpatient: 0.44%). The fallback to `charttime` (flagged `lab_nostore`) affects very few rows.

## Admissions and readmission gaps
- Dates are shifted into the 2100s for de-identification; intervals within a patient are real.
- Some "readmissions" start 0–1 minute after discharge (e.g. one patient with gaps of 00:01 and 00:00). These are transfers or back-to-back records, not true readmissions, so a naive 30-day readmission rule would count them.
- Several patients return within days (1–11 days), others after years.

## Index (EXPLAIN ANALYZE)
- Single-admission lookup (282 of 107,727 lab rows).
- Without index: Seq Scan, execution time 5.323 ms.
- With index on `hadm_id`: Bitmap Index Scan, execution time 0.153 ms (~35× faster).
- On full MIMIC-IV, with over 100 million lab rows, a sequential scan would read every row for each lookup. This is why the service in Part 6 indexes `(hadm_id, available_time)`.