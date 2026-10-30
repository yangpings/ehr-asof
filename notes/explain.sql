DROP INDEX IF EXISTS ix_labevents_hadm;
EXPLAIN ANALYZE SELECT * FROM labevents WHERE hadm_id = (SELECT min(hadm_id) FROM admissions);

CREATE INDEX ix_labevents_hadm ON labevents (hadm_id);
ANALYZE labevents;
EXPLAIN ANALYZE SELECT * FROM labevents WHERE hadm_id = (SELECT min(hadm_id) FROM admissions);