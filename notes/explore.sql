-- 1. How late do lab results arrive? (slowest 20 lab types)
SELECT d.label,
       count(*) AS n,
       round((percentile_cont(0.5) WITHIN GROUP (
           ORDER BY extract(epoch FROM l.storetime - l.charttime) / 3600))::numeric, 2) AS median_h
FROM labevents l JOIN d_labitems d USING (itemid)
WHERE l.storetime IS NOT NULL
GROUP BY d.label
ORDER BY median_h DESC
LIMIT 20;
-- 1b. Slowest *common inpatient* labs: at least 50 results, admission labs only.
--     Query 1 is dominated by rare tests (n = 1) and outpatient labs.
SELECT d.label,
       count(*) AS n,
       round((percentile_cont(0.5) WITHIN GROUP (
           ORDER BY extract(epoch FROM l.storetime - l.charttime) / 3600))::numeric, 2) AS median_h,
       round((percentile_cont(0.9) WITHIN GROUP (
           ORDER BY extract(epoch FROM l.storetime - l.charttime) / 3600))::numeric, 2) AS p90_h
FROM labevents l JOIN d_labitems d USING (itemid)
WHERE l.storetime IS NOT NULL AND l.hadm_id IS NOT NULL
GROUP BY d.label
HAVING count(*) >= 50
ORDER BY median_h DESC
LIMIT 15;

-- 2. What fraction of lab rows have no storetime? (all labs, then inpatient only)
SELECT round(avg((storetime IS NULL)::int), 4) AS share_no_storetime FROM labevents;
SELECT round(avg((storetime IS NULL)::int), 4) AS share_no_storetime_inpatient
FROM labevents WHERE hadm_id IS NOT NULL;

-- 3. Number each patient's admissions and find the gap to the next one.
SELECT subject_id, hadm_id, admittime, dischtime,
       ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY admittime) AS admission_number,
       LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime) AS next_admittime,
       LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime) - dischtime AS gap
FROM admissions
ORDER BY subject_id, admittime
LIMIT 20;