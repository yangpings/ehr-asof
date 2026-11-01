# ehr-asof

**Keywords:** EHR · MIMIC-IV · temporal leakage · point-in-time correctness · clinical prediction · AUROC · bootstrap confidence intervals

**Skills:** Python · pandas · NumPy · scikit-learn · SQL · PostgreSQL · SQLAlchemy 2.0 · FastAPI · Pydantic · Docker / Docker Compose · GitHub Actions CI · uv packaging · pytest · Hypothesis (property-based testing) · mypy (strict) · ruff · cProfile profiling

Point-in-time (as-of) features for EHR data, and a measurement of how much EHR prediction models rely on information that wasn't available at prediction time.

## Description

EHR tables give most events a clinical time, such as when a blood sample was drawn (`charttime`), but the hospital system often learned the result later (`storetime`). Diagnosis codes carry no time at all, because they are assigned after discharge. A model trained on "everything charted before 24 h" can therefore use results and codes that did not yet exist at 24 h. It looks excellent in a paper and fails silently in deployment. `ehr-asof` gives every event two clocks, builds features under either one, and measures the difference.

## Objective

1. Build a tested method for point-in-time-correct EHR features.
2. Measure how much temporal leakage inflates reported model performance, and what a hospital would get if it deployed the leaky model.

## Hypotheses

- **H1.** Features built on clinical time (naive) contain information that was not available at the prediction time.
- **H2.** Leakage inflates reported AUROC only for outcomes that the leaked information predicts (in-hospital mortality), not for others (length of stay).
- **H3.** A model trained on naive features performs worse when deployed on what is actually available than a model trained honestly.
- **H4.** The earlier the prediction time, the larger the inflation.

## Samples

| Dataset | Access | Size | Role |
| --- | --- | --- | --- |
| Synthetic, MIMIC-shaped (`ehr-asof synth`, seed 0) | Generated | 2,000 patients, 2,737 admissions | Known-answer validation: reporting delays and discharge-time codes are planted |
| MIMIC-IV Clinical Database Demo v2.2 | Open (ODbL) | 100 patients, 275 admissions, 79,307 inpatient lab results, 4,506 diagnosis codes | End-to-end run on real EHR data |

The demo shares the full MIMIC-IV schema, so the same code runs on full MIMIC-IV (credentialed) unchanged.

## Methods

- **Two clocks per event.** Labs: `event_time` = `charttime`, `available_time` = `storetime` (falls back to `charttime` when missing, flagged as `lab_nostore`). Diagnoses: `event_time` = `admittime` (the common shortcut), `available_time` = `dischtime`.
- **Features.** Latest value per lab, lab count and diagnosis flags, built in two modes: *naive* keeps events with `event_time <= cutoff`; *as-of* keeps events with `available_time <= cutoff`. Vocabulary fitted on training admissions only; static age, sex and emergency admission.
- **Cohort.** Admissions still in hospital and alive at `admittime + 24 h` (6, 12 and 48 h as sensitivity analyses). Outcomes: in-hospital mortality and length of stay > 7 days.
- **Model.** Median imputation with missingness indicators, standardization, logistic regression. Split by patient (70/30).
- **Three scores on the same test patients.** *Reported*: naive model on naive features. *Honest*: as-of model on as-of features. *Deployed*: naive model on as-of features.
- **Uncertainty.** Paired bootstrap over patients (1,000 resamples), scoring every model on the same resample, so differences get their own 95% interval.
- **Computational efficiency.** Profiling (cProfile) showed the bootstrap dominated run time: 1,000 resamples × 3 models plus 3 point estimates is 3,003 AUROC evaluations, and scikit-learn's `roc_auc_score` spends most of each call validating its inputs rather than computing. AUROC is therefore computed directly from the Mann–Whitney identity, AUROC = (R₊ − n₊(n₊+1)/2) / (n₊n₋), where R₊ is the sum of the positives' ranks and tied scores receive their average rank. A Hypothesis property test checks that it equals `roc_auc_score` on hundreds of random inputs with many ties.
- **Storage and serving.** Events are stored in Postgres with both clocks; the as-of filter runs in SQL on an index over `(hadm_id, available_time)`. On the demo, an index on `hadm_id` cut a single-admission lab lookup from 5.3 ms (sequential scan) to 0.15 ms (~35×).
- **Correctness checks.** Hypothesis property tests on hundreds of random event sets: as-of features equal the features computed after deleting every not-yet-available event, and a planted future value never appears as-of but does appear naive.

## Results

### Synthetic data (known answer)

1,935 training and 802 test admissions; 23 features; all 1,000 bootstrap resamples valid. 87.4% of test admissions had at least one naive feature value that wasn't available at 24 h (13.3% of all feature cells, and 61% of lab-count cells).

| Outcome (test n = 802) | Reported | Honest | Deployed | Reported − honest | Reported − deployed |
| --- | --- | --- | --- | --- | --- |
| In-hospital mortality (8.4%) | 0.996 [0.988, 1.000] | 0.884 [0.840, 0.923] | 0.857 [0.814, 0.899] | +0.112 [0.073, 0.155] | +0.139 [0.097, 0.184] |
| Length of stay > 7 days (20.1%) | 0.684 [0.640, 0.729] | 0.682 [0.640, 0.727] | 0.683 [0.640, 0.728] | +0.002 [−0.012, 0.015] | +0.001 [−0.013, 0.014] |

| Mortality, cutoff | Reported | Honest | Deployed | Reported − honest | Reported − deployed |
| --- | --- | --- | --- | --- | --- |
| 6 h | 0.991 | 0.843 | 0.772 | 0.148 | 0.219 |
| 12 h | 0.991 | 0.851 | 0.823 | 0.140 | 0.168 |
| 24 h | 0.996 | 0.884 | 0.857 | 0.112 | 0.139 |
| 48 h | 0.997 | 0.959 | 0.952 | 0.038 | 0.045 |

All property tests pass (200 random cases each). When the as-of filter is deliberately switched to clinical time, both fail, and Hypothesis shrinks the failure to a single lab result drawn at the cutoff and stored one minute later: the tests catch even a one-minute leak.

### Service

The API returns an admission's features as of any timestamp. For synthetic admission 20000001: as of 2150-01-01 (before admission) it uses 0 events; as of 2150-12-31 (after discharge) it uses 197 events.

**The leak, in one request.** The same admission, the same moment (24 hours after admission, 2150-07-25 20:37), in both modes:

| | As-of (what a deployed system knows) | Naive (clinical time) |
| --- | --- | --- |
| Events used | 12 | 16 |
| Lab results | 12 | 13: adds HbA1c (6.82%), drawn before 24 h but not yet resulted |
| Diagnosis flags | none | cardiac arrest (I469), septic shock (R6521), palliative care (Z515) |

At 24 hours the naive pipeline already "knows" that this patient will be coded for cardiac arrest and palliative care, codes that were only assigned at discharge. This is the information that inflates reported mortality AUROC.

| Load test (closed loop, 20 clients, 15 s, SQLite, 1 worker) | Result |
| --- | --- |
| Requests (errors) | 1,696 (0) |
| Throughput | 112.3 req/s |
| Latency p50 / p95 | 178 ms / 224 ms |
| Little's law check, L = λW | 112.3 × 0.178 s = 19.9 in flight (20 clients) |

The Little's law check confirms the measurement is internally consistent: with 20 clients always waiting on a response, throughput × latency should equal 20.

### Containers and failure drill

`docker compose up --build` starts Postgres 16, a one-shot loader (exits 0 after loading), and the API (non-root user, health check on `/health/live`). The API image is 740 MB on disk (168 MB compressed), mostly the scientific Python stack.

| Outage drill (database stopped under a running API) | `/health/live` | `/health/ready` |
| --- | --- | --- |
| Normal | 200 | 200 |
| `docker compose stop db` | 200 (0.55 ms) | 503 with `Retry-After: 5`, after 4.0 s |
| `docker compose start db` (no API restart) | 200 | 200 (6.3 ms) |

The API degraded cleanly and recovered on its own; the request ID on the 503 response matches the server's JSON log line. The full write-up is in [`docs/outage-drill.md`](docs/outage-drill.md).

### Computational performance

| | Experiment wall time (mortality, 1,000 resamples) | AUROC calls | Share of profiled time in AUROC |
| --- | --- | --- | --- |
| scikit-learn `roc_auc_score` | 6.1 s | 3,003 | 95% (8.8 of 9.3 s) |
| Rank-based (Mann–Whitney) AUROC | 2.7 s | 3,003 | 41% (0.33 of 0.82 s) |

Under the profiler the whole experiment went from 9.3 s to 0.8 s, and the AUROC calls from 8.8 s to 0.33 s (about 27× faster). Most of the remaining wall time is Python start-up and imports. The full test suite, which includes a 120-repetition bootstrap coverage test, now runs in under 30 seconds.

Point estimates and all bootstrap intervals are unchanged: the largest difference across every estimate and interval bound is 1.9 × 10⁻¹⁶ (floating-point rounding), so the speedup does not alter any result.

### MIMIC-IV demo (real data)

**Data flow.** The demo has 107,727 lab results; 28,420 (26%) are outpatient labs with no admission and are dropped, leaving 79,307 inpatient results for 275 admissions.

**Lab results arrive late.** Delay from draw (`charttime`) to result (`storetime`), inpatient lab types with at least 50 results:

| Lab | n | Median (h) | p90 (h) |
| --- | --- | --- | --- |
| Blasts (smear) | 57 | 2.80 | 19.03 |
| Schistocytes (smear) | 70 | 2.66 | 5.10 |
| Microcytes (smear) | 244 | 2.47 | 4.79 |
| Anisocytosis (smear) | 249 | 2.43 | 4.77 |
| Platelet smear | 264 | 2.28 | 4.23 |

The slowest common labs are manual blood-smear reads: a model predicting at 24 h from clinical time would use smear results drawn in the previous 2–5 hours (up to 19 h for blasts) that were not yet available. The same delays were computed independently in SQL (Postgres) and in Python (`ehr-asof delays`) and agree exactly.

**Diagnosis codes leak the most.** Codes become available at discharge, a median of 100–136 hours after admission for the most common codes (e.g. hypertension, hyperlipidemia), yet the common shortcut treats them as known at admission.

**The fallback rule is rarely needed.** 0.92% of all lab rows, and 0.44% of inpatient lab rows, have no `storetime`.

**AUROC on the demo** (24 h cutoff; 186 training and 59 test admissions; 315 features; 1,000 patient-level bootstrap resamples):

| Outcome (test prevalence) | Reported | Honest | Deployed | Reported − honest | Reported − deployed | Valid resamples |
| --- | --- | --- | --- | --- | --- | --- |
| In-hospital mortality (6.8%, 4 deaths) | 0.768 [0.454, 0.981] | 0.668 [0.314, 1.000] | 0.468 [0.162, 0.979] | +0.100 [−0.293, 0.576] | +0.300 [−0.017, 0.719] | 983 |
| Length of stay > 7 days (40.7%) | 0.717 [0.598, 0.835] | 0.562 [0.442, 0.683] | 0.520 [0.389, 0.657] | +0.155 [0.053, 0.236] | +0.196 [0.083, 0.287] | 1,000 |

In the demo test set, 93.2% of admissions had at least one feature value that wasn't available at 24 h (2.9% of feature cells; 17% of lab-count cells).

- **Mortality is uninformative at this size.** With 4 deaths among 59 test admissions, every interval spans most of the 0–1 range and 17 resamples had no deaths at all.
- **Length of stay shows inflation on real data, unlike the synthetic data.** The reported − honest gap is +0.155 with an interval excluding 0, and the deployed model is close to chance (0.520). In the synthetic generator, discharge codes were tied to death but not to stay length; in real records, longer stays plausibly accumulate more diagnosis codes and more labs, so the leaked information also predicts length of stay. Repeating the experiment over 10 random patient splits (seeds 0–9, 200 resamples each) gave reported − honest gaps of 0.155, 0.049, −0.034, 0.014, 0.048, 0.019, 0.086, 0.063, 0.010 and 0.124: positive in 9 of 10 splits, mean +0.053. The direction is consistent, but the first split (+0.155) was at the high end, so the size is uncertain. This is a hypothesis to test on full MIMIC-IV, not a finding.

## Discussion

On synthetic data, where the leak is planted and the answer is known, all four hypotheses hold. Naive features leaked for most admissions (H1). Mortality AUROC was inflated by 0.11 while length of stay was unaffected, even though both tasks used the same leaky features (87.4% of admissions affected): the leaked discharge codes (cardiac arrest, palliative care) predict death but not stay length (H2). Feature-level leakage alone doesn't tell you how much a result is inflated; it has to be measured per task. Shipping the naive model was worse than training honestly: 0.857 vs 0.884 (H3). The gap grew from 0.04 at 48 h to 0.15 at 6 h (H4). The practical lesson is that leakage is task-specific and has to be measured, not assumed, and that the cost of ignoring it is paid at deployment, not in the paper.

On the real MIMIC-IV demo, the picture changed in an instructive way. Length of stay, which showed no inflation on synthetic data, was inflated on the demo: +0.155 [0.053, 0.236] on the first split, and positive in 9 of 10 random splits (mean +0.053); the naive model was close to chance once deployed on what was actually available (0.520 on the first split). Hypothesis H2 as stated (inflation for mortality but not length of stay) held on synthetic data but not on real data: which outcomes are affected depends on how the leaked information relates to each outcome in real records, which a synthetic generator can only assume. This is the strongest argument for measuring leakage per task on real data rather than reasoning about it.

The method's correctness does not rest on sample size: it is established by property tests and by known-answer synthetic data. The MIMIC-IV demo shows that the pipeline runs on real EHR data and that the leak is real: diagnosis codes become available days after admission, and common lab results arrive hours after they are drawn.

The performance work illustrates a general point for evaluation-heavy research code: the expensive step was not modeling but a small statistic evaluated thousands of times, where a library's per-call overhead outweighed the computation itself. Measuring before optimizing located the cost, and an equivalent closed-form computation, verified against the reference implementation, removed it without changing any result. Resampling-based uncertainty estimates are only practical to run routinely, and in CI, if they are fast.

**Strengths.** Correctness proven by property tests rather than examples; known-answer validation; paired patient-level bootstrap with a patient-level split; the same feature code in the experiment and in the HTTP service (no training–serving skew); reproducible by anyone from open data, a lock file and CI on Python 3.11–3.13; a profiled, verified AUROC that keeps the full bootstrap fast enough to run in tests.

## Limitations

- **No real-world effect size.** The demo has 100 patients; its AUROCs have wide intervals and are not evidence about the size of the effect in practice.
- **One small split on the demo.** The demo results rest on a single train/test split with 59 test admissions (4 deaths) and 315 features fitted on 186 training admissions; the bootstrap reflects test-set sampling only; across 10 random splits the length-of-stay gap ranged from −0.034 to +0.155.
- **Synthetic effects are planted.** Their size reflects the generator's assumptions, not real patients.
- **`storetime` is a proxy** for when a clinician or a model's data feed actually saw a result; real feeds may lag further.
- **Availability rules are assumptions** (diagnoses known at discharge; labs without `storetime` fall back to `charttime`) and should be checked by a clinician for each site.
- **Scope.** Labs and diagnoses only (no vitals, notes or medications); one model family; one hospital system.

## Trial and error

What went wrong while building this, and what changed as a result. Analysis problems come first because they affect results; engineering problems follow because they affect reproducibility.

### Analysis and data

These shaped the analysis and are reported so the choices can be checked.

| Problem | Why it matters | How it was handled |
| --- | --- | --- |
| 26% of demo lab results (28,420 of 107,727) have no admission (`hadm_id`) | Outpatient labs cannot be placed on an inpatient timeline; keeping them would mix settings | Dropped in `read_tables`, with the count reported |
| The raw ranking of slowest lab types was topped by tests with a single result (e.g. Hemoglobin C, 227 h, n = 1) and by outpatient tests | A median over one result is not a finding; the naive ranking was misleading | Restricted to inpatient lab types with at least 50 results, and reported the p90 alongside the median |
| Lab delays could be wrong in either the SQL or the Python implementation | Delay estimates drive the leakage story | Computed independently in Postgres (SQL) and in Python (`delay_report`); both give, e.g., Blasts median 2.8 h and p90 19.0 h |
| Some consecutive admissions are 0–1 minute apart | These are transfers or back-to-back records, not readmissions; a naive 30-day readmission rule counts them | Noted as a data-quality issue for any readmission outcome (studied further in the companion project *cohort-diff*) |
| A test suite can pass without proving the as-of filter works | Passing example tests is weak evidence of point-in-time correctness | The as-of filter was deliberately broken (clinical time instead of availability time); both property tests failed and shrank to a single lab stored one minute after the cutoff |
| Feature-level leakage was identical for both outcomes (87.4% of admissions) but inflated only mortality AUROC | Counting leaked features overstates or understates the damage | Inflation is reported per task as reported − honest and reported − deployed AUROC with intervals |
| The bootstrap spent 95% of run time in scikit-learn's per-call input validation (3,003 AUROC calls) | Slow resampling discourages rigorous uncertainty estimates and makes CI impractical | Replaced with a rank-based Mann–Whitney AUROC, verified equal to scikit-learn on tied scores by a property test; results unchanged |
| On the demo, one train/test split showed a large length-of-stay inflation (+0.155, interval excluding 0) | With 59 test admissions, one split can be unrepresentative, and the bootstrap only reflects test-set sampling | Repeated over 10 random patient splits: positive in 9 of 10, mean +0.053. Reported the direction as consistent and the size as uncertain |
| A faster AUROC could silently mishandle tied scores | Ties are common (rounded probabilities, integer scores); wrong tie handling biases AUROC | Deliberately removed tie averaging from the rank computation: the constant-score test returned 0.75 instead of 0.5, and the property test against scikit-learn shrank the failure to one negative and one positive with the same score (AUROC 1.0 instead of 0.5), so the tests guard the shortcut |
| PhysioNet hosts two near-identical demo datasets (CSV and FHIR) | The code reads the CSV tables; the FHIR version is a different resource with different citations | Uses MIMIC-IV Clinical Database Demo v2.2 (CSV) and cites it |

### Engineering

| Part | What went wrong | Cause | Fix and lesson |
| --- | --- | --- | --- |
| Setup | `uv init` didn't create a Git repo, and later `git status` listed files from unrelated projects | The project sits inside a larger existing repo, so Git used the parent | Gave the project its own repo and ignored it in the parent. Lesson: check `git rev-parse --show-toplevel` before trusting Git in a nested folder |
| Setup | The first push had no CI run, and `.coverage` and `__pycache__` files were committed | `.gitignore` and the CI workflow were added after the first commit | Added them and untracked the generated files. Lesson: `.gitignore` before the first `git add` |
| Setup | `ruff format --check` crashed with a panic | A test file written by Windows PowerShell's `Set-Content` had a UTF-8 byte-order mark and CRLF line endings, which tripped a bug in ruff's error display | Recreated the file in the editor (UTF-8, LF). Lesson: tool crashes can come from file encoding; reproduce before blaming the code |
| Setup | CI's format check failed on files that looked identical to the expected output | Files had no final newline | `ruff format` before every commit; editor set to add a final newline |
| Data | The downloaded demo contained FHIR `.ndjson` files the reader couldn't use | PhysioNet hosts a CSV demo and a FHIR demo with almost the same name | Downloaded the CSV demo v2.2; the README cites the right resource |
| Events | Only 1 of 5 tests ran | Two test files were saved in the package folder instead of `tests/` | Moved them. Lesson: read the coverage table; it shows which files were actually collected |
| Statistics | The full test suite took about 5 minutes | Bootstrap tests call AUROC tens of thousands of times | Profiled and replaced the AUROC (see the analysis table above) |
| Profiling | The profiler's output file `prof.out` was committed | It wasn't added to `.gitignore` before `git add -A` | Ignored it and amended the commit before pushing. Lesson: read `git status` for unexpected files before committing |
| Containers | `docker compose up` failed: "did not find expected key" | Pasting into the editor changed the YAML indentation inside a multi-line `command: >` block | Rewrote the command as a one-line list and validated with `docker compose config -q` before running. Lesson: validate config files before running them; prefer formats that indentation can't break |
| Containers | During the outage, `/health/ready` took 4.0 s to return 503 | The readiness probe waits for the database connect timeout (3 s) plus the connection pool's pre-ping | Documented as a follow-up: give the readiness check its own short timeout so outages are detected faster |
| CI | Every job warned that `actions/checkout@v4` and `astral-sh/setup-uv@v5` target the deprecated Node.js 20 | GitHub is retiring Node 20 for Actions | Upgraded to `actions/checkout@v5` and `astral-sh/setup-uv@v7` (Node 24). Lesson: treat CI warnings as maintenance work before they become failures |
| Service | `curl localhost:8000` returned `{"detail":"invalid bearer token"}` and `ehr-asof serve` failed with WinError 10048 on ports 8000 and 8001 | Another local application (`claude-science`) was already listening on 8000 and 8001, on both IPv4 and IPv6, so requests reached it instead of the API | Found the owner with `Get-NetTCPConnection` and `Get-Process`; ran the API on a free port (`--port 8765`). Lesson: an unexpected response usually means you are not talking to the process you think; check who owns the port before debugging the code |

## Conclusions

Temporal leakage can make an EHR model look near-perfect on paper while it underperforms an honestly trained model in deployment. `ehr-asof` provides a tested way to build point-in-time-correct features and to report *reported*, *honest* and *deployed* performance side by side. The next step is to estimate the real-world size of the effect on full MIMIC-IV, where the same code runs unchanged.

## Repository layout

| Path | What it does |
| --- | --- |
| `src/ehr_asof/io.py` | Reads MIMIC-IV tables (demo layout or flat CSVs) |
| `src/ehr_asof/events.py` | One event table with `event_time` and `available_time`; delay report |
| `src/ehr_asof/features.py` | Naive and as-of feature matrices; leakage report |
| `src/ehr_asof/stats.py` | AUROC and the paired patient-level bootstrap |
| `src/ehr_asof/experiment.py` | Cohort, models, reported / honest / deployed |
| `src/ehr_asof/synth.py` | MIMIC-shaped synthetic data with planted delays |
| `src/ehr_asof/cli.py` | `synth`, `delays`, `experiment`, `load-db`, `serve` |
| `src/ehr_asof/db.py`, `api.py` | Postgres storage and the FastAPI service |
| `tests/` | Unit, property-based (Hypothesis), API and CLI tests |
| `notes/` | SQL exploration of the MIMIC-IV demo |
| `docs/` | Outage drill write-up |

## Reproduce

```bash
uv sync
uv run ehr-asof synth --out data/synth --patients 2000 --seed 0
uv run ehr-asof delays --data data/synth
uv run ehr-asof experiment --data data/synth --outcome mortality --out results/mortality.json
uv run ehr-asof experiment --data path/to/mimic-iv-demo --outcome mortality
```

Development: `uv run ruff format --check src tests`, `uv run ruff check src tests`, `uv run mypy src` (strict), `uv run pytest`.

## Citations

- Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV Clinical Database Demo (version 2.2). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/dp1f-ex47
- Johnson, A. E. W., Bulgarelli, L., Shen, L., et al. (2023). MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*, 10, 1. https://doi.org/10.1038/s41597-022-01899-x
- Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for biomedical research. *Nature Health*. https://doi.org/10.1038/s44360-026-00096-z

## License

Code: MIT. The MIMIC-IV demo data is distributed by PhysioNet under the Open Data Commons Open Database License v1.0 and is not included in this repository.