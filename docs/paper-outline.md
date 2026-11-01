# Paper outline (target: ML4H Findings, CHIL, or AMIA)

**Working title:** Reported, honest, deployed: measuring temporal leakage in EHR prediction models

1. **Introduction.** EHR events have a clinical time and an availability time; standard pipelines use the first. Leakage is a known reproducibility problem in ML-based science; how large is it in clinical prediction, and how much does it cost at deployment?
2. **Data and availability rules.** MIMIC-IV (full, credentialed) plus the open demo; synthetic data with a planted leak. Per-source rules: labs available at `storetime`; diagnoses at discharge. Clinician review of the rules.
3. **Method.** Two clocks per event; naive vs as-of features; reported / honest / deployed AUROC on the same patients; patient-level split; paired patient-level bootstrap; property tests proving the as-of filter.
4. **Results.** Synthetic (known answer): 0.996 vs 0.884 at 24 h, gap shrinking from 0.15 at 6 h to 0.04 at 48 h. Real data: lab-reporting delays; per-task inflation (mortality, length of stay), across cutoffs and splits. *(Full MIMIC-IV results to come.)*
5. **Discussion.** Leakage is task-specific: identical feature leakage inflated mortality but not length of stay on synthetic data, yet length of stay on real data. Which sources leak most. Deployment cost exceeds the reporting gap.
6. **Limitations.** `storetime` as a proxy; availability rules as assumptions; single center; model family.
7. **Next steps.** Audit widely used MIMIC benchmark pipelines; second dataset (eICU-CRD); vitals and notes.