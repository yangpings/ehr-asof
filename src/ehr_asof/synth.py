"""Synthetic data shaped like MIMIC-IV's hosp module.

The generator exists so the code and tests run anywhere. It is not a model of
real patients: it plants the two timing problems the project studies, so
every number it produces must be labelled *synthetic*.

1. Lab results become available (``storetime``) after the sample is taken
   (``charttime``). Some labs, such as cultures, take hours to days.
2. Diagnosis codes carry no timestamp. They are assigned at discharge, and
   sicker patients, including those who die, receive more severe codes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# (itemid, label, fluid, category, mean, sd, effect of severity, median delay in hours)
LAB_ITEMS: list[tuple[int, str, str, str, float, float, float, float]] = [
    (50912, "Creatinine", "Blood", "Chemistry", 1.0, 0.3, 0.6, 1.5),
    (50971, "Potassium", "Blood", "Chemistry", 4.1, 0.4, 0.3, 1.5),
    (50983, "Sodium", "Blood", "Chemistry", 139.0, 3.0, -2.0, 1.5),
    (51301, "White Blood Cells", "Blood", "Hematology", 8.0, 2.5, 3.0, 1.0),
    (51222, "Hemoglobin", "Blood", "Hematology", 12.5, 1.5, -1.0, 1.0),
    (50813, "Lactate", "Blood", "Blood Gas", 1.4, 0.5, 1.2, 0.5),
    (50852, "% Hemoglobin A1c", "Blood", "Chemistry", 5.7, 0.8, 0.2, 30.0),
    (51003, "Troponin T", "Blood", "Chemistry", 0.02, 0.02, 0.15, 3.0),
]

# ICD-10 codes; the last group is assigned mostly to patients who die.
COMMON_CODES = ["I10", "E119", "E785", "N179", "J189", "I509", "K219", "F329"]
TERMINAL_CODES = ["R6521", "I469", "Z515"]  # septic shock, cardiac arrest, palliative care


@dataclass(frozen=True)
class SynthConfig:
    n_patients: int = 2000
    admissions_per_patient: float = 1.4
    seed: int = 0
    start: str = "2150-01-01"


def generate(config: SynthConfig | None = None) -> dict[str, pd.DataFrame]:
    """Return MIMIC-IV-shaped tables: patients, admissions, labevents, d_labitems,
    diagnoses_icd and prescriptions."""
    cfg = config or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    start = pd.Timestamp(cfg.start)

    subject_ids = np.arange(10_000_000, 10_000_000 + cfg.n_patients)
    patients = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "gender": rng.choice(["F", "M"], size=cfg.n_patients),
            "anchor_age": rng.integers(18, 91, size=cfg.n_patients),
            "anchor_year": 2150,
            "anchor_year_group": "2017 - 2019",
            "dod": pd.NaT,
        }
    )
    frailty = rng.normal(0, 1, size=cfg.n_patients)

    adm_rows = []
    hadm_id = 20_000_000
    for i, sid in enumerate(subject_ids):
        n_adm = 1 + rng.poisson(cfg.admissions_per_patient - 1)
        t = start + pd.Timedelta(days=float(rng.uniform(0, 365)))
        for _ in range(n_adm):
            hadm_id += 1
            severity = 0.7 * frailty[i] + rng.normal(0, 1) + 0.02 * (patients.anchor_age[i] - 60)
            los_hours = float(
                np.clip(rng.lognormal(np.log(96) + 0.25 * severity, 0.6), 30, 60 * 24)
            )
            p_death = 1 / (1 + np.exp(-(-3.6 + 1.4 * severity)))
            died = bool(rng.random() < p_death)
            admit = t + pd.Timedelta(minutes=int(rng.integers(0, 24 * 60)))
            disch = admit + pd.Timedelta(hours=los_hours)
            adm_rows.append(
                {
                    "subject_id": sid,
                    "hadm_id": hadm_id,
                    "admittime": admit,
                    "dischtime": disch,
                    "deathtime": disch if died else pd.NaT,
                    "admission_type": rng.choice(
                        ["EW EMER.", "URGENT", "ELECTIVE"], p=[0.6, 0.25, 0.15]
                    ),
                    "admission_location": "EMERGENCY ROOM",
                    "discharge_location": "DIED"
                    if died
                    else rng.choice(["HOME", "SNF", "HOSPICE"], p=[0.75, 0.2, 0.05]),
                    "hospital_expire_flag": int(died),
                    "latent_severity": severity,
                }
            )
            if died:
                break  # no admissions after death
            t = disch + pd.Timedelta(days=float(rng.exponential(60)))
    admissions = pd.DataFrame(adm_rows)

    lab_rows = []
    labevent_id = 0
    for adm in admissions.itertuples(index=False):
        los = (adm.dischtime - adm.admittime).total_seconds() / 3600
        n_draws = max(1, int(los // 12))
        for k in range(n_draws):
            draw = adm.admittime + pd.Timedelta(hours=float(k * 12 + rng.uniform(0, 4)))
            # severity drifts upward towards death
            drift = 1.5 * (k / max(n_draws - 1, 1)) if adm.hospital_expire_flag else 0.0
            for itemid, _label, _fluid, _cat, mu, sd, eff, delay in LAB_ITEMS:
                if itemid == 50852 and k > 0:
                    continue  # HbA1c once per admission
                if rng.random() < 0.15:
                    continue
                labevent_id += 1
                value = mu + 0.5 * eff * (adm.latent_severity + drift) + rng.normal(0, sd)
                value = max(value, 0.01)
                store = draw + pd.Timedelta(hours=float(rng.lognormal(np.log(delay), 0.8)))
                lab_rows.append(
                    {
                        "labevent_id": labevent_id,
                        "subject_id": adm.subject_id,
                        "hadm_id": adm.hadm_id,
                        "specimen_id": labevent_id,
                        "itemid": itemid,
                        "charttime": draw,
                        "storetime": store if rng.random() > 0.02 else pd.NaT,
                        "value": f"{value:.2f}",
                        "valuenum": round(value, 2),
                        "valueuom": "",
                        "flag": None,
                    }
                )
    labevents = pd.DataFrame(lab_rows)

    d_labitems = pd.DataFrame(
        [(i, lab, fl, cat) for i, lab, fl, cat, *_ in LAB_ITEMS],
        columns=["itemid", "label", "fluid", "category"],
    )

    dx_rows = []
    rx_rows = []
    for adm in admissions.itertuples(index=False):
        codes: list[str] = []
        for code in COMMON_CODES:
            p = 1 / (1 + np.exp(-(-1.8 + 0.5 * adm.latent_severity)))
            if rng.random() < p:
                codes.append(code)
        if adm.hospital_expire_flag:
            codes += [c for c in TERMINAL_CODES if rng.random() < 0.7]
        elif rng.random() < 0.03:
            codes.append(rng.choice(TERMINAL_CODES))
        for seq, code in enumerate(codes, start=1):
            dx_rows.append(
                {
                    "subject_id": adm.subject_id,
                    "hadm_id": adm.hadm_id,
                    "seq_num": seq,
                    "icd_code": code,
                    "icd_version": 10,
                }
            )
        if "E119" in codes and rng.random() < 0.8:
            rx_rows.append(
                {
                    "subject_id": adm.subject_id,
                    "hadm_id": adm.hadm_id,
                    "starttime": adm.admittime + pd.Timedelta(hours=6),
                    "stoptime": adm.dischtime,
                    "drug": rng.choice(["Metformin", "Insulin", "Glipizide"]),
                }
            )
    diagnoses_icd = pd.DataFrame(dx_rows)
    prescriptions = pd.DataFrame(
        rx_rows, columns=["subject_id", "hadm_id", "starttime", "stoptime", "drug"]
    )

    return {
        "patients": patients,
        "admissions": admissions.drop(columns="latent_severity"),
        "labevents": labevents,
        "d_labitems": d_labitems,
        "diagnoses_icd": diagnoses_icd,
        "prescriptions": prescriptions,
    }
