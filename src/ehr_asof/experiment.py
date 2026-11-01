"""The experiment: how much does EHR leakage inflate reported performance?

Three numbers per task, all on the same held-out patients:

- ``reported``: trained and tested on naive features (what many papers report).
- ``honest``:   trained and tested on as-of features (what is achievable).
- ``deployed``: trained on naive features, run on as-of features (what a hospital
  actually gets if it ships the naive model).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ehr_asof.events import build_events
from ehr_asof.features import build_features, leakage_report, vocabulary
from ehr_asof.stats import paired_cluster_bootstrap

Outcome = Literal["mortality", "long_los"]


@dataclass(frozen=True)
class TaskConfig:
    outcome: Outcome = "mortality"
    horizon_hours: float = 24.0
    long_los_days: float = 7.0
    test_size: float = 0.3
    seed: int = 0
    n_boot: int = 1000


def make_cohort(tables: dict[str, pd.DataFrame], cfg: TaskConfig) -> pd.DataFrame:
    """Admissions still in hospital and alive at ``admittime + horizon``.

    Returns ``hadm_id, subject_id, cutoff, label`` plus static features that are
    known at admission (age, sex, emergency admission).
    """
    adm = tables["admissions"].copy()
    adm["cutoff"] = adm["admittime"] + pd.Timedelta(hours=cfg.horizon_hours)
    alive = adm["deathtime"].isna() | (adm["deathtime"] > adm["cutoff"])
    adm = adm[(adm["dischtime"] > adm["cutoff"]) & alive]
    if cfg.outcome == "mortality":
        label = adm["hospital_expire_flag"].astype(int)
    else:
        los_days = (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400
        label = (los_days > cfg.long_los_days).astype(int)
    pts = tables["patients"][["subject_id", "gender", "anchor_age", "anchor_year"]]
    adm = adm.assign(label=label).merge(pts, on="subject_id", how="left")
    adm["static:age"] = adm["anchor_age"] + (adm["admittime"].dt.year - adm["anchor_year"])
    adm["static:male"] = (adm["gender"] == "M").astype(float)
    adm["static:emergency"] = adm["admission_type"].astype(str).str.contains("EMER").astype(float)
    keep = [
        "hadm_id",
        "subject_id",
        "cutoff",
        "label",
        "static:age",
        "static:male",
        "static:emergency",
    ]
    return adm[keep].reset_index(drop=True)


def _model() -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )


def run(tables: dict[str, pd.DataFrame], cfg: TaskConfig | None = None) -> dict[str, Any]:
    """Run one task end to end and return a JSON-serializable result."""
    cfg = cfg or TaskConfig()
    cohort = make_cohort(tables, cfg)
    events = build_events(tables)
    split = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
    train_idx, test_idx = next(split.split(cohort, groups=cohort["subject_id"]))
    train, test = cohort.iloc[train_idx], cohort.iloc[test_idx]

    codes = vocabulary(events[events["hadm_id"].isin(train["hadm_id"])], min_count=5)
    static_cols = [c for c in cohort.columns if c.startswith("static:")]

    def matrix(part: pd.DataFrame, mode: Literal["naive", "asof"]) -> pd.DataFrame:
        feats = build_features(events, part[["hadm_id", "cutoff"]], mode, codes)
        return pd.concat([part.set_index("hadm_id")[static_cols], feats], axis=1)

    x = {
        (p, m): matrix(df, m)
        for p, df in (("train", train), ("test", test))
        for m in ("naive", "asof")
    }
    y_train, y_test = train["label"].to_numpy(), test["label"].to_numpy()
    naive_model = _model().fit(x["train", "naive"], y_train)
    asof_model = _model().fit(x["train", "asof"], y_train)
    scores = {
        "reported": naive_model.predict_proba(x["test", "naive"])[:, 1],
        "honest": asof_model.predict_proba(x["test", "asof"])[:, 1],
        "deployed": naive_model.predict_proba(x["test", "asof"])[:, 1],
    }
    intervals = paired_cluster_bootstrap(
        y_test,
        scores,
        clusters=test["subject_id"].to_numpy(),
        contrasts=[("reported", "honest"), ("reported", "deployed")],
        n_boot=cfg.n_boot,
        seed=cfg.seed,
    )
    return {
        "outcome": cfg.outcome,
        "horizon_hours": cfg.horizon_hours,
        "n_train": len(train),
        "n_test": len(test),
        "prevalence_test": float(np.mean(y_test)),
        "n_features": int(x["test", "naive"].shape[1]),
        "leakage_test": leakage_report(
            x["test", "naive"].drop(columns=static_cols),
            x["test", "asof"].drop(columns=static_cols),
        ),
        "auroc": {k: v.as_dict() for k, v in intervals.items()},
    }
