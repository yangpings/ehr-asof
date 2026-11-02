"""A risk model the service can serve: in-hospital mortality from events only.

Two logistic regressions are trained on the same admissions and cutoffs: one on
naive features (what a leaky pipeline learns) and one on as-of features (what is
honestly learnable). Serving both lets one request show the gap between the score
a paper would report and the score a deployed system would produce.

Train it on synthetic data only (``ehr-asof train``). A model fitted to credentialed
data is derived data and must never be committed or shipped in an image.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from sklearn.pipeline import Pipeline

from ehr_asof import __version__
from ehr_asof.events import build_events
from ehr_asof.experiment import TaskConfig, _model, make_cohort
from ehr_asof.features import build_features, vocabulary

Mode = Literal["naive", "asof"]


@dataclass
class RiskModel:
    codes: list[str]
    horizon_hours: float
    naive: Pipeline
    asof: Pipeline
    n_train: int
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    package_version: str = __version__

    def features(self, events: pd.DataFrame, cutoffs: pd.DataFrame, mode: Mode) -> pd.DataFrame:
        """The model's feature matrix: always the training vocabulary, in training order."""
        return build_features(events, cutoffs, mode, self.codes)

    def score(self, events: pd.DataFrame, cutoffs: pd.DataFrame) -> pd.DataFrame:
        """Reported, honest and deployed risk for each admission in ``cutoffs``.

        - ``reported``: naive model on naive features (what a leaky evaluation shows)
        - ``honest``:   as-of model on as-of features (what is achievable)
        - ``deployed``: naive model on as-of features (what shipping the leaky model gives)
        """
        x_naive = self.features(events, cutoffs, "naive")
        x_asof = self.features(events, cutoffs, "asof")
        return pd.DataFrame(
            {
                "reported": self.naive.predict_proba(x_naive)[:, 1],
                "honest": self.asof.predict_proba(x_asof)[:, 1],
                "deployed": self.naive.predict_proba(x_asof)[:, 1],
            },
            index=x_naive.index,
        )

    def info(self) -> dict[str, Any]:
        return {
            "outcome": "in-hospital mortality",
            "horizon_hours": self.horizon_hours,
            "n_features": len(self.codes) + 1,
            "n_train": self.n_train,
            "trained_at": self.trained_at,
            "package_version": self.package_version,
        }


def train_risk_model(
    tables: dict[str, pd.DataFrame], horizon_hours: float = 24.0, min_count: int = 5
) -> RiskModel:
    """Fit both models on every eligible admission at ``admittime + horizon``."""
    cohort = make_cohort(tables, TaskConfig(outcome="mortality", horizon_hours=horizon_hours))
    events = build_events(tables)
    codes = vocabulary(events[events["hadm_id"].isin(cohort["hadm_id"])], min_count=min_count)
    cutoffs = cohort[["hadm_id", "cutoff"]]
    y = cohort["label"].to_numpy()
    naive = _model().fit(build_features(events, cutoffs, "naive", codes), y)
    asof = _model().fit(build_features(events, cutoffs, "asof", codes), y)
    return RiskModel(
        codes=codes, horizon_hours=horizon_hours, naive=naive, asof=asof, n_train=len(cohort)
    )


def save(model: RiskModel, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load(path: str | Path) -> RiskModel:
    """Load a model file. Pickle runs code on load: only open files you created."""
    with Path(path).open("rb") as f:
        model = pickle.load(f)
    if not isinstance(model, RiskModel):
        raise TypeError(f"{path} does not contain a RiskModel")
    return model
