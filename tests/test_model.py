from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest

from ehr_asof.events import build_events
from ehr_asof.model import RiskModel, load, save, train_risk_model


@pytest.fixture(scope="session")
def risk_model(tables: dict[str, pd.DataFrame]) -> RiskModel:
    return train_risk_model(tables)


def test_scores_are_probabilities(risk_model: RiskModel, tables: dict[str, pd.DataFrame]) -> None:
    adm = tables["admissions"].head(5)
    cutoffs = pd.DataFrame(
        {"hadm_id": adm["hadm_id"], "cutoff": adm["admittime"] + pd.Timedelta(hours=24)}
    )
    scores = risk_model.score(build_events(tables), cutoffs)
    assert list(scores.columns) == ["reported", "honest", "deployed"]
    assert len(scores) == 5
    assert ((scores >= 0) & (scores <= 1)).all().all()


def test_save_load_roundtrip(risk_model: RiskModel, tmp_path: Path) -> None:
    path = save(risk_model, tmp_path / "models" / "risk.pkl")
    again = load(path)
    assert again.codes == risk_model.codes
    assert again.info() == risk_model.info()


def test_load_rejects_other_objects(tmp_path: Path) -> None:
    path = tmp_path / "not_a_model.pkl"
    path.write_bytes(pickle.dumps({"hello": 1}))
    with pytest.raises(TypeError):
        load(path)
