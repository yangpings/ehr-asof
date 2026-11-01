from __future__ import annotations

import pandas as pd
import pytest

from ehr_asof.experiment import TaskConfig, make_cohort, run


def test_cohort_excludes_discharged_or_dead_before_cutoff(tables: dict[str, pd.DataFrame]) -> None:
    cfg = TaskConfig(horizon_hours=24)
    cohort = make_cohort(tables, cfg)
    adm = tables["admissions"].set_index("hadm_id").loc[cohort["hadm_id"]]
    assert (adm["dischtime"].to_numpy() > cohort["cutoff"].to_numpy()).all()
    assert cohort["label"].isin([0, 1]).all()


@pytest.mark.slow
@pytest.mark.parametrize("outcome", ["mortality", "long_los"])
def test_run_end_to_end(tables: dict[str, pd.DataFrame], outcome: str) -> None:
    result = run(tables, TaskConfig(outcome=outcome, n_boot=100))  # type: ignore[arg-type]
    for key in ("reported", "honest", "deployed", "reported - honest"):
        iv = result["auroc"][key]
        assert iv["low"] <= iv["estimate"] <= iv["high"] or iv["n_valid"] == 0
    assert 0 <= result["leakage_test"]["cells_leaked"] <= 1
    assert result["leakage_test"]["flag_cells_leaked"] > 0  # diagnoses always leak in naive mode
