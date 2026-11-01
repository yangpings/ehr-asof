from __future__ import annotations

import pandas as pd

from ehr_asof.synth import SynthConfig, generate


def test_no_admission_starts_after_death(tables: dict[str, pd.DataFrame]) -> None:
    adm = tables["admissions"]
    death = adm.dropna(subset=["deathtime"]).groupby("subject_id")["deathtime"].min()
    later = adm.merge(death.rename("died_at"), on="subject_id")
    assert not (later["admittime"] > later["died_at"]).any()


def test_generator_is_deterministic() -> None:
    a = generate(SynthConfig(n_patients=30, seed=5))
    b = generate(SynthConfig(n_patients=30, seed=5))
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])
