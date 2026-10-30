from __future__ import annotations

import pandas as pd
import pytest

from ehr_asof.synth import SynthConfig, generate


@pytest.fixture(scope="session")
def tables() -> dict[str, pd.DataFrame]:
    """A small synthetic MIMIC-shaped dataset, generated once per test session."""
    return generate(SynthConfig(n_patients=300, seed=1))
