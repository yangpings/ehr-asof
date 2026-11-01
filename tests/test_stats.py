from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from ehr_asof.stats import auroc, paired_cluster_bootstrap


def test_auroc_perfect_and_constant() -> None:
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    with pytest.raises(ValueError):
        auroc([1, 1], [0.2, 0.3])


def test_identical_models_have_zero_difference() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.random(200)
    out = paired_cluster_bootstrap(
        y, {"a": s, "b": s}, clusters=np.arange(200), contrasts=[("a", "b")], n_boot=200
    )
    d = out["a - b"]
    assert d.estimate == 0.0 and d.low == 0.0 and d.high == 0.0


@pytest.mark.slow
def test_bootstrap_coverage_binormal() -> None:
    """95% intervals should cover the true AUROC about 95% of the time."""
    d = 1.0
    truth = norm.cdf(d / np.sqrt(2))
    rng = np.random.default_rng(1)
    hits = 0
    reps = 120
    for r in range(reps):
        y = np.r_[np.zeros(100), np.ones(100)].astype(int)
        s = np.r_[rng.normal(0, 1, 100), rng.normal(d, 1, 100)]
        iv = paired_cluster_bootstrap(
            y, {"m": s}, clusters=np.arange(200), contrasts=[], n_boot=300, seed=r
        )["m"]
        hits += iv.low <= truth <= iv.high
    assert 0.85 <= hits / reps <= 0.99


from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


@settings(max_examples=300, deadline=None)
@given(st.lists(st.tuples(st.booleans(), st.integers(0, 5)), min_size=2, max_size=60))
def test_auroc_matches_sklearn_including_ties(pairs: list[tuple[bool, int]]) -> None:
    y = np.array([p[0] for p in pairs], dtype=int)
    s = np.array([p[1] for p in pairs], dtype=float)  # small integers: many ties
    if y.min() == y.max():
        return
    assert auroc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
