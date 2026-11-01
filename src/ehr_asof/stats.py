"""Evaluation statistics: AUROC and a paired, patient-clustered bootstrap."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _average_ranks(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """1-based ranks, with tied values given the average of their ranks."""
    sorter = np.argsort(x, kind="mergesort")
    inverse = np.empty_like(sorter)
    inverse[sorter] = np.arange(x.size)
    xs = x[sorter]
    new_value = np.r_[True, xs[1:] != xs[:-1]]
    dense = np.cumsum(new_value)[inverse]
    bounds = np.r_[np.flatnonzero(new_value), x.size]
    ranks: NDArray[np.float64] = 0.5 * (bounds[dense] + bounds[dense - 1] + 1)
    return ranks


def auroc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the ROC curve via the Mann-Whitney rank identity.

    Equivalent to ``sklearn.metrics.roc_auc_score`` (the tests check this) but
    without its per-call input validation, which dominated bootstrap run time.
    Raises ``ValueError`` when only one class is present.
    """
    y = np.asarray(y_true).astype(bool)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both classes")
    ranks = _average_ranks(np.asarray(y_score, dtype=float))
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    n_valid: int

    def as_dict(self) -> dict[str, float]:
        return {
            "estimate": self.estimate,
            "low": self.low,
            "high": self.high,
            "n_valid": self.n_valid,
        }


Metric = Callable[[NDArray[np.int_], NDArray[np.float64]], float]


def paired_cluster_bootstrap(
    y_true: ArrayLike,
    scores: Mapping[str, ArrayLike],
    clusters: ArrayLike,
    contrasts: list[tuple[str, str]],
    metric: Metric = auroc,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, Interval]:
    """Percentile intervals for each score's metric and for each contrast ``a - b``.

    Resamples *clusters* (patients), keeping all of a chosen patient's rows, and
    scores every model on the same resample, so case-mix variation cancels in the
    differences. Resamples with a single class are skipped and counted.
    """
    y = np.asarray(y_true).astype(int)
    s = {k: np.asarray(v, dtype=float) for k, v in scores.items()}
    groups = np.asarray(clusters)
    uniq, inverse = np.unique(groups, return_inverse=True)
    rows_by_cluster = [np.flatnonzero(inverse == g) for g in range(uniq.size)]
    rng = np.random.default_rng(seed)

    names = list(s) + [f"{a} - {b}" for a, b in contrasts]
    draws: dict[str, list[float]] = {n: [] for n in names}
    for _ in range(n_boot):
        picked = rng.integers(0, uniq.size, size=uniq.size)
        idx = np.concatenate([rows_by_cluster[g] for g in picked])
        if np.unique(y[idx]).size < 2:
            continue
        vals = {k: metric(y[idx], v[idx]) for k, v in s.items()}
        for k, v in vals.items():
            draws[k].append(v)
        for a, b in contrasts:
            draws[f"{a} - {b}"].append(vals[a] - vals[b])

    point = {k: metric(y, v) for k, v in s.items()}
    for a, b in contrasts:
        point[f"{a} - {b}"] = point[a] - point[b]
    alpha = (1 - level) / 2
    out: dict[str, Interval] = {}
    for n in names:
        d = np.asarray(draws[n])
        lo, hi = np.quantile(d, [alpha, 1 - alpha]) if d.size else (np.nan, np.nan)
        out[n] = Interval(point[n], float(lo), float(hi), int(d.size))
    return out
