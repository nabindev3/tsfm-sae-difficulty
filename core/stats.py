"""Modality-agnostic statistical primitives.

Everything here operates on plain numpy arrays so it can be unit-tested on
synthetic data with no torch / no model in sight (Phase 1 "done when" criterion).

The two legacy repos each re-implemented:
  - a paired bootstrap over test indices for AUROC deltas, and
  - a one-sided label-permutation test for Delta(SAE - Raw),
with subtly different bookkeeping. This is the single shared implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------- #
# Result container — identical shape regardless of modality.
# --------------------------------------------------------------------------- #
@dataclass
class ProbeResult:
    """One probe's headline numbers. Modality-independent by construction.

    `delta_vs_baseline` / `delta_ci` / `perm_p` are populated only for probes
    that are compared against a reference rung (e.g. SAE-over-Raw); they stay
    None for the standalone rungs.
    """
    name: str
    auroc: float
    ci_low: float
    ci_high: float
    n_test: int
    best_C: float | None = None
    delta_vs_baseline: float | None = None
    delta_ci: tuple[float, float] | None = None
    perm_p: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LadderResult:
    """The full three-rung (+ diagnostic) ladder for one (modality, experiment)."""
    n_total: int
    n_train: int
    n_test: int
    hard_fraction: float
    probes: dict[str, ProbeResult] = field(default_factory=dict)
    # Headline deltas, keyed "A-B".
    deltas: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "hard_fraction": self.hard_fraction,
            "probes": {k: v.to_dict() for k, v in self.probes.items()},
            "deltas": self.deltas,
        }


# --------------------------------------------------------------------------- #
# Bootstrap utilities.
# --------------------------------------------------------------------------- #
def percentile_ci(samples: Sequence[float], alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided percentile CI. Returns (nan, nan) on empty input."""
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(arr, 100 * alpha / 2))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return lo, hi


def paired_bootstrap_auroc(
    y_test: np.ndarray,
    preds: dict[str, np.ndarray],
    pairs: Sequence[tuple[str, str]],
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    """Paired bootstrap over test indices.

    Resamples test indices ONCE per iteration and reuses them for every probe
    and every requested delta — the only correct way to get a CI on a paired
    AUROC difference (it neutralizes the "you just added dimensions" critique).

    Returns
    -------
    auroc_ci : {probe_name: (lo, hi)}
    delta_ci : {"a-b": (lo, hi)} for each (a, b) in `pairs`
    """
    y_test = np.asarray(y_test)
    names = list(preds.keys())
    rng = np.random.default_rng(seed)
    boot = {n: [] for n in names}
    boot_delta = {f"{a}-{b}": [] for a, b in pairs}

    if len(np.unique(y_test)) > 1:
        idx_all = np.arange(len(y_test))
        for _ in range(n_boot):
            idx = rng.choice(idx_all, size=len(idx_all), replace=True)
            if len(np.unique(y_test[idx])) < 2:
                continue
            per = {n: roc_auc_score(y_test[idx], preds[n][idx]) for n in names}
            for n in names:
                boot[n].append(per[n])
            for a, b in pairs:
                boot_delta[f"{a}-{b}"].append(per[a] - per[b])

    auroc_ci = {n: percentile_ci(boot[n]) for n in names}
    delta_ci = {k: percentile_ci(v) for k, v in boot_delta.items()}
    return auroc_ci, delta_ci


def paired_bootstrap_mean_delta(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """Bootstrap CI of mean(a - b) over a paired sample (e.g. causal ΔCRPS / Δnats).

    Used by the causal-ablation aggregation in both modalities.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, "paired arrays must have the same shape"
    n = len(a)
    rng = np.random.default_rng(seed)
    point = float((a - b).mean())
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boots.append((a[idx] - b[idx]).mean())
    return point, percentile_ci(boots)


# --------------------------------------------------------------------------- #
# Label-permutation test for an AUROC delta.
# --------------------------------------------------------------------------- #
def label_permutation_test(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_perm: int = 10000,
    seed: int = 42,
) -> dict:
    """One- and two-sided permutation test for AUROC(a) - AUROC(b).

    Null: both scores are independent of the true label. Shuffles `y`, recomputes
    the delta `n_perm` times. Direction of the one-sided test follows the sign of
    the observed delta.
    """
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("Need both classes present to compute AUROC.")
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)

    auc_a = float(roc_auc_score(y, score_a))
    auc_b = float(roc_auc_score(y, score_b))
    obs_delta = auc_a - auc_b

    rng = np.random.default_rng(seed)
    perm = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        try:
            perm[i] = roc_auc_score(y_perm, score_a) - roc_auc_score(y_perm, score_b)
        except ValueError:
            perm[i] = 0.0

    if obs_delta < 0:
        p_one = float((perm <= obs_delta).mean())
    else:
        p_one = float((perm >= obs_delta).mean())
    p_two = float((np.abs(perm) >= abs(obs_delta)).mean())

    return {
        "auroc_a": auc_a,
        "auroc_b": auc_b,
        "observed_delta": float(obs_delta),
        "n_perm": n_perm,
        "p_one_sided": p_one,
        "p_two_sided": p_two,
        "null_mean": float(perm.mean()),
        "null_std": float(perm.std(ddof=1)),
        "null_2p5_pct": float(np.percentile(perm, 2.5)),
        "null_97p5_pct": float(np.percentile(perm, 97.5)),
    }
