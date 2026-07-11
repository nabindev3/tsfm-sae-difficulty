"""End-to-end probe-ladder coverage on synthetic arrays — no model, no network.

The headline pipeline (`probing/probe.py`) downloads ETTh1 and loads a Chronos
checkpoint, so it can't run in CI. This exercises the shared ladder it now
delegates to (`core/probe.py`, ported from fm-difficulty-probe) on numpy arrays:
the same fit + paired-bootstrap code path, just fed synthetic features.

These tests assert contracts, not research conclusions — that the ladder builds
the right five rungs, returns well-formed AUROCs/CIs, and that the paired
bootstrap and CI helpers behave. Run with:  pytest tests/ -q
(after `pip install -e .`, so the fm-difficulty-probe `core` package resolves).
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from core import probe as P
from core import stats as ST


def _synthetic(n=400, d_cheap=8, d_act=24, seed=0):
    """Cheap features carry signal; raw acts carry a bit more; SAE ~ redundant
    with raw — mirrors the project's expected qualitative shape."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)                         # latent difficulty
    y = (z + 0.3 * rng.normal(size=n) > 0).astype(int)
    cheap = np.column_stack([z + rng.normal(size=n) for _ in range(d_cheap)])
    raw = np.column_stack([z + rng.normal(size=n) for _ in range(d_act)])
    sae = raw + 0.01 * rng.normal(size=raw.shape)  # ~redundant with raw
    return cheap, raw, sae, y


def test_build_ladder_shapes():
    cheap, raw, sae, _ = _synthetic()
    feats = P.build_ladder(cheap, raw, sae)
    assert set(feats) == {P.P1, P.P2, P.P3, P.P4, P.P5}
    assert feats[P.P1].shape[1] == cheap.shape[1]
    assert feats[P.P2].shape[1] == cheap.shape[1] + raw.shape[1]
    assert feats[P.P3].shape[1] == cheap.shape[1] + sae.shape[1]
    assert feats[P.P4].shape[1] == raw.shape[1]
    assert feats[P.P5].shape[1] == sae.shape[1]


def test_probe_ladder_runs_and_shapes():
    cheap, raw, sae, y = _synthetic()
    feats = P.build_ladder(cheap, raw, sae)
    n = len(y)
    train_mask = np.zeros(n, bool); train_mask[: n // 2] = True
    test_mask = ~train_mask
    folds = list(StratifiedKFold(3, shuffle=True, random_state=0)
                 .split(np.zeros((train_mask.sum(), 1)), y[train_mask]))

    result, preds = P.run_probe_ladder(feats, y, train_mask, test_mask, folds, n_boot=200)
    assert result.n_test == int(test_mask.sum())
    assert result.n_train == int(train_mask.sum())
    for pr in result.probes.values():
        assert 0.0 <= pr.auroc <= 1.0
        assert pr.ci_low <= pr.auroc <= pr.ci_high + 1e-6
        assert pr.best_C in P.DEFAULT_C_GRID
    # All three headline deltas present.
    for a, b in P.HEADLINE_PAIRS:
        assert f"{a}-{b}" in result.deltas
    assert set(preds) == set(feats)
    # Informative features: cheap baseline should beat chance on this synthetic data.
    assert result.probes[P.P1].auroc > 0.6


def test_probe_ladder_accepts_timeseries_folds():
    """The TSFM modality hands in TimeSeriesSplit folds; the ladder is agnostic."""
    cheap, raw, sae, y = _synthetic(n=300, seed=2)
    feats = P.build_ladder(cheap, raw, sae)
    n = len(y)
    train_mask = np.zeros(n, bool); train_mask[: 2 * n // 3] = True
    test_mask = ~train_mask
    folds = list(TimeSeriesSplit(n_splits=3).split(np.zeros((train_mask.sum(), 1)),
                                                    y[train_mask]))
    result, preds = P.run_probe_ladder(feats, y, train_mask, test_mask, folds, n_boot=100)
    assert set(preds) == set(feats)
    assert all(np.all(np.isfinite(p)) for p in preds.values())


def test_run_probe_ladder_rejects_empty_split():
    cheap, raw, sae, y = _synthetic(n=120)
    feats = P.build_ladder(cheap, raw, sae)
    n = len(y)
    train_mask = np.ones(n, bool)          # no test rows
    test_mask = np.zeros(n, bool)
    folds = list(StratifiedKFold(2).split(np.zeros((n, 1)), y))
    try:
        P.run_probe_ladder(feats, y, train_mask, test_mask, folds, n_boot=50)
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty test split")


def test_paired_bootstrap_ci_ordering():
    rng = np.random.default_rng(3)
    n = 200
    y = (rng.random(n) < 0.5).astype(int)
    good = y + rng.normal(scale=0.5, size=n)   # informative
    rand = rng.normal(size=n)                   # noise
    preds = {"good": good, "rand": rand}
    auroc_ci, delta_ci = ST.paired_bootstrap_auroc(
        y, preds, [("good", "rand")], n_boot=500
    )
    for name in preds:
        lo, hi = auroc_ci[name]
        assert lo <= hi
    d_lo, d_hi = delta_ci["good-rand"]
    assert d_lo <= d_hi


def test_percentile_ci_empty_is_nan():
    lo, hi = ST.percentile_ci([])
    assert np.isnan(lo) and np.isnan(hi)


def _run_standalone():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} core ladder tests passed.")


if __name__ == "__main__":
    _run_standalone()
