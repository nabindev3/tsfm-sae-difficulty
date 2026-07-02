"""Offline smoke tests for the modules added in the §4.8-4.15 experiment suite.

Same contract as test_probe_smoke.py: pure numpy/scipy on synthetic data, no
model and no network, sub-second. They don't assert any research conclusion;
they pin the behaviour the new analysis scripts depend on so a refactor can't
silently break them:

  * extended_baselines (§4.10/4.14): Hurst separates white noise from a random
    walk; STL trend/seasonal strength fire on the matching structure.
  * conformal_baseline (§4.9): the split-conformal correction is monotone in
    alpha, and the risk-coverage curve ranks a perfect signal at the oracle and
    a useless one no better than the unconditional mean.
  * power_analysis (§4.8): required-n shrinks with effect size and is infinite
    for a non-positive effect.

Run with:  pytest tests/ -q  (after `pip install -e .`). The eval/ analysis
helpers resolve via the pytest `pythonpath` set in pyproject.toml, so run
through pytest rather than executing this file directly.
"""
import numpy as np

from probing.extended_baselines import hurst_exponent, stl_strength
from eval.conformal_baseline import risk_coverage_curve, conformal_quantile
from eval.power_analysis import required_n


def test_hurst_separates_noise_from_random_walk():
    rng = np.random.default_rng(0)
    white = rng.standard_normal(512)
    rw = np.cumsum(rng.standard_normal(512))
    assert hurst_exponent(white) < 0.65   # ~0.5 for an uncorrelated series
    assert hurst_exponent(rw) > 0.70       # persistent / trending


def test_stl_strength_fires_on_matching_structure():
    rng = np.random.default_rng(0)
    t = np.arange(512)
    trended = 0.05 * t + 0.3 * rng.standard_normal(512)
    seasonal = 3 * np.sin(2 * np.pi * t / 24) + 0.3 * rng.standard_normal(512)
    white = rng.standard_normal(512)
    # trend strength no lower on a trend than on white; seasonal on a seasonal.
    assert stl_strength(trended, 24)[0] >= stl_strength(white, 24)[0]
    assert stl_strength(seasonal, 24)[1] >= stl_strength(white, 24)[1]


def test_conformal_quantile_monotone_in_alpha():
    scores = np.linspace(0.0, 1.0, 101)
    q10 = conformal_quantile(scores, 0.10)  # 90% nominal coverage
    q20 = conformal_quantile(scores, 0.20)  # 80% nominal coverage
    assert q10 is not None and q20 is not None
    assert q10 >= q20                        # tighter miscoverage -> larger correction
    assert conformal_quantile(np.array([]), 0.1) is None  # empty calibration set


def test_risk_coverage_oracle_beats_useless_signal():
    rng = np.random.default_rng(0)
    n = 120
    crps = rng.random(n)
    coverages = np.round(np.arange(0.1, 1.001, 0.1), 4)
    # A signal equal to the realised risk ranks easiest-first -> minimal AURC.
    _, _, _, aurc_perfect = risk_coverage_curve(crps, crps, coverages, rng, 50)
    # A constant signal retains an index-ordered (~random) subset -> ~mean risk.
    _, _, _, aurc_useless = risk_coverage_curve(np.zeros(n), crps, coverages, rng, 50)
    assert aurc_perfect < aurc_useless
    # At full coverage every window is retained, so the curve ends at the mean.
    curve, _, _, _ = risk_coverage_curve(crps, crps, coverages, rng, 50)
    assert abs(curve[-1] - crps.mean()) < 1e-9


def test_required_n_shrinks_with_effect_and_infinite_for_null():
    sd = 0.2
    n_big_effect = required_n(0.10, sd, 0.05, 0.80, one_sided=True)
    n_small_effect = required_n(0.05, sd, 0.05, 0.80, one_sided=True)
    assert n_small_effect > n_big_effect > 0           # smaller effect needs more n
    assert required_n(0.0, sd, 0.05, 0.80) == float("inf")   # null is unprovable
    assert required_n(-0.1, sd, 0.05, 0.80) == float("inf")  # wrong-sign too


def _run_standalone():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} smoke tests passed.")


if __name__ == "__main__":
    _run_standalone()
