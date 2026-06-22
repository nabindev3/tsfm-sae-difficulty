"""Pre-registered power analysis for the causal-ablation experiment (Exp5).

The §4.6 ablation is underpowered: the aggregate effect on hard windows is
+0.043 CRPS with 95% CI [-0.008, +0.095] at n=56 hard windows -- the bootstrap
cannot resolve it. Before "powering up" (lowering the hard-window threshold or
pooling hard windows across datasets) we PRE-REGISTER the analysis here, using
ONLY the already-collected causal_ablation.parquet to estimate the per-window
effect and noise. No new forecasts are run; this fixes the design before any
confirmatory data is added.

=========================  PRE-REGISTRATION  =========================
* Unit:           a test window i.
* Per-window stat: d_i = mean_f [ crps_ablate(f,i) - crps_sae_recon(i) ]
                   averaged over the top-5 difficulty-predictive features f
                   (the aggregate ablation effect; the SAE-recon condition is
                   the within-window control, so reconstruction loss cancels).
* Cohort:         "hard" = windows in the top tercile of crps_natural.
* Primary endpoint: mean_{i in hard} d_i.
* Hypotheses:     H0: mean d_i = 0   vs   H1: mean d_i > 0   (ONE-SIDED; the
                  direction is pre-specified from the 5/5 consistent positive
                  signs already observed -- declared here as the confirmatory
                  direction, not re-chosen after seeing new data).
* Test:           paired bootstrap (B=2000) percentile CI on mean_{hard} d_i,
                  reject H0 if the one-sided 95% lower bound > 0. Planning
                  curves additionally use the z-approximation
                  n = ((z_{1-a}+z_{1-b}) * sd / effect)^2.
* alpha = 0.05, target power = 0.80 (also report 0.90).
* Secondary endpoint: diff-in-diff mean_hard d_i - mean_easy d_i (same test).
* Decision rule:  collect hard windows (lower threshold and/or pool datasets)
                  until the projected n reaches 80% power for the effect size
                  estimated below, then run the confirmatory ablation ONCE.
* Effect/sd source: estimated from the pilot causal_ablation.parquet (this file)
                  -- reported transparently; the confirmatory run will re-estimate.
======================================================================
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def required_n(effect, sd, alpha, power, one_sided=True):
    za = stats.norm.ppf(1 - alpha) if one_sided else stats.norm.ppf(1 - alpha / 2)
    zb = stats.norm.ppf(power)
    if effect <= 0:
        return float("inf")
    return ((za + zb) * sd / effect) ** 2


def bootstrap_power(d_hard, n_grid, effect_assumed, sd, alpha, n_sim, rng, one_sided=True):
    """Simulation power: draw n_sim datasets of size n ~ N(effect_assumed, sd),
    reject if the (one-sided) bootstrap-equivalent z lower bound > 0. Uses the
    z-test as the per-sim decision (fast, matches the planning approximation)."""
    za = stats.norm.ppf(1 - alpha) if one_sided else stats.norm.ppf(1 - alpha / 2)
    powers = {}
    for n in n_grid:
        # parametric draws calibrated to the pilot effect + noise
        means = rng.normal(effect_assumed, sd / np.sqrt(n), size=n_sim)
        ses = sd / np.sqrt(n)
        z = means / ses
        powers[int(n)] = float((z > za).mean()) if one_sided else float((np.abs(z) > za).mean())
    return powers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", default="eval/results/causal_ablation.parquet")
    ap.add_argument("--out_dir", default="eval/results")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    df = pd.read_parquet(args.ablation)
    abl_cols = [c for c in df.columns if c.startswith("crps_ablate_")]
    feats = [c.replace("crps_ablate_", "") for c in abl_cols]

    natural = df["crps_natural"].values
    recon = df["crps_sae_recon"].values
    # aggregate per-window ablation effect (mean over features), recon-controlled
    d_agg = np.mean([df[c].values - recon for c in abl_cols], axis=0)

    # cohorts by tercile / alternative thresholds of natural CRPS
    out = {"n_total": int(len(df)), "features": feats, "alpha": args.alpha,
           "preregistration": "see module docstring", "thresholds": {}}
    print(f"n_total={len(df)}  features={feats}")
    print(f"{'top-frac':>9} {'n_hard':>6} {'effect':>8} {'sd':>7} {'boot 95% CI':>22} "
          f"{'power@n':>8} {'n@80%(1s)':>10} {'n@80%(2s)':>10}")

    primary = None
    for top_frac in [0.333, 0.40, 0.50]:
        thr = np.quantile(natural, 1 - top_frac)
        hard = natural >= thr
        d_h = d_agg[hard]
        n_h = int(hard.sum())
        eff = float(d_h.mean())
        sd = float(d_h.std(ddof=1))
        # paired bootstrap CI on the hard-cohort mean (matches the report method)
        boot = np.array([d_h[rng.integers(0, n_h, n_h)].mean() for _ in range(args.n_boot)])
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        ci_1s_lo = float(np.percentile(boot, 5.0))  # one-sided 95% lower bound
        # current power to detect THIS effect at THIS n (z-approx)
        se = sd / np.sqrt(n_h)
        za1 = stats.norm.ppf(1 - args.alpha)
        cur_power = float(1 - stats.norm.cdf(za1 - eff / se)) if eff > 0 else 0.0
        n80_1s = required_n(eff, sd, args.alpha, 0.80, one_sided=True)
        n80_2s = required_n(eff, sd, args.alpha, 0.80, one_sided=False)
        n90_1s = required_n(eff, sd, args.alpha, 0.90, one_sided=True)
        rec = {"n_hard": n_h, "threshold_crps": float(thr), "effect": eff, "sd": sd,
               "boot_ci95": ci, "boot_ci_onesided_lower95": ci_1s_lo,
               "current_power_onesided": cur_power,
               "n_for_80pct_power_onesided": n80_1s,
               "n_for_80pct_power_twosided": n80_2s,
               "n_for_90pct_power_onesided": n90_1s}
        out["thresholds"][f"top_{top_frac}"] = rec
        print(f"{top_frac:9.3f} {n_h:6d} {eff:+8.4f} {sd:7.4f} "
              f"[{ci[0]:+.4f},{ci[1]:+.4f}] {cur_power:8.2f} "
              f"{n80_1s:10.0f} {n80_2s:10.0f}")
        if abs(top_frac - 0.333) < 1e-6:
            primary = rec

    # diff-in-diff secondary (tercile)
    thr = np.quantile(natural, 1 - 0.333)
    hard, easy = natural >= thr, natural < thr
    did = float(d_agg[hard].mean() - d_agg[easy].mean())
    sd_did = float(np.sqrt(d_agg[hard].var(ddof=1) / hard.sum()
                          + d_agg[easy].var(ddof=1) / easy.sum()))  # SE of DiD
    out["diff_in_diff"] = {"effect": did, "se": sd_did, "z": did / sd_did,
                           "p_onesided": float(1 - stats.norm.cdf(did / sd_did))}

    # planning curves vs n (use the primary tercile effect+sd)
    eff, sd = primary["effect"], primary["sd"]
    n_grid = np.arange(40, 401, 10)
    mde80 = (stats.norm.ppf(1 - args.alpha) + stats.norm.ppf(0.80)) * sd / np.sqrt(n_grid)
    sim_power = bootstrap_power(d_agg[hard], n_grid, eff, sd, args.alpha,
                                n_sim=4000, rng=rng, one_sided=True)
    out["planning"] = {"effect_used": eff, "sd_used": sd,
                       "n_grid": n_grid.tolist(),
                       "mde_at_80pct_power": mde80.tolist(),
                       "sim_power_onesided": [sim_power[int(n)] for n in n_grid]}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "power_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)

    # plot: power vs n, and MDE vs n
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.plot(n_grid, [sim_power[int(n)] for n in n_grid], "-o", ms=3, color="#4c78a8")
    a1.axhline(0.8, color="gray", ls="--", label="80% power")
    a1.axvline(primary["n_hard"], color="red", ls=":", label=f"current n={primary['n_hard']}")
    a1.set_xlabel("n hard windows"); a1.set_ylabel("power (1-sided, α=0.05)")
    a1.set_title(f"Power to detect +{eff:.3f} CRPS"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
    a2.plot(n_grid, mde80, "-o", ms=3, color="#e45756")
    a2.axhline(eff, color="gray", ls="--", label=f"pilot effect {eff:+.3f}")
    a2.axvline(primary["n_hard"], color="red", ls=":", label=f"current n={primary['n_hard']}")
    a2.set_xlabel("n hard windows"); a2.set_ylabel("MDE @ 80% power (CRPS)")
    a2.set_title("Minimum detectable effect vs n"); a2.legend(fontsize=8); a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "power_analysis.png"), dpi=150)

    print(f"\nPRIMARY (tercile): effect={eff:+.4f}, sd={sd:.4f}, current n={primary['n_hard']}, "
          f"current power={primary['current_power_onesided']:.2f}")
    print(f"  -> need n={primary['n_for_80pct_power_onesided']:.0f} (1-sided) / "
          f"{primary['n_for_80pct_power_twosided']:.0f} (2-sided) for 80% power")
    print(f"  -> that is {primary['n_for_80pct_power_onesided']/primary['n_hard']:.1f}x "
          f"the current hard-window count")
    print(f"DiD (hard-easy): {did:+.4f} (SE {sd_did:.4f}, z={did/sd_did:.2f}, "
          f"p1={out['diff_in_diff']['p_onesided']:.3f})")
    print(f"\nSaved {args.out_dir}/power_analysis.json + power_analysis.png")


if __name__ == "__main__":
    main()
