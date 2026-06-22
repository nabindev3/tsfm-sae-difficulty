"""Conformal-prediction baseline + selective-prediction Pareto comparison.

Answers the reviewer question "you never beat the standard UQ method": adds the
model's *own* conformalized predictive uncertainty as a selective-prediction
signal and puts its risk-coverage frontier head-to-head with the P1 input-stats
probe, the oracle, and random.

Two deliverables:

(A) Split-conformal prediction intervals (CQR-style) on Chronos forecasts.
    - Nonconformity score per window: cqr_R = max_t max(q05(t)-y(t), y(t)-q95(t)).
    - Calibrate Q on the train(=calibration) split; conformal interval is the
      Chronos central-90 band widened by Q. Report marginal coverage on test
      (guaranteed >= 1-alpha under exchangeability) and the *conditional*
      coverage on probe-predicted hard vs easy windows -- i.e. does the standard
      UQ method silently under-cover exactly where our difficulty probe fires?

(B) Selective prediction. The deployable conformal signal is the per-window
    band width (q95-q05, available at inference without the truth; CQR
    calibration only adds a constant so it does not change the ranking). Rank
    test windows ascending by band width, retain the most-certain c*N, and plot
    mean-CRPS-on-retained vs the P1 probe, oracle, and random -- the same
    risk-coverage axes as eval/selective_prediction.py.

Inputs: the predictive-uncertainty parquet from extract_uncertainty.py, plus the
committed probe_scores + metadata. Risk axis is the canonical crps_raw label so
every signal is compared on identical per-window risk.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def risk_coverage_curve(rank_signal, crps, coverages, rng, n_boot):
    """Rank windows ascending by rank_signal (low = predicted easy/certain),
    retain the first c*N, return mean retained CRPS + bootstrap CI + AURC."""
    order = np.argsort(rank_signal)
    sorted_crps = crps[order]
    curve, lo, hi = [], [], []
    for c in coverages:
        k = max(1, int(round(c * len(crps))))
        kept = sorted_crps[:k]
        curve.append(float(kept.mean()))
        boots = [kept[rng.integers(0, k, k)].mean() for _ in range(n_boot)]
        lo.append(float(np.percentile(boots, 2.5)))
        hi.append(float(np.percentile(boots, 97.5)))
    return (np.array(curve), np.array(lo), np.array(hi),
            float(np.trapezoid(curve, coverages)))


def conformal_quantile(cal_scores, alpha):
    """Split-conformal correction: the ceil((n+1)(1-alpha))/n empirical quantile."""
    n = len(cal_scores)
    if n == 0:
        return None
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(cal_scores, level, method="higher"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uncertainty",
                    default="eval/results/uncertainty_ETTh1_chronos-t5-small.parquet")
    ap.add_argument("--probe_scores", default="activations/probe_scores.parquet")
    ap.add_argument("--metadata", default="activations/ETTh1_metadata.parquet")
    ap.add_argument("--out_dir", default="eval/results")
    ap.add_argument("--alphas", default="0.1,0.2", help="miscoverage levels to report")
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    for p in (args.uncertainty, args.probe_scores, args.metadata):
        if not os.path.exists(p):
            sys.exit(f"[conformal] missing {p}")
    os.makedirs(args.out_dir, exist_ok=True)

    unc = pd.read_parquet(args.uncertainty)
    scores = pd.read_parquet(args.probe_scores)
    meta = pd.read_parquet(args.metadata)[["start_ts", "crps_raw", "split", "crps_norm"]]

    # canonical risk + true hard label (top-25% train-normalised CRPS, per probe.py)
    hard_thresh = meta["crps_norm"].quantile(0.75)
    meta["is_hard"] = (meta["crps_norm"] >= hard_thresh).astype(int)

    test = (unc[unc.split == "test"]
            .merge(meta[["start_ts", "crps_raw", "is_hard"]], on="start_ts")
            .merge(scores[["start_ts", "pred_P1_InputStats", "pred_P3_InputStats_SAE"]],
                   on="start_ts"))
    crps = test["crps_raw"].values.astype(float)
    n = len(test)
    mean_crps = float(crps.mean())
    coverages = np.round(np.arange(0.10, 1.001, 0.05), 4)
    summary = {"n_test": n, "mean_crps_no_abstention": mean_crps}

    # --- repro check vs committed labels ---
    from scipy.stats import pearsonr, spearmanr
    summary["repro_crps"] = {
        "pearson": float(pearsonr(test.crps_recomputed, test.crps_raw)[0]),
        "spearman": float(spearmanr(test.crps_recomputed, test.crps_raw)[0]),
    }

    # ============ (A) conformal coverage ============
    cal = unc[unc.split == "train"]
    cov_block = {"n_calibration": int(len(cal)),
                 "raw_band_nominal_0.90_test_coverage": float(test.cover90_raw.mean())}
    if len(cal) >= 20:
        for a in [float(x) for x in args.alphas.split(",")]:
            Q = conformal_quantile(cal.cqr_R.values, a)
            covered = (test.cqr_R.values <= Q)
            ph = test.pred_P1_InputStats.values >= 0.5   # probe-predicted hard
            th = test.is_hard.values == 1                # truly hard
            def cov(mask):
                return float(covered[mask].mean()) if mask.sum() else None
            cov_block[f"alpha_{a}"] = {
                "nominal_coverage": 1 - a,
                "Q_correction": Q,
                "marginal_coverage_test": float(covered.mean()),
                "coverage_pred_hard": cov(ph), "n_pred_hard": int(ph.sum()),
                "coverage_pred_easy": cov(~ph), "n_pred_easy": int((~ph).sum()),
                "coverage_true_hard": cov(th), "n_true_hard": int(th.sum()),
                "coverage_true_easy": cov(~th), "n_true_easy": int((~th).sum()),
            }
    else:
        cov_block["note"] = ("no calibration (train) rows in uncertainty file; "
                             "skipped conformal coverage, Pareto comparison still valid")
    summary["conformal_coverage"] = cov_block

    # ============ (B) selective-prediction frontier ============
    sorted_truth = np.sort(crps)
    oracle = np.array([sorted_truth[:max(1, int(round(c*n)))].mean() for c in coverages])
    rand = np.array([[crps[rng.permutation(n)][:max(1, int(round(c*n)))].mean()
                      for c in coverages] for _ in range(args.n_bootstrap)]).mean(0)

    signals = {
        "conformal_band_width": test.band_width_90.values,   # the UQ baseline
        "predictive_std":       test.std_mean.values,        # alt UQ baseline
        "P1_InputStats":        test.pred_P1_InputStats.values,
        "P3_InputStats_SAE":    test.pred_P3_InputStats_SAE.values,
    }
    curves = {}
    i50 = int(np.argmin(np.abs(coverages - 0.5)))
    for name, sig in signals.items():
        c_, lo, hi, aurc = risk_coverage_curve(sig, crps, coverages, rng, args.n_bootstrap)
        curves[name] = {"curve": c_.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                        "aurc": aurc, "crps_at_0.5": float(c_[i50]),
                        "reduction_pct_at_0.5": 100*(mean_crps - c_[i50])/mean_crps}
    oracle_aurc = float(np.trapezoid(oracle, coverages))
    random_aurc = float(np.trapezoid(rand, coverages))
    summary["selective"] = {
        "coverages": coverages.tolist(),
        "oracle_aurc": oracle_aurc, "random_aurc": random_aurc,
        "oracle_curve": oracle.tolist(), "random_curve": rand.tolist(),
        "signals": curves,
        # fraction of oracle's AURC headroom captured: (random-sig)/(random-oracle)
        "oracle_headroom_captured": {
            name: (random_aurc - curves[name]["aurc"]) / (random_aurc - oracle_aurc)
            for name in curves},
    }
    with open(os.path.join(args.out_dir, "conformal_baseline.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- plot ----
    style = {"conformal_band_width": ("Conformal band width (UQ)", "#9c27b0", "-"),
             "predictive_std": ("Predictive std (UQ)", "#b39ddb", "--"),
             "P1_InputStats": ("P1 input-stats probe", "#4c78a8", "-"),
             "P3_InputStats_SAE": ("P3 stats+SAE", "#e45756", ":")}
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.axhline(mean_crps, color="gray", ls=":", label=f"No abstention ({mean_crps:.3f})")
    ax.plot(coverages, rand, color="black", ls="--", label=f"Random (AURC {random_aurc:.3f})")
    ax.plot(coverages, oracle, color="black", lw=2, label=f"Oracle (AURC {oracle_aurc:.3f})")
    for name, (lbl, col, ls) in style.items():
        c_ = np.array(curves[name]["curve"])
        ax.plot(coverages, c_, color=col, ls=ls, marker="o", ms=3.5,
                label=f"{lbl} (AURC {curves[name]['aurc']:.3f})")
        ax.fill_between(coverages, curves[name]["lo"], curves[name]["hi"], alpha=0.10, color=col)
    ax.set_xlabel("Coverage (fraction retained for forecasting)")
    ax.set_ylabel("Mean CRPS on retained (lower better)")
    ax.set_title("Selective prediction: conformal UQ vs input-stats probe — ETTh1 / chronos-t5-small")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "risk_coverage_conformal.png"), dpi=150)

    # ---- console ----
    print(f"n_test={n}  mean CRPS (no abstention)={mean_crps:.4f}")
    print(f"repro vs crps_raw: Pearson={summary['repro_crps']['pearson']:.3f} "
          f"Spearman={summary['repro_crps']['spearman']:.3f}")
    print("\n(A) CONFORMAL COVERAGE")
    print(f"  raw Chronos central-90 band coverage on test = "
          f"{cov_block['raw_band_nominal_0.90_test_coverage']:.3f} (nominal 0.90)")
    for a in [x for x in cov_block if x.startswith('alpha_')]:
        b = cov_block[a]
        print(f"  {a}: marginal={b['marginal_coverage_test']:.3f} (target {b['nominal_coverage']:.2f}) | "
              f"pred-hard={b['coverage_pred_hard']} (n={b['n_pred_hard']}) "
              f"pred-easy={b['coverage_pred_easy']} (n={b['n_pred_easy']}) | "
              f"true-hard={b['coverage_true_hard']} (n={b['n_true_hard']})")
    print("\n(B) SELECTIVE PREDICTION (AURC lower better; headroom = frac of oracle captured)")
    print(f"  {'oracle':28s} AURC={oracle_aurc:.4f}")
    print(f"  {'random':28s} AURC={random_aurc:.4f}")
    for name in curves:
        print(f"  {name:28s} AURC={curves[name]['aurc']:.4f}  "
              f"@50%={curves[name]['crps_at_0.5']:.4f} "
              f"({-curves[name]['reduction_pct_at_0.5']:+.1f}%)  "
              f"headroom={summary['selective']['oracle_headroom_captured'][name]*100:+.0f}%")
    print(f"\nSaved {args.out_dir}/conformal_baseline.json and risk_coverage_conformal.png")


if __name__ == "__main__":
    main()
