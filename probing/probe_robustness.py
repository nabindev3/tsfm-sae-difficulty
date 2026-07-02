"""Exp8: is the SAE null an artifact of (a) a weak baseline or (b) CRPS
top-25% binarization?

Two reviewer worries, answered with the committed activations + already-saved
forecasts (no new TSFM forward pass):

  (a) Stronger baseline. Augment the 8 classical P1 stats with STL trend &
      seasonal strength, the Hurst exponent, and the model's OWN forecast-
      interval width (band_width_90 from Exp1 — the natural difficulty signal).
      If the SAE still fails to beat this stronger baseline, the null is not a
      weak-baseline artifact.

  (b) Label variation. Repeat under a MASE-based binary label and a CONTINUOUS
      (un-binarized) CRPS target (Ridge -> test Spearman). If the SAE still
      doesn't help, the null is not a CRPS-binarization artifact.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from sae.sae_model import TopKSAE
from core.probe import build_ladder, run_probe_ladder, P1, P2, P3
from probing.probe import compute_input_stats
from probing.extended_baselines import compute_extended_baselines, interval_width_feature


def agg(acts):  # concat(mean, max, last) pooling, matching probe.py
    return np.concatenate([acts.mean(1).numpy(), acts.max(1).values.numpy(),
                           acts[:, -1, :].numpy()], axis=1)


def make_cv(y, train_mask):
    n_splits = max(2, min(5, int(np.bincount(y[train_mask]).min()) - 1, int(train_mask.sum()) // 3))
    return list(TimeSeriesSplit(n_splits=n_splits).split(
        np.zeros((int(train_mask.sum()), 1)), y[train_mask]))


def binary_ladder(stats_block, raw_agg, sae_agg, y, train_mask, test_mask, tag):
    feats = build_ladder(stats_block, raw_agg, sae_agg)
    ladder, _ = run_probe_ladder(feats, y, train_mask, test_mask,
                                 make_cv(y, train_mask), n_boot=2000, seed=42)
    ds = ladder.deltas[f"{P3}-{P1}"]
    dr = ladder.deltas[f"{P3}-{P2}"]
    return {"tag": tag, "P1_AUROC": ladder.probes[P1].auroc,
            "P2_AUROC": ladder.probes[P2].auroc, "P3_AUROC": ladder.probes[P3].auroc,
            "delta_sae_minus_stats": [ds["point"], ds["ci_low"], ds["ci_high"]],
            "delta_sae_minus_raw": [dr["point"], dr["ci_low"], dr["ci_high"]]}


def reg_spearman(block, target, train_mask, test_mask):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    m.fit(block[train_mask], target[train_mask])
    pred = m.predict(block[test_mask])
    return float(spearmanr(pred, target[test_mask]).correlation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="activations/ETTh1_metadata.parquet")
    ap.add_argument("--activations", default="activations/ETTh1_activations.safetensors")
    ap.add_argument("--sae_ckpt", default="sae/checkpoints/sae_topk_32.pt")
    ap.add_argument("--uncertainty", default="eval/results/uncertainty_ETTh1_chronos-t5-small.parquet")
    ap.add_argument("--series_url", default="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv")
    ap.add_argument("--channel", default="OT")
    ap.add_argument("--season_length", type=int, default=24)
    ap.add_argument("--context_length", type=int, default=512)
    ap.add_argument("--out", default="probing/results/robustness.json")
    args = ap.parse_args()

    meta = pd.read_parquet(args.metadata)
    raw_acts = load_file(args.activations)["encoder_embeddings"]
    state = torch.load(args.sae_ckpt, map_location="cpu", weights_only=True)
    dm, dh = state["W_enc"].shape
    sae = TopKSAE(d_model=dm, d_hidden=dh, k=32); sae.load_state_dict(state); sae.eval()
    series = pd.read_csv(args.series_url)[args.channel].values.astype(np.float64)

    train_mask = (meta["split"] == "train").values
    test_mask = (meta["split"] == "test").values

    print("Computing baselines (8 stats + STL/Hurst + interval width)...")
    input_stats = compute_input_stats(meta, season_length=args.season_length,
                                      series_url=args.series_url, channel=args.channel)
    ext = compute_extended_baselines(meta, series, args.context_length, args.season_length)
    iw = interval_width_feature(meta, args.uncertainty)[:, 0]
    # purge rows have no saved forecast -> NaN; they are never in train/test, but
    # fill with the train median so nothing can propagate NaN into a fit.
    med = np.nanmedian(iw[train_mask])
    iw = np.where(np.isnan(iw), med, iw)
    stats_strong = np.concatenate([input_stats, ext, iw[:, None]], axis=1)

    print("Aggregating raw + SAE activations...")
    raw_agg = agg(raw_acts)
    with torch.no_grad():
        sae_acts = torch.cat([sae(raw_acts[i:i+1].float())[0].cpu()
                              for i in range(raw_acts.shape[0])], dim=0)
    sae_agg = agg(sae_acts)

    out = {"n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
           "n_features": {"P1_stats": input_stats.shape[1],
                          "P1_strong": stats_strong.shape[1]}}

    # (a) baseline strength, CRPS top-25% label. Use the all-window 75th pct of
    # crps_norm to match probe.py / report §4.2 exactly (so P1 reproduces 0.654).
    crps = meta["crps_norm"].values
    y = (crps >= np.quantile(crps, 0.75)).astype(int)
    out["crps_label_weak_baseline"] = binary_ladder(input_stats, raw_agg, sae_agg, y, train_mask, test_mask, "8 stats")
    out["crps_label_strong_baseline"] = binary_ladder(stats_strong, raw_agg, sae_agg, y, train_mask, test_mask, "8 stats + STL + Hurst + interval-width")
    aw = roc_auc_score(y[test_mask], iw[test_mask])
    out["interval_width_alone_auroc"] = max(aw, 1 - aw)  # natural-signal AUROC (direction-agnostic)

    # (b1) MASE top-25% label, strong baseline (all-window quantile, as in §4.2)
    mase = meta["mase"].values
    ym = (mase >= np.quantile(mase, 0.75)).astype(int)
    out["mase_label_strong_baseline"] = binary_ladder(stats_strong, raw_agg, sae_agg, ym, train_mask, test_mask, "MASE top-25%, strong baseline")

    # (b2) continuous CRPS target (no binarization) -> test Spearman
    out["continuous_crps_spearman"] = {
        "stats8": reg_spearman(input_stats, crps, train_mask, test_mask),
        "stats_strong": reg_spearman(stats_strong, crps, train_mask, test_mask),
        "raw": reg_spearman(raw_agg, crps, train_mask, test_mask),
        "sae": reg_spearman(sae_agg, crps, train_mask, test_mask),
        "stats_strong_plus_sae": reg_spearman(np.concatenate([stats_strong, sae_agg], 1), crps, train_mask, test_mask),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    def d(x):  # format [point, lo, hi]
        return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
    print("\n=== (a) BASELINE STRENGTH (CRPS top-25% label) ===")
    for k in ("crps_label_weak_baseline", "crps_label_strong_baseline"):
        r = out[k]
        print(f"  {r['tag']:42s} P1={r['P1_AUROC']:.3f} P3={r['P3_AUROC']:.3f}  "
              f"Δ(SAE−stats)={d(r['delta_sae_minus_stats'])}")
    print(f"  interval-width-alone test AUROC = {out['interval_width_alone_auroc']:.3f} (the natural signal)")
    print("\n=== (b1) MASE LABEL (strong baseline) ===")
    r = out["mase_label_strong_baseline"]
    print(f"  P1={r['P1_AUROC']:.3f} P3={r['P3_AUROC']:.3f}  Δ(SAE−stats)={d(r['delta_sae_minus_stats'])}")
    print("\n=== (b2) CONTINUOUS CRPS TARGET — test Spearman (higher=better) ===")
    for k, v in out["continuous_crps_spearman"].items():
        print(f"  {k:24s} {v:+.3f}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
