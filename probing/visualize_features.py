"""Week 3 figure (b): do the difficulty-predictive SAE features visibly land on
regime changes / level shifts?

This is interpretability inspection, NOT the headline metric (that is probe.py).
It fits a plain L1 logistic on SAE features to RANK them, then for the top-ranked
features plots the raw series with the per-token feature activation overlaid for
the windows where each feature fires hardest. The pitch lives or dies on these.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sae.sae_model import TopKSAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default="activations/ETTh1_activations.safetensors")
    ap.add_argument("--metadata", default="activations/ETTh1_metadata.parquet")
    ap.add_argument("--sae_ckpt", default="sae/checkpoints/sae_topk_32.pt")
    ap.add_argument("--series_csv",
                    default="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv")
    ap.add_argument("--target_col", default="OT")
    ap.add_argument("--context_length", type=int, default=512)
    ap.add_argument("--d_hidden", type=int, default=None, help="Optional override; inferred from the SAE checkpoint if omitted")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--hard_quantile", type=float, default=0.85)
    ap.add_argument("--top_features", type=int, default=5)
    ap.add_argument("--out_dir", default="probing/results/features")
    args = ap.parse_args()

    acts = load_file(args.activations)["encoder_embeddings"]
    meta = pd.read_parquet(args.metadata)
    for need in ("split",):
        if need not in meta.columns:
            sys.exit(f"[viz] metadata lacks '{need}' - run extract_activations.py first.")
    label_col = "crps_norm" if "crps_norm" in meta.columns else "crps_raw"
    if label_col not in meta.columns:
        sys.exit("[viz] no crps label in metadata - run extract_activations.py first.")

    # Infer SAE dims from the checkpoint (single source of truth) and verify
    # they match the activations file's hidden dim.
    state = torch.load(args.sae_ckpt, map_location="cpu", weights_only=True)
    d_model_ckpt, d_hidden_ckpt = state["W_enc"].shape
    if d_model_ckpt != acts.shape[-1]:
        sys.exit(f"[viz] SAE expects d_model={d_model_ckpt}, activations have "
                 f"{acts.shape[-1]} — wrong checkpoint for these activations.")
    if args.d_hidden is not None and args.d_hidden != d_hidden_ckpt:
        sys.exit(f"[viz] --d_hidden={args.d_hidden} contradicts checkpoint "
                 f"d_hidden={d_hidden_ckpt}.")
    print(f"SAE checkpoint dims: d_model={d_model_ckpt}, d_hidden={d_hidden_ckpt}")
    sae = TopKSAE(d_model=d_model_ckpt, d_hidden=d_hidden_ckpt, k=args.k)
    sae.load_state_dict(state)
    sae.eval()

    # Per-window max-pooled codes (N, d_hidden) for ranking.
    pooled, per_window_codes = [], []
    with torch.no_grad():
        for i in range(acts.shape[0]):
            w = acts[i:i + 1].to(torch.float32)
            c, _, _ = sae(w.reshape(-1, w.shape[-1]))
            c = c.reshape(w.shape[1], -1).numpy()       # (seq, d_hidden)
            per_window_codes.append(c)
            pooled.append(c.max(axis=0))
    pooled = np.stack(pooled)

    tr = meta["split"].values == "train"
    thr = np.quantile(meta.loc[tr, label_col].values, args.hard_quantile)
    y = (meta[label_col].values >= thr).astype(int)

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty="l1", solver="liblinear",
                                           class_weight="balanced", max_iter=2000, C=0.3))
    clf.fit(pooled[tr], y[tr])
    coef = np.abs(clf.named_steps["logisticregression"].coef_.ravel())
    top = np.argsort(coef)[::-1][:args.top_features]
    print("Top difficulty-weighted SAE feature indices:", top.tolist())

    df = pd.read_csv(args.series_csv)
    series = df[args.target_col].values.astype(np.float64)
    starts = meta["start_ts"].values if "start_ts" in meta.columns else meta["start_idx"].values

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(args.out_dir, exist_ok=True)
    for feat in top:
        firing = pooled[:, feat]
        w_idx = int(np.argmax(firing))
        s = int(starts[w_idx])
        ctx = series[s:s + args.context_length]
        act = per_window_codes[w_idx][:len(ctx), feat]
        fig, ax1 = plt.subplots(figsize=(10, 3))
        ax1.plot(ctx, color="#333", lw=1, label="context series")
        ax1.set_ylabel("value")
        ax2 = ax1.twinx()
        ax2.fill_between(np.arange(len(act)), 0, act, color="#e45756", alpha=0.35)
        ax2.set_ylabel(f"SAE feat {feat} activation")
        ax1.set_title(f"Feature {feat} | window {w_idx} | "
                      f"{'HARD' if y[w_idx] else 'easy'} (coef={coef[feat]:.3f})")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"feat_{feat}.png"), dpi=150)
        plt.close(fig)
    print("Wrote feature plots to", args.out_dir)


if __name__ == "__main__":
    main()
