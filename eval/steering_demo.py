"""Steering demo — Golden-Gate-Claude for time series, scaled to our findings.

Take held-out test windows where the model is confident (low natural CRPS, the
'easy / flat' regime), pick the top-K difficulty-predictive features, clamp
each one to its 99th-percentile activation across the TRAIN split, re-run the
forecast through the SAE-recon hook, and visualize whether the forecast
distribution shifts in any directionally interpretable way.

Honest design: we DO NOT cherry-pick the most dramatic plot. We:
  - pick 4 confident windows (lowest natural CRPS),
  - try each of the top-K features on each window,
  - quantify the L2 distance between natural-mean and steered-mean forecast,
  - report the full matrix + plot the best/worst examples.

Given §4.6 shows the top-K features are *weakly* causally tied to forecast
quality, the steering effect is expected to be small. We surface that result
honestly rather than overclaim a 'Golden Gate' moment.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from chronos import ChronosPipeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sae.sae_model import TopKSAE
from extract_activations import compute_crps


def make_steering_hook(sae, feat_idx=None, clamp_value=None):
    """Hook that swaps the hidden state with SAE reconstruction. If feat_idx
    is given, the code value for that feature is CLAMPED to `clamp_value`
    at every token position before decoding (i.e., the feature is forced to
    fire at the given activation strength, regardless of input)."""
    W_enc = sae.W_enc.detach()
    b_enc = sae.b_enc.detach()
    W_dec = sae.W_dec.detach()
    b_dec = sae.b_dec.detach()
    k = sae.k

    def hook(module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        dtype = hidden.dtype
        b, s, d = hidden.shape
        flat = hidden.reshape(-1, d).float()
        pre = (flat - b_dec) @ W_enc + b_enc
        top_acts, top_idx = torch.topk(pre, k, dim=-1)
        codes = torch.zeros_like(pre)
        codes.scatter_(-1, top_idx, F.relu(top_acts))
        if feat_idx is not None:
            codes[:, feat_idx] = clamp_value
        recon = (codes @ W_dec + b_dec).reshape(b, s, d).to(dtype)
        if isinstance(output, tuple):
            return (recon,) + output[1:]
        return recon
    return hook


def feature_99pct_train(sae, activations, train_mask, max_tokens=20000, seed=42):
    """For each SAE feature, the 99th-percentile activation across TRAIN tokens
    (shape (d_hidden,)) — the 'crank-it-up' clamp value for steering.

    The full train set is ~250k tokens; the per-feature 99th percentile is
    stable under subsampling, so we cap at `max_tokens` rows. The dense
    (N, d_hidden) codes tensor and its column-wise quantile are what made the
    original run blow up — capping keeps this to seconds and a few hundred MB."""
    sae.eval()
    tr_acts = activations[train_mask].to(torch.float32)
    flat = tr_acts.reshape(-1, tr_acts.shape[-1])
    if flat.shape[0] > max_tokens:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(flat.shape[0], generator=g)[:max_tokens]
        flat = flat[idx]
    with torch.no_grad():
        pre = (flat - sae.b_dec) @ sae.W_enc + sae.b_enc
        top_acts, top_idx = torch.topk(pre, sae.k, dim=-1)
        codes = torch.zeros_like(pre)
        codes.scatter_(-1, top_idx, F.relu(top_acts))
    return codes.quantile(0.99, dim=0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default="activations/ETTh1_activations.safetensors")
    ap.add_argument("--metadata", default="activations/ETTh1_metadata.parquet")
    ap.add_argument("--sae_ckpt", default="sae/checkpoints/sae_topk_32.pt")
    ap.add_argument("--series_csv",
                    default="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv")
    ap.add_argument("--target_col", default="OT")
    ap.add_argument("--model", default="amazon/chronos-t5-small")
    ap.add_argument("--context_length", type=int, default=512)
    ap.add_argument("--prediction_length", type=int, default=96)
    ap.add_argument("--num_samples", type=int, default=20,
                    help="Forecast sample trajectories per window. 20 is plenty "
                         "for a mean-shift demo on CPU; raise to 100+ on GPU.")
    ap.add_argument("--top_features", type=int, nargs="+",
                    default=[1465, 2717, 1425, 3702, 3678],
                    help="Default: the top-5 from causal_ablation.py.")
    ap.add_argument("--n_windows", type=int, default=4,
                    help="Number of confident (lowest-CRPS) test windows to steer.")
    ap.add_argument("--max_train_tokens", type=int, default=20000,
                    help="Cap tokens used for the per-feature 99th-pct clamp value.")
    ap.add_argument("--threads", type=int, default=4,
                    help="Cap torch CPU threads. The original runaway was kernel-time "
                         "thread oversubscription across many forecasts; capping fixes it.")
    ap.add_argument("--out_dir", default="eval/results/steering")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Prevent CPU thread oversubscription (the cause of the original runaway:
    # 522% CPU, 11940s system time over 24 forecasts). Harmless on GPU.
    if args.threads and args.threads > 0:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading SAE...")
    state = torch.load(args.sae_ckpt, map_location="cpu", weights_only=True)
    d_model_ckpt, d_hidden_ckpt = state["W_enc"].shape
    sae = TopKSAE(d_model=d_model_ckpt, d_hidden=d_hidden_ckpt, k=32)
    sae.load_state_dict(state)
    sae.eval()

    acts = load_file(args.activations)["encoder_embeddings"]
    meta = pd.read_parquet(args.metadata)

    # Per-feature clamp value: 99th percentile of code activations across train tokens.
    train_mask_tensor = torch.as_tensor((meta["split"].values == "train"))
    p99 = feature_99pct_train(sae, acts, train_mask_tensor,
                              max_tokens=args.max_train_tokens, seed=args.seed)
    print("99th-pct clamp values for top features:")
    for f in args.top_features:
        print(f"  feat {f}: clamp = {p99[f]:.3f}")

    # Choose the most CONFIDENT test windows (lowest natural CRPS).
    test = meta[meta["split"] == "test"].copy().sort_values("crps_raw").reset_index(drop=True)
    chosen = test.head(args.n_windows).copy()
    print(f"\nSteering {args.n_windows} most-confident test windows "
          f"(natural CRPS range {chosen['crps_raw'].min():.3f}–"
          f"{chosen['crps_raw'].max():.3f}):")
    print(chosen[["window_id", "start_ts", "crps_raw"]].to_string(index=False))

    print(f"\nLoading {args.model} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipeline = ChronosPipeline.from_pretrained(args.model, device_map=device, dtype=dtype)
    num_layers = pipeline.model.model.config.num_layers
    mid = num_layers // 2
    hook_module = pipeline.model.model.encoder.block[mid].layer[-1]

    df = pd.read_csv(args.series_csv)
    series = df[args.target_col].values.astype(np.float64)

    def predict_under_hook(context, hook_fn):
        h = hook_module.register_forward_hook(hook_fn) if hook_fn is not None else None
        try:
            with torch.no_grad():
                f = pipeline.predict(context, prediction_length=args.prediction_length,
                                     num_samples=args.num_samples)
        finally:
            if h is not None:
                h.remove()
        return f.cpu().numpy() if torch.is_tensor(f) else np.asarray(f)

    recon_hook = make_steering_hook(sae)   # no clamp, just SAE recon
    results = []
    fig_grid = []
    for _, row in chosen.iterrows():
        s = int(row["start_ts"])
        context = torch.tensor(series[s:s + args.context_length], dtype=torch.float32)
        truth = series[s + args.context_length:
                       s + args.context_length + args.prediction_length]

        f_recon = predict_under_hook(context, recon_hook)[0]  # (num_samples, H)
        mean_recon = f_recon.mean(axis=0)

        per_feat = {}
        for feat in args.top_features:
            steer_hook = make_steering_hook(sae, feat_idx=int(feat),
                                             clamp_value=float(p99[feat]))
            f_steer = predict_under_hook(context, steer_hook)[0]
            mean_steer = f_steer.mean(axis=0)
            l2 = float(np.linalg.norm(mean_steer - mean_recon))
            l2_norm = l2 / (np.linalg.norm(mean_recon) + 1e-9)
            per_feat[feat] = {
                "l2_distance": l2,
                "l2_relative": float(l2_norm),
                "mean_steer": mean_steer.tolist(),
                "ci_lo": np.percentile(f_steer, 5, axis=0).tolist(),
                "ci_hi": np.percentile(f_steer, 95, axis=0).tolist(),
            }

        results.append({
            "window_id": int(row["window_id"]),
            "start_ts": s,
            "crps_natural": float(row["crps_raw"]),
            "context_last20": context[-20:].tolist(),
            "truth": truth.tolist(),
            "mean_recon": mean_recon.tolist(),
            "recon_ci_lo": np.percentile(f_recon, 5, axis=0).tolist(),
            "recon_ci_hi": np.percentile(f_recon, 95, axis=0).tolist(),
            "per_feature": {str(k): v for k, v in per_feat.items()},
        })

    # Summary numbers + plot.
    print("\nSteering shift magnitudes (L2 distance between steered-mean and recon-mean):")
    print(f"{'window_id':>10} {'feat':>6} {'L2':>10} {'L2/||recon||':>14}")
    for r in results:
        for fk, fv in r["per_feature"].items():
            print(f"{r['window_id']:>10} {fk:>6} {fv['l2_distance']:>10.4f} {fv['l2_relative']:>14.4%}")

    # Grid plot: one row per window, columns = recon + each top feature
    nF = len(args.top_features)
    nW = len(results)
    fig, axes = plt.subplots(nW, nF + 1, figsize=(3.2 * (nF + 1), 2.4 * nW),
                              sharex=True, sharey="row", squeeze=False)
    H = args.prediction_length
    x_ctx = np.arange(-20, 0)
    x_fc = np.arange(0, H)
    for ri, r in enumerate(results):
        # column 0 = natural recon
        axes[ri, 0].plot(x_ctx, r["context_last20"], color="#333", lw=1)
        axes[ri, 0].plot(x_fc, r["truth"], color="#333", lw=1, ls=":")
        axes[ri, 0].plot(x_fc, r["mean_recon"], color="#4c78a8", lw=1.5)
        axes[ri, 0].fill_between(x_fc, r["recon_ci_lo"], r["recon_ci_hi"],
                                  color="#4c78a8", alpha=0.18)
        axes[ri, 0].set_title(f"win {r['window_id']}\nSAE recon", fontsize=9)
        axes[ri, 0].axvline(0, c="k", lw=0.6, alpha=0.4)

        for ci, feat in enumerate(args.top_features, start=1):
            fv = r["per_feature"][str(feat)]
            axes[ri, ci].plot(x_ctx, r["context_last20"], color="#333", lw=1)
            axes[ri, ci].plot(x_fc, r["truth"], color="#333", lw=1, ls=":")
            axes[ri, ci].plot(x_fc, r["mean_recon"], color="#4c78a8", lw=1, alpha=0.4,
                              label="recon")
            axes[ri, ci].plot(x_fc, fv["mean_steer"], color="#e45756", lw=1.5,
                              label="steered")
            axes[ri, ci].fill_between(x_fc, fv["ci_lo"], fv["ci_hi"],
                                       color="#e45756", alpha=0.18)
            axes[ri, ci].set_title(
                f"feat {feat} (Δ_rel={fv['l2_relative']*100:.1f}%)", fontsize=9)
            axes[ri, ci].axvline(0, c="k", lw=0.6, alpha=0.4)
    fig.suptitle("Steering: clamp each top-5 feature to train-99th-percentile, "
                  "compare forecast vs SAE-recon baseline", fontsize=11)
    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "steering_grid.png")
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved {out_png}")

    with open(os.path.join(args.out_dir, "steering_results.json"), "w") as f:
        json.dump({"top_features": list(map(int, args.top_features)),
                   "clamp_99pct": {str(int(k)): float(p99[k]) for k in args.top_features},
                   "windows": results}, f, indent=2)
    print(f"Saved {os.path.join(args.out_dir, 'steering_results.json')}")


if __name__ == "__main__":
    main()
