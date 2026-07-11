"""Causal ablation of top-K difficulty-predictive SAE features.

For each test window we measure CRPS under three conditions, holding everything
else fixed:

  1. natural          — no SAE intervention (= the extraction's crps_raw).
  2. SAE reconstruct  — a forward hook on encoder.block[mid].layer[-1]
                        replaces its output with the SAE's reconstruction.
                        Isolates the *reconstruction loss* cost of inserting
                        the SAE into the forward pass.
  3. ablate(feat=k)   — same hook, but feature k is zeroed in the SAE codes
                        before decoding. Isolates the *causal contribution*
                        of that feature.

A feature is causally tied to forecast quality if ΔCRPS(ablate − sae_recon)
is significantly positive, especially on hard windows. Paired-bootstrap CIs.

Mirrors the Mishra (2026) ablation protocol at smaller scale (top-K instead
of all features) to make the result interpretable under CPU constraints.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from chronos import ChronosPipeline
from tqdm import tqdm

from sae.sae_model import TopKSAE
from extract_activations import compute_crps


def topk_difficulty_features(sae, activations, meta, k_features, hard_quantile=0.85):
    """Rank features by absolute L1 logistic coefficient on max-pooled codes,
    return top-k indices. Same logic as visualize_features.py, returned as a
    list of ints. Train-split only."""
    sae.eval()
    pooled = []
    with torch.no_grad():
        for i in range(activations.shape[0]):
            w = activations[i:i + 1].to(torch.float32)
            c, _, _ = sae(w.reshape(-1, w.shape[-1]))
            c = c.reshape(w.shape[1], -1).numpy()
            pooled.append(c.max(axis=0))
    pooled = np.stack(pooled)
    label_col = "crps_norm" if "crps_norm" in meta.columns else "crps_raw"
    tr = meta["split"].values == "train"
    thr = np.quantile(meta.loc[tr, label_col].values, hard_quantile)
    y = (meta[label_col].values >= thr).astype(int)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty="l1", solver="liblinear",
                                           class_weight="balanced",
                                           max_iter=2000, C=0.3))
    clf.fit(pooled[tr], y[tr])
    coef = np.abs(clf.named_steps["logisticregression"].coef_.ravel())
    top = np.argsort(coef)[::-1][:k_features].tolist()
    return [int(i) for i in top], coef


def make_sae_hook(sae, ablated_features=None):
    """Hook that replaces a module's hidden output with the SAE reconstruction.
    If `ablated_features` is given (list of ints), those latents are zeroed
    in the codes BEFORE decoding."""
    sae_W_enc = sae.W_enc.detach()
    sae_b_enc = sae.b_enc.detach()
    sae_W_dec = sae.W_dec.detach()
    sae_b_dec = sae.b_dec.detach()
    k = sae.k

    abl = None if ablated_features is None else torch.as_tensor(
        ablated_features, dtype=torch.long)

    def hook(module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        dtype = hidden.dtype
        b, s, d = hidden.shape
        flat = hidden.reshape(-1, d).float()
        x_centered = flat - sae_b_dec
        pre = x_centered @ sae_W_enc + sae_b_enc
        top_acts, top_idx = torch.topk(pre, k, dim=-1)
        codes = torch.zeros_like(pre)
        codes.scatter_(-1, top_idx, F.relu(top_acts))
        if abl is not None:
            codes[:, abl] = 0.0
        recon = codes @ sae_W_dec + sae_b_dec
        recon = recon.reshape(b, s, d).to(dtype)
        if isinstance(output, tuple):
            return (recon,) + output[1:]
        return recon
    return hook


def predict_with_hook(pipeline, hook_module, hook_fn, context, prediction_length, num_samples):
    h = hook_module.register_forward_hook(hook_fn) if hook_fn is not None else None
    try:
        with torch.no_grad():
            f = pipeline.predict(context, prediction_length=prediction_length,
                                 num_samples=num_samples)
    finally:
        if h is not None:
            h.remove()
    return f.cpu().numpy() if torch.is_tensor(f) else np.asarray(f)


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
    ap.add_argument("--num_samples", type=int, default=50,
                    help="Lower than the 100 used for headline labels — ablation "
                         "compares relative ΔCRPS, not absolute, so the extra "
                         "MC noise is acceptable for compute savings. For the "
                         "matched-MC baseline (Exp6) set this to 100 or 200.")
    ap.add_argument("--matched_natural", action="store_true",
                    help="Compute the natural baseline via a hook-free predict at "
                         "--num_samples (matched MC), instead of reusing crps_raw "
                         "(100 samples). Deletes the §4.6 MC-noise caveat.")
    ap.add_argument("--k_features", type=int, default=5)
    ap.add_argument("--hard_quantile", type=float, default=0.85)
    ap.add_argument("--max_windows", type=int, default=None,
                    help="Cap the number of test windows for time-bounded runs (None = all)")
    ap.add_argument("--out_dir", default="eval/results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading SAE checkpoint...")
    state = torch.load(args.sae_ckpt, map_location="cpu", weights_only=True)
    d_model_ckpt, d_hidden_ckpt = state["W_enc"].shape
    sae = TopKSAE(d_model=d_model_ckpt, d_hidden=d_hidden_ckpt, k=32)
    sae.load_state_dict(state)
    sae.eval()

    print("Loading activations + metadata for feature ranking...")
    acts = load_file(args.activations)["encoder_embeddings"]
    if acts.shape[-1] != d_model_ckpt:
        sys.exit(f"[ablation] SAE d_model={d_model_ckpt} but activations have "
                 f"{acts.shape[-1]} -- wrong checkpoint for these activations.")
    meta = pd.read_parquet(args.metadata)
    top, coef = topk_difficulty_features(sae, acts, meta, args.k_features,
                                         hard_quantile=args.hard_quantile)
    print(f"Top-{args.k_features} difficulty-predictive features: {top}")
    print(f"  abs coefs: {[round(coef[i], 4) for i in top]}")

    print(f"Loading {args.model} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipeline = ChronosPipeline.from_pretrained(args.model, device_map=device, dtype=dtype)
    num_layers = pipeline.model.model.config.num_layers
    mid = num_layers // 2
    hook_module = pipeline.model.model.encoder.block[mid].layer[-1]
    print(f"Hooking encoder.block[{mid}].layer[-1] (num_layers={num_layers}); "
          f"same layer the SAE was trained on.")

    df = pd.read_csv(args.series_csv)
    series = df[args.target_col].values.astype(np.float64)

    test = meta[meta["split"] == "test"].copy().reset_index(drop=True)
    if args.max_windows is not None:
        test = test.iloc[:args.max_windows].copy()
    print(f"Running ablation on {len(test)} test windows.")

    rows = []
    sae_recon_hook = make_sae_hook(sae, ablated_features=None)
    feat_hooks = {f: make_sae_hook(sae, ablated_features=[f]) for f in top}

    for _, row in tqdm(list(test.iterrows()), total=len(test)):
        s = int(row["start_ts"])
        context = torch.tensor(series[s:s + args.context_length], dtype=torch.float32)
        truth = series[s + args.context_length:
                       s + args.context_length + args.prediction_length]
        # SAE-recon baseline
        f_recon = predict_with_hook(pipeline, hook_module, sae_recon_hook,
                                    context, args.prediction_length, args.num_samples)
        crps_recon = compute_crps(f_recon[0], truth)
        # Matched-MC natural baseline: hook-free predict at the SAME num_samples
        # as recon/ablate, so Δ(recon − natural) carries no Monte-Carlo-count
        # asymmetry (the §4.6 caveat: 50-sample recon vs 100-sample crps_raw).
        # The original crps_raw is retained for reference only.
        if args.matched_natural:
            f_nat = predict_with_hook(pipeline, hook_module, None,
                                      context, args.prediction_length, args.num_samples)
            crps_nat_matched = float(compute_crps(f_nat[0], truth))
        else:
            crps_nat_matched = float(row["crps_raw"])
        out = {"window_id": int(row["window_id"]),
               "start_ts": s,
               "crps_natural": float(row["crps_raw"]),
               "crps_natural_matched": crps_nat_matched,
               "crps_sae_recon": float(crps_recon)}
        # Per-feature ablations
        for feat in top:
            f_abl = predict_with_hook(pipeline, hook_module, feat_hooks[feat],
                                      context, args.prediction_length, args.num_samples)
            out[f"crps_ablate_{feat}"] = float(compute_crps(f_abl[0], truth))
        rows.append(out)

    df_out = pd.DataFrame(rows)
    out_parquet = os.path.join(args.out_dir, "causal_ablation.parquet")
    df_out.to_parquet(out_parquet)
    print(f"Saved {out_parquet}")

    # Aggregate stats with paired bootstrap CIs
    rng = np.random.default_rng(args.seed)
    n = len(df_out)
    n_boot = 2000

    def _ci(arr):
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    # Use the matched-MC natural baseline for the recon delta when available, so
    # Δ(SAE recon − natural) is free of the Monte-Carlo-count caveat.
    natural_col = "crps_natural_matched" if "crps_natural_matched" in df_out.columns else "crps_natural"
    natural = df_out[natural_col].values
    recon = df_out["crps_sae_recon"].values
    delta_sae = recon - natural

    boot_delta_sae = []
    boot_delta_feat = {f: [] for f in top}
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boot_delta_sae.append((recon[idx] - natural[idx]).mean())
        for f in top:
            abl = df_out[f"crps_ablate_{f}"].values
            boot_delta_feat[f].append((abl[idx] - recon[idx]).mean())

    summary = {
        "n_windows": n,
        "top_features": top,
        "abs_probe_coefs": [float(coef[i]) for i in top],
        "natural_baseline": natural_col,
        "num_samples": args.num_samples,
        "matched_natural": bool(args.matched_natural),
        "mean_crps_natural": float(natural.mean()),
        "mean_crps_sae_recon": float(recon.mean()),
        "delta_sae_recon": {
            "point": float(delta_sae.mean()),
            "ci95": _ci(boot_delta_sae),
        },
        "per_feature_ablation_delta": {},
    }
    print(f"\nΔ(SAE recon − natural) = "
          f"{delta_sae.mean():+.4f}  95% CI {_ci(boot_delta_sae)}  "
          f"(reconstruction-loss baseline)")
    print(f"\nΔ(ablate feature − SAE recon)  [feature: point, 95% CI]:")
    for f in top:
        abl = df_out[f"crps_ablate_{f}"].values
        d = (abl - recon).mean()
        ci = _ci(boot_delta_feat[f])
        summary["per_feature_ablation_delta"][f] = {
            "point": float(d), "ci95": ci,
        }
        sig = " *" if (ci[0] > 0 or ci[1] < 0) else ""
        print(f"  feat {f:5d}: {d:+.4f}  CI {ci}{sig}")

    with open(os.path.join(args.out_dir, "causal_ablation.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {os.path.join(args.out_dir, 'causal_ablation.json')}")


if __name__ == "__main__":
    main()
