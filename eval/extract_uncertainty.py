"""Re-run Chronos forecasts to recover a per-window *predictive-uncertainty*
signal that the original extraction discarded.

The headline pipeline (`extract_activations.py`) computes CRPS from 100 forecast
samples per window and then throws the samples away, keeping only the scalar
`crps_raw`. A conformal / model-intrinsic UQ baseline (the "standard UQ method
you're currently not beaten against") needs the model's own predictive spread,
so we re-run `predict()` on the same windows and save, per window:

  - crps_recomputed : CRPS from the fresh samples (validates against crps_raw)
  - band_width_90   : mean over horizon of (q95 - q05) of the samples
                      -> the per-window width of Chronos's central 90% band.
                         This is the deployable conformal selective signal
                         (CQR calibration only adds a constant, so ranking by
                         this == ranking by the conformalized interval width).
  - band_width_80   : same with q90 - q10 (robustness alt)
  - std_mean        : mean over horizon of the sample std (alt model-UQ signal)
  - cqr_R           : CQR nonconformity score, max over horizon of
                      max(q05 - y, y - q95). Used ONLY for split-conformal
                      calibration + coverage (uses the truth y, so it is not a
                      deployable selective signal -- band_width_90 is).
  - cover90_raw     : per-window mean over horizon of 1[q05 <= y <= q95]
                      (uncalibrated nominal-90% step coverage; diagnostic).

Windowing matches `extract_activations.py` exactly (context 512, horizon 96,
stride 24, 70% temporal split, purge gap) so `start_ts` aligns 1:1 with the
saved metadata / probe_scores. Defaults reproduce the committed ETTh1 run.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

# Reuse the exact CRPS definition the labels were generated with.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract_activations import compute_crps, set_seed


def assign_split(start, context_length, prediction_length, split_idx):
    end_horizon = start + context_length + prediction_length
    if end_horizon <= split_idx:
        return "train"
    if start >= split_idx + context_length + prediction_length:
        return "test"
    return "purge"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ETTh1")
    ap.add_argument("--url",
                    default="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv")
    ap.add_argument("--channel", default="OT", help="Column of the CSV to forecast")
    ap.add_argument("--model", default="amazon/chronos-t5-small")
    ap.add_argument("--context_length", type=int, default=512)
    ap.add_argument("--prediction_length", type=int, default=96)
    ap.add_argument("--stride", type=int, default=24)
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--splits", default="train,test",
                    help="comma list of splits to forecast (train,test,purge)")
    ap.add_argument("--max_windows", type=int, default=None,
                    help="cap total windows processed (timing probe)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    set_seed(args.seed)
    want_splits = set(args.splits.split(","))

    print(f"Loading {args.dataset} channel={args.channel} from {args.url}")
    df = pd.read_csv(args.url)
    if args.channel not in df.columns:
        sys.exit(f"channel {args.channel!r} not in columns {list(df.columns)}")
    ts = df[args.channel].values.astype(np.float32)
    n = len(ts)
    starts = list(range(0, n - args.context_length - args.prediction_length + 1, args.stride))
    split_idx = int(n * 0.7)
    rows_meta = [(s, assign_split(s, args.context_length, args.prediction_length, split_idx))
                 for s in starts]
    todo = [(wid, s) for wid, (s, sp) in enumerate(rows_meta) if sp in want_splits]
    if args.max_windows:
        todo = todo[:args.max_windows]
    print(f"{len(starts)} total windows; forecasting {len(todo)} "
          f"(splits={sorted(want_splits)}) on {args.device}")

    dev = args.device
    dtype = torch.float32 if dev != "cuda" else torch.bfloat16
    pipe = ChronosPipeline.from_pretrained(args.model, device_map=dev, dtype=dtype)

    out_rows = []
    import time
    t0 = time.time()
    for bi in range(0, len(todo), args.batch_size):
        batch = todo[bi:bi + args.batch_size]
        ctx = torch.stack([torch.tensor(ts[s:s + args.context_length]) for _, s in batch])
        with torch.no_grad():
            fc = pipe.predict(ctx, prediction_length=args.prediction_length,
                              num_samples=args.num_samples)
        fc = fc.cpu().numpy() if torch.is_tensor(fc) else np.asarray(fc)
        for j, (wid, s) in enumerate(batch):
            samples = fc[j]  # (num_samples, pred_len)
            truth = ts[s + args.context_length: s + args.context_length + args.prediction_length]
            q05, q95 = np.quantile(samples, [0.05, 0.95], axis=0)
            q10, q90 = np.quantile(samples, [0.10, 0.90], axis=0)
            cqr_step = np.maximum(q05 - truth, truth - q95)  # >0 == outside band
            out_rows.append({
                "window_id": wid,
                "start_ts": s,
                "split": assign_split(s, args.context_length, args.prediction_length, split_idx),
                "crps_recomputed": float(compute_crps(samples, truth)),
                "band_width_90": float(np.mean(q95 - q05)),
                "band_width_80": float(np.mean(q90 - q10)),
                "std_mean": float(np.mean(samples.std(axis=0))),
                "cqr_R": float(np.max(cqr_step)),
                "cover90_raw": float(np.mean((truth >= q05) & (truth <= q95))),
            })
        done = bi + len(batch)
        rate = done / (time.time() - t0)
        print(f"  {done}/{len(todo)}  ({rate:.2f} win/s, "
              f"eta {(len(todo)-done)/max(rate,1e-9):.0f}s)", flush=True)

    out = pd.DataFrame(out_rows)
    out_path = args.out or f"eval/results/uncertainty_{args.dataset}_{args.model.split('/')[-1]}.parquet"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_parquet(out_path)
    print(f"\nSaved {out_path}  ({len(out)} windows, {time.time()-t0:.0f}s total)")

    # Reproducibility check vs the committed labels, if available.
    meta_path = f"activations/{args.dataset}_metadata.parquet"
    if os.path.exists(meta_path) and not args.max_windows:
        meta = pd.read_parquet(meta_path)[["start_ts", "crps_raw", "split"]]
        m = out.merge(meta, on="start_ts", suffixes=("", "_orig"))
        if len(m) > 5:
            from scipy.stats import pearsonr, spearmanr
            pr = pearsonr(m.crps_recomputed, m.crps_raw)[0]
            sr = spearmanr(m.crps_recomputed, m.crps_raw)[0]
            print(f"Reproducibility vs crps_raw: Pearson={pr:.3f} Spearman={sr:.3f} "
                  f"mean|Δ|={np.mean(np.abs(m.crps_recomputed-m.crps_raw)):.4f} "
                  f"(n={len(m)})")
    elif os.path.exists(meta_path):
        meta = pd.read_parquet(meta_path)[["start_ts", "crps_raw"]]
        m = out.merge(meta, on="start_ts")
        if len(m) > 2:
            print("crps_recomputed vs crps_raw (probe slice):")
            print(m[["start_ts", "crps_recomputed", "crps_raw"]].to_string(index=False))


if __name__ == "__main__":
    main()
