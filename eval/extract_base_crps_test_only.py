"""Test-only CRPS extraction for chronos-t5-base.

The cascade only needs base's per-window CRPS on the TEST split; computing it
on train/purge windows wastes hours. This script reads the existing small
metadata, filters to split='test', runs chronos-t5-base's predict (num_samples
matched to the small extraction so CRPS values are comparable), and writes
activations_base/ETTh1_metadata.parquet with [window_id, start_ts, split,
crps_raw] for the test windows only.

We deliberately do NOT compute activations here — the cascade doesn't need
them, and skipping the encoder hook + activation storage cuts both wall time
and peak memory.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from tqdm import tqdm

from extract_activations import compute_crps  # reuse identical CRPS implementation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ETTh1")
    ap.add_argument("--url",
                    default="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv")
    ap.add_argument("--target_col", default="OT")
    ap.add_argument("--small_metadata",
                    default="activations/ETTh1_metadata.parquet")
    ap.add_argument("--model", default="amazon/chronos-t5-base")
    ap.add_argument("--context_length", type=int, default=512)
    ap.add_argument("--prediction_length", type=int, default=96)
    ap.add_argument("--num_samples", type=int, default=100,
                    help="Match the small extraction's num_samples for fair CRPS comparison.")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--output_dir", default="activations_base")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading small metadata from {args.small_metadata} ...")
    meta = pd.read_parquet(args.small_metadata)
    test = meta[meta["split"] == "test"].copy().reset_index(drop=True)
    if len(test) == 0:
        sys.exit("[base_crps] no test windows in small metadata.")
    print(f"  {len(test)} test windows to score")

    print(f"Loading series from {args.url} ...")
    df = pd.read_csv(args.url)
    ts_data = df[args.target_col].values.astype(np.float64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading {args.model} on {device} (dtype={dtype}) ...")
    pipeline = ChronosPipeline.from_pretrained(args.model, device_map=device, dtype=dtype)

    rows = []
    starts = test["start_ts"].values.astype(int)
    n = len(starts)
    for i in tqdm(range(0, n, args.batch_size)):
        bs = starts[i:i + args.batch_size]
        ctx = []
        truth = []
        for s in bs:
            ctx.append(torch.tensor(ts_data[s:s + args.context_length], dtype=torch.float32))
            truth.append(ts_data[s + args.context_length:
                                  s + args.context_length + args.prediction_length])
        batch_ts = torch.stack(ctx)

        with torch.no_grad():
            f = pipeline.predict(batch_ts,
                                 prediction_length=args.prediction_length,
                                 num_samples=args.num_samples)
        f = f.cpu().numpy() if torch.is_tensor(f) else np.asarray(f)

        for j, s in enumerate(bs):
            crps = compute_crps(f[j], truth[j])
            rows.append({
                "window_id": int(test.iloc[i + j]["window_id"]),
                "start_ts": int(s),
                "split": "test",
                "crps_raw": float(crps),
            })

    out = pd.DataFrame(rows)
    print(f"\nbase CRPS  mean={out['crps_raw'].mean():.4f}  std={out['crps_raw'].std():.4f}")
    print(f"small CRPS mean={meta.loc[meta['split']=='test','crps_raw'].mean():.4f}  "
          f"(reference)")

    path = os.path.join(args.output_dir, "ETTh1_metadata.parquet")
    out.to_parquet(path)
    print(f"Saved {path}  ({len(out)} test windows)")


if __name__ == "__main__":
    main()
