import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from safetensors.torch import save_file
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_crps(samples, truth):
    # samples: (num_samples, prediction_length)
    # truth: (prediction_length,)
    crps_vals = []
    for i in range(len(truth)):
        s = samples[:, i]
        t = truth[i]
        mae = np.mean(np.abs(s - t))
        diffs = np.abs(s[:, None] - s[None, :])
        mean_diff = np.mean(diffs)
        crps_vals.append(mae - 0.5 * mean_diff)
    return np.mean(crps_vals)

def compute_mase(forecast_mean, truth, context, season_length=24):
    # forecast_mean: (prediction_length,)
    # truth: (prediction_length,)
    # context: (context_length,)
    # Seasonal-naive scaling (m=season_length) is the M-competition / Chronos-paper
    # standard for seasonal data (ETTh is hourly with strong daily seasonality, m=24).
    # Using lag-1 here would make MASE non-comparable to published numbers and
    # would change which windows are labelled "hard".
    mae = np.mean(np.abs(forecast_mean - truth))
    if len(context) > season_length:
        naive_errs = np.abs(context[season_length:] - context[:-season_length])
    else:
        naive_errs = np.abs(np.diff(context))
    naive_mae = np.mean(naive_errs)
    if naive_mae == 0 or np.isnan(naive_mae):
        naive_mae = 1e-5
    return mae / naive_mae

def extract_and_cache(dataset_name, url, model_id, context_length, prediction_length, stride, batch_size, output_dir, max_batches, season_length=24, layer_idx=None, skip_predict=False, hook_target="residual", channel="OT", num_samples=100):
    print(f"Loading dataset {dataset_name} (channel={channel})...")
    df = pd.read_csv(url)
    if channel not in df.columns:
        raise SystemExit(f"channel {channel!r} not in {url} columns {list(df.columns)}")
    ts_data = df[channel].values
    
    total_length = len(ts_data)
    # We need both context and horizon
    valid_starts = list(range(0, total_length - context_length - prediction_length + 1, stride))
    print(f"Found {len(valid_starts)} windows with context_length={context_length}, horizon={prediction_length}, and stride={stride}")

    # Define temporal split point (e.g. 70% train)
    split_idx = int(total_length * 0.7)

    print(f"Loading model {model_id}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bf16 on CPU is slow and not universally supported; use fp32 on CPU.
    model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipeline = ChronosPipeline.from_pretrained(
        model_id,
        device_map=device,
        dtype=model_dtype,
    )
    
    # Global state to capture hook outputs
    captured_acts = []
    def hook(module, input, output):
        # T5LayerFF returns a tuple where the first element is the hidden states
        hidden_states = output[0] if isinstance(output, tuple) else output
        captured_acts.append(hidden_states.detach().cpu().to(torch.float16))
        
    # Register hook on chosen encoder block. Default layer is mid (num_layers //
    # 2); pass layer_idx to probe early/late layers. A T5 encoder block's
    # `.layer` is [T5LayerSelfAttention, T5LayerFF]; hook_target selects which
    # sub-layer's residual output we capture:
    #   "residual"  -> layer[-1] (post-FF residual stream; the headline target)
    #   "attention" -> layer[0]  (post-self-attention residual)
    num_layers = pipeline.model.model.config.num_layers
    chosen_layer = (num_layers // 2) if layer_idx is None else int(layer_idx)
    if not (0 <= chosen_layer < num_layers):
        raise ValueError(f"layer_idx {chosen_layer} out of range [0, {num_layers})")
    sublayer_idx = {"residual": -1, "attention": 0}[hook_target]
    print(f"Hooking encoder.block[{chosen_layer}].layer[{sublayer_idx}] "
          f"(target={hook_target}, num_layers={num_layers})")
    handle = pipeline.model.model.encoder.block[chosen_layer].layer[sublayer_idx].register_forward_hook(hook)
    
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = []
    all_embeddings = []
    
    print("Extracting activations and computing metrics...")
    window_id = 0
    # Process in batches
    for i in tqdm(range(0, len(valid_starts), batch_size)):
        if max_batches is not None and i // batch_size >= max_batches:
            break
            
        batch_starts = valid_starts[i:i+batch_size]
        batch_ts = []
        batch_truths = []
        
        for start in batch_starts:
            end_context = start + context_length
            end_horizon = end_context + prediction_length
            
            context_ts = ts_data[start:end_context]
            horizon_ts = ts_data[end_context:end_horizon]
            
            batch_ts.append(torch.tensor(context_ts, dtype=torch.float32))
            batch_truths.append(horizon_ts)
            
        # Clear captured acts
        captured_acts.clear()
        
        # Convert list to 2D tensor to ensure batched execution in the pipeline
        batch_ts_tensor = torch.stack(batch_ts)
        
        with torch.no_grad():
            # Run embed to trigger the encoder hook exactly once per window (no num_samples duplication)
            pipeline.embed(batch_ts_tensor)

            # The hook should have captured exactly one tensor of shape (batch, seq_len, d_model)
            batch_embeddings = torch.cat(captured_acts, dim=0)
            all_embeddings.append(batch_embeddings)

            if skip_predict:
                forecasts = None
            else:
                # predict() is where ~all the wall time goes (num_samples=100
                # sampling). Skip it when we only need activations from a
                # different layer and can reuse existing CRPS labels.
                captured_acts.clear()
                forecasts = pipeline.predict(
                    batch_ts_tensor,
                    prediction_length=prediction_length,
                    num_samples=num_samples
                )
                if torch.is_tensor(forecasts):
                    forecasts = forecasts.cpu().numpy()
                else:
                    forecasts = np.asarray(forecasts)

        # Build metadata. When --skip_predict, we only have start_ts + split.
        for j, start in enumerate(batch_starts):
            end_horizon = start + context_length + prediction_length
            if end_horizon <= split_idx:
                split = "train"
            elif start >= split_idx + context_length + prediction_length:
                split = "test"
            else:
                split = "purge"

            row = {
                "window_id": window_id,
                "dataset": dataset_name,
                "start_ts": start,
                "split": split,
            }
            if not skip_predict:
                samples = forecasts[j]
                forecast_mean = np.mean(samples, axis=0)
                truth = batch_truths[j]
                context = batch_ts[j].numpy()
                row["crps_raw"] = compute_crps(samples, truth)
                row["mase"] = compute_mase(forecast_mean, truth, context, season_length=season_length)
            metadata.append(row)
            window_id += 1

    # Remove hook
    handle.remove()

    # Concatenate all embeddings
    print("Concatenating and saving...")
    final_tensor = torch.cat(all_embeddings, dim=0)
    print(f"Final embeddings shape: {final_tensor.shape}")
    
    # Save with safetensors
    safetensors_path = os.path.join(output_dir, f"{dataset_name}_activations.safetensors")
    save_file({"encoder_embeddings": final_tensor}, safetensors_path)
    print(f"Saved activations to {safetensors_path}")
    
    # Save metadata with Parquet
    df_meta = pd.DataFrame(metadata)
    
    print("Split counts:\n", df_meta['split'].value_counts().to_string())
    train_rows = df_meta['split'] == "train"
    if train_rows.sum() < 2:
        raise RuntimeError(
            f"Only {int(train_rows.sum())} train windows — series too short for this "
            f"context/horizon/stride. Lower stride or use a longer series."
        )
    if "crps_raw" in df_meta.columns:
        # Normalize CRPS using TRAIN-split stats only. Test stats would leak.
        crps_mean = df_meta.loc[train_rows, 'crps_raw'].mean()
        crps_std = df_meta.loc[train_rows, 'crps_raw'].std()
        df_meta['crps_norm'] = (df_meta['crps_raw'] - crps_mean) / (crps_std + 1e-8)
    else:
        print("NOTE: --skip_predict was set; metadata has no CRPS/MASE columns. "
              "Downstream tools must point --metadata at a labeled metadata file "
              "from a prior full extraction (same windows / start_ts).")
    
    parquet_path = os.path.join(output_dir, f"{dataset_name}_metadata.parquet")
    df_meta.to_parquet(parquet_path, engine='pyarrow')
    print(f"Saved metadata to {parquet_path}")
    print(f"Processed {window_id} windows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract activations from Chronos")
    parser.add_argument("--dataset", type=str, default="ETTh1", help="Dataset name")
    parser.add_argument("--url", type=str, default="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv", help="Dataset URL or path")
    parser.add_argument("--model", type=str, default="amazon/chronos-t5-base", help="Model ID")
    parser.add_argument("--context_length", type=int, default=512, help="Context length")
    parser.add_argument("--prediction_length", type=int, default=96, help="Prediction length / horizon")
    parser.add_argument("--stride", type=int, default=24, help="Stride for sliding window")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for extraction")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches to process (for testing)")
    parser.add_argument("--season_length", type=int, default=24, help="Seasonal period for MASE scaling (24 = daily, hourly data)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="activations", help="Output directory")
    parser.add_argument("--layer_idx", type=int, default=None, help="Encoder block index to hook (default mid = num_layers // 2)")
    parser.add_argument("--hook_target", type=str, default="residual", choices=["residual", "attention"],
                        help="Which sub-layer residual to capture: 'residual' (post-FF, default) or 'attention' (post-self-attention)")
    parser.add_argument("--skip_predict", action="store_true", help="Skip CRPS/MASE labelling (fast layer-only extraction; reuse labels from a prior full run)")
    parser.add_argument("--channel", type=str, default="OT", help="CSV column to forecast (default OT; ETTh1 also has HUFL,HULL,MUFL,MULL,LUFL,LULL)")
    parser.add_argument("--num_samples", type=int, default=100, help="Chronos forecast samples per window for CRPS/MASE labels (100=headline fidelity; lower e.g. 20-30 is ~5x faster and keeps the hard/easy ranking robust for replication runs)")

    args = parser.parse_args()
    set_seed(args.seed)

    extract_and_cache(
        dataset_name=args.dataset,
        url=args.url,
        model_id=args.model,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        max_batches=args.max_batches,
        season_length=args.season_length,
        layer_idx=args.layer_idx,
        skip_predict=args.skip_predict,
        hook_target=args.hook_target,
        channel=args.channel,
        num_samples=args.num_samples,
    )
