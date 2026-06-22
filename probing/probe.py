import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from sklearn.model_selection import TimeSeriesSplit
import warnings
# NOTE: scipy and statsmodels are imported lazily inside the two functions that
# use them (compute_spectral_entropy / compute_input_stats), so importing this
# module for a single helper does not pay the statsmodels import cost. This
# mirrors the unified modalities/tsfm.py adapter in fm-difficulty-probe.

# Suppress only the known-benign notices instead of blanket-ignoring every
# warning: statsmodels' ADF/ACF helpers warn on short or near-constant windows,
# and the shared core.probe ladder's liblinear fit emits a benign n_jobs notice.
# Real deprecation/data warnings stay visible. (Mirrors core.probe's scoped
# filter in fm-difficulty-probe.)
warnings.filterwarnings("ignore", message=".*n_jobs.*liblinear.*")
warnings.filterwarnings("ignore", module="statsmodels")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(_REPO_ROOT, 'sae'))
sys.path.append(_REPO_ROOT)
from sae_model import TopKSAE
# Shared, unit-tested probe ladder from the fm-difficulty-probe `core` package
# (installed editable: `pip install -e ../fm-difficulty-probe --no-deps`). The
# ladder fit + paired bootstrap live there so they are tested on synthetic numpy
# arrays with no model/network; see tests/test_core_synthetic.py.
from core.probe import build_ladder, run_probe_ladder, P1, P2, P3, P4, P5

def compute_spectral_entropy(ts):
    import scipy.signal
    import scipy.stats
    f, Pxx = scipy.signal.welch(ts)
    if np.sum(Pxx) == 0:
        return 0.0
    Pxx = Pxx / np.sum(Pxx)
    return scipy.stats.entropy(Pxx)

def compute_input_stats(df_meta, context_length=512, season_length=24,
                        series_url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
                        channel="OT"):
    """Eight classical features per window. The report claims eight; the prior
    version computed four. The missing ones (volatility, seasonal autocorr,
    trend slope, range) are precisely the ones a regime-shift detector would
    target, so they need to be in the baseline for an honest comparison.

    series_url/channel MUST match the series the activations were extracted from:
    `start_ts` indexes into this array, so a mismatch silently computes the
    baseline from the wrong series (the previous hardcoded ETTh1/OT would corrupt
    every non-ETTh1 / non-OT run)."""
    from statsmodels.tsa.stattools import acf, adfuller
    df_raw = pd.read_csv(series_url)
    if channel not in df_raw.columns:
        sys.exit(f"[probe] channel {channel!r} not in {series_url} columns {list(df_raw.columns)}")
    ts_data = df_raw[channel].values.astype(np.float64)

    stats = []
    for _, row in df_meta.iterrows():
        start = int(row["start_ts"])
        x = ts_data[start:start + context_length]
        n = len(x)

        var = float(np.var(x))
        volatility = float(np.mean(np.abs(np.diff(x)))) if n > 1 else 0.0
        acf_vals = acf(x, nlags=max(1, season_length), fft=False) if n > season_length else np.zeros(season_length + 1)
        lag1_acf = float(acf_vals[1]) if len(acf_vals) > 1 else 0.0
        seasonal_acf = float(acf_vals[season_length]) if len(acf_vals) > season_length else 0.0
        try:
            adf_p = float(adfuller(x, autolag="AIC")[1])
        except Exception:
            adf_p = 1.0
        entropy = compute_spectral_entropy(x)
        trend_slope = float(np.polyfit(np.arange(n), x, 1)[0]) if n > 1 else 0.0
        rng = float(x.max() - x.min())

        stats.append([var, volatility, lag1_acf, seasonal_acf,
                      adf_p, entropy, trend_slope, rng])
    return np.array(stats)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="activations/ETTh1_metadata.parquet")
    parser.add_argument("--activations", type=str, default="activations/ETTh1_activations.safetensors")
    parser.add_argument("--sae_ckpt", type=str, default="sae/checkpoints/sae_topk_32.pt")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_hidden", type=int, default=4096)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--series_url",
                        default="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
                        help="CSV the windows index into for input stats; MUST match the extraction series.")
    parser.add_argument("--channel", default="OT", help="Series column for input stats (match extraction).")
    parser.add_argument("--season_length", type=int, default=24,
                        help="Seasonal period for the seasonal-ACF stat (24=hourly ETTh, 96=15-min ETTm).")
    parser.add_argument("--scores_out", default="activations/probe_scores.parquet",
                        help="Per-test-window probe scores output path.")
    parser.add_argument("--results_out", default="probing/results/probe_results.json",
                        help="Probe results JSON output path.")
    args = parser.parse_args()

    print("Loading data...")
    df_meta = pd.read_parquet(args.metadata)
    tensors = load_file(args.activations)
    raw_acts = tensors["encoder_embeddings"] # (batch, seq, d_model)
    
    print("Computing input statistics...")
    input_stats = compute_input_stats(df_meta, season_length=args.season_length,
                                      series_url=args.series_url, channel=args.channel)
    
    print("Loading SAE...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Hard-fail if the checkpoint can't be loaded. The probe must NEVER silently
    # run on a random SAE -- the resulting numbers would be noise but look
    # identical to a real result in the JSON and the report.
    if not os.path.exists(args.sae_ckpt):
        sys.exit(f"[probe] SAE checkpoint '{args.sae_ckpt}' not found. "
                 f"Train the SAE first; refusing to probe with random weights.")
    state = torch.load(args.sae_ckpt, map_location=device, weights_only=True)
    if "W_enc" not in state:
        sys.exit(f"[probe] '{args.sae_ckpt}' is not a TopKSAE checkpoint (no W_enc).")
    d_model_ckpt, d_hidden_ckpt = state["W_enc"].shape
    print(f"Auto-detected SAE dims from checkpoint: d_model={d_model_ckpt}, d_hidden={d_hidden_ckpt}")
    sae = TopKSAE(d_model=d_model_ckpt, d_hidden=d_hidden_ckpt, k=args.k).to(device)
    sae.load_state_dict(state)
    sae.eval()
    
    print("Aggregating activations...")
    # Aggregation: concat(mean, max, last-token) per feature
    raw_mean = raw_acts.mean(dim=1).numpy()
    raw_max = raw_acts.max(dim=1).values.numpy()
    raw_last = raw_acts[:, -1, :].numpy()
    raw_agg = np.concatenate([raw_mean, raw_max, raw_last], axis=1)
    
    sae_acts_list = []
    with torch.no_grad():
        for i in range(raw_acts.shape[0]):
            x = raw_acts[i:i+1].to(device).to(torch.float32)
            acts, _, _ = sae(x)
            sae_acts_list.append(acts.cpu())
    sae_acts = torch.cat(sae_acts_list, dim=0) # (batch, seq, d_hidden)
    
    sae_mean = sae_acts.mean(dim=1).numpy()
    sae_max = sae_acts.max(dim=1).values.numpy()
    sae_last = sae_acts[:, -1, :].numpy()
    sae_agg = np.concatenate([sae_mean, sae_max, sae_last], axis=1)
    
    # Label definition: Top 25% CRPS in test set is "hard"
    threshold = df_meta['crps_norm'].quantile(0.75)
    y = (df_meta['crps_norm'] >= threshold).astype(int).values
    
    train_mask = (df_meta['split'] == 'train').values
    test_mask = (df_meta['split'] == 'test').values
    
    if test_mask.sum() == 0 or train_mask.sum() == 0:
        print("Not enough train/test split data. Need full extraction.")
        return
    
    print(f"Train samples: {train_mask.sum()}, Test samples: {test_mask.sum()}")
    
    y_train, y_test = y[train_mask], y[test_mask]
    
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print("Warning: Only one class present in train or test split. AUROC cannot be computed properly.")
        
    # Assemble the five rungs from the three building blocks. The diagnostic
    # rungs P4/P5 isolate where signal lives: if P4 ~ chance, raw activations
    # carry no difficulty signal; if P5 ~ chance, SAE features carry none; if
    # P4/P5 are non-trivial but P2/P3 still lose to P1, input-stats are
    # DOMINATING the L1 logistic when concatenated.
    features = build_ladder(input_stats, raw_agg, sae_agg)

    # Inner CV uses TimeSeriesSplit so consecutive (overlapping) windows do not
    # leak across folds when picking C. The outer temporal/purge split already
    # protects the test set, but a shuffled inner CV would still bias the
    # chosen regularization toward overfitting. Pre-materialize the folds over
    # the TRAIN rows and hand them to the shared (modality-agnostic) ladder.
    n_splits = max(2, min(5, int(np.bincount(y_train).min()) - 1, train_mask.sum() // 3))
    cv_splits = list(TimeSeriesSplit(n_splits=n_splits).split(np.zeros((train_mask.sum(), 1)), y_train))

    print("Fitting probe ladder (shared core.probe.run_probe_ladder)...")
    ladder, preds = run_probe_ladder(
        features, y, train_mask, test_mask, cv_splits,
        # 1e-4 ... 1.0 (core default) covers the high-dim sparse-feature regime;
        # the prior 0.01 lower bound left CV pinned to the top of the grid.
        n_boot=2000, seed=42,
    )

    for name, pr in ladder.probes.items():
        print(f"  {name} point AUROC = {pr.auroc:.3f}  (C={pr.best_C})  "
              f"95% CI [{pr.ci_low:.3f}, {pr.ci_high:.3f}]")
    d_raw = ladder.deltas[f"{P2}-{P1}"]
    d_sae = ladder.deltas[f"{P3}-{P1}"]
    d_sor = ladder.deltas[f"{P3}-{P2}"]
    print("\n--- Incremental Predictive Power (ΔAUROC, paired bootstrap) ---")
    print(f"Δ Raw - Stats : {d_raw['point']:+.3f}  95% CI [{d_raw['ci_low']:+.3f}, {d_raw['ci_high']:+.3f}]")
    print(f"Δ SAE - Stats : {d_sae['point']:+.3f}  95% CI [{d_sae['ci_low']:+.3f}, {d_sae['ci_high']:+.3f}]")
    print(f"Δ SAE - Raw   : {d_sor['point']:+.3f}  95% CI [{d_sor['ci_low']:+.3f}, {d_sor['ci_high']:+.3f}]")

    # Canonical core rung names -> the legacy column/JSON labels the rest of the
    # pipeline (eval/selective_prediction.py, cascade.py, calibration.py,
    # populate_report.py) reads by name. Keep this map in sync with those.
    LEGACY = {P1: "P1_InputStats", P2: "P2_InputStats_Raw",
              P3: "P3_InputStats_SAE", P4: "P4_RawOnly", P5: "P5_SAEOnly"}

    df_test = df_meta[test_mask].copy()
    for name, p in preds.items():
        df_test[f"pred_{LEGACY[name]}"] = p
    os.makedirs(os.path.dirname(args.scores_out) or ".", exist_ok=True)
    df_test.to_parquet(args.scores_out)
    print(f"\nSaved {args.scores_out}")

    # Save probe results JSON
    import json
    os.makedirs(os.path.dirname(args.results_out) or ".", exist_ok=True)

    final_results = {
        "n_total": ladder.n_total,
        "n_train": ladder.n_train,
        "n_test": ladder.n_test,
        "hard_fraction": ladder.hard_fraction,
        "P1_AUROC": ladder.probes[P1].auroc,
        "P1_CI_lower": ladder.probes[P1].ci_low,
        "P1_CI_upper": ladder.probes[P1].ci_high,
        "P2_AUROC": ladder.probes[P2].auroc,
        "P2_CI_lower": ladder.probes[P2].ci_low,
        "P2_CI_upper": ladder.probes[P2].ci_high,
        "P3_AUROC": ladder.probes[P3].auroc,
        "P3_CI_lower": ladder.probes[P3].ci_low,
        "P3_CI_upper": ladder.probes[P3].ci_high,
        "delta_raw": d_raw["point"],
        "delta_raw_CI_lower": d_raw["ci_low"],
        "delta_raw_CI_upper": d_raw["ci_high"],
        "delta_sae": d_sae["point"],
        "delta_sae_CI_lower": d_sae["ci_low"],
        "delta_sae_CI_upper": d_sae["ci_high"],
        "delta_sae_over_raw": d_sor["point"],
        "delta_sae_over_raw_CI_lower": d_sor["ci_low"],
        "delta_sae_over_raw_CI_upper": d_sor["ci_high"],
        # Diagnostic probes (not in headline table)
        "P4_RawOnly_AUROC": ladder.probes[P4].auroc,
        "P4_RawOnly_CI_lower": ladder.probes[P4].ci_low,
        "P4_RawOnly_CI_upper": ladder.probes[P4].ci_high,
        "P5_SAEOnly_AUROC": ladder.probes[P5].auroc,
        "P5_SAEOnly_CI_lower": ladder.probes[P5].ci_low,
        "P5_SAEOnly_CI_upper": ladder.probes[P5].ci_high,
        "chosen_C": {LEGACY[name]: pr.best_C for name, pr in ladder.probes.items()},
    }
    with open(args.results_out, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Saved {args.results_out}")

if __name__ == "__main__":
    main()
