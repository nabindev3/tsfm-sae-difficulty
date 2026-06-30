"""Multi-condition sweep runner: datasets × backbones × seeds × hook-targets.

This turns the "future work" row of the README (multi-dataset / multi-backbone /
seeds / attention) from a scaffold into runnable infrastructure. It does NOT
invent any numbers — it shells out to the same real pipeline the headline run
uses (extract_activations.py -> sae/train_sae.py -> probing/probe.py), one
isolated output directory per condition, then aggregates the genuine
`probe_results.json` each probe writes into a single tidy table.

The actual compute is heavy (each condition is a full extraction + SAE train +
probe; on GPU this is the right place to run it). Use `--dry-run` first to print
the exact commands without executing, then drop it to run for real. Resume-safe:
a condition whose `probe_results.json` already exists is skipped unless --force.

Examples
--------
    # See the plan (no compute):
    python experiments/run_sweep.py --dry-run \
        --datasets ETTh1 ETTh2 --models small base --seeds 42 7 \
        --hook-targets residual attention

    # Run a single extra seed on the headline condition for real:
    python experiments/run_sweep.py --datasets ETTh1 --models small --seeds 7
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import subprocess
from itertools import product

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable  # the venv interpreter running this script

# Known single-series CSVs (same family/source as the headline ETTh1 run).
DATASET_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
}
# Per-dataset sampling cadence. ETTh* is hourly (daily season m=24); ETTm* is
# 15-minute (daily season m=96). Stride is chosen so each series yields ~700
# windows -> comparable window counts and per-dataset compute.
DATASET_PARAMS = {
    "ETTh1": {"stride": 24, "season": 24},
    "ETTh2": {"stride": 24, "season": 24},
    "ETTm1": {"stride": 96, "season": 96},
    "ETTm2": {"stride": 96, "season": 96},
}
# Backbone short-name -> HuggingFace id.
MODEL_IDS = {
    "small": "amazon/chronos-t5-small",
    "base":  "amazon/chronos-t5-base",
    "large": "amazon/chronos-t5-large",
}


def _run(cmd, dry_run):
    print("  $ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=REPO)


def condition_tag(dataset, model, hook, layer, seed, channel="OT", horizon=96):
    layer_s = "mid" if layer is None else f"L{layer}"
    ch = "" if channel == "OT" else f"_{channel}"
    hz = "" if horizon == 96 else f"_H{horizon}"
    return f"{dataset}_{model}_{hook}_{layer_s}_s{seed}{ch}{hz}"


def run_condition(dataset, model, hook, layer, seed, args):
    tag = condition_tag(dataset, model, hook, layer, seed, args.channel, args.horizon)
    run_dir = os.path.join(args.out_root, tag)
    acts_dir = os.path.join(run_dir, "activations")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    out_json = os.path.join(run_dir, "probe_results.json")
    os.makedirs(run_dir, exist_ok=True)

    if os.path.exists(out_json) and not args.force:
        print(f"[skip] {tag}: probe_results.json already present (use --force to redo)")
        return out_json

    print(f"[run ] {tag}")
    acts_file = os.path.join(acts_dir, f"{dataset}_activations.safetensors")
    meta_file = os.path.join(acts_dir, f"{dataset}_metadata.parquet")
    ckpt_file = os.path.join(ckpt_dir, "sae_topk_32.pt")
    scores_file = os.path.join(run_dir, "probe_scores.parquet")
    p = DATASET_PARAMS[dataset]
    stride = p["stride"] * args.stride_mult  # mult>1 -> fewer windows (lean replication)

    extract = [PY, "extract_activations.py",
               "--dataset", dataset, "--url", DATASET_URLS[dataset],
               "--channel", args.channel, "--num_samples", str(args.num_samples),
               "--batch_size", str(args.batch_size), "--prediction_length", str(args.horizon),
               "--stride", str(stride), "--season_length", str(p["season"]),
               "--model", MODEL_IDS[model], "--output_dir", acts_dir,
               "--hook_target", hook, "--seed", str(seed)]
    if layer is not None:
        extract += ["--layer_idx", str(layer)]

    train = [PY, "sae/train_sae.py",
             "--activations", acts_file, "--metadata", meta_file,
             "--output_dir", ckpt_dir, "--seed", str(seed)]

    # Pass the actual series/channel/season so the P1 input-stats baseline is
    # computed from THIS dataset, not a hardcoded ETTh1/OT, and write per-run
    # outputs so conditions never clobber each other.
    probe = [PY, "probing/probe.py",
             "--metadata", meta_file, "--activations", acts_file,
             "--sae_ckpt", ckpt_file,
             "--series_url", DATASET_URLS[dataset], "--channel", args.channel,
             "--season_length", str(p["season"]),
             "--scores_out", scores_file, "--results_out", out_json]

    _run(extract, args.dry_run)
    _run(train, args.dry_run)
    _run(probe, args.dry_run)

    if not args.dry_run and not os.path.exists(out_json):
        raise SystemExit(f"[sweep] expected {out_json} after probe; not found.")
    return out_json


def aggregate(rows, out_root):
    """Collect each condition's probe_results.json into one tidy table."""
    summary = []
    for r in rows:
        tag, jpath = r["tag"], r["json"]
        rec = {"condition": tag, **{k: r[k] for k in ("dataset", "model", "hook", "layer", "seed")}}
        if os.path.exists(jpath):
            with open(jpath) as f:
                d = json.load(f)
            for key in ("n_test", "hard_fraction", "P1_AUROC", "P2_AUROC", "P3_AUROC",
                        "delta_sae", "delta_sae_over_raw"):
                rec[key] = d.get(key)
        else:
            rec["status"] = "not_run"
        summary.append(rec)

    os.makedirs(out_root, exist_ok=True)
    jout = os.path.join(out_root, "sweep_summary.json")
    with open(jout, "w") as f:
        json.dump(summary, f, indent=2)

    cout = os.path.join(out_root, "sweep_summary.csv")
    cols = ["condition", "dataset", "model", "hook", "layer", "seed", "n_test",
            "hard_fraction", "P1_AUROC", "P2_AUROC", "P3_AUROC",
            "delta_sae", "delta_sae_over_raw"]
    with open(cout, "w") as f:
        f.write(",".join(cols) + "\n")
        for rec in summary:
            f.write(",".join("" if rec.get(c) is None else str(rec.get(c)) for c in cols) + "\n")
    print(f"\nWrote {jout}\nWrote {cout}  ({len(summary)} conditions)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["ETTh1"], choices=list(DATASET_URLS))
    ap.add_argument("--models", nargs="+", default=["small"], choices=list(MODEL_IDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--channel", default="OT",
                    help="Series column to forecast (default OT; ETT also has HUFL,HULL,MUFL,MULL,LUFL,LULL).")
    ap.add_argument("--num-samples", type=int, default=100,
                    help="Chronos samples per window for CRPS labels (100=headline; 20-30 ~5x faster for replication).")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Extraction batch size. Default 8 (memory-safe on 16GB); the old default 32 x 100 samples OOM-kills.")
    ap.add_argument("--stride-mult", type=int, default=1,
                    help="Multiply the per-dataset stride to thin the window count (e.g. 2 -> ~half the windows) for lean CPU replication.")
    ap.add_argument("--horizon", type=int, default=96,
                    help="Forecast horizon (prediction_length). Default 96; sweep 24/336/720 to test horizon-dependence of the null.")
    ap.add_argument("--hook-targets", nargs="+", default=["residual"],
                    choices=["residual", "attention"])
    ap.add_argument("--layers", nargs="+", default=["mid"],
                    help="Encoder block indices, or 'mid' for num_layers//2 (the default).")
    ap.add_argument("--out-root", default=os.path.join(REPO, "experiments", "runs"))
    ap.add_argument("--force", action="store_true", help="Re-run conditions even if results exist.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = ap.parse_args()

    layers = [None if x == "mid" else int(x) for x in args.layers]
    combos = list(product(args.datasets, args.models, args.hook_targets, layers, args.seeds))
    print(f"{'DRY RUN — ' if args.dry_run else ''}{len(combos)} condition(s):\n")

    rows = []
    for dataset, model, hook, layer, seed in combos:
        jpath = run_condition(dataset, model, hook, layer, seed, args)
        rows.append({"tag": condition_tag(dataset, model, hook, layer, seed, args.channel, args.horizon),
                     "json": jpath, "dataset": dataset, "model": model,
                     "hook": hook, "layer": ("mid" if layer is None else layer),
                     "seed": seed})

    if not args.dry_run:
        aggregate(rows, args.out_root)
    else:
        print("\n(no aggregation in --dry-run; drop the flag to execute and summarize)")


if __name__ == "__main__":
    main()
