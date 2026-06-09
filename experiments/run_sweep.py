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
import shutil
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
# Backbone short-name -> HuggingFace id.
MODEL_IDS = {
    "small": "amazon/chronos-t5-small",
    "base":  "amazon/chronos-t5-base",
    "large": "amazon/chronos-t5-large",
}

# probe.py writes these fixed paths; we copy the JSON out after each run.
PROBE_JSON = os.path.join(REPO, "probing", "results", "probe_results.json")


def _run(cmd, dry_run):
    print("  $ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=REPO)


def condition_tag(dataset, model, hook, layer, seed):
    layer_s = "mid" if layer is None else f"L{layer}"
    return f"{dataset}_{model}_{hook}_{layer_s}_s{seed}"


def run_condition(dataset, model, hook, layer, seed, args):
    tag = condition_tag(dataset, model, hook, layer, seed)
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

    extract = [PY, "extract_activations.py",
               "--dataset", dataset, "--url", DATASET_URLS[dataset],
               "--model", MODEL_IDS[model], "--output_dir", acts_dir,
               "--hook_target", hook, "--seed", str(seed)]
    if layer is not None:
        extract += ["--layer_idx", str(layer)]

    train = [PY, "sae/train_sae.py",
             "--activations", acts_file, "--metadata", meta_file,
             "--output_dir", ckpt_dir, "--seed", str(seed)]

    probe = [PY, "probing/probe.py",
             "--metadata", meta_file, "--activations", acts_file,
             "--sae_ckpt", ckpt_file]

    _run(extract, args.dry_run)
    _run(train, args.dry_run)
    _run(probe, args.dry_run)

    if not args.dry_run:
        if not os.path.exists(PROBE_JSON):
            raise SystemExit(f"[sweep] expected {PROBE_JSON} after probe; not found.")
        shutil.copy2(PROBE_JSON, out_json)
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
        rows.append({"tag": condition_tag(dataset, model, hook, layer, seed),
                     "json": jpath, "dataset": dataset, "model": model,
                     "hook": hook, "layer": ("mid" if layer is None else layer),
                     "seed": seed})

    if not args.dry_run:
        aggregate(rows, args.out_root)
    else:
        print("\n(no aggregation in --dry-run; drop the flag to execute and summarize)")


if __name__ == "__main__":
    main()
