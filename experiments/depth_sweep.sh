#!/usr/bin/env bash
# Full encoder-depth sweep (Exp9): does the SAE-vs-input-stats null hold at EVERY
# encoder block, not just mid (block 3) and late (block 5) as in §4.2?
#
# Cheap by construction: activations are extracted embed-only (--skip_predict),
# reusing the committed ETTh1 CRPS labels -- forecast difficulty does not depend
# on which layer we hook -- so NO forecast sampling runs. Each layer = one fast
# embed pass + SAE train + probe. Resumable: a layer whose probe JSON exists is
# skipped. The throttling note (memory) applies: launch under caffeinate on AC,
#   caffeinate -i bash experiments/depth_sweep.sh
set -euo pipefail
PY=${PY:-$HOME/.venvs/tsfm-sae-difficulty/bin/python}
MODEL=${MODEL:-amazon/chronos-t5-small}
LABELS=${LABELS:-activations/ETTh1_metadata.parquet}   # canonical labelled metadata (crps_norm + split)
LAYERS=${LAYERS:-"0 1 2 3 4 5"}                          # small T5 encoder: blocks 0..5
mkdir -p probing/results/depth logs

[ -f "$LABELS" ] || { echo "missing $LABELS (run the headline ETTh1 extraction first)"; exit 1; }

for L in $LAYERS; do
  outjson=probing/results/depth/layer${L}.json
  if [ -f "$outjson" ]; then echo "[skip] layer $L (have $outjson)"; continue; fi
  adir=activations_depth/L${L}; cdir=sae/checkpoints_depth/L${L}
  acts=$adir/ETTh1_activations.safetensors
  echo "== layer $L: embed-only extract (no predict) =="
  $PY extract_activations.py --dataset ETTh1 --model "$MODEL" \
      --layer_idx "$L" --skip_predict --output_dir "$adir" --batch_size 8 \
      > logs/depth_L${L}_extract.log 2>&1
  echo "== layer $L: train SAE on the train split =="
  $PY sae/train_sae.py --activations "$acts" --metadata "$LABELS" \
      --output_dir "$cdir" > logs/depth_L${L}_sae.log 2>&1
  echo "== layer $L: probe (reusing committed labels) =="
  $PY probing/probe.py --metadata "$LABELS" --activations "$acts" \
      --sae_ckpt "$cdir/sae_topk_32.pt" \
      --scores_out "$adir/probe_scores.parquet" --results_out "$outjson" \
      > logs/depth_L${L}_probe.log 2>&1
  echo "   -> $outjson"
done

echo "== aggregate depth sweep =="
$PY - <<'EOF'
import json, glob, os
rows=[]
for p in sorted(glob.glob("probing/results/depth/layer*.json")):
    d=json.load(open(p)); L=int(os.path.basename(p)[5:-5])
    rows.append(dict(layer=L, P1=d.get("P1_AUROC"), P2=d.get("P2_AUROC"),
                     P3=d.get("P3_AUROC"), delta_sae=d.get("delta_sae"),
                     delta_sae_over_raw=d.get("delta_sae_over_raw")))
print(f"{'layer':>5} {'P1':>7} {'P2':>7} {'P3':>7} {'dSAE-stats':>11} {'dSAE-raw':>10}")
for r in rows:
    g=lambda k: (f"{r[k]:+.3f}" if isinstance(r[k],(int,float)) else "  n/a")
    print(f"{r['layer']:>5} {r['P1'] or 0:7.3f} {r['P2'] or 0:7.3f} {r['P3'] or 0:7.3f} "
          f"{g('delta_sae'):>11} {g('delta_sae_over_raw'):>10}")
json.dump(rows, open("probing/results/depth/depth_sweep_summary.json","w"), indent=2)
print("\nSaved probing/results/depth/depth_sweep_summary.json")
EOF
