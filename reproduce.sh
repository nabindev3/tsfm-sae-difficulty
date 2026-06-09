#!/usr/bin/env bash
# One-command pipeline. Stops at the first failure. The probe REFUSES to run on
# unlabelled data, so this also enforces the correct order.
set -euo pipefail
# Interpreter lives OUTSIDE the project tree (see README "Running"). Override
# with e.g. PY=./venv/bin/python if you keep one locally.
PY=${PY:-$HOME/.venvs/tsfm-sae-difficulty/bin/python}

echo "== [1/5] smoke test =="
$PY smoke_test.py

echo "== [2/5] extract activations + CRPS/MASE labels (FULL series) =="
echo "   NOTE: using chronos-t5-small for CPU compute."
$PY extract_activations.py --dataset ETTh1 --model amazon/chronos-t5-small --batch_size 4

echo "== [3/5] train SAE on the TRAIN split only =="
$PY sae/train_sae.py --activations activations/ETTh1_activations.safetensors \
    --metadata activations/ETTh1_metadata.parquet --epochs 5

echo "== [4/5] difficulty probe (headline result) =="
$PY probing/probe.py --sae_ckpt sae/checkpoints/sae_topk_32.pt

echo "== [5/5] feature visualization =="
$PY probing/visualize_features.py --sae_ckpt sae/checkpoints/sae_topk_32.pt

echo "== [6/7] selective-prediction analysis (positive result) =="
$PY eval/selective_prediction.py

echo "== [7/8] populate report =="
$PY eval/populate_report.py

echo "== [8/8] steering demo (clamp top-5 features on confident windows) =="
echo "   ~4 min on CPU (24 chronos forecasts); set STEERING=0 to skip."
if [ "${STEERING:-1}" != "0" ]; then
    $PY eval/steering_demo.py
fi

echo
echo "Done. Results in probing/results/. "
echo "Target-tier cascade (optional): re-run step 2 with --model amazon/chronos-t5-small"
echo "  --output_dir activations_small, then:"
echo "  \$PY eval/cascade.py --cheap_meta activations_small/ETTh1_metadata.parquet \\"
echo "       --base_meta activations/ETTh1_metadata.parquet"
