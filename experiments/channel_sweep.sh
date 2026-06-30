#!/usr/bin/env bash
# Exp4: run the lean probe pipeline on ETTh1's other six channels (the headline
# uses OT only) to test whether the difficulty signal is channel-dependent.
# Lean fidelity (num_samples=20, ~350 windows) for CPU feasibility; the OT
# channel at the same lean config comes from the Exp2 run (ETTh1_small_*_s42).
# Not set -e: one channel failing must not abort the rest.
set -uo pipefail
PY=${PY:-$HOME/.venvs/tsfm-sae-difficulty/bin/python}
for ch in HUFL HULL MUFL MULL LUFL LULL; do
  echo "=== CHANNEL $ch ==="
  "$PY" experiments/run_sweep.py --datasets ETTh1 --models small --channel "$ch" \
      --num-samples 20 --batch-size 4 --stride-mult 2 || echo "FAILED channel $ch"
done
echo "ALL CHANNELS DONE"
