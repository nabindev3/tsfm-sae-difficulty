#!/usr/bin/env bash
# Exp7: does the SAE-vs-stats null depend on forecast horizon? Sweep
# H in {24, 96, 336, 720} on ETTh1 OT (small, lean num_samples=20, stride x2).
# H=96 is the Exp2 lean run (same tag) and is resume-skipped. H=720 is the
# slowest (7.5x the decode of H=96). Not set -e: one horizon must not abort all.
set -uo pipefail
PY=${PY:-$HOME/.venvs/tsfm-sae-difficulty/bin/python}
for H in 24 96 336 720; do
  echo "=== HORIZON $H ==="
  "$PY" experiments/run_sweep.py --datasets ETTh1 --models small --horizon "$H" \
      --num-samples 20 --batch-size 4 --stride-mult 2 || echo "FAILED horizon $H"
done
echo "ALL HORIZONS DONE"
