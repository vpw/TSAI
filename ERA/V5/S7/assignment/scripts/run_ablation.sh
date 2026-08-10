#!/usr/bin/env bash
# Runs ON the GPU box. Trains every arm of the input-path ablation.
#
#   run_ablation.sh [tokens_per_arm]
#
# Arm order is chosen so the most informative result lands first if the budget runs out:
#
#   kronecker_32   the baseline being challenged (the shipped scheme)
#   fourier_2048   the claim: same job, 4x smaller code, 4x fewer trainable params
#   fourier_8192   matched-dimension control -- isolates "dense vs sparse code" from "size"
#   dense          the upper-bound control (a full V x d_model table)
#   naive_2048     the negative control: permutation-invariant, should be clearly worst
#   kronecker_48   what widening the window costs, since that is the session's own remedy
#
# Everything except the embedding module is pinned: architecture, optimiser, schedule, seed,
# sequence length, token budget, and the token stream itself (same sampler seed per arm).
set -uo pipefail
cd "$(dirname "$0")/.."

TOKENS="${1:-30000000}"
PY="${PY:-$HOME/s7venv/bin/python}"
[ -x "$PY" ] || PY=python3

ARCH="--d-model 512 --n-layers 8 --n-heads 8 --seq-len 512 --batch-seqs 32 --micro-seqs 16"
EVAL="--light-eval-tokens 200000 --final-eval-tokens 1000000"
mkdir -p runs logs

run() {
  local name="$1"
  echo "=== $name  start $(date -u +%H:%M:%S) ==="
  if "$PY" train_arm.py --arm "$name" --tokens "$TOKENS" $ARCH $EVAL > "logs/${name}.log" 2>&1; then
    tail -12 "logs/${name}.log"
    echo "--- $name done $(date -u +%H:%M:%S)"
  else
    echo "!!! $name FAILED - tail of log:"
    tail -25 "logs/${name}.log"
    echo "!!! continuing with the remaining arms"
  fi
}

for arm in kronecker_32 fourier_2048 fourier_8192 dense naive_2048 kronecker_48; do
  run "$arm"
done

echo
echo "=== all arms finished $(date -u +%H:%M:%S) ==="
ls -la runs/
