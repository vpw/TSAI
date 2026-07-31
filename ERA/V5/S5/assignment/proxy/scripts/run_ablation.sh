#!/usr/bin/env bash
# Runs ON the GPU box. Executes every arm of the ablation, in the order that gets the most
# useful result first if the budget runs out: A and B (the hypothesis), then C and D (what
# the floor and the tiering are worth), then the seed repeat that sets the noise band, then
# the three short transition probes.
#
#   run_ablation.sh [tokens_per_arm]
#
# Model size and token count were chosen by benchmarking this exact T4: 33.5k tokens/s at
# d_model=384 / 8 layers, which is what makes 5.75 full-length-equivalent runs fit the
# GPU-hour budget. See RESULTS.md for the honest statement of what this scale does and does
# not license.
set -uo pipefail
cd "$(dirname "$0")/.."

TOKENS="${1:-75000000}"
PY="${PY:-$HOME/s5venv/bin/python}"
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"

ARCH="--d-model 384 --n-layers 8 --n-heads 6 --seq-len 1024 --batch-seqs 16 --micro-seqs 8"
EVAL="--light-eval-tokens 250000 --final-eval-tokens 1000000"
mkdir -p results logs

run() {
  local name="$1"; shift
  echo "=== $name  start $(date -u +%H:%M:%S) ==="
  if "$PY" scripts/train.py "$@" $ARCH $EVAL > "logs/${name}.log" 2>&1; then
    tail -14 "logs/${name}.log"
    echo "--- $name done $(date -u +%H:%M:%S)"
  else
    echo "!!! $name FAILED — tail of log:"
    tail -20 "logs/${name}.log"
    echo "!!! continuing with the remaining arms"
  fi
}

# The four mixture arms: identical in everything except the mixture.
run A_proposed      --arm A_proposed      --tokens "$TOKENS" --seed 1234
run B_web_heavy     --arm B_web_heavy     --tokens "$TOKENS" --seed 1234
run C_no_floor      --arm C_no_floor      --tokens "$TOKENS" --seed 1234
run D_verified_only --arm D_verified_only --tokens "$TOKENS" --seed 1234

# Seed repeat of arm A: sets the noise band any claimed difference has to clear.
run A_seed2 --arm A_proposed --tokens "$TOKENS" --seed 8675 --tag seed2

# Transition probes: quarter length, mixture switches main -> anneal at the midpoint.
ETOK=$(( TOKENS / 4 ))
run E1_hard_frozen    --arm E1_hard_frozen    --tokens $ETOK --seed 1234
run E2_hard_trainable --arm E2_hard_trainable --tokens $ETOK --seed 1234
run E3_warm_trainable --arm E3_warm_trainable --tokens $ETOK --seed 1234

echo "=== all arms attempted $(date -u +%H:%M:%S) ==="
ls -la results/
