#!/usr/bin/env bash
# Runs ON the GPU box, once, to make a python that can train this ablation.
#
# The box already carries a CUDA driver from the S5 run; all this needs is a venv with a
# CUDA build of torch. Idempotent -- safe to re-run.
set -uo pipefail

VENV="${VENV:-$HOME/s7venv}"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q numpy
# cu121 wheels match the driver already on this AMI.
"$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu121

"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"device {p.name}  {p.total_memory/1e9:.1f} GB  sm_{p.major}{p.minor}")
PY
