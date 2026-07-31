#!/usr/bin/env bash
# Runs ON the GPU box. Brings up a Python environment, records what hardware we actually
# got, then rebuilds the token pools from source.
#
# Only the two Indic pools are uploaded (they are S4's cleaned output and exist nowhere
# else); everything else is re-fetched here, because this box downloads from Hugging Face
# far faster than the workstation can upload.
set -euo pipefail
cd "$(dirname "$0")/.."          # -> proxy/

echo "=== hardware ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || true
echo "cpus: $(nproc)   mem: $(free -g | awk '/^Mem:/{print $2}') GB   disk: $(df -h . | awk 'NR==2{print $4}') free"

PY=""
for cand in "$HOME/s5venv/bin/python" python3; do
  if $cand -c "import torch" 2>/dev/null; then PY="$cand"; break; fi
done

if [ -z "$PY" ]; then
  echo "=== installing torch ==="
  python3 -m venv "$HOME/s5venv" 2>/dev/null || python3 -m venv --system-site-packages "$HOME/s5venv"
  "$HOME/s5venv/bin/pip" install -q --upgrade pip
  "$HOME/s5venv/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu121
  PY="$HOME/s5venv/bin/python"
fi
echo "python: $PY"
"$PY" -m pip install -q "numpy<2" pyarrow tokenizers

echo "=== torch / cuda ==="
"$PY" - <<'EOF'
import torch, json
d = {"torch": torch.__version__, "cuda": torch.version.cuda,
     "cudnn": torch.backends.cudnn.version(), "device_count": torch.cuda.device_count()}
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    d.update(gpu=p.name, vram_gb=round(p.total_memory/1e9, 1),
             sm=f"{p.major}.{p.minor}", bf16=torch.cuda.is_bf16_supported())
print(json.dumps(d, indent=1))
open("results/environment.json", "w").write(json.dumps(d, indent=1))
EOF

echo "=== building pools ==="
"$PY" scripts/fetch_raw.py
"$PY" scripts/build_pools.py
TOKENIZERS_PARALLELISM=true "$PY" scripts/tokenize_pools.py
echo "=== ready ==="
