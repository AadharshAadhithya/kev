#!/usr/bin/env bash
# Serve the local decision model over the TypeSafe POST /v1/systemone contract on port 8009.
#
# Usage: RUN=<run dir or Hub id> PORT=8009 bash serve_kev.sh
#   RUN  default jaredpalmer/kev-0.6b; use a locally trained run (e.g. runs/delta-v1) to serve
#        a model fine-tuned on your own data.
#
# KEV_DTYPE: bf16 on Ampere and newer is faster; on CPU bf16 is emulated and roughly 6x slower
#            than float32 (measured on a Ryzen laptop: >8 s versus ~1-2 s per decision).
# KEV_MERGE=0 keeps the LoRA separate from the base weights. Merging in fp32 peaks around 3.6 GB,
#            which does not fit a 4 GB card alongside another resident model.
set -u
cd "$(dirname "$0")" || exit 1

export KEV_DTYPE="${KEV_DTYPE:-bf16}"
export KEV_ATTN="${KEV_ATTN:-sdpa}"
export KEV_MERGE="${KEV_MERGE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
RUN="${RUN:-jaredpalmer/kev-0.6b}"
PORT="${PORT:-8009}"

echo "serve: run=$RUN port=$PORT dtype=$KEV_DTYPE attn=$KEV_ATTN merge=$KEV_MERGE"
exec uv run --extra serve python -m kev.serve --run "$RUN" --port "$PORT"
