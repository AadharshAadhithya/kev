#!/usr/bin/env bash
# Train kev inside an isolated cgroup so an out-of-memory event kills only this process,
# never the desktop session or the agent's service (learned the hard way on a 14 GB laptop).
#
# Usage: bash run_train.sh [epochs] [out_dir] [suite_dir]
#   epochs     default 2
#   out_dir    default runs/gate-v1
#   suite_dir  default $SUITE, otherwise ~/Downloads/kev-data
#
# Env: MEM_LIMIT (default 5000M), OMP_NUM_THREADS (default 4)
set -u
cd "$(dirname "$0")" || exit 1

EPOCHS="${1:-2}"
OUT="${2:-runs/gate-v1}"
SUITE="${3:-${SUITE:-$HOME/Downloads/kev-data}}"
LIMIT="${MEM_LIMIT:-5000M}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

TRAIN="python -m kev.train --suite $SUITE --device cuda --dtype bf16 --checkpointing 1 \
--batch 1 --accum 8 --epochs $EPOCHS --out $OUT"
echo "train: epochs=$EPOCHS out=$OUT suite=$SUITE limit=$LIMIT"

# systemd-run is Linux-only; on other platforms run directly and keep the same command.
if command -v systemd-run >/dev/null 2>&1; then
    exec systemd-run --user --scope -p MemoryMax="$LIMIT" -p MemorySwapMax=0 --unit=kev-train-1 -- \
        bash -c "source .venv/bin/activate && $TRAIN"
fi
source .venv/bin/activate
exec $TRAIN
