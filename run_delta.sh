#!/usr/bin/env bash
# Fine-tune from a PUBLISHED ADAPTER instead of the base model (delta mode).
#
# Why this exists: the trainer always builds a fresh LoRA on the base checkpoint, so fine-tuning
# on new data overwrites what the released model already knows. Measured on our data, the
# upstream decision-v2 suite fell from 0.835 to 0.331 when training from base. With --init_from
# the LoRA and the pointer head are warm started and that skill is kept (0.825) while the model
# adapts to the new domain (0.875 on the new domain versus 0.630 from base).
#
# Usage: bash run_delta.sh <source> [epochs] [out_dir]
#   source  a local run directory or a Hub id, e.g. jaredpalmer/kev-0.6b
#
# Env: SUITE (default ~/Downloads/kev-data), MEM_LIMIT (default 5000M)
set -eu
SRC="${1:?usage: bash run_delta.sh <source> [epochs] [out_dir]}"
EPOCHS="${2:-2}"
OUT="${3:-runs/delta-v1}"
SUITE="${SUITE:-$HOME/Downloads/kev-data}"
LIMIT="${MEM_LIMIT:-5000M}"

cd "$(dirname "$0")" || exit 1
TRAIN="python -m kev.train --suite $SUITE --device cuda --dtype bf16 --checkpointing 1 \
--batch 1 --accum 8 --epochs $EPOCHS --init_from $SRC --out $OUT"
echo "delta: source=$SRC epochs=$EPOCHS out=$OUT suite=$SUITE limit=$LIMIT"

if command -v systemd-run >/dev/null 2>&1; then
    exec systemd-run --user --scope -p MemoryMax="$LIMIT" -p MemorySwapMax=0 --unit=kev-train-delta -- \
        bash -c "source .venv/bin/activate && $TRAIN"
fi
source .venv/bin/activate
exec $TRAIN
