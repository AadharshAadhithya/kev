#!/usr/bin/env bash
# The Modal studies that were queued when the workspace hit its spend limit (PLAN.md, "Overnight autoresearch").
# Run after raising the limit in the Modal dashboard (workspace Settings -> Billing -> spend limit):
#     bash scripts/overnight_queue.sh            # ~$70 of H100 time at the admission bounds below, sequential
# Each study is immutable: a name that already exists under runs/ is skipped. Nothing here reads the locked test.
set -euo pipefail
cd "$(dirname "$0")/.."

run() {  # run <name> <suite> <plan> <transfer> <budget> <timeout>
  if [ -d "runs/$1" ]; then echo "skip $1 (exists)"; return; fi
  echo "== $1"; uv run modal run modal_app.py::study --suite "$2" --plan "$3" --name "$1" --transfer "$4" --budget "$5" --timeout "$6" 2>&1 | tee "runs/$1.log" | grep -aE "bound|launching|^trial-|^\{|pulled|Error" || true
}

# 1. Data mix at 4B on v4: compositional arm only; one epoch (killed mid-training at 20:55)
run mix-4b-v4-b evals/v4/decision-v4 experiments/auto/mix-4b-v4.json evals/v4/transfer-v4 15 4800
# 2. Data mix at 4B on v5 (20k public, 4x synthetic): both arms; compositional-only one epoch
run mix-4b-v5-b evals/v5/decision-v5 experiments/auto/mix-4b-v5.json evals/v5/transfer-v5 25 7200
# 3. Oversampling / subsampling at 4B on v4: synthetic_repeat 3, public_frac 0.33 (the v3-size regime with none pairs)
cat > experiments/auto/mix-4b-knobs.json <<'EOF'
[
  {"base": "Qwen/Qwen3-4B-Base", "seed": 0, "epochs": 2, "batch": 4, "accum": 2, "dtype": "bf16", "checkpointing": 1, "p_none_pair": 0.25, "synthetic_repeat": 3},
  {"base": "Qwen/Qwen3-4B-Base", "seed": 0, "epochs": 2, "batch": 4, "accum": 2, "dtype": "bf16", "checkpointing": 1, "p_none_pair": 0.25, "public_frac": 0.33},
  {"base": "Qwen/Qwen3-4B-Base", "seed": 0, "epochs": 2, "batch": 4, "accum": 2, "dtype": "bf16", "checkpointing": 1, "p_none_pair": 0.25, "option_isolation": 1},
  {"base": "Qwen/Qwen3-4B-Base", "seed": 0, "epochs": 2, "batch": 4, "accum": 2, "dtype": "bf16", "checkpointing": 1, "p_none_pair": 0.25, "public_frac": 0.33, "synthetic_repeat": 2, "option_isolation": 1}
]
EOF
run mix-4b-knobs evals/v4/decision-v4 experiments/auto/mix-4b-knobs.json evals/v4/transfer-v4 25 4800
# 4. Then: uv run python -m kev.autoresearch leaderboard   (rebuilds runs/leaderboard.md and the PLAN.md log section)
uv run python -m kev.autoresearch leaderboard
