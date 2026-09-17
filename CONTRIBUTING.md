# Contributing

Thanks for looking. This is a small research prototype; the bar for a change is "does it help test or explain the mechanism".

## Setup

```bash
uv sync --extra serve
cd playground && npm install && cd ..
```

## Checks before a pull request

```bash
# fast, no GPU, no weights
uv run python -m pytest tests/test_unit.py -q

# playground
cd playground && npm run lint && npx tsc --noEmit -p . && cd ..

# if you changed the model, data or API: retrain the smoke model and run the server tests
uv run python -m kev.train --n_per_source 40 --accum 4 --out runs/smoke
uv run --extra serve python -m kev.serve --run runs/smoke --port 8009 &
KEV_BASE_URL=http://127.0.0.1:8009 uv run --extra serve python -m pytest tests/test_api.py -q
```

CI runs the first two. The server tests and any training need a machine with a GPU (Apple MPS or CUDA).

## If you change results

If a change alters `evaluate.py` output or you train a new reference checkpoint:

1. Run the full evaluation and commit the new `runs/<name>/eval.json`.
2. Update the numbers in `README.md` ("reproducing" section) and in `MODEL_CARD.md`.
3. Copy the training log to `runs/logs/train_<name>.log` and regenerate `docs/training.png` with `python -m kev.plot`.
4. Say in the pull request which run the numbers come from and what flags produced it.

Do not commit weights. Attach them to a GitHub Release if they need to be shared.

## Style

- Python: compact, few comments, type hints where they help. Follow the existing files.
- Agent notes and gotchas go in `AGENTS.md`.
- README follows the Vercel Labs house style: short declarative sentences, Title Case sections, tables over prose, no first person. MODEL_CARD.md is formal. Keep numbers in both in sync.
