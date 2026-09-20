---
language: en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3-0.6B-Base
base_model_relation: adapter
pipeline_tag: text-classification
tags:
  - decision-model
  - calibration
  - lora
  - multiple-choice
  - typesafe
  - decision-model
datasets:
  - legacy-datasets/banking77
  - google/boolq
  - fancyzhx/ag_news
  - nyu-mll/multi_nli
  - SetFit/sst5
  - Yelp/yelp_review_full
  - CogComp/trec
  - fancyzhx/dbpedia_14
  - SetFit/amazon_reviews_multi_en
  - stanfordnlp/imdb
metrics:
  - accuracy
  - brier_score
  - expected_calibration_error
model-index:
  - name: kev-0.6b
    results:
      - task: { type: text-classification, name: typed decision (choice / noul / score) }
        dataset: { type: mixed, name: "decision-v4 development (1,204 records; ten trained public sources + programmatic policy pairs)" }
        metrics:
          - { type: accuracy, value: 0.805 }
          - { type: expected_calibration_error, value: 0.078, name: "ECE, raw probabilities" }
      - task: { type: text-classification, name: typed decision, out-of-domain }
        dataset: { type: mixed, name: "transfer-v4 development (764 records; six never-trained sources + held-out policy structures)" }
        metrics:
          - { type: accuracy, value: 0.598 }
          - { type: brier_score, value: 0.521 }
---

# kev-0.6b

`kev-0.6b` is a **decision model**: one document (the *state*) and a set of typed questions in, a probability distribution per question out, in one forward pass. No text generation. It is a LoRA adapter (r=16) plus a pointer head on `Qwen/Qwen3-0.6B-Base`, and it serves TypeSafe's public `/v1/systemone` contract.

**The small member of the kev family.** It is the best 0.6B checkpoint under a frozen, checksummed evaluation protocol (three seeds, eight one-knob mutations). Out of domain it is a 0.6B model — use `kev-4b` for accuracy; use this one where memory or latency rule the 4B out, and measure on your own data.

- Hub: `jaredpalmer/kev-0.6b` (this repo; trial `v4-06b-hardened/00-trial-0`, seed 0 of 3)
- Code, suites, results, and the full research log: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev) — see `PLAN.md`, `runs/leaderboard.md`, and `evals/v4/*/manifest.json`

## What changed since kev-0.5b

| | kev-0.5b | kev-0.6b (this) |
|---|---|---|
| backbone | Qwen2.5-0.5B | Qwen3-0.6B-Base |
| training records | 9,000 (six sources) | 10,896 (ten public sources + 896 programmatic policy pairs) |
| none-of-the-above | augmentation fix only | + minimal pairs: same state rendered with the true option present and removed |
| in-distribution accuracy (decision-v4 dev) | 0.712 | **0.805** |
| out-of-domain accuracy (transfer-v4 dev) | 0.575 | 0.598 |
| none-option present, accuracy | 0.25 (transfer-v1) | 0.78 |
| seeds behind the number | 1 | 3 (transfer 0.595–0.605) |

Jev (`typesafe-ai/jev` via Vercel AI Gateway) on the same frozen development sets: **0.845** in-distribution, **0.857** out-of-domain. Per-source transfer accuracy for this checkpoint: QNLI 0.85, SciQ 0.86, TweetEval-offensive 0.69, PAWS 0.56, Emotion 0.50, MMLU 0.46; held-out policy structures near chance.

## Known limits

- **Out of domain it is a 0.6B model.** Transfer accuracy is flat at ~0.60 across every hyperparameter we tried (eight one-knob mutations, three seeds). The same recipe at 4B reaches 0.72–0.75 and at 8B 0.74–0.77; capacity, not data, is the bottleneck at this size.
- **Held-out policy reasoning fails**: on programmatic policy pairs whose rule structure was never trained, both-siblings-correct is 6–11% (kev-4b 0.73, Jev 0.86).
- **Ordinal hedging**: on 3-level Score questions with date arithmetic it collapses to the middle level.
- Confident-error rate out of domain is 5% (≥0.9 confidence and wrong); raw ECE 0.08 in-domain, 0.16 out of domain. Probabilities are usable in-domain; treat them as advisory elsewhere.
- **Locked test, read once** (`runs/locked/kev-06b-preview-ungated/`): in-distribution accuracy **0.819** (Brier 0.264, ECE 0.072), out-of-domain **0.631** (Brier 0.489, ECE 0.115, confident errors 3.7%). Both are slightly above the development numbers, so the development set was not over-fitted by selection. Out of domain, the none-of-the-above option is still chosen wrongly when the true option is present (accuracy 0.25 on those 36 items); in-domain the fix holds (0.79). This partition will not be read again for this checkpoint.

## Architecture

Prefill-only causal LM with a block-causal attention mask: a shared state prefix, one isolated branch per question, and a pointer readout over option boundary tokens. Questions packed into one request get exactly the probabilities they would get alone (measured max delta 4e-6). Details in the repository README.

## Training

Frozen suite `evals/v4/decision-v4` (manifest pins dataset and base-model revisions): 10,000 public records (1,000 per source) plus two programmatic policy arms of 448 records each, two epochs, LoRA r=16 on attention and MLP projections, pointer head from scratch, cross-entropy on the option distribution, bf16 autocast with fp32 master weights on one H100 (~10 min). Augmentation: option permutation, none-of-the-above insertion, distractors, and none minimal pairs on 25% of Choice records. No Jev outputs were used for training.

## Evaluation protocol

Development partitions select models; a locked test partition exists and is read at most once per promoted candidate. Every number above carries the suite hash, code hashes, and git commit in `result.json`. Comparisons use a record-clustered paired bootstrap. See `PLAN.md` for the corrections we made to our own earlier claims.

## Use

```python
from typesafe import TypeSafeClient   # any TypeSafe-compatible client
client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008", model="kev-latest")
```

Serve with `uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.6b --port 8008` from the repository.

## License

Apache-2.0 for the adapter and head. The base model is Apache-2.0 (Qwen3). Training datasets carry their own licenses.
