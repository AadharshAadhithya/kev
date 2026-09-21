---
language: en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3.5-0.8B-Base
base_model_relation: adapter
pipeline_tag: text-classification
tags:
  - decision-model
  - calibration
  - lora
  - multiple-choice
  - typesafe
  - qwen3.5
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
  - name: Kev-0.8B
    results:
      - task: { type: text-classification, name: typed decision (choice / noul / score) }
        dataset: { type: mixed, name: "decision-v7 development (1,204 records; ten trained public sources + programmatic policy data)" }
        metrics:
          - { type: accuracy, value: 0.825 }
          - { type: expected_calibration_error, value: 0.110, name: "ECE, raw probabilities" }
      - task: { type: text-classification, name: typed decision, out-of-domain }
        dataset: { type: mixed, name: "transfer-v4 development (764 records; six never-trained sources + held-out policy structures)" }
        metrics:
          - { type: accuracy, value: 0.652 }
          - { type: brier_score, value: 0.499 }
---

# Kev-0.8B

Kev-0.8B is a **decision model**: one document (the *state*) and a set of typed questions in, a probability distribution per question out, in one forward pass. No text generation. It is a LoRA adapter (r=16, 11.3M trainable parameters) plus a pointer head on `Qwen/Qwen3.5-0.8B-Base` (revision `dc7cdfe2`), serving TypeSafe's public `/v1/systemone` contract.

**The small member of the Kev family.** Same data and recipe as the 0.6B it replaces, on the Qwen3.5 base: in-distribution 0.825 (Kev-0.6B 0.801), out of domain 0.652 (0.620), and it is the first small Kev that learns any rule composition (held-out pairs 0.42 vs 0.08). Three seeds of the base recipe: transfer 0.622 / 0.634 / **0.643**; this checkpoint is seed 2 (selected on development accuracy) followed by a 9-minute **delta fine-tune** on 1,425 generated records (date-bearing policy cases with explicit day counts; evidence-free cases with uniform targets) mixed with 2,000 replayed training records — the same delta as Kev-4B and Kev-9B. Locked test against the pre-delta checkpoint: out of domain 0.668 → **0.684** (+2.2 pp [−0.8, +5.5]), Brier 0.473 → 0.460. Out of domain it is still a sub-1B model: use Kev-4B for accuracy; use this one where memory rules the 4B out, and measure on your own data.

- Hub: `jaredpalmer/kev-0.8b` (this repo; trial `night2-08b-du2/00-trial-0`). The pre-delta checkpoint is at revision `v7-base`.
- Demo: [huggingface.co/spaces/jaredpalmer/kev](https://huggingface.co/spaces/jaredpalmer/kev) runs Kev-4B and Kev-0.8B on ZeroGPU with the same encoder and API code as `kev.serve`.
- Code, suites, results, and the full research log: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev) — `PLAN.md` (the Qwen3.5 port is under History), `runs/leaderboard.md`

## Results (same frozen items for every row)

| | Kev-0.6B (Qwen3) | **Kev-0.8B** | Kev-4B | Kev-9B | Jev |
|---|---|---|---|---|---|
| in-distribution accuracy (decision-v7 dev, 1,204 records) | 0.801 | **0.825** | 0.872 | 0.872 | 0.845 |
| out-of-domain accuracy (transfer-v4 dev, 764 records) | 0.620 | **0.652** | 0.797 | 0.822 | 0.857 |
| out-of-domain Brier | 0.536 | **0.499** | 0.299 | 0.286 | 0.211 |
| confident errors out of domain (p ≥ 0.9 and wrong) | 10.8% | 9.9% | 6.9% | 8.7% | 3.7% |
| coverage at ≤ 5% error (share of decisions automatable) | – | 0.23 | 0.54 | 0.47 | 0.70 |
| held-out policy structures, both siblings correct | 0.08 | **0.42** | 0.78 | 0.83 | 0.86 |
| option-order flip rate | 0.07 | 0.08 | 0.08 | 0.03 | 0.00 |
| none-option present, accuracy | 0.80 | 0.83 | 0.92 | 0.90 | – |
| as served (built-in T = 2.41): Brier / ECE / confident errors | – | 0.430 / 0.054 / 0.3% | | | |

Per-source out-of-domain accuracy (Kev-0.8B / Jev): QNLI 0.85 / 0.93, SciQ 0.91 / 0.99, TweetEval-offensive 0.68 / 0.81, PAWS 0.55 / 0.79, MMLU 0.42 / 0.90, Emotion 0.54 / 0.59, authorization 0.97 / 1.00, deadline (3-level date arithmetic) 0.38 / 0.93, (A or B) and C 0.66 / 0.91, (A and B) or not C 0.56 / 0.97, if A then not B else C 0.59 / 0.78.

Paired against Kev-0.6B on the same items (record-clustered bootstrap), before the delta: +5.7 pp [+1.2, +10.0] out of domain; the delta adds +0.5 pp [−3.2, +3.8] on development and +2.2 pp on the locked test.

**Locked test, read once per checkpoint** (`runs/locked/kev-08b-night2-du-ungated/`; pre-delta `runs/locked/kev-08b-q35-ungated/`): in-distribution **0.834** (Brier 0.268, ECE 0.100), out-of-domain **0.684** (Brier 0.460, ECE 0.154, confident errors 8.7%, held-out pairs 0.45). Pre-delta: 0.827 / 0.668; Kev-0.6B on the same test items: 0.808 / 0.642.

## Known limits

- **Out of domain it is a sub-1B model.** Knowledge (MMLU 0.41) and paraphrase (PAWS 0.59) are near the untrained base; the same recipe reaches 0.79 at 4B and 0.81 at 9B on these items.
- **Slow on a Mac for its size.** The DeltaNet kernels have no MPS implementation; a five-question request takes ~0.33 s in bf16 on an M5 (Kev-0.6B: 0.12 s). On CUDA with `flash-linear-attention` it is fast.
- Requires `transformers >= 5.17` and `peft >= 0.21`.
- Ordinal hedging on date arithmetic (`deadline` 0.38): collapses to the middle level. `KEV_DATE_FACTS=1` (day counts appended to the state) helps the larger models more than this one.
- Confident-error rate out of domain is 9.9% for the raw logits; the built-in temperature (T = 2.41, fitted on the in-distribution development rows and stored in `head.pt`) brings it to 0.3% and ECE from 0.179 to 0.054 without changing any answer. `KEV_TEMPERATURE=1.0` gives the raw values. Probabilities are usable in-domain; treat them as advisory elsewhere.

## Training

Frozen suite `evals/v7/decision-v7`: 10,000 public records (1,000 per source), 896 policy minimal-pair records over nine template families, 1,680 records from 60 randomly generated rule structures in four rendering styles. Two epochs, LoRA r=16 α=32 on attention, MLP and DeltaNet projections; pointer head from scratch; cross-entropy on the option distribution; lr 1e-4 (OneCycle), batch 8, bf16 autocast with fp32 master weights; option permutation, none-of-the-above insertion, distractors, none minimal pairs on 25% of Choice records; ~20 min on one H100. Then the delta: `--init_from jaredpalmer/kev-0.8b@v7-base --data evals/night2/dates_unknowable.jsonl --replay 2000 --lr 4e-5 --epochs 1`, 9 minutes. No Jev outputs were used for training.

## Evaluation protocol

Development partitions select models; the locked test partition is read at most once per candidate. Every number carries suite hash, code hashes and git commit in `result.json`.

## Use

```bash
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8008
```

Any TypeSafe-compatible client works: `TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008", model="kev-latest")`.

## License

Apache-2.0 for the adapter and head; the Qwen3.5 base is Apache-2.0; datasets carry their own licenses.
