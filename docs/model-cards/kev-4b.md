---
language: en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3-4B-Base
base_model_relation: adapter
pipeline_tag: text-classification
tags:
  - decision-model
  - calibration
  - lora
  - multiple-choice
  - typesafe
  - research-preview
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
  - name: kev-4b (research preview)
    results:
      - task: { type: text-classification, name: typed decision (choice / noul / score) }
        dataset: { type: mixed, name: "decision-v4 development (1,204 records; ten trained public sources + programmatic policy pairs)" }
        metrics:
          - { type: accuracy, value: 0.843 }
          - { type: expected_calibration_error, value: 0.066, name: "ECE, raw probabilities" }
      - task: { type: text-classification, name: typed decision, out-of-domain }
        dataset: { type: mixed, name: "transfer-v4 development (764 records; six never-trained sources + held-out policy structures)" }
        metrics:
          - { type: accuracy, value: 0.759 }
          - { type: brier_score, value: 0.346 }
---

# kev-4b — research preview

`kev-4b` is a **decision model**: one document (the *state*) and a set of typed questions in, a probability distribution per question out, in one forward pass. No text generation. It is a LoRA adapter (r=16) plus a pointer head on `Qwen/Qwen3-4B-Base`, serving TypeSafe's public `/v1/systemone` contract.

**Research preview, not a versioned release.** It is the best 4B checkpoint under a frozen, checksummed protocol after ~30 controlled 4B trials, and the first kev whose out-of-domain accuracy is within ten points of Jev on the same items. It does not pass the release screen we set in advance (held-out policy pairs: 0.62 both-correct, screen 70%).

- Hub: `jaredpalmer/kev-4b` (this repo; trial `lowdrift-4b-v4/01-trial-1`)
- Code, suites, every trial with hashes and paired bootstraps: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev) — `PLAN.md`, `runs/leaderboard.md`

## Results (same frozen items for every row)

| | kev-0.5b | kev-0.6b preview | **kev-4b preview** | Jev |
|---|---|---|---|---|
| in-distribution accuracy (decision-v4 dev, 1,200 q) | 0.712 | 0.805 | **0.843** | 0.845 |
| out-of-domain accuracy (transfer-v4 dev, 560 q) | 0.575 | 0.598 | **0.759** | 0.857 |
| out-of-domain Brier | 0.50 | 0.521 | **0.346** | 0.211 |
| confident errors out of domain (p ≥ 0.9 and wrong) | – | 5.2% | 5.5% | 3.7% |
| held-out policy structures, both siblings correct | – | 0.11 | 0.62 | 0.86 |
| option-order flip rate | 0.21 | 0.02 | 0.00 | 0.00 |

Per-source out-of-domain accuracy (kev-4b / Jev): QNLI 0.89 / 0.93, SciQ 0.99 / 0.99, TweetEval-offensive 0.71 / 0.81, PAWS 0.64 / 0.79, MMLU 0.68 / 0.90, Emotion 0.60 / 0.59, deadline (3-level date arithmetic) 0.60 / 0.93.

Seeds: the recipe was run at three seeds on this suite (transfer 0.759 / 0.758 / 0.759; in-distribution 0.843 / 0.853 / 0.855) and twice more on a superset suite (0.755 / 0.761); the spread is ~1 pp. The improvement over the default learning rate is +4.7 pp, 95% CI [+0.4, +9.6], record-clustered paired bootstrap.

**Locked test, one exploratory read** (`runs/locked/kev-4b-preview-ungated/`, labelled ungated because the checkpoint fails the held-out-pair screen): in-distribution **0.852** (Brier 0.221), out-of-domain **0.794** (Brier 0.296, confident errors 3.7%). Both above the development numbers, as for kev-0.6b, so development-set selection did not overfit. This partition will not be read again for this checkpoint.

## What we learned building it

- **Capacity dominates out of domain.** With public examples and synthetic budget held equal, 0.6B → 4B is +14–19 pp; 4B → 8B is +1–7 pp.
- **Fine-tuning erodes base capability, and the learning rate controls it.** The 4B base, zero-shot with a letter readout, scores 0.688 on the same MMLU items and 0.787 on PAWS; the default recipe (lr 2e-4) trained down to 0.60–0.66 / 0.56–0.71. Lowering lr to 5e-5 recovers most of it and is the single largest recipe improvement we found; fewer LoRA target modules and smaller ranks help less.
- **More public training data raises in-distribution accuracy and lowers transfer** at 4B (10k vs 3.4k records: −3 pp). Knowledge MCQ sources (ARC, OpenBookQA, CommonsenseQA) raise in-distribution accuracy to 0.86 without moving transfer.
- Programmatic contrastive policy pairs teach the trained rule structures (both-correct 0.85–1.0) but transfer to unseen structures only partially (0.5–0.6 at 4B, 0.03–0.11 at 0.6B).

## Known limits

- Held-out policy reasoning (unseen rule compositions, date arithmetic with grace periods) is far from Jev.
- Product-shaped questions with no training analogue are not guaranteed: on the TypeSafe docs example ("two charges on my card" → *Is there a billing problem?*) this checkpoint answers 0.22 while kev-0.6b answers 0.97 and picks the return reason (wrong size, 0.54) correctly. Lower drift from the base means fewer task-specific priors; measure on your own inputs.
- Out-of-domain probabilities are usable but not calibrated (raw ECE 0.096); temperature fitted in-domain does not transfer.
- 4B fp32 needs ~16 GB; on a 32 GB Mac use `KEV_DTYPE=bf16`. Latency on an H100 is ~45 ms per packed request; on an M5 several hundred ms.

## Training

Frozen suite `evals/v4/decision-v4`: 10,000 public records (1,000 per source, ten sources) plus two programmatic policy arms of 448 records, two epochs, LoRA r=16 on attention and MLP projections, pointer head from scratch, cross-entropy on the option distribution, **lr 5e-5** (OneCycle), effective batch 8, bf16 autocast with fp32 master weights, gradient checkpointing, one H100 (~40 min). Augmentation: option permutation, none-of-the-above insertion, distractors, none minimal pairs on 25% of Choice records. No Jev outputs were used for training.

## Evaluation protocol

Development partitions select models; the locked test partition is read at most once per candidate. Every number carries suite hash, code hashes, and git commit in `result.json`. See `PLAN.md` for the corrections we made to our own earlier claims.

## Use

```bash
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8008      # KEV_DTYPE=bf16 on a 32 GB Mac
```

Any TypeSafe-compatible client works: `TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008", model="kev-latest")`.

## License

Apache-2.0 for the adapter and head; Qwen3 base is Apache-2.0; datasets carry their own licenses.
