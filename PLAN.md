# Research plan

Working plan for the next round of data, evaluation, and training work. Numbers refer to
`evals/decision-v1` (in-distribution) and `evals/transfer-v1` (eval-only) unless stated.

## Where we are

- kev-0.5b vs Jev, in-distribution (720 clean dev questions): 79.7% vs 81.1%, macro
  difference -1.8 pp, 95% CI [-5.5, +1.7]. Not distinguishable.
- kev-0.5b vs Jev, transfer (640 clean dev questions, 8 sources never trained on):
  63.3% vs 82.3%, macro difference **-19.1 pp, 95% CI [-23.1, -15.0]**. This is the real gap.
- kev is better calibrated out of domain (ECE 0.052 vs 0.075; it is less confident where it
  is wrong). Jev puts exact zeros on true answers on Emotion (NLL 4.58).
- Jev's option-order flip rate is 0.000 on 48 permuted items; kev-0.5b's is 0.208.
- kev-0.5b's none-of-the-above shortcut is catastrophic out of domain: it picks "none" 75%
  of the time when the true option is present. kev2 (training now) addresses this
  in-distribution; the fix must be verified out of domain too.
- Nimble (bespokelabs) shipped code and hashes but not data (`data/` is git-ignored, no Hub
  dataset). Their transferable idea is contrastive pairs with an evidence-ablation filter.
  Their eval is same-generator synthetic; not a benchmark for us.

## Principles

1. A source is either **trainable** or **eval-only**, recorded in the suite manifest.
   Training refuses eval-only sources. Eval-only status never changes for MMLU.
2. Development partitions select models; the locked test is read only for promoted
   candidates via `--allow-test`.
3. Every result carries suite hash, code hashes, git commit, coverage counts, and a
   record-clustered bootstrap when compared.
4. One GPU job at a time. New work queues behind the current run with `--wait-pid`.
5. No Jev distillation. Jev is a reference to measure against, not a teacher.

## Steps

### 1. Graduate part of transfer-v1 into training; keep the rest held out  [now]

Trainable in the next suite (`decision-v2`): banking77, boolq, agnews, mnli, sst5, yelp
(existing) + **trec, dbpedia14, amazon, imdb** (graduated: same task families, large public
train splits, fix K-variety and domain breadth).

Eval-only forever: **mmlu** (knowledge probe), **emotion**, **tweet_offensive** (noisy-label
honesty checks), **qnli** (reading transfer), plus new eval-only sources so the held-out set
does not shrink: **paws** (paraphrase Noul), **sciq** (passage MCQ).

Deliverables:
- `TRAINABLE` / `EVAL_ONLY` tables in `kev/data.py`; `kev.suite` writes `trainable_sources`
  and `eval_only_sources` to the manifest; `kev.train --suite` and `kev.experiment` refuse a
  suite whose training partition contains an eval-only source.
- `evals/decision-v2` (10 trainable sources; train/calibration/development/test) and
  `evals/transfer-v2` (6 eval-only sources; development/test), both with exact-match
  exclusion against each other's training states.

### 2. Programmatic contrastive pairs  [now, CPU only]

Templated policy + case generator with certain labels: return windows, eligibility
thresholds, authorization by named role, date arithmetic, quantity limits. Each item is a
pair differing in one fact so the label flips; an ablation check confirms the label is
undeterminable when the deciding sentence is removed. Siblings share a family id and a split.

Deliverables:
- `kev/contrastive.py`: generator, ablation check, `paired_flip` metric (answer changes when
  the fact changes; immune to majority-class guessing).
- Trainable source `contrastive_policy` (in decision-v2 training) and eval-only families
  (held out by family) in transfer-v2.
- `kev.benchmark` reports `paired_flip_rate` for sources that carry `pair_id`.

### 3. Score kev-0.5b, kev2, and Jev on decision-v2 / transfer-v2  [after 1-2]

kev on CPU while the GPU is busy; Jev remotely (cents). Commit reports and comparison JSON.

### 4. Permutation invariance as architecture  [smoke suite, cheap]

Experiment in `kev/model.py`: option-position-agnostic encoding (shared position ids for
every option span, or per-option scoring against the same decide state). Measure flip rate
and packed-vs-separate equality on the smoke suite before any full run.

### 5. Training runs, in order, via `kev.experiment` on decision-v2  [GPU, sequential]

a. Baseline: current data + none-of-the-above fix (kev2 recipe) on decision-v2.
b. + graduated transfer sources.
c. + programmatic contrastive pairs.
d. Qwen3-0.6B-Base on the best of a-c.

Each trial: same suite, same development set, record-clustered bootstrap against (a),
gates (coverage, isolation, packing, none-present/absent, permutation), transfer-v2 as the
out-of-domain check. Promote to the locked test only if the development CI excludes zero.

### 6. LLM-generated contrastive pairs  [later, needs API keys]

Nimble-shaped recipe: generator != verifier, ablation filter, product domains (triage,
rubric grading, compliance). Families held out. Labels marked synthetic; report agreement,
not accuracy, until spot-checked.

### 7. Weekend: larger backbones on the winning data  [Modal, $100 cap]

Qwen3-4B-Base then Qwen3-8B-Base with the recipe from step 5. Not Qwen3.5 (hybrid attention
breaks the branch mask).

## Status

- [x] transfer-v1 frozen and scored (kev-0.5b, Jev); comparison committed
- [ ] 1. trainable/eval-only manifest flag; decision-v2 + transfer-v2 frozen
- [ ] 2. programmatic contrastive pairs + paired_flip metric
- [ ] 3. kev-0.5b / kev2 / Jev on v2 suites
- [ ] 4. permutation-invariant encoding experiment
- [ ] 5a-d training runs (queued after current GPU work)
- [ ] 6. LLM contrastive pairs
- [ ] 7. Modal runs
