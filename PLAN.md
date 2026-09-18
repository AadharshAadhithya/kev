# Research plan

Working plan for the next round of infrastructure, data, evaluation, and training work.
Numbers refer to [`evals/decision-v1`](evals/decision-v1/manifest.json) (in-distribution) and
[`evals/transfer-v1`](evals/transfer-v1/manifest.json) (eval-only) unless stated. Every number
below has a committed artifact; follow the links.

## Where we are

- **kev-0.5b vs Jev, in-distribution** (720 clean dev questions): 79.7% vs 81.1%, macro
  difference -1.8 pp, 95% CI [-5.5, +1.7]. Not distinguishable.
  Artifacts: [`runs/kev-vs-jev-v1.json`](runs/kev-vs-jev-v1.json),
  [`runs/research-kev-v01/report.json`](runs/research-kev-v01/report.json),
  [`runs/research-jev-v1/report.json`](runs/research-jev-v1/report.json),
  figure [`docs/kev-vs-jev.png`](docs/kev-vs-jev.png).
  *Annotation:* kev was fine-tuned on these six datasets, so parity here is expected and
  says little. Jev's exposure to the same public sets is unknown.
- **kev-0.5b vs Jev, transfer** (640 clean dev questions, 8 sources never trained on;
  zero exact-match overlap with any kev training state): 63.3% vs 82.3%, macro difference
  **-19.1 pp, 95% CI [-23.1, -15.0]**. This is the real gap.
  Artifacts: [`runs/kev-vs-jev-transfer-v1.json`](runs/kev-vs-jev-transfer-v1.json),
  [`runs/transfer-kev-v01/report.json`](runs/transfer-kev-v01/report.json),
  [`runs/transfer-jev-v1/report.json`](runs/transfer-jev-v1/report.json).
  Per task (kev / Jev): IMDB .925/.938, QNLI .787/.875, DBpedia-14 .762/.988, TREC .600/.912,
  Amazon .575/.675, TweetEval-offensive .525/.800, MMLU .475/.900, Emotion .412/.500.
  *Annotation:* MMLU 47.5% is the Qwen2.5-0.5B base level; the readout adds nothing where
  knowledge is the bottleneck. DBpedia and TREC are format/label-set transfer failures, not
  knowledge failures, and are the cheapest to fix with data (step 1).
- **kev is better calibrated out of domain** (ECE 0.052 vs 0.075; mean confidence 0.685 vs
  accuracy 0.633, i.e. it knows it does not know). Jev puts exact zeros on true answers on
  Emotion (NLL 4.58 at floor 1e-9; NLL is floor-sensitive, see
  [`nll_floor_sensitivity`](runs/kev-vs-jev-transfer-v1.json)). We report accuracy and Brier
  for Jev, not NLL.
- **Jev's option-order flip rate is 0.000** on 48 permuted items; kev-0.5b's is 0.208
  ([`permutation`](runs/transfer-jev-v1/report.json)). *Annotation:* zero over 48 items is
  architectural or served invariance, not training. Design target for step 4.
- **kev-0.5b's none-of-the-above shortcut is catastrophic out of domain**: it picks "none"
  75% of the time when the true option is present (`none_present` acc 0.25; Jev 0.688).
  Root cause and in-distribution fix: commit
  [`61a7643`](https://github.com/jaredpalmer/kev/commit/61a7643) (the none option was correct
  100% of the time it appeared in training). kev2 trains on the fixed data; the fix must be
  verified out of domain too.
- **Evaluator bugs fixed** (both mine): clean eval silently added none-options via a leaked
  default; the none-removed probe never counted anything. Tests:
  [`tests/test_research.py`](tests/test_research.py). Ordinal loss replaced with the ranked
  probability score (proper scoring rule).
- **Nimble** ([repo](https://github.com/bespokelabsai/nimble),
  [model](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B)): LoRA r=16 on Qwen3.5-9B,
  2,676 fully synthetic examples (GPT-5.6 generates and separately verifies), letter-code
  next-token readout, one prompt per field, 26-option cap. 90.1% vs Jev 93.2% on their own
  324-example synthetic holdout (6 source families, same generator family as training).
  *What is actually open:* code and hashes. `data/` is git-ignored and there is no Hub
  dataset, so there is nothing to mix into our training today.
  *Transferable idea:* contrastive pairs (two contexts differing in <=8 words on one fact so
  the label flips) plus an evidence-ablation filter (remove either evidence sentence and the
  fact must become undeterminable). Siblings stay in one split.
  *Not transferable:* Qwen3.5 (hybrid DeltaNet; our block-causal mask needs standard
  attention in every layer, verified earlier), the 26-option letter readout (our pointer head
  has no K cap and gives exact packed isolation), and their eval as a benchmark.
- **Compute.** MBP (M5, MPS, fp32, batch 1): 0.34 s/record, 1h45m per 9k x 2 run, one job at
  a time, training slows the playground 10x. Modal H100 at
  [$3.95/h](https://modal.com/pricing): ~25x per trial once training is batched, ~100x
  throughput with parallel containers, ~$0.15-0.30 per 0.5B trial. Nimble's 9B ran ~100 ms per
  example on H100 and 444 ms median on an M5 Pro.

## Principles

1. A source is either **trainable** or **eval-only**, recorded in the suite manifest.
   Training refuses eval-only sources. Eval-only status never changes for MMLU.
2. Development partitions select models; the locked test is read only for promoted
   candidates via `--allow-test` ([`kev/suite.py`](kev/suite.py) `load_split`).
3. Every result carries suite hash, code hashes, git commit, coverage counts, and a
   record-clustered bootstrap when compared ([`kev/experiment.py`](kev/experiment.py),
   [`kev/benchmark.py`](kev/benchmark.py) `paired_bootstrap`).
4. One GPU job per device. Locally that means queuing behind the current run with
   `--wait-pid`; on Modal it means one trial per container.
5. No Jev distillation. Jev is a reference to measure against, not a teacher.
6. Historical checkpoints (`runs/kev`, `runs/kev2`) overlap the suites' training data:
   exploratory only; never promoted.
7. The MBP path stays supported and documented as optional. Modal is the default for
   anything longer than a smoke run.

## Steps

### 0. Modal as the default compute  [now]

Why first: every later step is a training run or an eval sweep, and each is 25-100x faster
on Modal. Sequencing this ahead of the data work is what makes 30-40 trials an evening.

- CUDA support in [`kev/train.py`](kev/train.py), [`kev/benchmark.py`](kev/benchmark.py),
  [`kev/experiment.py`](kev/experiment.py): `--device cuda`, TF32 matmul, optional bf16
  autocast (`--dtype`), CUDA sync for latency, peak-memory tracking, git sha from env inside
  containers.
- Batched training: pad to the longest sequence in the batch, batched block-causal mask
  `[B,1,L,L]`, SDPA attention (accepts arbitrary masks; FlashAttention does not). Parity test:
  batched bf16 vs batch-1 fp32 on the smoke suite, probabilities within tolerance; bf16 will
  move the third decimal and the baseline gets re-scored on the same hardware.
- [`modal_app.py`](modal_app.py): image via `uv_sync` from `pyproject.toml`/`uv.lock` with a
  CUDA torch index; Volumes for the HF cache and `runs/`; `HF_TOKEN` as a Modal Secret;
  one H100 container per trial, fanned out with `.map`; results pulled back into `runs/` so
  git provenance is unchanged. Docs used:
  [images](.agents/skills/modal/references/guide/images.md),
  [volumes](.agents/skills/modal/references/api/Volume.md),
  [gpu](.agents/skills/modal/references/guide/gpu.md),
  [secrets](.agents/skills/modal/references/guide/secrets.md).
- Smoke on a cheap GPU, then the backbone study (step 5d-lite: Qwen2.5-0.5B vs
  Qwen3-0.6B-Base, two seeds each) on H100s in parallel, replacing the locally queued copy.

### 1. Graduate part of transfer-v1 into training; keep the rest held out

Trainable in the next suite (`decision-v2`): banking77, boolq, agnews, mnli, sst5, yelp
(existing) + **trec, dbpedia14, amazon, imdb** (graduated: same task families, large public
train splits; they fix the K-variety and domain-breadth failures visible in the transfer
table above).

Eval-only forever: **mmlu** (knowledge probe), **emotion**, **tweet_offensive** (noisy-label
honesty checks), **qnli** (reading transfer), plus new eval-only sources so the held-out set
does not shrink: **paws** (paraphrase Noul), **sciq** (passage MCQ). Excluded on purpose:
Rotten Tomatoes (SST parent), SNLI (MNLI sibling).

Deliverables:
- `TRAINABLE` / `EVAL_ONLY` tables in [`kev/data.py`](kev/data.py); `kev.suite` writes both
  lists to the manifest; `kev.train --suite` and `kev.experiment` refuse a suite whose training
  partition contains an eval-only source.
- `evals/decision-v2` (10 trainable sources; train/calibration/development/test) and
  `evals/transfer-v2` (6 eval-only sources; development/test), with exact-match exclusion
  against each other's training states (`--exclude-states-from`).

### 2. Programmatic contrastive pairs  [CPU only]

Templated policy + case generator with certain labels: return windows, eligibility
thresholds, authorization by named role, date arithmetic, quantity limits. Each item is a
pair differing in one fact so the label flips; an ablation check confirms the label is
undeterminable when the deciding sentence is removed (Nimble's filter, in code, no LLM).
Siblings share a family id and a split.

Deliverables:
- `kev/contrastive.py`: generator, ablation check, `paired_flip` metric (did the answer
  change when the fact changed; immune to majority-class guessing).
- Trainable source `contrastive_policy` (decision-v2 training) and eval-only families (held
  out by family) in transfer-v2.
- `kev.benchmark` reports `paired_flip_rate` for sources that carry `pair_id`.

### 3. Score kev-0.5b, kev2, and Jev on decision-v2 / transfer-v2

kev on Modal (or CPU if Modal is not yet up); Jev remotely via
[`kev/jev.py`](kev/jev.py) (AI SDK 7 `experimental_evaluate`, ~$0.015 per 600 records,
5xx-retry only). Commit reports and comparison JSON as for v1.

### 4. Permutation invariance as architecture  [smoke suite, cheap]

Experiment in [`kev/model.py`](kev/model.py): option-position-agnostic encoding (shared
position ids for every option span, or per-option scoring against the same decide state).
Measure flip rate and packed-vs-separate equality on the smoke suite before any full run.
Target: Jev's 0.000 without the `--perm_kl` loss.

### 5. Training runs, in order, via `kev.experiment` on decision-v2  [Modal]

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

### 7. Larger backbones on the winning data  [Modal, $100 cap]

Qwen3-4B-Base then Qwen3-8B-Base with the recipe from step 5 (Qwen3 tech report base
MMLU-Pro: 4B low 50s, 8B mid-to-high 50s, 14B low 60s; Jev 84.6). Not Qwen3.5 (hybrid
attention breaks the branch mask). Rough cost at ~40% MFU: 8B on 200k examples ~ $25.

## Status

- [x] transfer-v1 frozen and scored (kev-0.5b, Jev); comparison committed
      ([`1054ea7`](https://github.com/jaredpalmer/kev/commit/1054ea7))
- [x] Modal CLI installed (`modal==1.5.5`, dev dependency) and authenticated (workspace `jp-1083`)
- [ ] 0. CUDA + batched training; `modal_app.py`; smoke; backbone study on Modal
- [ ] 1. trainable/eval-only manifest flag; decision-v2 + transfer-v2 frozen
- [ ] 2. programmatic contrastive pairs + paired_flip metric
- [ ] 3. kev-0.5b / kev2 / Jev on v2 suites
- [ ] 4. permutation-invariant encoding experiment
- [ ] 5a-d training runs
- [ ] 6. LLM contrastive pairs
- [ ] 7. 4B / 8B runs

Local runs in flight and untouched: `runs/kev2` (9k x 2, none-fix, on the MBP GPU) and the
locally queued backbone study `runs/mbp-comparison-v1` (to be cancelled once the Modal copy is
running).
