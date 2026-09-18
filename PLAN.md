# Research plan

## Current decision

Use **Qwen3-0.6B-Base as the research baseline**. Keep the released Qwen2.5 checkpoint as a historical reference, not as an equally funded development track. Preserve the MacBook training path; run the controlled studies on Modal H100s.

On the same v2 recipe and data, Qwen3's development accuracy was 81.6% / 79.3% across two seeds, versus 74.2% / 65.8% for Qwen2.5. Transfer accuracy was 62.0% / 62.1% versus 60.5% / 48.2%. This supports our backbone choice, not a claim that every Qwen3 model dominates every Qwen2.5 model.

Evidence: [ablation-v2 ledger](runs/ablation-v2/results.jsonl), [Qwen3 seed 0](runs/ablation-v2/06-trial-6/result.json), [Qwen3 seed 1](runs/ablation-v2/07-trial-7/result.json).

## Evidence and corrections

- Original kev versus Jev on familiar sources: 79.7% versus 81.1% micro accuracy; macro difference −1.8 points, 95% CI [−5.5, +1.7]. An interval including zero is not evidence of equivalence. [Comparison](runs/kev-vs-jev-v1.json), [figure](docs/kev-vs-jev.png).
- Original kev versus Jev on transfer-v1: 63.3% versus 82.3%, macro difference −19.1 points, CI [−23.1, −15.0]. The overlap check covered 1,360 decision-v1 training/calibration states, **not every example used to train the released checkpoint**. Public pretraining overlap is unknown for both models. [Comparison](runs/kev-vs-jev-transfer-v1.json), [manifest](evals/transfer-v1/manifest.json).
- Lower aggregate ECE on transfer-v1 did not establish generally better calibration. Qwen3 trial 6 gets 50% authorization accuracy with 99.6% mean confidence on transfer-v2. Its familiar-source ECE falls from .073 to .026 after fitting temperature on calibration; this does not establish transfer calibration. [Result](runs/ablation-v2/06-trial-6/result.json).
- Jev's zero observed argmax flips do **not** prove architectural invariance. Its probabilities move under permutation. [Jev transfer-v1 report](runs/transfer-jev-v1/report.json).
- A none-present accuracy of 25% means 75% wrong, not necessarily 75% selecting none. Report actual none-option mass and selection separately. [Original comparison diagnostics](runs/kev-vs-jev-transfer-v1.json).
- MMLU and domain-transfer failures do not by themselves identify a knowledge versus readout bottleneck. We need a controlled capacity/data experiment.
- Two v2 issues were found: siblings shuffled their sentence order independently, and calibration took the first slice of a family-ordered synthetic list. Pair metrics also compared option indices rather than semantic keys. Fix these under **v3**, with tests; do not rewrite v2 again.

## Nimble research

[Bespoke Nimble](https://github.com/bespokelabsai/nimble), [model card](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B), [dataset guide](https://github.com/bespokelabsai/nimble/blob/main/docs/DATASET.md).

The inspected release trained a Qwen3.5-9B LoRA adapter on 2,676 synthetic contrastive records. It reported 90.1% agreement with synthetic reference labels versus Jev's 93.2% on 324 records from six source families. These are useful external research results, not measurements on our evaluation suite or proof of broad parity. In the inspected checkout, `data/` was ignored and the dataset guide required separately supplied files; no directly usable dataset was found during that review.

Adopt minimal factual edits, executable labels where possible, evidence checks, and grouped splits. A check on structured fact dictionaries does not independently validate the English rendering. Separate model calls also do not eliminate correlated synthetic-label errors. Hard outcome labels with a proper scoring rule are sufficient; teacher probabilities are not required.

Nimble scores letter tokens and supports a hybrid recurrent backbone through separate field execution. Our current packed mask alone does not isolate recurrent state, so Qwen3.5 is not a drop-in replacement. This does not make hybrid models fundamentally unusable.

## v3 protocol (approved)

### 1. Correctness and immutable artifacts

- Preserve all existing suite versions and run directories. New freezes and Modal trials refuse to overwrite existing paths, including failure artifacts.
- Match sentence order and option order inside a minimal pair. Change one decisive fact only.
- Stratify calibration by family while keeping pairs/groups together.
- Reject incomplete or duplicated pairs. Compare semantic answer keys, not their positions.
- Preserve locked-test bytes. New held-out structures/renderings are appended only to a new version. Do not score the locked test during this study.
- Verify exact semantic-state separation among newly generated groups. Do not claim fuzzy or pretraining decontamination. Inherited legacy test overlap with newly generated controls is not certified by this audit.

### 2. Compositional policy data

Generate an explicit rule tree, facts, executable reference label, and a textual rendering. Include numeric comparisons (`<`, `≤`, `>`, `≥`, `=`, inclusive ranges), entity equality, elapsed-date comparisons, AND, OR, NOT, exceptions, and conditional precedence.

For each situation create both:

1. Relevant intervention: change one fact and require the answer to change.
2. Irrelevant intervention: change a routing reference and require the answer to stay unchanged.

Keep all four records in one split and one bootstrap unit. Require removal of the decisive fact to make the relevant decision unknown in the executable rule. Test truth tables and numeric/date boundaries, then manually inspect rendered samples.

Eight rule structures are trainable; three new compositions are transfer-development only; three further structures and a reserved rendering style are locked-test only. Authorization and deadline template families remain excluded from training, although related logical primitives are intentionally trained. This measures compositional/template transfer, not unseen primitive knowledge.

### 3. Matched data-versus-capacity experiment

| Backbone | Corrected legacy policy data | Compositional policy data |
|---|---|---|
| Qwen3-0.6B-Base | Control | Data effect |
| Qwen3-4B-Base | Capacity effect | Combined effect |

- Identical 3,000 public training records from ten permitted sources in both arms.
- Exactly 448 synthetic training records per arm, replacing rather than adding examples.
- Shared, family-stratified calibration and shared development/transfer sets.
- Two epochs, LoRA rank 16, identical learning rate per comparison, effective batch size eight. Memory-saving microbatching must preserve per-record loss weights, including the last partial batch.
- Start with one seed per cell; repeat the cells with a second seed if smoke validation and the budget permit. Do not select and report only the better seed.
- Record total forward tokens, steps, memory, training/evaluation wall time, full resolved configuration, dependency lock hash, source hashes, and base revisions. Equal record exposure is not equal FLOPs.

### 4. Selection and uncertainty

Primary score remains macro development NLL, with separate familiar-task retention and transfer reporting. Also report Brier, raw/calibrated ECE, confident-error rate at probability ≥0.9, accuracy at 50%/80% coverage (including ties), relevant-pair both-correct rate, and irrelevant-pair invariance/both-correct rates.

Temperature is fit on calibration only and applied unchanged to transfer. Neither the agent nor a candidate can adjust the evaluator through the trial config.

Research screening includes complete coverage, isolation, original-task retention, transfer accuracy/Brier non-regression, at least 70% held-out pair correctness, and at most 10% confidently wrong transfer answers. These are predeclared provisional thresholds, not production guarantees. A development win can become a candidate for a locked test; it must never automatically become a released model.

### 5. Compute and spending

No local training job needs to be interrupted. The Modal workspace currently reports $9.46 metered usage before this phase (covered by credits). Verified H100 rate: $3.95/GPU-hour, plus CPU/memory/storage. [Modal pricing](https://modal.com/pricing).

Use at most four concurrent containers, no automatic trial retries, and an 1,800-second per-trial timeout. Launch-time cost admission uses the GPU rate plus bounded CPU/memory requests; it excludes image build/startup/storage and is not an account-level hard spending cap. Keep the phase within the previously discussed $100 total budget, with an initial study compute bound below $20. Preserve failures rather than silently retraining or overwriting them.

Modal documentation: [images](https://modal.com/docs/guide/images), [volumes](https://modal.com/docs/guide/volumes), [GPU](https://modal.com/docs/guide/gpu), [secrets](https://modal.com/docs/guide/secrets).

## Status and deferred work

- [x] Modal CUDA/batched path and backbone-v1 study completed; MBP path retained.
- [x] v2 source policy, PAWS/SciQ conversion, contrastive prototype, and data-ablation study completed. Findings remain exploratory.
- [x] v3 minimal-pair/calibration/metric corrections and regression tests ([tests](tests/test_v3.py)).
- [x] v3 compositional generator with truth-table and boundary tests ([generator](kev/composition.py)).
- [x] v3 frozen matched suites ([decision-v3](evals/v3/decision-v3/manifest.json), [transfer-v3](evals/v3/transfer-v3/manifest.json)); both arms share 3,000 public records and 448 synthetic records.
- [x] Modal smoke and the matched 0.6B/4B comparison, seed 0 ([ledger](runs/v3-data-capacity-s0/results.jsonl)).
      Paired, record-clustered bootstrap, transfer-v3 development:
      capacity (4B vs 0.6B, same data): +17.5 pp acc CI [+13.0, +22.1] on legacy data, +19.0 pp CI [+12.3, +25.0] on compositional data;
      data (compositional vs legacy, same backbone): +3.6 pp CI [-0.4, +7.8] at 0.6B, +5.1 pp CI [0.0, +9.9] at 4B; Brier -0.053 CI [-0.102, -0.005] at 0.6B.
      Held-out compositional structures, both siblings correct: 0.6B 3%/6%; 4B 45%/52%. Held-out authorization: 4B 100% both arms (0.6B 50%).
      Held-out deadline (3-level score) stays near chance for all cells. No cell passes the 70% held-out-pair screen; none is a locked-test candidate.
      Cost: 4 H100 trials, 0.6B ~4.5 min and 4B ~13.5 min wall each, admission bound $8.86.
- [ ] Second seed for the four cells (budget permitting); then decide whether any candidate warrants a locked test.
- [ ] Deferred: option-order architecture experiments. Do not infer Jev's architecture from zero argmax flips.
- [ ] Deferred: LLM-authored product scenarios, with a separate verification model and retained provenance; needs explicit API/budget decisions.
- [ ] Deferred: 8B runs after the data-versus-capacity result, not as an automatic escalation.
- [ ] Deferred: final release/model-card/Hub updates until generalization and calibration justify them.

Relevant code: [suite builder](kev/study_v3.py), [rule generator](kev/composition.py), [experiment runner](kev/experiment.py), [benchmark](kev/benchmark.py), [Modal app](modal_app.py), [v3 tests](tests/test_v3.py).

## Autoresearch log

Maintained by `kev.autoresearch`; full table in [`runs/leaderboard.md`](runs/leaderboard.md). Selection uses development partitions only.

- **Qwen3-0.6B-Base** incumbent (v4 suites): transfer 0.605, dev 0.800, seeds [2], knobs `{"epochs": 2, "p_none_pair": 0.25}`
- **Qwen3-4B-Base**: no eligible trial yet
- **Qwen3-8B-Base**: no eligible trial yet

| round | base | trials | best transfer | best knobs | incumbent after | spend |
|---|---|---|---|---|---|---|
