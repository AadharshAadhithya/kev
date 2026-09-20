# Plan: move Kev to the Qwen3.5 base family

Status: **proposal for review**, 2026-09-20. Nothing here has been started except the probe in section 2. Numbers are on the same frozen items as every other number in this repository ([`evals/v4/transfer-v4`](evals/v4/transfer-v4/manifest.json), development partition, 764 records); untrained bases are read zero-shot from next-token letter logits with [`scripts/base_mmlu_probe.py`](scripts/base_mmlu_probe.py), the same probe used for every untrained row in the README.

## 1. What "latest Qwen base" means, checked

Every Qwen release after Qwen3 shares one text architecture, `qwen3_5_text`, and only Qwen3.5 has Base checkpoints ([Hub listing, `author=Qwen`](https://huggingface.co/Qwen)):

| family | released | Base weights | architecture (from `config.json`) |
|---|---|---|---|
| Qwen3 | 2025 | 0.6B, 1.7B, 4B, 8B, 14B, 32B ([Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base)) | dense attention, every layer |
| **Qwen3.5** | Feb–Mar 2026 | **0.8B, 2B, 4B, 9B, 35B-A3B** ([Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base), [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base)) | hybrid: 32 text layers, **24 Gated DeltaNet ("linear_attention") + 8 full attention**, `full_attention_interval: 4` ([9B config](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/main/config.json)) |
| Qwen3.6 | Apr 2026 | none ([27B](https://huggingface.co/Qwen/Qwen3.6-27B), 35B-A3B, post-trained only) | `qwen3_5_text`, 64 layers, 16 attention / 48 DeltaNet |
| Qwen3.8 | Aug 2026 | none ([27B](https://huggingface.co/Qwen/Qwen3.8-27B), 2.4T-A95B, post-trained only) | `qwen3_5_text`, same shape as 3.6-27B |

So: **Qwen3.5 Base is the newest generation Kev can train on**, and building for its architecture builds for Qwen3.6/3.8 the day their Base weights appear. Model details for the two sizes we would use (read from the configs):

| | Qwen3.5-4B-Base | Qwen3.5-9B-Base |
|---|---|---|
| Hub revision (pin) | `1001bb4d826a52d1f399e183466143f4da7b741b` | `68c46c4b3498877f3ef123c856ecfde50c39f404` |
| text layers | 32 (8 attention, 24 DeltaNet) | 32 (8 attention, 24 DeltaNet) |
| hidden | 2560 | 4096 |
| attention heads / KV heads | 16 / 4 | 16 / 4 |
| DeltaNet value heads | 32 | 32 |
| vocab | 248,320 | 248,320 |
| context | 262,144 | 262,144 |
| checkpoint parameters (incl. vision tower) | 4.7B | 9.7B |

The checkpoints are `Qwen3_5ForConditionalGeneration` (a vision tower is bundled). Loading through `AutoModelForCausalLM` gives the text-only `Qwen3_5ForCausalLM` with `.model` = `Qwen3_5TextModel`; verified on Qwen3.5-0.8B-Base under transformers 5.17.0 (forward on CPU, top token for "The capital of France is" → " Paris"). Our five delimiter tokens (`kev/model.py` [`SPECIAL`](kev/model.py#L10)) exist in the Qwen3.5 tokenizer with ids 248049–248062, so the encoding scheme carries over unchanged.

## 2. The probe: is the new base actually better for Kev?

Run 2026-09-20 on Modal H100s with [`modal_probe35.py`](modal_probe35.py) (own image: transformers 5.17, [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) for the DeltaNet kernel). Reports and per-item rows: [`runs/probes/`](runs/probes/). Paired comparisons are record-clustered bootstraps ([`kev.benchmark.paired_bootstrap`](kev/benchmark.py)).

| untrained base, zero-shot | acc | Brier | MMLU | PAWS | QNLI | SciQ | TweetEval | Emotion | authorization | **deadline** | rule: (A∨B)∧C | rule: (A∧B)∨¬C | rule: if-then-not |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.567 | 0.505 | 0.42 | 0.71 | 0.76 | 0.93 | 0.57 | 0.29 | 0.50 | 0.28 | 0.38 | 0.62 | 0.44 |
| Qwen3.5-0.8B | 0.566 | 0.510 | 0.47 | 0.66 | 0.72 | 0.90 | 0.62 | 0.21 | 0.55 | 0.28 | 0.50 | 0.56 | 0.50 |
| Qwen3-4B | 0.671 | 0.392 | 0.68 | 0.79 | 0.78 | 0.97 | 0.65 | 0.29 | 1.00 | 0.55 | 0.56 | 0.41 | 0.47 |
| Qwen3.5-4B | 0.692 | 0.395 | 0.69 | 0.84 | 0.84 | 0.99 | 0.65 | 0.28 | 0.95 | **0.68** | 0.53 | 0.44 | 0.50 |
| Qwen3-8B | 0.726 | 0.366 | 0.75 | 0.84 | 0.89 | 1.00 | 0.68 | 0.30 | 1.00 | 0.53 | 0.59 | 0.62 | 0.62 |
| Qwen3.5-9B | 0.729 | 0.356 | 0.74 | 0.84 | 0.90 | 0.99 | 0.69 | 0.31 | 1.00 | **0.82** | 0.59 | 0.47 | 0.44 |
| *Kev-8B (trained, Qwen3-8B)* | *0.796* | *0.337* | *0.70* | *0.78* | *0.91* | *1.00* | *0.79* | *0.56* | *1.00* | *0.60* | *0.97* | *0.91* | *0.59* |
| *Jev* | *0.857* | *0.211* | *0.90* | *0.79* | *0.93* | *0.99* | *0.81* | *0.59* | *1.00* | *0.93* | *0.91* | *0.97* | *0.78* |

Paired deltas, new base minus old base at matched size:

- Qwen3.5-0.8B vs Qwen3-0.6B: **+0.8 pp** [−3.3, +4.8]
- Qwen3.5-4B vs Qwen3-4B: **+2.1 pp** [−1.5, +5.8]
- Qwen3.5-9B vs Qwen3-8B: **−0.3 pp** [−4.9, +3.9]

Sources: [`runs/probes/qwen35-9b-base-base-transfer-v4/report.json`](runs/probes/qwen35-9b-base-base-transfer-v4/report.json), [`qwen35-4b`](runs/probes/qwen35-4b-base-base-transfer-v4/report.json), [`qwen35-2b`](runs/probes/qwen35-2b-base-base-transfer-v4/report.json), [`qwen35-08b`](runs/probes/qwen35-08b-base-base-transfer-v4/report.json), [`qwen3-8b`](runs/probes/qwen3-8b-base-transfer-v4/report.json), [`qwen3-4b`](runs/probes/qwen3-4b-base-transfer-v4/report.json), [`qwen3-06b`](runs/probes/qwen3-06b-base-transfer-v4/report.json); Kev-8B from [`runs/v7-final/00-trial-0/result.json`](runs/v7-final/00-trial-0/result.json); Jev from [`runs/jev-transfer-v4/report.json`](runs/jev-transfer-v4/report.json).

### Reading

1. **Overall, the new generation is not a better base on our suite.** At 9B vs 8B the difference is zero within noise; at 4B it is +2 pp with a CI that includes zero. MMLU and PAWS, the two places Kev loses most to Jev, are unchanged (0.74 vs 0.75, 0.84 vs 0.84). A version bump alone would not have been worth a port.
2. **But the gain is concentrated exactly where Kev is stuck.** On `deadline` (day-precision date arithmetic to a 3-level ordinal), Qwen3.5-9B gets **33/40** items zero-shot where Qwen3-8B gets **21/40**, and Qwen3.5-4B gets 0.68 where Qwen3-4B gets 0.55. This is the one family that no training data moved — two purpose-built families in `decision-v7`/`v8` left it at 0.45–0.60 for every Kev ([PLAN.md, "Final v7/v8 read"](PLAN.md#toward-v02-crossing-the-release-screen-2026-09-19)) — and it was the deciding family for the predeclared held-out-pair screen, which Kev-8B missed by one pair (0.69 vs 0.70).
3. **The rule-composition columns are irrelevant to the choice of base.** Every base is near chance on them zero-shot; Kev learns them from the synthetic data (Kev-8B 0.91–0.97). What carries over from the base is knowledge and arithmetic, and the low-learning-rate recipe was found precisely because it preserves base capability ([PLAN.md, "The learning rate is the lever"](PLAN.md#overnight-autoresearch-branch-researchovernight-1-pr-3)).

**Verdict: proceed**, on the narrow hypothesis that a Kev trained on Qwen3.5-9B keeps most of the base's 0.82 on `deadline`, which would lift held-out-pair correctness from 0.69 to roughly 0.80 and put the overall out-of-domain number at ~0.81 (Jev 0.857). The probe does not justify expecting gains on knowledge or paraphrase. Everything below is scoped to test that hypothesis at controlled cost, with the existing Kev-8B as the paired baseline on the same items.

## 3. Why the port is real work: the hybrid breaks the packed mask

Kev's forward pass today packs the state and every question into one sequence and uses a block-causal additive mask so that a question sees the state but never another question ([`encode`](kev/model.py#L30), [`branch_mask_batch`](kev/model.py#L75), [`hidden_batch`](kev/model.py#L145)). The mask is applied inside attention. In Qwen3.5, 24 of 32 layers are Gated DeltaNet ([Yang et al., 2024](https://arxiv.org/abs/2412.06464); implementation [`Qwen3_5GatedDeltaNet` in transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)): a recurrent state updated token by token, plus a short causal convolution. Neither respects a per-token attention mask, so in a packed sequence question 2's tokens would read a recurrent state that has already absorbed question 1's tokens. Isolation, the property the README verifies to 4e-6, would fail by construction. `transformers` confirms this: in `Qwen3_5TextModel.forward` the mask dict is only consulted for `full_attention` layers (`attention_mask=causal_mask_mapping[self.config.layer_types[i]]`); the DeltaNet path only uses the 2D mask to zero padding.

### The design: one state pass, then questions as a batch of rows

Replace "one row, masked" with "state once, branches as separate rows continuing from the state":

1. **State pass.** Run the state tokens (`<state> …`, positions `0..Ls-1`) once with `use_cache=True`. The cache holds the KV of the 8 attention layers and, for the 24 DeltaNet layers, the recurrent state and the conv state (`cache_params.layers[i].recurrent_states`, `conv_states` in the transformers implementation).
2. **Branch pass.** Build a batch of `Q` rows, one per question: `<q> instr <opt> o </opt> … <decide>`, right-padded, positions continuing from `Ls` exactly as [`encode`](kev/model.py#L30) already assigns them. Replicate the cache along the batch dimension and run the rows. The chunked prefill path takes `initial_state=recurrent_state` when a cache with previous state is present (`torch_chunk_gated_delta_rule(..., initial_state=...)`), so a multi-token continuation from a cached state is supported by the library, not something we would hack in. **This exact pattern is already running on Qwen3.5-4B in the wild:** SemIf's [`shared.py`](https://github.com/TheoLeeCJ/SemIf/blob/master/src/semif_phase1/shared.py) prefills the state once, replicates the cache with `cache.reorder_cache(torch.zeros(Q))`, and scores right-padded suffixes in one batch (positions continuing from the prefix length, `logits_to_keep` at each row's last real token); its [MLX backend](https://github.com/TheoLeeCJ/SemIf/blob/master/src/semif_phase1/mlx_backend.py) does the same with MLX-LM's prompt cache (`entry.merge([entry] * Q)`, `prepare(lengths, right_padding)`). We adopt the `reorder_cache` idiom rather than `batch_repeat_interleave`, since it is what has been exercised on this architecture. They also quantified the cost: bf16 reuse changed 5–6 of 777 argmaxes versus fresh scoring ([results](https://github.com/TheoLeeCJ/SemIf/blob/master/docs/RESULTS.md)), the same magnitude as the bf16 noise we measured on our own prefix cache, so the fp32 parity tests stay.
3. **Readout.** Gather `</opt>` and `<decide>` hidden states per row and apply the unchanged [`PointerHead`](kev/model.py) — [`_readout`](kev/model.py#L162) already takes per-question index lists.

Properties: isolation is exact **by construction** (rows are independent tensors; no mask to get wrong), on any architecture; the state is computed once, as today; FLOPs equal the packed form (the packed sequence also processes every branch token once); the serving prefix cache we shipped ([`prefix`](kev/model.py#L183), [`probs_with_prefix`](kev/model.py#L206)) is literally step 1 + step 2 and becomes the *only* path rather than an optimization. What we lose: the single-row form and the `option_isolation` flag (which cost −5.8 pp at 4B anyway, [PLAN.md](PLAN.md#overnight-autoresearch-branch-researchovernight-1-pr-3)).

**Training** through a cache is the one uncertain mechanical step (cache tensors may be updated in place or detached). Fallback that is exactly correct and simple: for training, build rows as `[state + branch_q]` with the state duplicated per row. Cost is `Q×` the state tokens instead of `1×`; our training records average ~1.3 questions, so ≤ 1.5× the state compute, and the state is short (≤ 384 tokens). Start there; optimize to the cache-continuation form only if it is measurably needed.

**Equivalence check that gates everything:** on the existing Qwen3-based Kev-4B, the row-batched forward must reproduce the packed-mask forward's probabilities to fp32 noise on the 24-record parity set used for the serving changes. If it does, the same code path is the reference for the hybrid.

## 4. Dependencies and their risks

| item | today | needed | evidence / risk |
|---|---|---|---|
| transformers | `>=4.51,<4.58` ([pyproject](pyproject.toml)) | `>=5.17` (qwen3_5 is not in 4.x) | 5.17.0 loads Qwen3.5-0.8B-Base and runs a forward (verified in a scratch venv). **Risk:** 5.x API changes for the existing Qwen3 path; gate with the parity tests in [`tests/test_v3.py`](tests/test_v3.py) (merged-vs-unmerged, prefix-vs-full, bucket padding) and a bf16 dev-set re-score of the published Kev-4B (must match [`runs/v7-rc3/01-trial-1/result.json`](runs/v7-rc3/01-trial-1/result.json) within bf16 noise). |
| peft | `>=0.15` | a version tested with transformers 5 (`>=0.18`, used in the probe image) | LoRA target names on DeltaNet layers differ (`in_proj_qkvz`, `in_proj_ba`, `out_proj`); attention and MLP names are unchanged. |
| DeltaNet kernel | – | [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) + triton on CUDA | Without it transformers falls back to a reference PyTorch implementation ("correct but much slower", its own warning). CUDA-only. **Mac serving runs the fallback**; speed unmeasured — measured in phase 1 before committing to a Mac story. |
| Modal image | pins `uv.lock` | a second image (or the upgraded lock) | [`modal_probe35.py`](modal_probe35.py) already builds the transformers-5 image; the main app follows once the lock is upgraded. |
| suites | base revisions pinned per suite ([`kev.suite.freeze`](kev/suite.py#L146), [`validated_trial`](kev/experiment.py#L43) requires a 40-hex `base_revision` for unpinned bases) | pass `base_revision` per trial (already supported) or freeze `decision-v9` = v7 with Qwen3.5 revisions pinned | Dev/test bytes stay identical to v4 either way, so every number remains comparable. |
| vision tower | – | ignored | `AutoModelForCausalLM` loads the text model only (verified). Weight download is the full 9.7B checkpoint. |
| vocabulary | 152k | 248k | Embedding matrix is frozen; the pointer head reads hidden states, not logits. No change. |

## 5. Work plan

Each phase has a stop condition. Costs are Modal H100 at the rates we have been paying (Kev-4B trial ≈ $3.50, Kev-8B ≈ $6; [PLAN.md, "Compute and spending"](PLAN.md#5-compute-and-spending)).

### Phase 0 — MMLU-Pro, a buried-state variant, and cross-benchmarks with SemIf (independent of the port; ~4 h, ~$3)
- Add `mmlu_pro` to [`kev/data.py`](kev/data.py) as an **eval-only** source ([TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), [Wang et al., 2024](https://arxiv.org/abs/2406.01574); 10 options, `answer_index`), rendered as a Choice with neutral keys like the other MCQ converters. Pin the dataset revision (`b189ec765a…`).
- Freeze `transfer-v5` = transfer-v4 items + 200 MMLU-Pro items (new version; v4 numbers stay valid, and v5 minus the new source equals v4 byte-for-byte for comparability).
- Score Jev, the untrained bases, and Kev-4B/8B on it. Expectation, from openjev's report on the same benchmark: Jev ~0.83, untrained one-token readouts ~0.6, Kev well below Jev — the honest number for a one-pass model on a benchmark designed for chain-of-thought. It replaces the saturated 4-way MMLU in the knowledge column of the README.
- Add an eval-only **buried-state variant** to `transfer-v5`: the same record with the state embedded in unrelated background text (SemIf's "irrelevant context" perturbation and localjev's 2,048-word distraction condition both found this is where small models fall over; we train at ≤ 384 state tokens and have never measured it).
- **SemIf-style untrained baseline on our suite**: Qwen3.5-4B *instruct* (their pinned revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) with their exact prompt ([`core.direct_messages`](https://github.com/TheoLeeCJ/SemIf/blob/master/src/semif_phase1/core.py): system instruction + JSON `{evidence, criterion, options}` through the chat template, letter logits) on `transfer-v4`, alongside our base-model probe row. Our probe uses the *base* model and a plain prompt; theirs may be the stronger untrained readout and is the one people will compare against.
- **Kev on SemIf's fixtures**: convert their committed [`authored144.jsonl`](https://github.com/TheoLeeCJ/SemIf/blob/master/benchmarks/data/authored144.jsonl) and [`perturbations108.jsonl`](https://github.com/TheoLeeCJ/SemIf/blob/master/benchmarks/data/perturbations108.jsonl) (three families: evidence interpretation, rule application, candidate selection; 2–3 options with ids and descriptions) into System One requests and score Kev-4B/8B with the labels they ship. Their reported direct-logit numbers are 0.813 balanced accuracy on the 144 and 0.723 on the 36 perturbation bases. Also run **live Jev** on the 144 rows (cents) — their Jev column is agreement with published outputs on a different subset, not a live read.

### Phase 1 — transformers 5 branch and row-batched forward (½–1 day, ~$2)
- Branch `qwen35`. Upgrade the lock to transformers 5.x / peft ≥ 0.18; run the full test suite and the bf16 re-score of Kev-4B; fix what 5.x moved.
- Implement `DecisionModel.forward_rows` (state pass + row batch) alongside the packed path; add `test_rows_match_packed` on the smoke checkpoint (fp32, CPU, < 1e-4) and on Kev-4B (bf16, 24 records, 0 argmax flips).
- Load Qwen3.5-0.8B-Base through `DecisionModel`, run the isolation and packing checks from [`kev.evaluate`](kev/evaluate.py) on the untrained model (they test the *mechanism*, not accuracy). **Stop if** the DeltaNet cache continuation does not reproduce a single-row forward to fp32 noise; that would mean the library's chunked-prefill-with-initial-state path is not usable and we would need to write the continuation ourselves before spending on training.
- Measure the CPU/MPS fallback speed of the 0.8B and 4B forward. Record it; it decides whether "serves on a Mac" survives for Qwen3.5-based Kevs or waits on MLX.

### Phase 2 — the controlled experiment (~1 day wall, ~$25)
Same data, same recipe, new base. Nothing else changes, so any difference is the base.
- `decision-v7` training partition, lr 5e-5, 2 epochs, LoRA r=16, `p_none_pair 0.25` (the released recipe, [`experiments/v7-final.json`](experiments/v7-final.json)); LoRA on attention + MLP projections in all layers, DeltaNet in/out projections included (one ablation without them at 4B).
- Cells: Qwen3.5-4B × 2 seeds, Qwen3.5-9B × 2 seeds, scored on `transfer-v4` (and `transfer-v5` once Phase 0 lands). Paired bootstraps against Kev-4B and Kev-8B on the same items ([`kev.autoresearch compare`](kev/autoresearch.py)).
- Read-outs, in order of what the hypothesis predicts: `deadline` (≥ 0.75 expected if base skill is retained), held-out pairs (≥ 0.70 on both seeds is the old release screen), overall transfer, MMLU/PAWS retention (should be ≥ Kev-8B's 0.70/0.78; if lower, the recipe is eroding the new base more and a small lr sweep at 3e-5/2e-5 is the next $10), Brier and confident-error rate.
- **Stop if** Kev-9B(3.5) is not above Kev-8B on transfer with a CI excluding zero *and* deadline did not move. Then the result is written up in PLAN.md as a negative and the family stays on Qwen3.

### Phase 3 — promotion (only if Phase 2 passes; ~½ day, ~$5)
- One locked-test read for each winning size ([`modal_app.py::locked_test`](modal_app.py)), gated if the pair screen passes on both seeds.
- Cards for `kev-4b`/`kev-9b` on Qwen3.5 (naming: the Hub repo is by size, so `jaredpalmer/kev-9b`; `kev-8b` stays as is), README table and figures regenerated from result files ([`scripts/plot_family.py`](scripts/plot_family.py), [`scripts/plot_tweet.py`](scripts/plot_tweet.py)), the "untrained base" rows switched to Qwen3.5.
- Serving: the row-batched path is the prefix-cache path; re-run the serving benchmark in the README's "Serving performance" section on CUDA.
- **Mac serving through MLX-LM**, promoted from "deferred" on SemIf's evidence that MLX-LM runs Qwen3.5 with prompt-cache replication today ([docs/MLX.md](https://github.com/TheoLeeCJ/SemIf/blob/master/docs/MLX.md): MLX 0.32.2, MLX-LM pinned past the Qwen recurrent q/k-norm fix, vision weights stripped by the sanitizer, 4/8-bit in memory). Kev's additions are small: load the fp32-merged LoRA weights (we already merge at load) into the MLX-LM Qwen3.5 text model, take hidden states from the inner model instead of `lm_head` logits, and apply the pointer head in MLX. Gate with the same parity tests against the PyTorch fp32 path. Measure the torch fallback first (Phase 1); if it is within ~2× of today's MPS latency, MLX waits.

### Deferred, deliberately
- Qwen3.5-35B-A3B-Base: 70 GB in bf16, MoE, serving footprint out of the laptop story; only after the 9B result.
- A browser/WebGPU path (SemIf's distribution advantage, via wllama/GGUF). Kev's pointer head and per-token positions do not fit llama.cpp as-is; with the row-batched design a GGUF export of the merged model plus a small JS head is conceivable, but it is its own project.
- Post-trained Qwen3.6/3.8 as bases: changes the recipe and removes the untrained-base comparison; revisit when Base weights exist.

## 6. Competitive context: SemIf

[SemIf](https://github.com/TheoLeeCJ/SemIf) (formerly OpenJev; 2.3k stars) is an **untrained** readout: frozen Qwen3.5-4B instruct, chat-template JSON prompt, probabilities from letter logits, with prefix-reuse and parallel-suffix modes and a WebGPU browser demo. Its own results page states the probabilities are "uncalibrated as decision confidence" and that "the next justified phase is targeted training for decision semantics and calibration" — i.e. Kev. What it means for us:

- **It validates the port design** (section 3) and hands us the MLX path (Phase 3). Its author solved the hybrid-cache replication problem on exactly our target base.
- **It sharpens the claim Kev has to make.** Their headline is "3.8 pp behind Jev" on *agreement with Jev's published answers* over 102 curated public cases. On labelled frozen items, untrained Qwen3.5-4B scores 0.692 on our suite and Kev-4B 0.790: training the readout is worth about +10 pp at 4B, and it is what buys calibration (Brier, confident-error rate), which they do not measure. Once a Qwen3.5-based Kev-4B exists the comparison is apples to apples on the same base, which is the cleanest possible demonstration of what training adds.
- **Their evaluation is thinner than ours** (144 authored rows, agreement rather than ground truth, no live Jev, no paired uncertainty). Phase 0 scores both ways on frozen items and runs live Jev on their rows; the neutral-ground position is worth more than winning any single cell.
- **Their distribution is stronger than ours** (browser demo, "no waitlist"). Not something this plan addresses; noted in deferred work.

## 7. Decision criteria, stated in advance

Ship a Qwen3.5-based Kev-9B as the new top of the family **only if**, on the same 764 items: transfer accuracy ≥ Kev-8B's 0.796 with the paired CI excluding zero, `deadline` ≥ 0.75, MMLU and PAWS not below Kev-8B by more than seed noise (~1 pp), and Brier ≤ 0.34. Ship a Qwen3.5-based Kev-4B if it beats Kev-4B (0.790) by the same test. Otherwise the outcome is a documented negative result and the Qwen3 family remains the release. Locked test read once per candidate, as always.

## 8. Budget and time

| phase | wall time | Modal |
|---|---|---|
| 0 MMLU-Pro, buried-state variant, SemIf cross-benchmarks | 4 h | ~$3 |
| 1 transformers 5 + row forward + hybrid smoke | ½–1 day | ~$2 |
| 2 controlled experiment | ~1 day | ~$25 |
| 3 promotion (+ MLX serving if the torch fallback is slow) | ½–1 day | ~$5 |
| total | ~3–4 days | **~$37** of the remaining ~$240 |
