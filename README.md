# Kev

Jev-inspired decision model. Typed questions in, calibrated probabilities out, one forward pass.

<p>
  <a href="https://github.com/jaredpalmer/kev/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/jaredpalmer/kev/ci.yml?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="https://huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd"><img alt="Weights: Kev-0.6B · 4B · 8B" src="https://img.shields.io/badge/WEIGHTS-0.6B%20%C2%B7%204B%20%C2%B7%208B-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="https://huggingface.co/datasets/jaredpalmer/kev-suites"><img alt="Frozen eval suites" src="https://img.shields.io/badge/EVAL%20SUITES-frozen-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="PLAN.md"><img alt="Research log" src="https://img.shields.io/badge/RESEARCH%20LOG-PLAN.md-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
</p>

![Kev playground](docs/playground.png)

Kev is a LoRA adapter and a small readout head on top of a Qwen3 base model (0.6B, 4B or 8B). It reads a document once and answers many typed questions about it in parallel, in a single prefill pass with no decoding. The document and every question are packed into one sequence; a block-causal mask lets each question see the document but never another question. A pointer head then scores each question's options against its decision token and applies softmax. Those probabilities are the output. The head is trained with cross-entropy against labelled outcomes, so the probabilities are learned rather than generated as text.

The architecture follows the reconstruction of TypeSafe's Jev in [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked). The API follows TypeSafe's [System One](https://docs.typesafe.ai/api) contract, so the official `typesafe-sdk` works against a local Kev server with a `base_url` change.

## Highlights

- **Three question types.** `noul` (yes/no), `choice` (2–255 options), `score` (ordered levels). One shared readout.
- **One pass, many answers.** The state is encoded once. Questions run as isolated branches under a block-causal mask.
- **Isolation is exact.** A question cannot see a sibling question. Packed and separate requests agree to `4e-6`.
- **Probabilities, not prose.** Trained with cross-entropy on labelled outcomes. Out of domain, Kev-4B has Brier 0.33 and 8% confident errors on sources it never saw.
- **Drop-in API.** `POST /v1/systemone` with TypeSafe's request and response shapes. Their SDK's quickstart runs unmodified.
- **A family, measured the same way.** 0.6B, 4B and 8B checkpoints scored on frozen, checksummed suites with a locked test, against the real Jev on the same items. Out of domain: Kev-4B 0.79, Kev-8B 0.80, Jev 0.86.
- **Runs on a laptop; trains in the cloud.** Kev-0.6B trains in minutes on one H100 (or ~2 h on an Apple M5); the 4B/8B recipes take 40–70 min on one H100 via Modal and serve on a 32 GB Mac in bf16.

![Kev family vs Jev on sources Kev never trained on](docs/kev-family.png)

## Installation

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and Node 20+ for the playground. Serving is tested on Apple Silicon (MPS); training and evaluation on CUDA (H100 via Modal) and MPS.

```bash
git clone https://github.com/jaredpalmer/kev.git && cd kev
uv sync --extra serve
cd playground && npm install && cd ..
```

### Download the weights

All checkpoints are on the Hugging Face Hub in the [Kev collection](https://huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd). `--run` accepts a Hub id; the base model downloads on first load. **Kev-4B is the one to start with**: the best accuracy per byte, and it serves on a 32 GB Mac in bf16.

```bash
KEV_DTYPE=bf16 uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
```

| checkpoint | base | in-distribution (dev / locked test) | out-of-domain (dev / locked test) | serve on a Mac | card |
|---|---|---|---|---|---|
| [Kev-8B](https://huggingface.co/jaredpalmer/kev-8b) | Qwen3-8B-Base | 0.863 / **0.870** | **0.796** / 0.780 | bf16, ~0.5 s | [card](docs/model-cards/kev-8b.md) |
| [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b) | Qwen3-4B-Base | 0.854 / 0.856 | 0.790 / **0.806** | bf16, ~0.3 s | [card](docs/model-cards/kev-4b.md) |
| [Kev-0.6B](https://huggingface.co/jaredpalmer/kev-0.6b) | Qwen3-0.6B-Base | 0.801 / 0.808 | 0.620 / 0.642 | fp32, ~0.1 s | [card](docs/model-cards/kev-0.6b.md) |
| [Kev-0.5B](https://huggingface.co/jaredpalmer/kev-0.5b) · prototype | Qwen2.5-0.5B | 0.712 / – | 0.561 / – | fp32, ~0.2 s | [card](docs/model-cards/kev-0.5b.md) |
| Jev (hosted reference) | – | 0.845 / – | 0.857 / – | | |

Same frozen items for every row (`evals/v4`: 1,200 in-distribution questions from the trained sources; 764 out-of-domain records from six public sources Kev never trained on plus held-out programmatic policy rules). Development partitions selected the checkpoints; the locked test was read once per checkpoint and is recorded in each card. Latencies are five 3-way questions on a ~230-token state, M5. Kev-0.5B is the original laptop-trained prototype (September 2026), kept on the Hub for reference; it was never scored on the locked test. The same three checkpoints, as tarballs with SHA-256 sums, are attached to the [GitHub release](https://github.com/jaredpalmer/kev/releases/tag/kev-family).

![Kev benchmarks](docs/kev-benchmark.png)

## Quick Start

Start the server:

```bash
KEV_DTYPE=bf16 uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
```

Ask it something:

```bash
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
  "model": "kev-latest",
  "questions": {
    "department":  {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                 "shipping": "Delivery status, delays, lost packages",
                                 "billing": "Charges, invoices, payment problems"}},
    "escalate":    {"type": "noul",  "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  }}'
```

```json
{
  "model": "kev-latest",
  "answers": {
    "department":  { "type": "choice", "choice": "returns", "confidence": 0.83,
                     "probabilities": { "returns": 0.89, "shipping": 0.04, "billing": 0.07 } },
    "escalate":    { "type": "noul", "noul": 0.54 },
    "frustration": { "type": "score", "score": 1.25, "confidence": 0.88,
                     "legend": { "0": "Calm", "1": "Frustrated", "2": "Very angry" },
                     "probabilities": { "0": 0.00, "1": 0.75, "2": 0.25 } }
  },
  "usage": { "input_tokens": 101, "output_tokens": 161 },
  "latency_ms": 277
}
```

(Kev-4B, bf16 on an M5.)

Or use the TypeSafe SDK:

```python
from typesafe_sdk import TypeSafeClient, Choice, Noul, Score

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009", model="kev-latest")
r = client.system_one(
    state="I was charged twice. Please fix this ASAP.",
    questions={
        "billing": Noul(instructions="Is this ticket about billing?"),
        "tone": Choice(instructions="What is the customer's tone?", criteria={"calm": None, "frustrated": None, "angry": None}),
        "urgency": Score(instructions="How urgent is this ticket?", criteria=["can wait", "this week", "today"]),
    },
)
r.nouls["billing"].noul, r.choices["tone"].choice, r.scores["urgency"].score
```

### Playground

```bash
cd playground && npm run dev -- -p 3001
```

Open [localhost:3001](http://localhost:3001). Load a preset, edit the state and questions, press `⌘↵`. **Packed vs separate** compares one N-question request with N single-question requests. **Permute** re-asks a Choice under six option orders. The **Isolation probe** and **Boundary forgery** presets reproduce the two experiments from the blog post.

**Chess** at [localhost:3001/chess](http://localhost:3001/chess): the legal moves are the options of one Choice question, the board is the state, and a Score rates the position in the same request. Play the model or watch it play itself; games are kept in `localStorage`.

![Kev chess: every move is a Choice question](docs/chess.png)

### Serving performance

`kev.serve` folds the LoRA into the base weights in fp32 before any cast (exact; in bf16 it is *closer* to the fp32 numbers than running the adapter unmerged), uses SDPA on Apple GPUs, pads to 64-token buckets on MPS (per-shape kernel warm-up), and keeps a small LRU of **state-prefix KV caches** so a repeated state only pays for its question branches — exact, because under the block-causal mask the state's activations do not depend on the questions. Kev-4B, bf16, M5, five 3-way questions, median model time:

| | before | now |
|---|---|---|
| short state (232 tokens), new text every call | 442 ms | **296 ms** |
| long state (772 tokens), new text every call | 1290 ms | **861 ms** |
| long state, repeated (agent loops, permute/separate probes) | 1214 ms | **242 ms** |

Every change is parity-tested against the exact fp32 path (`tests/test_v3.py`); on 24 out-of-domain records the served bf16 probabilities are within 0.017 of fp32 with zero argmax flips. Knobs: `KEV_MERGE=0`, `KEV_ATTN=eager`, `KEV_SHAPE_BUCKET=1`, `KEV_PREFIX_CACHE=0` restore the old behaviour. On an H100 the same request is ~40 ms.

## API

### `POST /v1/systemone`

```jsonc
{
  "state": "…",                          // string | object | array — the content to evaluate
  "model": "kev-latest",
  "questions": {
    "<id>": {                            // you choose the id; the model never sees it
      "type": "noul" | "choice" | "score",
      "instructions": "…",               // string | object | array
      "criteria": …                      // noul: {true?, false?}  choice: {option: description|null}  score: [level, …]
    }
  }
}
```

| Answer type | Fields | Derived from the distribution `p` |
|---|---|---|
| `noul` | `noul` | `p[yes]` |
| `choice` | `choice`, `probabilities`, `confidence` | `argmax`, `p` by option key, `(p_max − 1/K) / (1 − 1/K)` |
| `score` | `score`, `legend`, `probabilities`, `confidence` | `Σ k·p[k]`, level index → text, `p` by level index |

Structured `instructions`, `criteria` and `state` are flattened to labelled text. Option and branch delimiters cannot be forged from user text. Validation errors return `422`.

### Other endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/models` | Model, base and run info |
| `POST` | `/v1/systemone/permute` | One Choice question under N option orders |
| `POST` | `/v1/systemone/separate` | Each question in its own pass, for comparison |

There is no authentication. The server is intended for local use.

## How It Works

```mermaid
flowchart LR
    A[API request<br/>state + typed questions] --> B[render to text<br/>api.to_record]
    B --> C[pack into one sequence<br/>model.encode]
    C --> D[block-causal mask +<br/>branch position ids]
    D --> E[causal LM backbone<br/>Qwen base + LoRA<br/>prefill only]
    E --> F[pointer readout<br/>decide token · option tokens]
    F --> G[softmax per question]
    G --> H[API response<br/>choice · confidence · score]
```

**Packing.** The state and every question go into one token sequence. Reserved tokens mark the structure.

```
<state> …state…
<q> instructions <opt> option 1 </opt> <opt> option 2 </opt> … <decide>    ← question 1
<q> instructions <opt> option 1 </opt> <opt> option 2 </opt> … <decide>    ← question 2
```

**Mask.** Position `i` attends to `j` when `j ≤ i` and `j` is in the state or in the same question as `i`. State tokens are computed once. Question branches never see each other.

```mermaid
flowchart TB
    subgraph S[state prefix — computed once]
        s1[t1] --> s2[t2] --> s3[t3]
    end
    subgraph Q1[question 1 branch]
        q1a[q] --> q1b[opt] --> q1c[opt] --> q1d[decide]
    end
    subgraph Q2[question 2 branch]
        q2a[q] --> q2b[opt] --> q2c[opt] --> q2d[decide]
    end
    S --> Q1
    S --> Q2
    Q1 -. no attention .- Q2
```

**Positions.** Each branch restarts its position ids after the state. Every question sees "state, then one question". Question order does not matter.

**Readout.** A pointer head scores each option's `</opt>` hidden state against the `<decide>` hidden state and applies softmax. `K` is whatever the request sends. `<decide>` follows all options, so the model reads the full list before scoring; this is what makes "none of the above" work.

**Training.** LoRA (r=16) on the backbone, head from scratch, cross-entropy on the option distribution. Training data and live requests go through the same renderer, so the model never meets a format at inference that it did not see in training.

## Training

Training data are public datasets converted to TypeSafe-shaped requests plus programmatic policy data, frozen into checksummed suites (`evals/`). Kev-4B and Kev-8B train on `decision-v7`: ten public sources at 1,000 records each, nine policy template families as minimal pairs (896 records), and 1,680 records from 60 randomly generated rule structures rendered in four styles; two epochs, LoRA r=16, **lr 5e-5** — the single largest recipe improvement we found, because the default 2e-4 erodes what the base model already knows (details and the base-model probe in [PLAN.md](PLAN.md)). Kev-0.6B trains on the same suite at lr 1e-4.

```bash
# sanity run, ~1 minute
uv run python -m kev.train --n_per_source 40 --accum 4 --out runs/smoke

# Kev-0.6B on a Mac (~2 h on an M5) or a few minutes on one H100
uv run python -m kev.train --suite evals/v7/decision-v7 --base Qwen/Qwen3-0.6B-Base --epochs 2 --lr 1e-4 --p_none_pair 0.25 --out runs/kev-0.6b

# the Kev-4B recipe (one H100 via Modal, ~50 min; see below)
uv run python -m kev.train --suite evals/v7/decision-v7 --base Qwen/Qwen3-4B-Base --epochs 2 --lr 5e-5 \
    --batch 4 --accum 2 --dtype bf16 --checkpointing 1 --p_none_pair 0.25 --device cuda --out runs/kev-4b
```

| Flag | Default | Purpose |
|---|---|---|
| `--base` | `Qwen/Qwen3-0.6B-Base` | Causal LM backbone (any Qwen2.5/Qwen3 base; Qwen3-4B/8B for the released checkpoints) |
| `--n_per_source` | `1000` | Records sampled per dataset |
| `--holdout` | – | Sources to exclude, e.g. `mnli,sst5`, for out-of-source evaluation |
| `--perm_kl` | `0` | Optional symmetric KL between predictions under two option orders |
| `--ord_w` | `0` | Optional ranked probability score for ordered levels |
| `--suite` | – | Train on a frozen suite's training partition |
| `--batch`, `--dtype` | `1`, `fp32` | Padded batch size; `bf16` autocast on CUDA (fp32 master weights) |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--p_none_pair` | `0` | Fraction of Choice records that also emit a none-of-the-above minimal pair (true option present / removed) |
| `--option_isolation` | `0` | Option spans as isolated sub-branches with shared positions: exact permutation invariance (costs accuracy at 4B) |
| `--lora_targets`, `--head_lr`, `--weight_decay` | `all`, `=lr`, `0.01` | Low-drift knobs; none beat plain lr 5e-5 |

The released checkpoints used cross-entropy without either extra loss (`--perm_kl`, `--ord_w` are research knobs; neither beat the plain recipe). The optional ordinal loss compares cumulative probabilities, a proper scoring rule, rather than absolute error of the expected level.

On a Mac, run one training job at a time; two jobs on the same Apple GPU slow each other by about 10×. The MBP path is kept working, but anything longer than a smoke run goes to Modal.

### Modal

Studies run as one H100 container per trial, in parallel, with results pulled back into `runs/` and ranked by the same code that runs locally.

```bash
uv run modal token new                                    # once; opens the browser
KEV_GPU=T4 uv run modal run modal_app.py::smoke           # end-to-end check, ~1 minute of GPU

uv run modal deploy modal_app.py                          # once; studies run on the deployed app and survive disconnects
uv run modal run modal_app.py::study \
    --suite evals/v7/decision-v7 --plan experiments/v7-final.json \
    --name my-study --transfer evals/v4/transfer-v4 --budget 30 --timeout 7200
uv run modal run modal_app.py::pull --name my-study       # results -> runs/my-study, ranked
uv run modal run modal_app.py::locked_test --trial my-study/00-trial-0 --name my-candidate   # one read, ever
```

A plan is a JSON list of 1–8 trials over an allowlisted set of training parameters (`kev/experiment.py`). Each trial records the local git commit, the suite hash, and the hashes of the shipped `kev/*.py`; the container refuses to run if they differ from what the launcher hashed. Training uses TF32 and optional bf16; evaluation is fp32-exact (TF32 alone moves probabilities by ~1e-3, enough to trip the isolation gate). Measured: 0.019 s/record for Qwen2.5-0.5B at batch 8 on an H100 vs 0.34 s/record on an M5, ~$0.15–0.30 per 0.5B trial.

## Evaluation

```bash
uv run python -m kev.benchmark --run jaredpalmer/kev-4b --suite evals/v4/transfer-v4 --out runs/my-eval      # out of domain
uv run python -m kev.benchmark --run jaredpalmer/kev-4b --suite evals/v4/decision-v4 --out runs/my-eval-id   # in distribution
uv run python -m kev.benchmark --remote http://127.0.0.1:8009 --suite evals/v4/transfer-v4 --out runs/my-remote   # any System One endpoint
```

Every number in this README comes from these commands on frozen suites (accuracy, Brier, ECE, confident-error rate, held-out-pair correctness, permutation flips, packed-vs-separate isolation, per-source breakdowns, record-clustered paired bootstraps against a reference run via `kev.compare`).

### Frozen research suites

Every number in this README after the table above comes from frozen, checksummed suites under `evals/`: separate training, calibration, development, and locked test partitions; pinned dataset and base-model revisions; per-record provenance. Development partitions select models; the locked test is read once per published candidate (`--allow-test`, or `modal_app.py::locked_test`, which refuses a second read). Manifests and dev/test partitions are in git; training partitions over 10 MB are fetched from the [`jaredpalmer/kev-suites`](https://huggingface.co/datasets/jaredpalmer/kev-suites) mirror and verified against the manifest hash on first use.

| suite | trains on | evaluates | used for |
|---|---|---|---|
| `decision-v4` / `v6` | 10–13 public sources + programmatic policy pairs | 1,200 in-distribution questions | model selection, Kev-4B / Kev-8B |
| `transfer-v4` | nothing | QNLI, SciQ, TweetEval, PAWS, MMLU, Emotion + held-out policy rule structures | out-of-domain, every trial |
| `decision-v1`, `transfer-v1` | six sources | first prototype (Kev-0.5B) vs Jev comparison | historical |

```bash
uv run python -m kev.benchmark --run jaredpalmer/kev-4b --suite evals/v4/transfer-v4 --out runs/my-eval
uv run python -m kev.experiment --suite evals/v4/decision-v4 --plan experiments/auto/lowdrift-4b-v4.json --out runs/my-study --transfer evals/v4/transfer-v4
uv run python -m kev.autoresearch leaderboard        # rebuild runs/leaderboard.md from every study
```

Trials are configuration-only: `kev.experiment` refuses configs outside a bounded allowlist, records code, suite, and git hashes, checks complete coverage, isolation, and packing, scores the transfer suite, and never reads the locked test. `kev.autoresearch` runs bounded hill-climb rounds over that allowlist on Modal and keeps the leaderboard; 90 trials so far, all in [`runs/leaderboard.md`](runs/leaderboard.md).

### Comparison with Jev

`kev.jev` scores the same frozen partitions against the real `typesafe-ai/jev` through Vercel AI Gateway (AI SDK 7 `experimental_evaluate`, cost-capped; about two cents per suite). Jev is the hosted reference product; its training exposure to these public datasets is unknown, so this is a shared-item comparison, not a controlled ablation.

| out-of-domain, `transfer-v4` dev (764 records) | Qwen3-8B base, untrained | Qwen3-30B-A3B base, untrained | Kev-0.5B (prototype) | Kev-0.6B | Kev-4B | Kev-8B | Jev |
|---|---|---|---|---|---|---|---|
| accuracy | 0.726 | 0.707 | 0.561 | 0.620 | 0.790 | 0.796 | **0.857** |
| Brier (lower is better) | 0.366 | 0.365 | 0.534 | 0.536 | 0.328 | 0.337 | **0.211** |
| confident errors (p ≥ 0.9 and wrong) | 0.6% | 0.5% | 2.0% | 10.8% | 8.2% | 9.9% | 3.7% |
| held-out policy rules, both siblings correct | 0.55 | 0.44 | 0.09 | 0.08 | 0.73 | 0.69 | **0.86** |
| MMLU / PAWS | 0.75 / 0.84 | 0.79 / 0.82 | 0.40 / 0.50 | 0.50 / 0.59 | 0.65 / 0.72 | 0.70 / 0.78 | 0.90 / 0.79 |

The two untrained columns are the base models read zero-shot from next-token letter logits (`scripts/base_mmlu_probe.py`): Kev-8B beats its own base by +5.8 pp [+1.8, +10.0] and the untrained 30B-A3B by +9.7 pp [+4.7, +14.4], while both untrained models beat every Kev on knowledge (MMLU) and paraphrase (PAWS) — the fine-tune still gives back some of what the base knows.

What the controlled studies established (record-clustered paired bootstraps, seeds replicated; full log in [PLAN.md](PLAN.md)):

- **Capacity dominates out of domain.** With public examples and synthetic budget held equal, 0.6B → 4B is +14–19 pp; 4B → 8B is +1.5–2 pp.
- **Fine-tuning erodes base capability, and the learning rate controls it.** The 4B base scores 0.69 on the same MMLU items zero-shot; the default recipe trained it down to 0.60–0.66. lr 5e-5 recovers most of it: +4.7 pp [+0.4, +9.6], replicated at three seeds on 4B and 8B.
- **More public data raises in-distribution accuracy and lowers or flattens transfer.** Knowledge MCQ sources lift MMLU a few points without moving the total.
- **Programmatic contrastive policy pairs** teach the trained rule structures (0.85–1.0) and transfer partially to unseen ones (0.5–0.67 at 4B/8B, near chance at 0.6B); none-of-the-above minimal pairs fixed the "none" shortcut in-domain (0.75 → 0.93 at 4B).
- **Structural diversity in the synthetic rules** (60 random rule trees with negation anywhere, instead of eight fixed shapes) fixed the two failing held-out compositions (0.62–0.66 → 0.75–0.97) and produced the released 4B/8B checkpoints (+3 pp transfer). Day-precision date arithmetic (`deadline`) did not move under any data we generated (0.45–0.60; untrained bases 0.53; Jev 0.93).
- Fourteen one-knob mutations around the low-lr recipe, anchoring to the base model's own distribution, and weight-space interpolation all land within ±1 pp: the remaining gap to Jev is MMLU, PAWS, Emotion, and date arithmetic, not hyperparameters.

The prototype's first comparison (Kev-0.5B on `decision-v1`/`transfer-v1`: −1.8 pp in-distribution with a CI including zero, **−19.1 pp [−23.1, −15.0]** out of domain) is kept in `runs/kev-vs-jev-v1.json` and `runs/kev-vs-jev-transfer-v1.json`. Figures: `uv run python scripts/plot_family.py` (per-source figure above), `uv run python scripts/plot_tweet.py` (benchmark card); both read saved result files only and share `scripts/chartstyle.py`.

The mechanism tests below are from `kev.evaluate` on the prototype; every released checkpoint reproduces the isolation and packing results exactly (max delta 4e-6, checked on every trial).

| Mechanism test | Result |
|---|---|
| Isolation — secret in sibling question / absent / in state | `p = 0.03` / `0.03` / **`0.99`** |
| Packed vs separate — max probability difference | **`3.7e-6`**, packed 2.0× faster |
| Permutation — argmax flips under 4 option orders | 7.4% |
| IIA — log-odds shift from one irrelevant option | 0.13 mean, 0.34 p90 |
| Boundary forgery — fake delimiters in option text | option count unchanged, forged option `p ≤ 0.09` |

## Limitations

- **Out of domain it trails Jev by 6–7 points** at 4B/8B and by 24 at 0.6B. The gap is concentrated in knowledge (MMLU 0.69–0.75 vs 0.90), paraphrase (PAWS), noisy-label emotion, and date arithmetic. Fine-tuning still loses some of what the base model knows even at lr 5e-5.
- **Held-out rule reasoning** (unseen compositions of policy conditions) is 0.62–0.73 both-siblings-correct depending on the seed; Jev is 0.86. The remaining failures are concentrated in day-precision date arithmetic, which no data we generated has moved.
- **Calibration is in-distribution.** Temperature fitted in-domain does not transfer; out-of-domain probabilities are usable but not calibrated (ECE ~0.1).
- **Product-shaped questions** with no training analogue are not guaranteed; the low-drift 4B/8B recipes carry fewer task priors than the 0.6B and can answer differently on the same input. Measure on your own data.
- **Context.** Trained at 384 state / 1,024 branch tokens; serving caps at 8,192. Jev allows ~32k per branch.
- **Serving.** One request at a time; the state prefix is cached across requests (`KEV_PREFIX_CACHE`, states ≥ 384 tokens) but there is no cross-request batching. 8B needs bf16 (`KEV_DTYPE=bf16`) on a 32 GB Mac.
- **Score confidence** uses a stand-in formula. TypeSafe has not published theirs.

## Development

```bash
uv run python -m pytest tests/test_unit.py -q                                      # no weights, no server; runs in CI
KEV_BASE_URL=http://127.0.0.1:8009 uv run --extra serve python -m pytest tests/test_api.py -q   # against a running server
cd playground && npm run lint && npx tsc --noEmit -p .
```

`tests/test_api.py` runs the TypeSafe docs' example requests and the official SDK against the local server.

<details>
<summary>Troubleshooting</summary>

- **`MPS backend out of memory` while training.** Do not enable `output_hidden_states`; read `last_hidden_state` from the bare backbone. Do not add tokens with peft `trainable_token_indices`. Lower `--n_per_source` on small machines.
- **Playground shows `connecting…` and buttons do nothing.** Next.js 16 dev only trusts the hostname it started with. Use `localhost:3001` or add your host to `allowedDevOrigins` in `next.config.ts`. Nothing is logged; verify hydration with a browser, not `curl`.
- **`Dataset scripts are no longer supported`.** Use `legacy-datasets/banking77`; already wired in `data.py`.

</details>

## Authors

- Jared Palmer ([@jaredpalmer](https://github.com/jaredpalmer))

Built with [Devin](https://devin.ai). Architecture claims from [Archer Hume](https://archerhume.com/posts/jevs-architecture-unmasked). API contract from [TypeSafe](https://docs.typesafe.ai/api). Backbone: [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B). Related work: [Hydragen](https://arxiv.org/abs/2402.05099), [DeFT](https://arxiv.org/abs/2404.00242), [FIRST](https://arxiv.org/abs/2406.15657).

## License

[Apache-2.0](LICENSE). The base model is distributed under the Qwen license. Datasets carry their own licenses; see the [model cards](docs/model-cards/#training-data).
