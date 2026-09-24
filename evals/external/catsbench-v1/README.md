---
license: mit
language:
  - en
task_categories:
  - time-series-forecasting
  - question-answering
configs:
  - config_name: default
    data_files:
      - split: validation
        path: development.jsonl
---

# CaTSBench Kev

CaTSBench Kev converts numeric time series and human captions from
[`mhfisher/CaTSBench`](https://huggingface.co/datasets/mhfisher/CaTSBench) into labelled TypeSafe System One requests.
It is an evaluation dataset for Kev, Jev, and other decision models. It contains both questions about the visible series
and questions about a hidden suffix.

The validation split has 459 records, 950 questions, and 402 source-series groups. Each record contains one shared state
and one or more Choice, Noul, or Score questions. There are no training or test records in this release.

## Construction

The source dataset is pinned to revision `63e35cded0d5a8e3a3fd87d487da267cf873a846`. The builder sampled 500 test
series with human captions using seed `catsbench-pilot-v1`.

`Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` selected applicable question
templates. The model did not produce labels. Deterministic Python functions calculated every label from the numeric
series. Forecast records expose only the prefix used by the question.

The builder removed duplicate states and excluded two underpowered templates, `future_change` and
`observed_monotonic_decrease`. It then applied Kev's training-context admission rule with the released Kev-4B tokenizer.

## Files

- `development.jsonl` contains the benchmark records.
- `manifest.json` pins sources, revisions, hashes, context limits, and the construction protocol.
- `proposals.jsonl` preserves every raw Qwen template-selection response.
- `audit.json` contains source-backed label, prefix, leakage, duplication, and balance checks.
- The other JSONL partitions are empty so the folder can also be loaded as a Kev frozen suite.

## Record format

```json
{
  "state": "{\"domain\":\"agriculture\",\"times\":[0,1,2],\"values\":[96.46,102.01,104.59]}",
  "questions": {
    "direction": {
      "type": "choice",
      "instructions": "From the first observation to the last, did the value decrease, stay stable, or increase?",
      "criteria": {
        "decrease": "decrease",
        "stable": "stable",
        "increase": "increase"
      },
      "label": "increase",
      "src": "catsbench_observation"
    }
  }
}
```

`group_id` keeps observation and forecast records derived from the same source series together. Use it when splitting or
bootstrapping the dataset.

## Validation

The audit recomputes every label from the pinned source archive and verifies that each forecast state is a strict prefix.
It reports no invalid records, duplicate states, caption leakage, or metadata-statistic leakage. The forecast-direction
family has only one stable example, so per-class results for that family are preliminary.

## Initial results

The initial run uses `jaredpalmer/kev-4b` at revision
`1da696f7938f77c4cdf5471e92fd342baff41778`. It evaluates every record without truncation or rejection.

| Split | Questions | Accuracy | NLL | ECE |
| --- | ---: | ---: | ---: | ---: |
| All | 950 | 62.8% | 0.855 | 0.093 |
| Observation | 859 | 66.2% | 0.771 | 0.060 |
| Forecast | 91 | 30.8% | 1.655 | 0.434 |

Results by question family show where the model succeeds and fails. The majority column is the accuracy obtained by
always predicting that family's most common label.

| Question family | Questions | Kev-4B | Majority |
| --- | ---: | ---: | ---: |
| Forecast direction | 38 | 26.3% | 57.9% |
| Forecast new high | 23 | 43.5% | 52.2% |
| Forecast new low | 30 | 26.7% | 63.3% |
| Observation direction | 208 | 85.6% | 52.4% |
| Observation maximum position | 317 | 48.3% | 41.3% |
| Observation relative change | 303 | 70.6% | 35.6% |
| Observation second-half comparison | 31 | 77.4% | 51.6% |

Kev-4B performs above the majority baseline on every observation family and below it on every forecast family. The
forecast result also has poor calibration, with an ECE of 0.434. This first run therefore supports using observation and
forecast questions as separate benchmark sections.

The complete outputs are in [`baselines/kev-4b/report.json`](baselines/kev-4b/report.json) and
[`baselines/kev-4b/summary.json`](baselines/kev-4b/summary.json). Reproduce the run from a Kev checkout with:

```bash
uv run python -m kev.benchmark \
  --run jaredpalmer/kev-4b \
  --suite evals/external/catsbench-v1 \
  --out runs/kev-4b-catsbench-v1 \
  --device cuda
```

## License

CaTSBench and this derived dataset use the MIT license.
