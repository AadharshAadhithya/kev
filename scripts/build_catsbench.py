#!/usr/bin/env python3
"""Build a deterministic Kev/Jev benchmark from CaTSBench time series."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kev.data import materialize
from kev.suite import digest, read_jsonl, record_digest, write_json, write_jsonl


DATASET = "mhfisher/CaTSBench"
REVISION = "63e35cded0d5a8e3a3fd87d487da267cf873a846"
GENERATOR_VERSION = 3

OBSERVED_TEMPLATES = (
    "observed_direction",
    "observed_monotonic_decrease",
    "observed_second_half_higher",
    "observed_extreme_position",
    "observed_change",
)
FORECAST_TEMPLATES = (
    "future_direction",
    "future_new_high",
    "future_new_low",
    "future_change",
)
TEMPLATES = frozenset((*OBSERVED_TEMPLATES, *FORECAST_TEMPLATES))

# These fields describe the series. CaTSBench metadata also contains statistics
# computed over the complete series; those are deliberately excluded.
SAFE_METADATA = (
    "attribute",
    "country",
    "region",
    "state",
    "town",
    "location",
    "port",
    "border",
    "means",
    "mode",
    "severity",
    "geotype",
    "income group",
    "category by income",
    "population",
)


@dataclass(frozen=True)
class SourceSeries:
    source_id: str
    domain: str
    values: tuple[float, ...]
    metadata: dict[str, Any]
    caption: str
    caption_kind: str


def _zip_root(split: str) -> str:
    return f"{split}_data"


def _source_id(path: str) -> str:
    return Path(path).stem


def _domain(source_id: str, split: str) -> str:
    match = re.fullmatch(rf"(.+)_\d+_{re.escape(split)}", source_id)
    return (match.group(1) if match else source_id).replace("_", " ")


def _numbers(text: str) -> tuple[float, ...]:
    values = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            values.append(float(line))
    return tuple(values)


def load_archive(path: Path, split: str, captions: str = "human") -> list[SourceSeries]:
    """Load aligned series, metadata, and captions from a CaTSBench archive."""
    root = _zip_root(split)
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        prefix = f"{root}/time series/"
        series_names = sorted(
            name for name in names if name.startswith(prefix) and name.endswith(".txt")
        )
        rows = []
        for series_name in series_names:
            source_id = _source_id(series_name)
            metadata_name = f"{root}/metadata/{source_id}.json"
            human_name = f"{root}/human_rewritten_captions/{source_id}.txt"
            synthetic_name = f"{root}/synth_gt_captions/{source_id}.txt"
            if metadata_name not in names:
                continue
            if human_name in names:
                caption_name, caption_kind = human_name, "human"
            elif captions == "any" and synthetic_name in names:
                caption_name, caption_kind = synthetic_name, "synthetic"
            else:
                continue
            values = _numbers(archive.read(series_name).decode("utf-8"))
            if len(values) < 3 or not all(math.isfinite(value) for value in values):
                continue
            rows.append(
                SourceSeries(
                    source_id=source_id,
                    domain=_domain(source_id, split),
                    values=values,
                    metadata=json.loads(archive.read(metadata_name).decode("utf-8")),
                    caption=archive.read(caption_name).decode("utf-8").strip(),
                    caption_kind=caption_kind,
                )
            )
    return rows


def _metadata_value(metadata: dict[str, Any], wanted: str) -> Any | None:
    for key, value in metadata.items():
        if key.strip().lower() == wanted:
            return value
    return None


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    for pattern in (r"(\d{4})-(\d{1,2})-(\d{1,2})", r"(\d{4})/(\d{1,2})/(\d{1,2})"):
        match = re.fullmatch(pattern, value)
        if match:
            try:
                return date(*(int(part) for part in match.groups()))
            except ValueError:
                return None
    match = re.fullmatch(r"\d{4}", value)
    return date(int(value), 1, 1) if match else None


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year, month = value.year + month_index // 12, month_index % 12 + 1
    return date(year, month, min(value.day, 28))


def infer_times(metadata: dict[str, Any], length: int) -> tuple[list[int | str], str]:
    """Return real timestamps only when CaTSBench's range agrees with its length."""
    lowered = {str(key).strip().lower(): value for key, value in metadata.items()}
    start_value = next((v for k, v in lowered.items() if "start" in k and "date" in k), None)
    end_value = next((v for k, v in lowered.items() if "end" in k and "date" in k), None)
    frequency = next((v for k, v in lowered.items() if "frequency" in k), None)
    start, end = _parse_date(start_value), _parse_date(end_value)
    unit = str(frequency or "").strip().lower()
    if start and end:
        if "year" in unit:
            times = [date(start.year + i, start.month, min(start.day, 28)) for i in range(length)]
        elif "month" in unit:
            times = [_add_months(start, i) for i in range(length)]
        elif "week" in unit:
            times = [start + timedelta(weeks=i) for i in range(length)]
        elif "day" in unit:
            times = [start + timedelta(days=i) for i in range(length)]
        else:
            times = []
        if times and times[-1] == end:
            if "year" in unit and all(item.month == 1 and item.day == 1 for item in times):
                return [str(item.year) for item in times], unit
            return [item.isoformat() for item in times], unit
    return list(range(length)), "index"


def safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    lowered = {str(key).strip().lower(): value for key, value in metadata.items()}
    return {
        key.replace(" ", "_"): lowered[key]
        for key in SAFE_METADATA
        if key in lowered and isinstance(lowered[key], (str, int, float, bool))
    }


def _relative_change(start: float, end: float) -> float | None:
    if abs(start) < 1e-12:
        return None
    return (end - start) / abs(start)


def direction_label(change: float) -> str:
    if change < -0.02:
        return "decrease"
    if change > 0.02:
        return "increase"
    return "stable"


def change_score(change: float) -> int:
    """Map relative change to ordered bins from large decrease to large increase."""
    if change < -0.10:
        return 0
    if change < -0.02:
        return 1
    if change <= 0.02:
        return 2
    if change <= 0.10:
        return 3
    return 4


def _choice(instructions: str, label: str, options: Iterable[str], src: str) -> dict[str, Any]:
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {option: option.replace("_", " ") for option in options},
        "label": label,
        "src": src,
    }


def _noul(instructions: str, label: bool, src: str) -> dict[str, Any]:
    return {
        "type": "noul",
        "instructions": instructions,
        "criteria": {"true": "yes", "false": "no"},
        "label": label,
        "src": src,
    }


def _score(instructions: str, label: int, src: str) -> dict[str, Any]:
    return {
        "type": "score",
        "instructions": instructions,
        "criteria": [
            "large decrease (more than 10%)",
            "small decrease (2% to 10%)",
            "stable (within 2%)",
            "small increase (2% to 10%)",
            "large increase (more than 10%)",
        ],
        "label": label,
        "src": src,
    }


def _complement(source_id: str, template: str) -> bool:
    return bool(source_id) and int(record_digest(f"{source_id}/{template}")[:8], 16) % 2 == 1


def observation_questions(
    values: tuple[float, ...], selected: set[str], source_id: str = ""
) -> dict[str, dict[str, Any]]:
    src = "catsbench_observation"
    questions = {}
    change = _relative_change(values[0], values[-1])
    if change is not None and "observed_direction" in selected:
        questions["direction"] = _choice(
            "From the first observation to the last, did the value decrease, stay stable, or increase?",
            direction_label(change),
            ("decrease", "stable", "increase"),
            src,
        )
    if "observed_monotonic_decrease" in selected:
        predicate = all(right < left for left, right in zip(values, values[1:]))
        complement = _complement(source_id, "observed_monotonic_decrease")
        questions["monotonic_decrease"] = _noul(
            (
                "Does at least one observation after the first have a value greater than or equal to the previous observation?"
                if complement
                else "Does every observation after the first have a lower value than the observation before it?"
            ),
            not predicate if complement else predicate,
            src,
        )
    if "observed_second_half_higher" in selected:
        half = len(values) // 2
        predicate = sum(values[-half:]) / half > sum(values[:half]) / half
        complement = _complement(source_id, "observed_second_half_higher")
        questions["second_half_higher"] = _noul(
            (
                "Is the mean of the last half less than or equal to the mean of the first half?"
                if complement
                else "Is the mean of the last half of the observations higher than the mean of the first half?"
            ),
            not predicate if complement else predicate,
            src,
        )
    if "observed_extreme_position" in selected:
        maximum = max(range(len(values)), key=values.__getitem__)
        third = int(3 * (maximum + 0.5) / len(values))
        questions["maximum_position"] = _choice(
            "Where does the first occurrence of the maximum value fall in the observed series?",
            ("early", "middle", "late")[min(third, 2)],
            ("early", "middle", "late"),
            src,
        )
    if change is not None and "observed_change" in selected:
        questions["relative_change"] = _score(
            "Rate the relative change from the first observation to the last.",
            change_score(change),
            src,
        )
    return questions


def forecast_questions(
    prefix: tuple[float, ...],
    suffix: tuple[float, ...],
    selected: set[str],
    source_id: str = "",
) -> dict[str, dict[str, Any]]:
    src = "catsbench_forecast"
    questions = {}
    change = _relative_change(prefix[-1], suffix[-1])
    horizon = len(suffix)
    if change is not None and "future_direction" in selected:
        questions["direction"] = _choice(
            f"Over the next {horizon} observations, will the value decrease, stay stable, or increase?",
            direction_label(change),
            ("decrease", "stable", "increase"),
            src,
        )
    if "future_new_high" in selected:
        predicate = max(suffix) > max(prefix)
        complement = _complement(source_id, "future_new_high")
        questions["new_high"] = _noul(
            (
                f"Will every one of the next {horizon} observations remain at or below the highest value observed so far?"
                if complement
                else f"Will any of the next {horizon} observations exceed every value observed so far?"
            ),
            not predicate if complement else predicate,
            src,
        )
    if "future_new_low" in selected:
        predicate = min(suffix) < min(prefix)
        complement = _complement(source_id, "future_new_low")
        questions["new_low"] = _noul(
            (
                f"Will every one of the next {horizon} observations remain at or above the lowest value observed so far?"
                if complement
                else f"Will any of the next {horizon} observations fall below every value observed so far?"
            ),
            not predicate if complement else predicate,
            src,
        )
    if change is not None and "future_change" in selected:
        questions["relative_change"] = _score(
            f"Rate the relative change from the latest visible value to the value {horizon} observations ahead.",
            change_score(change),
            src,
        )
    return questions


def parse_proposal(text: str) -> set[str]:
    """Extract registered template IDs from a model response."""
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end >= start:
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            payload = None
        values = payload.get("templates", []) if isinstance(payload, dict) else []
        if isinstance(values, list):
            selected = {value for value in values if isinstance(value, str) and value in TEMPLATES}
            if selected:
                return selected
    return {template for template in TEMPLATES if re.search(rf"\b{re.escape(template)}\b", text)}

class QwenProposer:
    """Use a local Hugging Face model only to select registered templates."""

    def __init__(self, model_id: str):
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForMultimodalLM,
            AutoTokenizer,
        )

        self.torch = torch
        config = AutoConfig.from_pretrained(model_id)
        self.revision = getattr(config, "_commit_hash", None)
        architectures = set(getattr(config, "architectures", ()) or ())
        self.multimodal = any(
            "Multimodal" in name or "ConditionalGeneration" in name for name in architectures
        )
        loader = AutoModelForMultimodalLM if self.multimodal else AutoModelForCausalLM
        self.processor = AutoTokenizer.from_pretrained(model_id, revision=self.revision)
        kwargs: dict[str, Any] = {
            "dtype": "auto",
            "low_cpu_mem_usage": True,
            "revision": self.revision,
        }
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        self.model = loader.from_pretrained(model_id, **kwargs).eval()
        self.device = self.model.device

    def __call__(self, row: SourceSeries) -> tuple[set[str], str]:
        cutoff = min(max(3, 2 * len(row.values) // 3), len(row.values) - 2)
        prompt = (
            "A caption describes a complete time series. Select only benchmark templates that test the specific "
            "phenomena discussed by the caption. Observation templates use the complete series. Future templates "
            f"show zero-based indices 0 through {cutoff - 1} and hide indices {cutoff} through "
            f"{len(row.values) - 1}. Select future templates when the caption discusses a movement or extreme in "
            "that hidden suffix. The hidden values are supplied only so you can classify relevance; they will not "
            f"appear in the benchmark input.\nSeries values: {json.dumps(row.values)}\n\n"
            "Templates:\n"
            "- observed_direction: overall start-to-end direction is discussed\n"
            "- observed_monotonic_decrease: every consecutive step decreasing is explicitly discussed\n"
            "- observed_second_half_higher: earlier and later periods are compared\n"
            "- observed_extreme_position: timing of a maximum or peak is discussed\n"
            "- observed_change: magnitude of the overall change is discussed\n"
            "- future_direction: direction after the visible cutoff is discussed\n"
            "- future_new_high: a new high after the cutoff is discussed\n"
            "- future_new_low: a new low after the cutoff is discussed\n"
            "- future_change: magnitude of change after the cutoff is discussed\n\n"
            "Choose between one and five IDs. Do not answer the tasks. Return exactly one JSON object with no prose: "
            "{\"templates\":[\"template_id\"]}\n\nCaption: "
            + row.caption
        )
        messages = [{"role": "user", "content": prompt}]
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            preserve_thinking=False,
        )
        inputs = self.processor(rendered, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=128)
        generated = output[0, inputs.input_ids.shape[1] :]
        response = self.processor.decode(generated, skip_special_tokens=True)
        return parse_proposal(response), response


def _state(row: SourceSeries, times: list[int | str], values: tuple[float, ...]) -> str:
    payload: dict[str, Any] = {
        "domain": row.domain,
        **safe_metadata(row.metadata),
        "time_axis": "timestamp" if times and isinstance(times[0], str) else "index",
        "times": times,
        "values": list(values),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _record(
    row: SourceSeries,
    split: str,
    mode: str,
    state: str,
    questions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "state": state,
        "questions": questions,
        "_meta": {
            "id": f"catsbench/{row.source_id}/{mode}",
            "group_id": f"catsbench/{row.source_id}",
            "source": "catsbench",
            "dataset": DATASET,
            "revision": REVISION,
            "split": split,
            "source_id": row.source_id,
            "domain": row.domain,
            "mode": mode,
            "caption_kind": row.caption_kind,
            "generator_version": GENERATOR_VERSION,
            "text_sha256": record_digest(state),
        },
    }
    record["_meta"]["row_sha256"] = record_digest(record)
    materialize(record)
    return record


def build_records(
    rows: Iterable[SourceSeries],
    split: str,
    proposals: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    records = []
    seen_states: set[str] = set()
    for row in rows:
        selected = proposals.get(row.source_id, set()) if proposals is not None else set(TEMPLATES)
        if not selected:
            continue
        times, _ = infer_times(row.metadata, len(row.values))
        observed = observation_questions(row.values, selected, row.source_id)
        if observed:
            state = _state(row, times, row.values)
            if state not in seen_states:
                records.append(_record(row, split, "observation", state, observed))
                seen_states.add(state)
        if len(row.values) >= 5:
            cutoff = min(max(3, 2 * len(row.values) // 3), len(row.values) - 2)
            prefix, suffix = row.values[:cutoff], row.values[cutoff:]
            forecast = forecast_questions(prefix, suffix, selected, row.source_id)
            if forecast:
                state = _state(row, times[:cutoff], prefix)
                if state not in seen_states:
                    records.append(_record(row, split, "forecast", state, forecast))
                    seen_states.add(state)
    return records


def _download(split: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=DATASET,
            filename=f"{split}_data.zip",
            repo_type="dataset",
            revision=REVISION,
        )
    )


def _load_proposals(path: Path) -> tuple[dict[str, set[str]], str | None, str | None]:
    rows = read_jsonl(path)
    proposals = {
        str(row["source_id"]): {
            value for value in row.get("templates", []) if isinstance(value, str) and value in TEMPLATES
        }
        for row in rows
    }
    models = {row.get("model") for row in rows if row.get("model")}
    revisions = {row.get("revision") for row in rows if row.get("revision")}
    if len(models) > 1 or len(revisions) > 1:
        raise ValueError(f"{path}: proposals contain mixed model provenance")
    return proposals, next(iter(models), None), next(iter(revisions), None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Local CaTSBench ZIP; downloaded if omitted")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--captions", choices=("human", "any"), default="human")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0, help="Skip this many rows after seeded shuffle")
    parser.add_argument("--seed", default="catsbench-pilot-v1")
    parser.add_argument("--proposer-model", help="Optional Hugging Face instruct model")
    parser.add_argument("--proposals", type=Path, help="Reuse saved template proposals")
    parser.add_argument("--proposals-out", type=Path, help="Save generated template proposals")
    parser.add_argument(
        "--exclude-template",
        action="append",
        choices=sorted(TEMPLATES),
        default=[],
        help="Exclude a selected template; repeat for more than one",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    if args.proposer_model and args.proposals:
        raise ValueError("use either --proposer-model or --proposals")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    rows = load_archive(args.archive or _download(args.split), args.split, args.captions)
    random.Random(args.seed).shuffle(rows)
    rows = rows[args.offset : args.offset + args.limit]

    proposals: dict[str, set[str]] | None = None
    proposer_model = None
    proposer_revision = None
    if args.proposals:
        proposals, proposer_model, proposer_revision = _load_proposals(args.proposals)
    elif args.proposer_model:
        proposer = QwenProposer(args.proposer_model)
        proposer_model = args.proposer_model
        proposer_revision = proposer.revision
        proposed = {row.source_id: proposer(row) for row in rows}
        proposals = {source_id: result[0] for source_id, result in proposed.items()}
        proposal_path = args.proposals_out or args.out.with_name("proposals.jsonl")
        write_jsonl(
            proposal_path,
            (
                {
                    "source_id": source_id,
                    "templates": sorted(result[0]),
                    "response": result[1],
                    "model": args.proposer_model,
                    "revision": proposer.revision,
                }
                for source_id, result in proposed.items()
            ),
        )

    excluded = set(args.exclude_template)
    if proposals is None and excluded:
        proposals = {row.source_id: set(TEMPLATES) - excluded for row in rows}
    elif proposals is not None:
        proposals = {
            source_id: selected - excluded for source_id, selected in proposals.items()
        }

    records = build_records(rows, args.split, proposals)
    write_jsonl(args.out, records)
    counts = Counter(
        (record["_meta"]["mode"], question["type"])
        for record in records
        for question in record["questions"].values()
    )
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "source_split": args.split,
        "captions": args.captions,
        "seed": args.seed,
        "offset": args.offset,
        "input_series": len(rows),
        "records": len(records),
        "questions": sum(len(record["questions"]) for record in records),
        "counts": {f"{mode}/{kind}": count for (mode, kind), count in sorted(counts.items())},
        "proposer_model": proposer_model,
        "proposer_revision": proposer_revision,
        "excluded_templates": sorted(excluded),
        "series_with_records": len({record["_meta"]["source_id"] for record in records}),
        "output": args.out.name,
        "sha256": digest(args.out),
        "generator_version": GENERATOR_VERSION,
    }
    write_json(args.out.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
