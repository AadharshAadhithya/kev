#!/usr/bin/env python3
"""Audit a CaTSBench-derived Kev/Jev dataset against its numeric source."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_catsbench as cats
from kev.data import materialize
from kev.suite import read_jsonl, write_json


LEAKAGE_MARKERS = (
    "caption",
    "historical average",
    "historical maximum",
    "historical minimum",
    "standard deviation",
)


def _task_report(labels: dict[tuple[str, str], Counter[str]]) -> dict[str, Any]:
    report = {}
    for (mode, qid), counts in sorted(labels.items()):
        total = sum(counts.values())
        report[f"{mode}/{qid}"] = {
            "n": total,
            "labels": dict(sorted(counts.items())),
            "majority_accuracy": max(counts.values()) / total,
        }
    return report


def audit(
    records: list[dict[str, Any]],
    sources: dict[str, cats.SourceSeries],
    proposals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    labels: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    modes: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    ids: set[str] = set()
    groups: set[str] = set()
    states: Counter[str] = Counter()

    for record in records:
        meta = record.get("_meta", {})
        record_id = str(meta.get("id", ""))
        source_id = str(meta.get("source_id", ""))
        mode = str(meta.get("mode", ""))
        if not record_id or record_id in ids:
            errors.append(f"duplicate or missing record id: {record_id!r}")
        ids.add(record_id)
        groups.add(str(meta.get("group_id", "")))
        modes[mode] += 1
        domains[str(meta.get("domain", ""))] += 1
        states[str(record.get("state", ""))] += 1

        source = sources.get(source_id)
        if source is None:
            errors.append(f"{record_id}: source {source_id!r} is absent from the archive")
            continue
        try:
            materialize(record)
        except Exception as exc:
            errors.append(f"{record_id}: TypeSafe validation failed: {exc}")
            continue
        try:
            state = json.loads(record["state"])
            visible = tuple(float(value) for value in state["values"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{record_id}: invalid state: {exc}")
            continue

        lowered = record["state"].lower()
        for marker in LEAKAGE_MARKERS:
            if marker in lowered:
                errors.append(f"{record_id}: state contains leakage marker {marker!r}")

        if mode == "observation":
            if visible != source.values:
                errors.append(f"{record_id}: observation state differs from the source series")
            expected = cats.observation_questions(source.values, set(cats.TEMPLATES), source_id)
        elif mode == "forecast":
            cutoff = len(visible)
            if cutoff >= len(source.values) or visible != source.values[:cutoff]:
                errors.append(f"{record_id}: state is not a strict source-series prefix")
            expected = cats.forecast_questions(
                source.values[:cutoff], source.values[cutoff:], set(cats.TEMPLATES), source_id
            )
        else:
            errors.append(f"{record_id}: unknown mode {mode!r}")
            continue

        for qid, question in record["questions"].items():
            if qid not in expected:
                errors.append(f"{record_id}/{qid}: unknown or inapplicable question")
                continue
            if question.get("label") != expected[qid].get("label"):
                errors.append(
                    f"{record_id}/{qid}: label {question.get('label')!r} does not match "
                    f"{expected[qid].get('label')!r}"
                )
            labels[(mode, qid)][str(question.get("label"))] += 1

    task_report = _task_report(labels)
    question_count = sum(item["n"] for item in task_report.values())
    majority_correct = sum(max(counts.values()) for counts in labels.values())
    macro_majority = (
        sum(max(counts.values()) / sum(counts.values()) for counts in labels.values()) / len(labels)
        if labels
        else 0.0
    )
    warnings = []
    for task, item in task_report.items():
        if item["n"] < 20:
            warnings.append(f"{task}: only {item['n']} examples")
        if item["majority_accuracy"] > 0.8:
            warnings.append(f"{task}: majority baseline is {item['majority_accuracy']:.1%}")
        rarest = min(item["labels"].values())
        if len(item["labels"]) > 1 and rarest < 5:
            warnings.append(f"{task}: rarest observed label has only {rarest} examples")

    proposal_report = None
    if proposals is not None:
        source_ids = [str(row.get("source_id", "")) for row in proposals]
        template_counts = Counter(
            template for row in proposals for template in row.get("templates", [])
        )
        unknown = sorted(
            {
                template
                for row in proposals
                for template in row.get("templates", [])
                if template not in cats.TEMPLATES
            }
        )
        parse_mismatches = sum(
            cats.parse_proposal(str(row.get("response", ""))) != set(row.get("templates", []))
            for row in proposals
        )
        if len(source_ids) != len(set(source_ids)):
            errors.append("proposal file contains duplicate source IDs")
        if unknown:
            errors.append(f"proposal file contains unknown templates: {unknown}")
        if parse_mismatches:
            errors.append(f"{parse_mismatches} saved proposals do not match their raw responses")
        proposal_report = {
            "rows": len(proposals),
            "empty": sum(not row.get("templates") for row in proposals),
            "raw_parse_mismatches": parse_mismatches,
            "template_counts": dict(sorted(template_counts.items())),
            "models": sorted({str(row.get("model")) for row in proposals if row.get("model")}),
            "revisions": sorted(
                {str(row.get("revision")) for row in proposals if row.get("revision")}
            ),
        }

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "records": len(records),
        "questions": question_count,
        "source_groups": len(groups),
        "mode_records": dict(sorted(modes.items())),
        "domains": dict(sorted(domains.items())),
        "duplicate_states": sum(count - 1 for count in states.values() if count > 1),
        "task_families": task_report,
        "baselines": {
            "per_task_majority_micro": majority_correct / question_count if question_count else 0.0,
            "per_task_majority_macro": macro_majority,
        },
        "proposals": proposal_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--captions", choices=("human", "any"), default="human")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = {
        row.source_id: row for row in cats.load_archive(args.archive, args.split, args.captions)
    }
    report = audit(
        read_jsonl(args.records),
        sources,
        read_jsonl(args.proposals) if args.proposals else None,
    )
    write_json(args.out, report)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
