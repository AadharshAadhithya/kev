import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi

from kev.composition import DEV_SHAPES, TEST_SHAPES, TRAIN_SHAPES, check_group, generate as compose
from kev.contrastive import generate
from kev.data import materialize
from kev.model import encode, load_tokenizer
from kev.suite import SPLITS, digest, load_split, record_digest, write_json

BASES = ("Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-4B-Base")
FAMILIES = ("return_window", "spend_threshold", "age_eligibility", "quantity_limit")


def semantic_hash(r):
    state = r["state"]
    if isinstance(state, dict) and "policy" in state and "case" in state:
        state = {"policy": state["policy"], "sentences": sorted(s.rstrip(".") for s in state["case"].split(". "))}
    return record_digest(state)


def grouped_split(records, calibration_groups):
    groups = defaultdict(dict)
    for r in records:
        m = r["_meta"]
        groups[m["family"]].setdefault(m["group_id"], []).append(r)
    train, calibration = [], []
    for family in sorted(groups):
        units = list(groups[family].values())
        if len(units) <= calibration_groups:
            raise ValueError("not enough groups per family")
        calibration.extend(r for group in units[:calibration_groups] for r in group)
        train.extend(r for group in units[calibration_groups:] for r in group)
    return train, calibration


def legacy(pairs, seed, families=FAMILIES, source="legacy_policy", excluded=()):
    candidates, _ = generate(3 * pairs, seed, families)
    records, seen, counts = [], set(excluded), Counter()
    for a, b in zip(candidates[::2], candidates[1::2]):
        family = a["_meta"]["family"]
        hashes = {semantic_hash(a), semantic_hash(b)}
        if counts[family] >= pairs or hashes & seen:
            continue
        for r in (a, b):
            m = r["_meta"]
            m.update(source=source, group_id=m["pair_id"], variant="clean")
            m["text_sha256"] = semantic_hash(r)
        records.extend((a, b)); seen.update(hashes); counts[family] += 1
    if any(counts[f] != pairs for f in families):
        raise ValueError("insufficient unique legacy pairs")
    return records


def validate_training(records, manifest):
    from kev.data import EVAL_ONLY
    allowed = set(manifest.get("trainable_sources", []))
    forbidden = set(EVAL_ONLY) | set(manifest.get("eval_only_sources", [])) | set(manifest.get("holdout_sources", []))
    for r in records:
        m = r["_meta"]
        if m["source"] in forbidden or (allowed and m["source"] not in allowed):
            raise ValueError(f"eval-only or undeclared training source: {m['source']}")
        if m["source"] == "compositional" and m["family"] not in TRAIN_SHAPES:
            raise ValueError("held-out compositional structure in training")
    if not records:
        raise ValueError("empty training partition")


def freeze(out, source="evals/decision-v2", transfer="evals/transfer-v2", public_train=None):
    """public_train: optional larger public training pool (a frozen suite dir). Its train+calibration partitions replace
    the source suite's public train/calibration; development/test still come from `source` so results stay comparable."""
    out, source, transfer = Path(out), Path(source), Path(transfer)
    if out.exists():
        raise FileExistsError("v3 destination already exists; choose a new version")
    original = json.loads((source / "manifest.json").read_text())
    old_transfer = json.loads((transfer / "manifest.json").read_text())
    revisions = {base: HfApi().model_info(base).sha for base in BASES}
    tokenizers = [load_tokenizer(base, revision=sha) for base, sha in revisions.items()]
    parts = {s: [] for s in SPLITS}
    for split in ("train", "calibration", "development"):
        parts[split] = [r for r in load_split(source, split) if r["_meta"]["source"] != "contrastive"]
    if public_train:
        pool = Path(public_train)
        dev_test_states = {r["_meta"]["text_sha256"] for split in ("development", "test") for r in load_split(source, split, allow_test=True)}
        for split in ("train", "calibration"):
            fresh = [r for r in load_split(pool, split) if r["_meta"]["source"] != "contrastive"]
            leaked = [r for r in fresh if r["_meta"]["text_sha256"] in dev_test_states]
            if leaked: raise ValueError(f"{len(leaked)} public {split} records collide with development/test states")
            parts[split] = fresh
        source_dirs = [pool]
    else:
        source_dirs = []
    reserved = set()
    for split in ("train", "calibration", "development"):
        reserved.update(semantic_hash(r) for r in parts[split])
    sources = [p for p in source.glob("*.json*")] + [p for p in transfer.glob("*.json*")] + [p for d in source_dirs for p in d.glob("*.json*")]
    parent_hashes = {str(p): digest(p) for p in sources}

    def admit_groups(records):
        groups = defaultdict(list)
        for r in records:
            groups[r["_meta"]["group_id"]].append(r)
        for group in groups.values():
            if group[0]["_meta"]["source"] in ("compositional", "composition_holdout"):
                check_group(group)
            hashes = {semantic_hash(r) for r in group}
            if hashes & reserved:
                raise ValueError("semantic state collision across groups; choose a new generation seed")
            for r in group:
                rec = materialize(r)
                for tok in tokenizers:
                    e = encode(tok, rec, strict=True)
                    if len(e["ids"]) > 2048:
                        raise ValueError("packed token limit")
            reserved.update(hashes)
        return records

    old_train, old_cal = grouped_split(admit_groups(legacy(64, "v3-control")), 8)
    new_train, new_cal = grouped_split(admit_groups(compose(16, "v3-composition")), 2)
    if len(old_train) != len(new_train) or len(old_cal) != len(new_cal):
        raise ValueError("synthetic arms have unequal record budgets")
    parts["train"] += old_train + new_train
    parts["calibration"] += old_cal + new_cal
    parts["development"] += admit_groups(legacy(12, "v3-control-dev", excluded=reserved))
    parts["development"] += admit_groups(compose(4, "v3-composition-dev"))
    transfer_dev = [r for r in load_split(transfer, "development") if r["_meta"]["source"] != "contrastive"]
    transfer_dev += admit_groups(legacy(20, "v3-legacy-transfer", ("authorization", "deadline"), source="legacy_holdout"))
    transfer_dev += admit_groups(compose(8, "v3-composition-transfer", DEV_SHAPES, styles=(1,), source="composition_holdout"))
    final_extra = admit_groups(compose(8, "v3-locked", TEST_SHAPES, styles=(2,), source="composition_holdout"))
    for split in ("train", "calibration"):
        random.Random(f"v3-{split}").shuffle(parts[split])
    manifest = {"version": 3, "base_revisions": revisions, "dataset_revisions": original["dataset_revisions"],
        "parent_files": parent_hashes, "holdout_sources": [],
        "trainable_sources": [s for s in original["trainable_sources"] if s != "contrastive"] + ["legacy_policy", "compositional"],
        "eval_only_sources": old_transfer["eval_only_sources"] + ["legacy_holdout", "composition_holdout"],
        "context": original["context"],
        "protocol": {"train_shapes": TRAIN_SHAPES, "transfer_shapes": DEV_SHAPES, "locked_shapes": TEST_SHAPES,
                     "train_render_styles": [0, 1], "locked_render_styles": [2],
                     "public_train_records": len(parts["train"]) - len(old_train) - len(new_train),
                     "public_train_pool": str(public_train) if public_train else str(source),
                     "synthetic_records_per_arm": len(old_train), "calibration": "shared; stratified by family and group",
                     "arm_selection": "train_sources selects public sources plus exactly one synthetic arm",
                     "primary": "macro development NLL; transfer and confident-error checks required; no automatic release",
                     "legacy_test": "Inherited v2 locked test bytes retained without inspecting examples; only new structures added to transfer test",
                     "contamination": "Exact semantic checks among new groups, not fuzzy or pretraining decontamination. Inherited legacy test overlap with new synthetic controls is not certified."},
        "files": {}, "code_hashes": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}}
    validate_training(parts["train"], manifest)
    for name, partitions, inherited in (("decision-v3", parts, source),
            ("transfer-v3", {"train": [], "calibration": [], "development": transfer_dev, "test": final_extra}, transfer)):
        folder = out / name
        folder.mkdir(parents=True, exist_ok=False)
        m = copy.deepcopy(manifest)
        if name.startswith("transfer"):
            m.update(trainable_sources=[], holdout_sources=manifest["eval_only_sources"], eval_only=True)
        for split, records in partitions.items():
            path = folder / f"{split}.jsonl"
            payload = b""
            inherited_records = inherited_questions = 0
            if split == "test":
                old = json.loads((inherited / "manifest.json").read_text())["files"]["test.jsonl"]
                if digest(inherited / "test.jsonl") != old["sha256"]:
                    raise ValueError("inherited test checksum mismatch")
                payload = (inherited / "test.jsonl").read_bytes()
                inherited_records, inherited_questions = old["records"], old["questions"]
            payload += "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode()
            path.write_bytes(payload)
            m["files"][path.name] = {"sha256": digest(path), "records": inherited_records + len(records),
                                     "questions": inherited_questions + sum(len(r["questions"]) for r in records)}
        write_json(folder / "manifest.json", m)
        print(name, {s: v["records"] for s, v in m["files"].items()}, flush=True)
    if any(digest(p) != h for p, h in parent_hashes.items()):
        raise ValueError("parent artifacts changed")
    return manifest


def smoke_subset(source, out):
    source, out = Path(source), Path(out)
    manifest = json.loads((source / "manifest.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    manifest["files"] = {}
    for split in SPLITS:
        records = []
        if split != "test":
            groups = defaultdict(list)
            for r in load_split(source, split):
                groups[(r["_meta"]["source"], r["_meta"]["group_id"])].append(r)
            taken = Counter()
            for (name, _), group in groups.items():
                if taken[name] < 2:
                    records.extend(group); taken[name] += 1
        path = out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        manifest["files"][path.name] = {"sha256": digest(path), "records": len(records),
                                        "questions": sum(len(r["questions"]) for r in records)}
    manifest["smoke_only"] = True
    write_json(out / "manifest.json", manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke-from")
    ap.add_argument("--public-train", help="frozen suite whose train/calibration public partitions replace the source's (larger pool)")
    a = ap.parse_args()
    if a.smoke_from:
        smoke_subset(a.smoke_from, a.out)
    else:
        freeze(a.out, public_train=a.public_train)


if __name__ == "__main__":
    main()
