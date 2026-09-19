import random

import pytest

from kev.contrastive import generate, paired_flip


def test_rendered_pairs_change_one_sentence_not_order():
    rows, _ = generate(20, 17)
    for a, b in zip(rows[::2], rows[1::2]):
        left = a["state"]["case"].split(". ")
        right = b["state"]["case"].split(". ")
        assert len(left) == len(right)
        assert sum(x != y for x, y in zip(left, right)) == 1


def test_pair_metric_compares_semantics_not_indices():
    rows = [
        {"pair_id": "p", "sibling": "a", "keys": ["deny", "allow"], "label": 1, "p": [0, 1]},
        {"pair_id": "p", "sibling": "b", "keys": ["allow", "deny"], "label": 1, "p": [0, 1]},
    ]
    assert paired_flip(rows)["flip_rate"] == 1
    assert paired_flip(rows)["both_correct_rate"] == 1


def test_pair_metric_refuses_missing_sibling():
    with pytest.raises(ValueError, match="incomplete"):
        paired_flip([{"pair_id": "p", "sibling": "a", "keys": ["x", "y"], "label": 0, "p": [1, 0]}])


def test_compositional_truth_tables_and_unknowns():
    from itertools import product
    from kev.composition import evaluate_rule
    atoms = [{"kind": "flag", "fields": [k], "threshold": 0} for k in ("a", "b", "c")]
    for a, b, c in product((True, False), repeat=3):
        facts = dict(a=a, b=b, c=c)
        assert evaluate_rule(("and", ("or", 0, 1), 2), atoms, facts) == ((a or b) and c)
        assert evaluate_rule(("unless", 0, 1), atoms, facts) == (a and not b)
        assert evaluate_rule(("if", 0, 1, 2), atoms, facts) == (b if a else c)
    assert evaluate_rule(("and", 0, 1), atoms, {"a": False}) is False
    assert evaluate_rule(("and", 0, 1), atoms, {"a": True}) is None
    assert evaluate_rule(("or", 0, 1), atoms, {"a": False}) is None


@pytest.mark.parametrize("kind,threshold,value,expected", [
    ("lt", 10, 10, False), ("le", 10, 10, True), ("gt", 10, 10, False),
    ("ge", 10, 10, True), ("eq", 10, 11, False), ("range", 10, 20, True), ("range", 10, 21, False),
])
def test_rule_boundary_labels(kind, threshold, value, expected):
    from kev.composition import atom_value
    assert atom_value({"kind": kind, "threshold": threshold, "fields": ["x"]}, {"x": value}) == expected


def test_compositional_pairs_validate_and_keep_invariance():
    from kev.composition import SHAPES, TRAIN_SHAPES, DEV_SHAPES, TEST_SHAPES, check_group, generate as compose
    from kev.benchmark import labels, prediction_rows
    assert not set(TRAIN_SHAPES) & (set(DEV_SHAPES) | set(TEST_SHAPES))
    records = compose(4, "test", tuple(SHAPES))
    rows = []
    for i in range(0, len(records), 4):
        group = records[i:i + 4]
        assert check_group(group)
        for r in group:
            q = r["questions"]["decision"]
            keys, y = labels(q)
            rows += prediction_rows(r, {"probabilities": {"decision": {k: int(i == y) for i, k in enumerate(keys)}}})
    summary = paired_flip(rows)
    assert summary["both_correct_rate"] == 1
    assert summary["invariance_rate"] == 1
    assert summary["invariant_both_correct_rate"] == 1
    records[0]["state"]["case"] = "The facts were changed."
    with pytest.raises(ValueError, match="rendered facts"):
        check_group(records[:4])


def test_calibration_covers_every_family_without_splitting_groups():
    from kev.study_v3 import grouped_split, legacy
    train, calibration = grouped_split(legacy(10, "split-test"), 2)
    assert {r["_meta"]["family"] for r in train} == {r["_meta"]["family"] for r in calibration}
    assert not {r["_meta"]["group_id"] for r in train} & {r["_meta"]["group_id"] for r in calibration}
    assert len(calibration) == 16


def test_selective_metrics_include_confidence_ties():
    from kev.benchmark import metrics
    rows = [{"p": [0.99, 0.01], "label": y, "type": "noul"} for y in [0, 1]]
    report = metrics(rows)
    assert report["confident_error_rate"] == .5
    assert report["selective"]["0.5"] == {"coverage": 1.0, "accuracy": .5, "confidence_cutoff": .99}


def test_gate_rejects_confident_transfer_failure():
    from kev.experiment import gate_report
    coverage = {"requested_records": 2, "evaluated_records": 2, "requested_questions": 2,
                "evaluated_questions": 2, "rejected_records": 0, "truncated_records": 0}
    report = {"coverage": coverage, "transfer": {"coverage": coverage,
              "clean": {"confident_error_rate": .5}, "paired_flip": {"pairs": 1, "both_correct_rate": 0}}}
    result = gate_report(report, {"passed": True})
    assert not result["passed"]
    assert not result["checks"]["heldout_pairs_at_least_70pct"]
    assert not result["checks"]["transfer_confident_errors_below_10pct"]


def test_uneven_microbatches_have_equal_record_weight():
    import torch
    from kev.train import accumulation_records
    x = torch.arange(10, dtype=torch.float32)
    gradients = []
    for batch, accum in ((8, 1), (3, 3), (2, 4)):
        w = torch.tensor(1.0, requires_grad=True)
        for mb, start in enumerate(range(0, len(x), batch)):
            chunk = x[start:start + batch]
            ((w * chunk).sum() / accumulation_records(len(x), batch, accum, mb)).backward()
        gradients.append(w.grad.item())
    assert gradients[0] == gradients[2]
    assert accumulation_records(10, 3, 3, 2) == 9
    assert accumulation_records(10, 3, 3, 3) == 1


def test_v3_training_refuses_heldout_structure():
    from kev.study_v3 import validate_training
    r = {"_meta": {"source": "compositional", "family": "held_and_or"}}
    with pytest.raises(ValueError, match="held-out"):
        validate_training([r], {"trainable_sources": ["compositional"]})


def test_date_and_entity_atoms():
    from kev.composition import atom_value
    a = {"kind": "elapsed", "fields": ["start", "end"], "threshold": 2}
    assert atom_value(a, {"start": "2028-02-28", "end": "2028-03-01"}) is True
    assert atom_value(a, {"start": "2028-02-28", "end": "2028-03-02"}) is False
    a = {"kind": "match", "fields": ["signer", "approver"], "threshold": 0}
    assert atom_value(a, {"signer": "Mira", "approver": "Mira"}) is True
    assert atom_value(a, {"signer": "Mira", "approver": "Noah"}) is False
    assert atom_value(a, {"signer": "Mira"}) is None




def test_unpinned_base_requires_full_sha_in_trial():
    from kev.experiment import validated_trial
    manifest = {"base_revisions": {"pinned": "a" * 40}, "trainable_sources": []}
    with pytest.raises(ValueError, match="base_revision"):
        validated_trial({"base": "other"}, manifest)
    with pytest.raises(ValueError, match="base_revision"):
        validated_trial({"base": "other", "base_revision": "main"}, manifest)
    assert validated_trial({"base": "other", "base_revision": "b" * 40}, manifest)["base_revision"] == "b" * 40
    with pytest.raises(ValueError, match="conflicts"):
        validated_trial({"base": "pinned", "base_revision": "b" * 40}, manifest)


def test_none_pair_is_minimal_and_relabelled():
    import random
    from kev.data import none_pair, materialize
    req = {"state": "The shoes are the wrong size.", "questions": {"reason": {"type": "choice", "instructions": "Why?",
           "criteria": {"size": "Wrong size", "damage": "Damaged", "color": "Wrong color"}, "label": "size", "src": "t"}}}
    present, absent = none_pair(req, random.Random(3))
    pk, ak = list(present["questions"]["reason"]["criteria"]), list(absent["questions"]["reason"]["criteria"])
    assert len(pk) == 4 and present["questions"]["reason"]["label"] == "size"
    assert [k for k in pk if k != "size"] == ak                     # same order, true option removed, nothing else moved
    assert absent["questions"]["reason"]["label"] == ak[-1] or absent["questions"]["reason"]["label"] in ak
    assert absent["questions"]["reason"]["label"] not in req["questions"]["reason"]["criteria"]
    materialize(present); materialize(absent)
    assert none_pair({"state": "s", "questions": {"q": {"type": "noul", "instructions": "i", "label": True, "src": "t"}}}, random.Random(0)) == []


def test_option_isolation_mask_rule():
    from kev.model import branch_mask_batch, OPT_NONE, OPT_DECIDE
    seg = [0, 0, 1, 1, 1, 1, 1, 1, 1]           # state x2, then q: instr x2, option0 x2, option1 x2, decide
    opt = [OPT_NONE, OPT_NONE, OPT_NONE, OPT_NONE, 0, 0, 1, 1, OPT_DECIDE]
    m = branch_mask_batch([seg], "cpu", opts=[opt])[0, 0] == 0
    assert m[6, 4] == False and m[7, 5] == False      # option1 never sees option0
    assert m[6, 2] and m[6, 3] and m[6, 0]           # option sees instruction and state
    assert m[7, 6] and m[5, 4]                        # option sees itself (causal within span)
    assert all(m[8, j] for j in range(9))             # decide sees everything in its question
    assert m[3, 4] == False                           # instruction never sees options (causal)


def test_missing_partition_is_fetched_and_verified(tmp_path, monkeypatch):
    import json
    from kev import suite as S
    evals = tmp_path / "evals" / "x" / "decision-x"; evals.mkdir(parents=True)
    payload = b'{"state": "s", "questions": {}, "_meta": {}}\n'
    import hashlib
    S.write_json(evals / "manifest.json", {"files": {"train.jsonl": {"sha256": hashlib.sha256(payload).hexdigest(), "records": 1}}})
    served = tmp_path / "served.jsonl"; served.write_bytes(payload)
    calls = []
    def fake_download(repo, path, repo_type, revision):
        calls.append((repo, path, repo_type, revision)); return str(served)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    assert len(S.load_split(evals, "train")) == 1
    assert calls == [(S.SUITES_DATASET, "x/decision-x/train.jsonl", "dataset", S.SUITES_REVISION)]
    # a tampered mirror is rejected by the manifest hash
    (evals / "train.jsonl").unlink(); served.write_bytes(b'{"tampered": 1}\n')
    with pytest.raises(ValueError, match="checksum"):
        S.load_split(evals, "train")


def test_random_rule_structures_exclude_heldout_and_cover_negation():
    from kev.composition import SHAPES, DEV_SHAPES, TEST_SHAPES, canonical, push_negation, sample_trees, generate as compose, check_group
    assert canonical(("or", ("not", 0), ("and", 1, 2))) == canonical(SHAPES["held_or_not"])       # order/numbering-independent
    assert canonical(("not", ("and", 0, 1))) != canonical(("or", ("not", 0), ("not", 1)))
    assert canonical(push_negation(("not", ("and", 0, 1)))) == canonical(("or", ("not", 0), ("not", 1)))   # De Morgan
    trees = sample_trees(30, "t")
    held = {canonical(SHAPES[s]) for s in DEV_SHAPES + TEST_SHAPES}
    assert len({canonical(t) for t in trees}) == 30 and not any(canonical(t) in held or canonical(push_negation(t)) in held for t in trees)
    assert any("not(" in canonical(t) for t in trees)
    recs = compose(1, "t", styles=(3, 4), trees={f"rand{i}": t for i, t in enumerate(trees[:5])})
    for i in range(0, len(recs), 4):
        assert check_group(recs[i:i + 4])


def test_ordinal_threshold_families_are_balanced_minimal_pairs():
    import collections
    from kev.contrastive import ORDINAL_FAMILIES, generate
    from kev.data import materialize
    recs, rep = generate(30, "t", families=list(ORDINAL_FAMILIES))
    assert all(v["pairs"] == 30 for v in rep.values())
    for a, b in zip(recs[::2], recs[1::2]):
        assert a["questions"]["decision"]["type"] == "score" and a["questions"]["decision"]["label"] != b["questions"]["decision"]["label"]
        assert sum(x != y for x, y in zip(a["state"]["case"].split(". "), b["state"]["case"].split(". "))) == 1
        materialize(a)
    counts = collections.Counter((r["_meta"]["family"], r["questions"]["decision"]["label"]) for r in recs)
    assert all(counts[(f, level)] >= 8 for f in ORDINAL_FAMILIES for level in (0, 1, 2))   # every level appears in every family


def test_remote_predictor_maps_system_one_answers_and_retries(monkeypatch):
    import io, json
    from kev.benchmark import RemotePredictor
    rec = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B"}, "label": "a", "src": "t"},
                                       "y": {"type": "noul", "instructions": "i", "label": True, "src": "t"}}}
    calls = []
    class Resp:
        def __init__(self, body): self.body = body
        def read(self): return json.dumps(self.body).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def urlopen(req, timeout):
        calls.append(json.loads(req.data))
        if len(calls) == 1: raise OSError("503")
        return Resp({"model": "openjev-x", "answers": {"q": {"type": "choice", "probabilities": {"a": 0.7, "b": 0.3}}, "y": {"type": "noul", "noul": 0.2}}, "usage": {"input_tokens": 12}})
    p = RemotePredictor("http://example.test/", retries=2); monkeypatch.setattr(p._request, "urlopen", urlopen); monkeypatch.setattr("time.sleep", lambda s: None)
    out = p(rec)
    assert out["probabilities"] == {"q": {"a": 0.7, "b": 0.3}, "y": {"true": 0.2, "false": 0.8}} and p.served_model == "openjev-x" and len(calls) == 2
    assert calls[0]["model"] == "kev-latest" and "label" not in json.dumps(calls[0])        # labels never leave the machine


def test_top_bins_and_confidence_bias():
    from kev.benchmark import metrics
    rows = [{"p": [0.99, 0.01], "label": 0, "type": "noul"}, {"p": [0.99, 0.01], "label": 1, "type": "noul"}, {"p": [0.6, 0.4], "label": 0, "type": "noul"}]
    m = metrics(rows)
    assert m["top_bins"]["0.99"] == {"n": 2, "errors": 1, "error_rate": 0.5} and m["top_bins"]["0.9"]["n"] == 2
    assert abs(m["confidence_bias"] - ((0.99 + 0.99 + 0.6) / 3 - 2 / 3)) < 1e-9


def test_anchor_loss_aligns_by_key_and_skips_changed_option_sets():
    import torch
    from kev.train import anchor_loss
    q = {"keys": ["b", "a"]}
    z = torch.tensor([0.0, 0.0])
    # teacher puts 0.9 on 'a'; student uniform -> KL(teacher||student) > 0 and the same for either key order
    l1 = anchor_loss(z, {"keys": ["a", "b"]}, {"a": 0.9, "b": 0.1}, "cpu"); l2 = anchor_loss(z, q, {"a": 0.9, "b": 0.1}, "cpu")
    assert l1 is not None and abs(l1.item() - l2.item()) < 1e-6 and l1.item() > 0
    assert anchor_loss(z, {"keys": ["a", "b", "none"]}, {"a": 0.9, "b": 0.1}, "cpu") is None      # none-option inserted -> skip
    assert anchor_loss(z, q, None, "cpu") is None
    peaked = torch.tensor([10.0, -10.0])                                                       # student already matches teacher's argmax key 'b'? keys=[b,a]: p(b)=1
    assert anchor_loss(peaked, q, {"b": 1.0, "a": 0.0}, "cpu").item() < 1e-3


def test_anchor_trial_validation():
    from kev.experiment import validated_trial
    m = {"base_revisions": {"m": "x"}, "trainable_sources": ["arc", "boolq"]}
    with pytest.raises(ValueError, match="anchor"):
        validated_trial({"base": "m", "anchor_w": 0.5}, m)
    with pytest.raises(ValueError, match="anchor_sources"):
        validated_trial({"base": "m", "anchor": "runs/anchors/x.json", "anchor_w": 0.5, "anchor_sources": "mmlu"}, m)
    assert validated_trial({"base": "m", "anchor": "runs/anchors/x.json", "anchor_w": 0.5, "anchor_sources": "arc"}, m)["anchor_w"] == 0.5
