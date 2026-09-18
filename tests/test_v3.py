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


