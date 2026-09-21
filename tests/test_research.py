import copy
import random

import pytest
import torch

from kev import evaluate
from kev.data import materialize
from kev.train import question_loss


def choice_request():
    return {"state": "The shoes are the wrong size.", "questions": {"reason": {
        "type": "choice", "instructions": "Why return the shoes?",
        "criteria": {"size": "Wrong size", "damage": "Damaged", "color": "Wrong color"},
        "label": "size", "src": "fixture",
    }}}


def test_clean_evaluation_does_not_add_options(monkeypatch):
    observed = []

    def predict(tok, model, req):
        observed.append(copy.deepcopy(req))
        rec = materialize(req)
        return rec, [torch.ones(len(q["options"])) / len(q["options"]) for q in rec["questions"]]

    monkeypatch.setattr(evaluate, "_probs", predict)
    evaluate.test_accuracy(None, None, [choice_request() for _ in range(100)], random.Random(1))
    assert all(set(r["questions"]["reason"]["criteria"]) == {"size", "damage", "color"} for r in observed)


def test_none_removed_relabels_and_counts_complete_pairs(monkeypatch):
    observed = []

    def predict(tok, model, req):
        rec = materialize(req)
        observed.append(copy.deepcopy(req))
        return rec, [torch.nn.functional.one_hot(torch.tensor(q["label"]), len(q["options"])).float() for q in rec["questions"]]

    monkeypatch.setattr(evaluate, "_probs", predict)
    report = evaluate.test_none_of_the_above(None, None, [choice_request()], random.Random(1))
    assert report["n"] == 1
    assert report["true_option_present"]["acc"] == 1
    assert report["true_option_removed"]["picks_none_rate"] == 1
    assert observed[1]["questions"]["reason"]["label"] != "size"


def test_score_loss_is_proper_at_true_distribution():
    logits = torch.tensor([0.2, 0.8]).log().requires_grad_()
    loss = sum(p * question_loss(logits, {"label": y, "qtype": "score"}, "cpu", 0.5) for y, p in enumerate([0.2, 0.8]))
    loss.backward()
    assert logits.grad.abs().max().item() < 1e-6


def frozen_request(i=0):
    r = choice_request()
    r["_meta"] = {"id": f"item-{i}", "group_id": f"item-{i}", "source": "fixture", "variant": "clean"}
    return r


def test_contrast_cases_preserve_groups_and_relabel():
    from kev.suite import contrast_cases
    original = frozen_request()
    present, absent, permuted = contrast_cases(original)
    assert present["questions"]["reason"]["label"] == "size"
    assert absent["questions"]["reason"]["label"] == "none_of_these"
    for record in (present, absent, permuted):
        assert record["_meta"]["group_id"] == original["_meta"]["id"]
        materialize(record)
    assert original == frozen_request()


def test_source_sampling_does_not_depend_on_other_sources(monkeypatch):
    from kev import data

    def convert(split, n, rng):
        value = rng.randrange(1000000)
        rng.origins = [{"row": value, "row_sha256": str(value), "text_sha256": str(value)}]
        return [choice_request()]

    monkeypatch.setattr(data, "SOURCES", {"agnews": (convert, "train", "test"), "mnli": (convert, "train", "test")})
    alone = data.build(1, only=["mnli"])
    together = data.build(1)
    assert alone[0] == next(r for r in together if r["_meta"]["source"] == "mnli")


def test_strict_encoding_rejects_truncation():
    from types import SimpleNamespace
    from kev.model import encode

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=list(range(len(text))))

        def convert_tokens_to_ids(self, text):
            return 1000

    rec = {"state": "abcdefgh", "questions": [{"instr": "q", "options": ["a", "b"], "label": 0}]}
    assert encode(Tokenizer(), rec, max_state=4)["state_truncated"]
    with pytest.raises(ValueError, match="state exceeds"):
        encode(Tokenizer(), rec, max_state=4, strict=True)


def test_locked_split_and_hash_verification(tmp_path):
    import json
    from kev.suite import digest, load_split, write_json
    for name in ("development", "test"):
        (tmp_path / f"{name}.jsonl").write_text(json.dumps(frozen_request()) + "\n")
    manifest = {"files": {f"{name}.jsonl": {"sha256": digest(tmp_path / f"{name}.jsonl"), "records": 1} for name in ("development", "test")}}
    write_json(tmp_path / "manifest.json", manifest)
    assert len(load_split(tmp_path, "development")) == 1
    with pytest.raises(ValueError, match="locked test"):
        load_split(tmp_path, "test")
    (tmp_path / "development.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        load_split(tmp_path, "development")


def test_api_payload_excludes_answers_and_metadata():
    from kev.benchmark import api_request
    clean = api_request(frozen_request())
    assert set(clean) == {"state", "questions"}
    assert set(clean["questions"]["reason"]) == {"type", "instructions", "criteria"}


def test_failed_prediction_cannot_produce_partial_score(tmp_path):
    import json
    from kev.benchmark import evaluate_records

    def fail(record):
        raise ValueError("invalid prediction")

    out = tmp_path / "evaluation"
    with pytest.raises(ValueError):
        evaluate_records([frozen_request()], fail, out)
    failure = json.loads((out / "failure.json").read_text())
    assert failure["coverage"]["requested_records"] == 1
    assert failure["coverage"]["evaluated_records"] == 0
    assert failure["coverage"]["rejected_records"] == 1
    assert not (out / "report.json").exists()


def test_missing_answers_and_nonfinite_probabilities_fail():
    from kev.benchmark import prediction_rows, validate_distribution
    with pytest.raises(ValueError, match="answer IDs"):
        prediction_rows(frozen_request(), {"probabilities": {}})
    with pytest.raises(ValueError, match="non-finite"):
        validate_distribution({"x": float("nan"), "y": 0.5}, ["x", "y"])
    with pytest.raises(ValueError, match="sum"):
        validate_distribution({"x": 0, "y": 0}, ["x", "y"])


def test_task_macro_and_record_bootstrap():
    from kev.benchmark import prediction_rows, summarize, paired_bootstrap
    pred = {"probabilities": {"reason": {"size": 0.8, "damage": 0.1, "color": 0.1}}}
    rows = prediction_rows(frozen_request(), pred)
    report = summarize(rows)
    assert report["objective"] == pytest.approx(-report["clean"]["nll"])
    assert paired_bootstrap(rows, rows, samples=50)["ci95"] == [0, 0]
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap(rows, [])


def test_trial_config_cannot_change_evaluator_or_read_test():
    from kev.experiment import validated_trial
    manifest = {"base_revisions": {"model": "pinned"}}
    assert validated_trial({"base": "model"}, manifest)["ord_w"] == 0
    for extra in ({"test": True}, {"command": "echo x"}, {"lr": -1}, {"epochs": 2.5}):
        with pytest.raises(ValueError):
            validated_trial({"base": "model", **extra}, manifest)


def test_batched_mask_matches_single_and_pads_are_invisible():
    from kev.model import branch_mask, branch_mask_batch
    a, b = [0, 0, 1, 1, 2], [0, 1, 1]
    m = branch_mask_batch([a, b], "cpu")
    assert m.shape == (2, 1, 5, 5)
    assert torch.equal(m[0:1], branch_mask(a, "cpu"))
    assert torch.equal(m[1:2, :, :3, :3], branch_mask(b, "cpu"))
    allowed = m[1, 0] == 0
    assert not allowed[:3, 3:].any()          # real tokens never attend to padding
    assert allowed[3, 3] and allowed[4, 4]    # padded rows keep the diagonal, so softmax is finite
    assert not allowed[3, 1:3].any()          # pads belong to no question segment (state stays visible; rows are discarded)


def test_eval_only_sources_cannot_be_trained(tmp_path):
    import json
    from kev.data import EVAL_ONLY, TRAINABLE, ALL_SOURCES
    from kev.suite import digest, write_json
    from kev.experiment import load_plan
    assert "mmlu" in EVAL_ONLY and not set(TRAINABLE) & set(EVAL_ONLY) and set(TRAINABLE) | set(EVAL_ONLY) == set(ALL_SOURCES)
    r = frozen_request(); r["_meta"]["source"] = "mmlu"
    for name in ("train", "calibration", "development"):
        (tmp_path / f"{name}.jsonl").write_text(json.dumps(r) + "\n")
    write_json(tmp_path / "manifest.json", {"base_revisions": {"m": "x"}, "files": {f"{n}.jsonl": {"sha256": digest(tmp_path / f"{n}.jsonl"), "records": 1} for n in ("train", "calibration", "development")}})
    (tmp_path / "plan.json").write_text('[{"base": "m"}]')
    with pytest.raises(ValueError, match="eval-only"):
        load_plan(tmp_path, tmp_path / "plan.json")


def test_contrastive_pairs_are_checked_and_labelled_by_code():
    from kev import contrastive
    from kev.contrastive import FAMILIES, UNDETERMINED, check_pair, generate, label_of, paired_flip
    records, report = generate(5, seed=7)
    assert len(records) == 2 * 5 * len(FAMILIES) and all(v["pairs"] == 5 for v in report.values())
    for a, b in zip(records[::2], records[1::2]):
        assert a["_meta"]["pair_id"] == b["_meta"]["pair_id"] and a["_meta"]["family_id"] == b["_meta"]["family_id"]
        assert a["questions"]["decision"]["label"] != b["questions"]["decision"]["label"]
        assert a["state"]["policy"] == b["state"]["policy"]
        materialize(a); materialize(b)
    # a family whose label leaks into the policy text (no evidence needed) must be rejected by the ablation check
    def leaky(rng):
        def evaluate(f): return True
        item = {"policy": "Everything is allowed.", "sentences": [("Filler.", {}), ("Age is 30.", {"age": 30})], "evaluate": evaluate,
                "question": {"type": "noul", "instructions": "Allowed?"}}
        other = {**item, "sentences": [("Filler.", {}), ("Age is 10.", {"age": 10})], "evaluate": lambda f: False}
        return item, other
    assert check_pair(*leaky(None)) == "ablation_failed"
    # a pair whose two items do not differ in exactly one sentence is rejected
    a, b = FAMILIES["authorization"](__import__("random").Random(1))
    b["sentences"][2] = ("The refund amount is $1.", {})
    assert check_pair(a, b) == "not_exactly_one_sentence_differs"
    assert label_of(a, drop=0) == UNDETERMINED
    # paired_flip: a constant model never flips; a perfect model flips every pair and gets both right
    rows = []
    for rec in records[:8]:
        q = rec["questions"]["decision"]; keys = list(q["criteria"]) if q["type"] == "choice" else (["false", "true"] if q["type"] == "noul" else [str(i) for i in range(len(q["criteria"]))])
        y = keys.index(q["label"]) if q["type"] == "choice" else int(q["label"])
        rows.append({"pair_id": rec["_meta"]["pair_id"], "sibling": rec["_meta"]["sibling"], "keys": keys, "label": y, "p": [1.0 if i == y else 0.0 for i in range(len(keys))]})
    assert paired_flip(rows) == {"pairs": 4, "flip_rate": 1.0, "both_correct_rate": 1.0}
    constant = [{**r, "p": [1.0] + [0.0] * (len(r["keys"]) - 1)} for r in rows]
    assert paired_flip(constant)["flip_rate"] == 0.0


def test_permuted_variants_pair_with_their_parent_not_their_group():
    from kev.benchmark import prediction_rows, summarize
    from kev.suite import contrast_cases
    rows = []
    for i in range(2):
        r = frozen_request(i); r["_meta"]["group_id"] = "shared-pair"     # siblings share a bootstrap group
        variants = contrast_cases(r)
        for rec in [r] + variants:
            rec["_meta"].setdefault("group_id", "shared-pair")
            keys = list(rec["questions"]["reason"]["criteria"])
            rows += prediction_rows(rec, {"probabilities": {"reason": {k: (0.7 if k == rec["questions"]["reason"]["label"] else 0.3 / (len(keys) - 1)) for k in keys}}})
    report = summarize(rows)
    assert report["permutation"]["n"] == 2 and report["permutation"]["flip_rate"] == 0.0


def test_contrastive_eval_split_is_stratified_by_family():
    from collections import Counter
    from kev.contrastive import generate
    recs, _ = generate(6, seed="t", families=["authorization", "deadline"])
    dev, test = [], []
    for i in range(0, len(recs), 2):
        (dev if (i // 2) % 2 == 0 else test).extend(recs[i : i + 2])
    for part in (dev, test):
        fams = Counter(r["_meta"]["family"] for r in part)
        assert set(fams) == {"authorization", "deadline"} and all(v == 6 for v in fams.values())
        assert all(a["_meta"]["pair_id"] == b["_meta"]["pair_id"] for a, b in zip(part[::2], part[1::2]))


def test_coverage_cannot_split_equal_confidence_ties():
    from kev.benchmark import coverage_at_error
    correct = [True] * 90 + [False] * 10
    assert coverage_at_error([0.99] * 100, correct, 0.05) == 0.0
    assert coverage_at_error([0.99] * 100, correct[::-1], 0.05) == 0.0
    assert coverage_at_error([0.99] * 100, correct, 0.1) == 1.0
    assert coverage_at_error([], [], 0.05) == 0.0


def test_risk_curve_thresholds_and_nonmonotone_risk():
    from kev.benchmark import coverage_at_error, risk_coverage_curve
    curve = risk_coverage_curve([0.99, 0.99, 0.9, 0.8], [True, False, True, True])
    assert [p["accepted"] for p in curve] == [2, 3, 4]
    assert [p["threshold"] for p in curve] == [0.99, 0.9, 0.8]
    assert [p["risk"] for p in curve] == pytest.approx([0.5, 1 / 3, 0.25])
    assert coverage_at_error([0.99, 0.99, 0.9, 0.8], [True, False, True, True], 0.25) == 1.0


@pytest.mark.parametrize("confidence,correct,budget", [
    ([float("nan")], [True], 0.05), ([1.1], [True], 0.05),
    ([0.5], [], 0.05), ([[0.5]], [True], 0.05), ([0.5], [True], -0.1),
])
def test_selective_metrics_reject_invalid_inputs(confidence, correct, budget):
    from kev.benchmark import coverage_at_error
    with pytest.raises(ValueError):
        coverage_at_error(confidence, correct, budget)


def test_fixed_threshold_does_not_reselect_using_evaluation_labels():
    from kev.benchmark import select_threshold, evaluate_threshold
    threshold = select_threshold([0.99, 0.98, 0.97, 0.6], [True, True, True, False], 0.05)
    assert threshold == 0.97
    result = evaluate_threshold([0.99, 0.7, 0.6], [False, True, True], threshold)
    assert result["accepted"] == 1 and result["errors"] == 1 and result["risk"] == 1.0
    assert result["coverage"] == pytest.approx(1 / 3)
    empty = evaluate_threshold([0.99], [True], None)
    assert empty["coverage"] == 0 and empty["risk"] is None


def test_global_coverage_bootstrap_recomputes_full_statistic():
    from kev.benchmark import metrics, paired_bootstrap
    candidate, reference = [], []
    for i in range(20):
        common = {"id": str(i), "group": "one-cluster", "source": "fixture", "task": "fixture",
                  "question": "q", "variant": "clean", "type": "choice", "keys": ["a", "b"], "label": 0}
        candidate.append({**common, "p": [0.99, 0.01] if i < 18 else [0.4, 0.6]})
        reference.append({**common, "p": [0.6, 0.4] if i < 18 else [0.01, 0.99]})
    assert metrics(candidate)["acc"] == metrics(reference)["acc"]
    result = paired_bootstrap(candidate, reference, samples=50, metric="coverage_at_5pct_error", aggregation="micro")
    assert result["micro_coverage_at_5pct_error_delta"] == pytest.approx(0.9)
    assert result["ci95"] == pytest.approx([0.9, 0.9])
    assert result["groups"] == 1
    assert paired_bootstrap(candidate, candidate, samples=50, metric="aurc", aggregation="micro")["ci95"] == [0, 0]


def test_bootstrap_rejects_duplicate_or_inconsistent_pairs():
    from kev.benchmark import paired_bootstrap, prediction_rows
    rows = prediction_rows(frozen_request(), {"probabilities": {"reason": {"size": 0.8, "damage": 0.1, "color": 0.1}}})
    with pytest.raises(ValueError, match="duplicate"):
        paired_bootstrap(rows + rows, rows)
    with pytest.raises(ValueError, match="group"):
        paired_bootstrap(rows, [{**rows[0], "group": "different"}])


def test_temperature_preserves_argmax_but_not_cross_question_ranking():
    import numpy as np
    from kev.benchmark import probabilities_at_temperature
    rows = [{"p": [0.6, 0.2, 0.2]}, {"p": [0.55, 0.449, 0.001]}]
    calibrated = [probabilities_at_temperature(r, 2.0) for r in rows]
    assert max(rows[0]["p"]) > max(rows[1]["p"])
    assert calibrated[0].max() < calibrated[1].max()
    assert all(np.argmax(r["p"]) == p.argmax() for r, p in zip(rows, calibrated))


def test_calibration_uses_logits_without_probability_floor_distortion():
    import numpy as np
    from kev.benchmark import probabilities_at_temperature
    p = probabilities_at_temperature({"p": [1.0, 0.0], "logits": [0.0, -100.0]}, 2.0)
    assert p[1] == pytest.approx(np.exp(-50), rel=1e-6, abs=0)
    for temperature in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            probabilities_at_temperature({"p": [0.5, 0.5]}, temperature)


def test_unknowable_not_in_raw_or_calibrated_accuracy_denominator():
    from kev.benchmark import prediction_rows, summarize
    rows = prediction_rows(frozen_request(), {"probabilities": {"reason": {"size": 0.8, "damage": 0.1, "color": 0.1}}})
    rows.append({**rows[0], "id": "unk", "source": "unknowable", "task": "unknowable_test"})
    result = summarize(rows, temperature=2.0)
    assert result["clean"]["n"] == result["calibrated_clean"]["n"] == 1
    assert result["metric_policy"]["selective_ties"] == "whole_confidence_groups"


def test_logits_are_recorded_in_option_order():
    from kev.benchmark import prediction_rows
    prediction = {"probabilities": {"reason": {"size": 0.8, "damage": 0.1, "color": 0.1}},
                  "logits": {"reason": {"color": -2.0, "damage": -2.0, "size": 0.0}}, "inference_temperature": 1.0}
    row = prediction_rows(frozen_request(), prediction)[0]
    assert row["logits"] == [0.0, -2.0, -2.0] and row["inference_temperature"] == 1.0
    prediction["logits"]["reason"]["size"] = float("inf")
    with pytest.raises(ValueError, match="logit"):
        prediction_rows(frozen_request(), prediction)


def test_risk_curve_area_has_explicit_tie_policy():
    from kev.benchmark import area_under_risk_coverage
    assert area_under_risk_coverage([0.99, 0.9, 0.8], [True, True, False]) == pytest.approx(1 / 9)
    assert area_under_risk_coverage([0.99] * 10, [True] * 9 + [False]) == pytest.approx(0.1)
    assert area_under_risk_coverage([0.99] * 10, [False] + [True] * 9) == pytest.approx(0.1)


@pytest.mark.parametrize("options", [{}, {"label_smoothing": 0.05}, {"brier_w": 0.5}, {"focal_gamma": 1.0}])
def test_training_losses_match_definitions(options):
    z = torch.tensor([0.4, -0.7, 1.2], requires_grad=True)
    q = {"label": 1, "qtype": "choice"}
    ce = torch.nn.functional.cross_entropy(z[None], torch.tensor([1]))
    target = torch.nn.functional.one_hot(torch.tensor(1), 3).float()
    expected = ce
    if "label_smoothing" in options:
        epsilon = options["label_smoothing"]
        expected = -(((1 - epsilon) * target + epsilon / 3) * z.log_softmax(-1)).sum()
    elif "brier_w" in options:
        expected = ce + options["brier_w"] * (z.softmax(-1) - target).square().sum()
    elif "focal_gamma" in options:
        expected = (1 - z.softmax(-1)[1]) ** options["focal_gamma"] * ce
    actual = question_loss(z, q, "cpu", 0, **options)
    assert torch.allclose(actual, expected)
    actual.backward()
    assert torch.isfinite(z.grad).all()


def test_brier_mixture_is_proper_and_soft_targets_unchanged():
    distribution = torch.tensor([0.2, 0.3, 0.5])
    logits = distribution.log().requires_grad_()
    expected_loss = sum(p * question_loss(logits, {"label": i, "qtype": "choice"}, "cpu", 0, brier_w=0.5)
                        for i, p in enumerate(distribution))
    expected_loss.backward()
    assert logits.grad.abs().max() < 1e-6
    soft = {"label": 0, "qtype": "choice", "target": [1 / 3] * 3}
    base = question_loss(logits, soft, "cpu", 0)
    for options in ({"label_smoothing": 0.05}, {"brier_w": 0.5}, {"focal_gamma": 1.0}):
        assert torch.equal(question_loss(logits, soft, "cpu", 0, **options), base)


@pytest.mark.parametrize("bad", [{"label_smoothing": -0.1}, {"label_smoothing": 1.1}, {"brier_w": float("nan")},
                                  {"focal_gamma": -1}, {"brier_w": 0.5, "focal_gamma": 1.0}])
def test_loss_options_fail_before_training(bad):
    with pytest.raises(ValueError):
        question_loss(torch.zeros(3), {"label": 0, "qtype": "choice"}, "cpu", 0, **bad)


def test_trial_accepts_one_registered_loss_change():
    from kev.experiment import validated_trial
    manifest = {"base_revisions": {"model": "pinned"}}
    for flag, value in (("label_smoothing", 0.05), ("brier_w", 0.5), ("focal_gamma", 1.0)):
        assert validated_trial({"base": "model", flag: value}, manifest)[flag] == value
    with pytest.raises(ValueError):
        validated_trial({"base": "model", "brier_w": 0.5, "focal_gamma": 1.0}, manifest)


def test_temperature_fit_requires_raw_rows_and_uses_true_logit_nll():
    from kev.benchmark import fit_temperature, nll_at_temperature
    row = {"variant": "clean", "source": "fixture", "task": "fixture", "p": [1.0, 0.0],
           "logits": [0.0, -100.0], "label": 1, "inference_temperature": 1.0}
    assert nll_at_temperature(row, 2.0) == pytest.approx(50.0)
    assert fit_temperature([row], aggregation="micro") == pytest.approx(4.0)
    with pytest.raises(ValueError, match="raw logits"):
        fit_temperature([{**row, "inference_temperature": 2.0}])


def test_training_plan_matches_registered_screen():
    import json
    from pathlib import Path
    from kev.experiment import load_plan
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "experiments/calibration-audit-protocol.json").read_text())
    trials = load_plan(root / protocol["data"]["decision_suite"], root / "experiments/calibration-screen-4b.json")
    loss_keys = {"label_smoothing", "brier_w", "focal_gamma"}
    assert len(trials) == len(protocol["screen"]["arms"]) == 4
    assert all({k: v for k, v in t.items() if k not in loss_keys} == {k: v for k, v in trials[0].items() if k not in loss_keys} for t in trials)
    assert trials[0]["init_from"] == protocol["parents"]["4b"]
    for trial, arm in zip(trials, protocol["screen"]["arms"]):
        assert all(trial[k] == arm[k] for k in loss_keys)


def test_modal_compute_bound_includes_requested_memory_and_cpu():
    from modal_app import TRIAL_CPU, TRIAL_MEMORY, compute_bound
    expected = 3.95 + TRIAL_CPU * 0.04730 + TRIAL_MEMORY[1] / 1024 * 0.008
    assert compute_bound("H100", 3600, 1) == pytest.approx(expected)
    with pytest.raises(ValueError):
        compute_bound("unknown", 3600, 1)


def test_modal_worker_preserves_object_dependency_environment():
    from modal_app import worker_environment
    env = worker_environment("kev-calibration-audit", "H100", "named-secret")
    assert env["KEV_APP_NAME"] == "kev-calibration-audit"
    assert env["KEV_GPU"] == "H100" and env["KEV_HF_SECRET"] == "named-secret"
    assert "HF_TOKEN" not in env
    assert "KEV_HF_SECRET" not in worker_environment("kev-research", "H100")


def test_screen_requires_beating_continuation_control_not_just_parent():
    from scripts.review_calibration_screen import screen_checks
    def result(cov):
        return {"micro": {"coverage_at_5pct_error": cov, "acc": 0.8, "aurc": 0.05}, "sources": {"x": {"acc": 0.8}}}
    rule = {"coverage_delta_vs_ce_control_min": 0.05, "coverage_delta_vs_recalibrated_parent_min": 0.05,
            "accuracy_delta_vs_each_min": -0.01, "aurc_delta_vs_each_max": 0, "per_source_accuracy_delta_min": -0.05}
    checks = screen_checks(result(0.6), {"parent": result(0.5), "ce-control": result(0.59)}, rule)
    assert checks["coverage_vs_parent"] and not checks["coverage_vs_ce-control"]


def test_final_audit_partition_remains_bound_to_registration():
    import json
    from pathlib import Path
    from kev.suite import digest, load_split
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "experiments/calibration-audit-protocol.json").read_text())
    suite = root / protocol["data"]["development_suite"]
    assert digest(suite / "test.jsonl") == protocol["data"]["fresh_test_sha256"]
    assert digest(suite / "calibration.jsonl") == protocol["data"]["fresh_threshold_sha256"]
    with pytest.raises(ValueError, match="locked test"):
        load_split(suite, "test")


def test_tempered_replay_records_effective_temperature():
    from scripts.calibration_audit import tempered
    from kev.benchmark import fit_temperature
    raw = {"variant": "clean", "source": "fixture", "task": "fixture", "p": [0.9, 0.1],
           "logits": [2.197224577, 0.0], "label": 0, "inference_temperature": 1.0}
    rows = tempered([raw], 2.0)
    assert raw["inference_temperature"] == 1.0 and rows[0]["inference_temperature"] == 2.0
    with pytest.raises(ValueError, match="raw logits"):
        fit_temperature(rows)
