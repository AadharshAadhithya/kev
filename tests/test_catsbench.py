from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_catsbench as cats
import audit_catsbench

from kev.data import materialize


def test_direction_uses_the_same_stable_band_as_score() -> None:
    assert cats.direction_label(-0.021) == "decrease"
    assert cats.direction_label(-0.02) == "stable"
    assert cats.direction_label(0.02) == "stable"
    assert cats.direction_label(0.021) == "increase"


def test_maximum_position_uses_observation_midpoints() -> None:
    questions = cats.observation_questions(
        (1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 7.0, 6.0),
        {"observed_extreme_position"},
    )
    assert questions["maximum_position"]["label"] == "late"


def test_change_score_boundaries() -> None:
    assert [cats.change_score(value) for value in (-0.11, -0.10, -0.02, 0.02, 0.10, 0.11)] == [
        0,
        1,
        2,
        2,
        3,
        4,
    ]


def test_builds_observation_and_leak_free_forecast_records() -> None:
    row = cats.SourceSeries(
        source_id="yield_1_test",
        domain="yield",
        values=(10.0, 9.0, 8.0, 7.0, 6.0, 5.0),
        metadata={
            "attribute": "crop yield",
            "start date": "2017",
            "end date": "2022",
            "sampling frequency": "yearly",
            "mean": 7.5,
            "maximum": 10.0,
        },
        caption="Yield falls throughout the period.",
        caption_kind="human",
    )

    records = cats.build_records([row], "test")

    assert [record["_meta"]["mode"] for record in records] == ["observation", "forecast"]
    assert {question["type"] for question in records[0]["questions"].values()} == {
        "choice",
        "noul",
        "score",
    }
    forecast_state = json.loads(records[1]["state"])
    assert forecast_state["values"] == [10.0, 9.0, 8.0, 7.0]
    assert forecast_state["times"] == ["2017", "2018", "2019", "2020"]
    assert "caption" not in records[1]["state"]
    assert "mean" not in records[1]["state"]
    assert "maximum" not in records[1]["state"]
    assert "5.0" not in records[1]["state"]
    expected = cats.forecast_questions(
        row.values[:4], row.values[4:], set(cats.TEMPLATES), row.source_id
    )
    assert records[1]["questions"]["direction"]["label"] == "decrease"
    assert records[1]["questions"]["new_low"] == expected["new_low"]
    for record in records:
        assert materialize(record)


def test_infer_times_falls_back_when_range_disagrees_with_length() -> None:
    times, unit = cats.infer_times(
        {
            "start date": "2017-01-01",
            "end date": "2020-01-01",
            "sampling frequency": "weekly",
        },
        12,
    )
    assert times == list(range(12))
    assert unit == "index"


def test_load_archive_prefers_human_and_can_fall_back_to_synthetic(tmp_path: Path) -> None:
    archive_path = tmp_path / "test_data.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for source_id, caption_dir, caption in (
            ("yield_1_test", "human_rewritten_captions", "human caption"),
            ("yield_2_test", "synth_gt_captions", "synthetic caption"),
        ):
            archive.writestr(f"test_data/time series/{source_id}.txt", "1\n2\n3\n")
            archive.writestr(f"test_data/metadata/{source_id}.json", '{"attribute":"yield"}')
            archive.writestr(f"test_data/{caption_dir}/{source_id}.txt", caption)

    human = cats.load_archive(archive_path, "test", "human")
    any_caption = cats.load_archive(archive_path, "test", "any")

    assert [(row.source_id, row.caption_kind) for row in human] == [("yield_1_test", "human")]
    assert [(row.source_id, row.caption_kind) for row in any_caption] == [
        ("yield_1_test", "human"),
        ("yield_2_test", "synthetic"),
    ]


def test_parse_proposal_keeps_only_registered_templates() -> None:
    response = 'Here is the result:\n```json\n{"templates":["future_direction","invented"]}\n```'
    assert cats.parse_proposal(response) == {"future_direction"}
    assert cats.parse_proposal("Selected: observed_extreme_position") == {
        "observed_extreme_position"
    }
    assert cats.parse_proposal("not json") == set()


def test_binary_question_polarity_uses_both_labels() -> None:
    labels = {
        cats.forecast_questions(
            (1.0, 2.0, 3.0), (4.0, 5.0), {"future_new_high"}, f"series-{i}"
        )["new_high"]["label"]
        for i in range(20)
    }
    assert labels == {False, True}


def test_audit_recomputes_labels_and_detects_bad_label() -> None:
    row = cats.SourceSeries(
        source_id="yield_2_test",
        domain="yield",
        values=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        metadata={},
        caption="Yield rises and reaches a new high.",
        caption_kind="human",
    )
    records = cats.build_records([row], "test")
    report = audit_catsbench.audit(records, {row.source_id: row})
    assert report["valid"] is True
    records[0]["questions"]["direction"]["label"] = "decrease"
    report = audit_catsbench.audit(records, {row.source_id: row})
    assert report["valid"] is False
    assert "does not match" in report["errors"][0]


def test_build_records_deduplicates_identical_states() -> None:
    rows = [
        cats.SourceSeries(
            source_id=f"yield_{i}_test",
            domain="yield",
            values=(1.0, 2.0, 3.0),
            metadata={},
            caption="Yield rises.",
            caption_kind="human",
        )
        for i in range(2)
    ]
    records = cats.build_records(rows, "test")
    assert len(records) == 1
