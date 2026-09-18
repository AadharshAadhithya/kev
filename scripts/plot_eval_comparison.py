import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASK_LABELS = {
    "agnews": "AG News",
    "agnews_yn": "AG News · Yes/no",
    "banking77": "Banking77",
    "boolq": "BoolQ",
    "mnli": "MNLI",
    "sst5": "SST-5 · Score",
    "yelp": "Yelp · Score",
    "yelp_yn": "Yelp · Yes/no",
    "amazon": "Amazon 5-star · Score",
    "dbpedia14": "DBpedia-14",
    "emotion": "Emotion-6",
    "imdb": "IMDB",
    "mmlu": "MMLU · 4-way",
    "qnli": "QNLI",
    "trec": "TREC-6",
    "tweet_offensive": "TweetEval · Offensive",
}
SHARED_CAVEATS = [
    "Development-suite comparison; the locked final test has not been evaluated.",
    "Not a final model-selection claim and not a fair architecture ablation.",
    "Accuracy is top-1 exact match, including Score questions; perturbations are excluded from plotted accuracy.",
    "Macro accuracy weights tasks equally; micro accuracy weights clean questions equally.",
    "Only the saved paired macro-difference interval is shown; no per-task uncertainty is estimated.",
    "Behaviour probes (ECE, permutation flip rate, none-of-the-above) are point estimates without uncertainty.",
    "Jev is a hosted alias with an unexposed revision, evaluated through the Vercel AI SDK.",
    "NLL is not plotted: Jev returns rounded zeros, making NLL clipping-floor sensitive (EPS 1e-9; saved alternatives 1e-6 and 1e-3).",
]
PROFILES = {
    "baseline": {
        "eyebrow": "PRELIMINARY  /  BASELINE SNAPSHOT",
        "status": "preliminary baseline snapshot",
        "suite": "evals/decision-v1/development.jsonl",
        "candidate_dir": "runs/research-kev-v01",
        "reference_dir": "runs/research-jev-v1",
        "paired": "runs/kev-vs-jev-v1.json",
        "out": "docs/kev-vs-jev.png",
        "candidate_label": "Original kev: Qwen2.5-0.5B + LoRA",
        "candidate_description": "Original released Qwen2.5-0.5B + LoRA checkpoint",
        "extra_units": "including perturbations",
        "caveats": [
            "Original kev checkpoint was fine-tuned on all six sources; Jev training exposure is unknown.",
            "This is not an out-of-domain equivalence claim.",
        ],
        "footnotes": [
            "Scope: original kev was fine-tuned on all six sources; Jev training exposure is unknown. Not an out-of-domain equivalence claim or fair architecture ablation.",
            "Status: the locked final test has not been evaluated. More runs are pending; this development snapshot is not a final model-selection claim.",
            "Probability caveat: NLL is omitted. Jev returns rounded zeros ({reference_zeros:,} in this run); NLL is clipping-floor sensitive (EPS 1e-9; saved alternatives 1e-6 and 1e-3).",
        ],
        "png_title": "kev vs. Jev — preliminary development baseline",
    },
    "transfer": {
        "eyebrow": "TRANSFER  /  OUTSIDE KEV'S TRAINING DATA",
        "status": "transfer snapshot on eval-only sources",
        "suite": "evals/transfer-v1/development.jsonl",
        "candidate_dir": "runs/transfer-kev-v01",
        "reference_dir": "runs/transfer-jev-v1",
        "paired": "runs/kev-vs-jev-transfer-v1.json",
        "out": "docs/kev-vs-jev-transfer.png",
        "candidate_label": "Released kev-0.5b: Qwen2.5-0.5B + LoRA, CPU",
        "candidate_description": "Released kev-0.5b Qwen2.5-0.5B + LoRA checkpoint, CPU inference",
        "extra_units": "including contrast cases",
        "caveats": [
            "Eval-only suite: eight public sources kev never trained on.",
            "Contamination check is exact-match only: 0 overlaps with kev's 1,360 training/calibration states; no semantic or near-duplicate check.",
            "Jev's exposure to these public datasets is unknown.",
            "Emotion NLL for Jev is floor-driven by rounded zeros and is not plotted.",
        ],
        "footnotes": [
            "Scope: eval-only suite of eight public sources kev never trained on. Exact-match check found 0 overlaps with kev's 1,360 training/calibration states; no semantic check was run.",
            "Status: the locked final test has not been evaluated. Jev's exposure to these public datasets is unknown; this is not a fair architecture ablation.",
            "Probability caveat: NLL is omitted. Jev returned rounded zeros ({reference_zeros:,} in this run); Emotion NLL {emotion_nll:.2f} is floor-driven (EPS 1e-9; saved alternatives 1e-6 and 1e-3).",
        ],
        "png_title": "kev vs. Jev — transfer to eval-only sources",
    },
}


def load_json(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(np.isclose(actual, expected, rtol=0, atol=1e-12), message)


def validate_report(report, rows, name):
    index = {(row["id"], row["question"]): row for row in rows}
    require(len(index) == len(rows), f"{name}: duplicate question IDs")
    require(report["split"] == "development", f"{name}: expected development split")
    coverage = report["coverage"]
    require(len(rows) == coverage["evaluated_questions"] == coverage["requested_questions"],
            f"{name}: question coverage mismatch")
    require(len({row["id"] for row in rows}) == coverage["evaluated_records"] == coverage["requested_records"],
            f"{name}: record coverage mismatch")
    require(coverage["rejected_records"] == coverage["truncated_records"] == 0,
            f"{name}: incomplete coverage")
    clean = [row for row in rows if row["variant"] == "clean"]
    require({row["task"] for row in clean} == set(report["tasks"]),
            f"{name}: task sets differ")
    populations = [("clean", clean)]
    populations += [(task, [row for row in clean if row["task"] == task]) for task in report["tasks"]]
    populations += [(variant, [row for row in rows if row["variant"] == variant]) for variant in report["variants"]]
    for label, subset in populations:
        metrics = report["clean"] if label == "clean" else report["tasks"].get(label) or report["variants"][label]
        require(len(subset) == metrics["n"], f"{name}/{label}: question count mismatch")
        accuracy = np.mean([np.argmax(row["p"]) == row["label"] for row in subset])
        close(accuracy, metrics["acc"], f"{name}/{label}: accuracy mismatch")
        confidence = np.mean([np.max(row["p"]) for row in subset])
        close(confidence, metrics["mean_conf"], f"{name}/{label}: mean confidence mismatch")
    return index, len({row["id"] for row in clean})


def behaviour(report):
    variants, permutation = report["variants"], report["permutation"]
    require({"none_present", "none_absent"} <= set(variants), "Missing none-of-the-above variants")
    return {
        "ece_clean": report["clean"]["ece"],
        "mean_confidence_clean": report["clean"]["mean_conf"],
        "permutation_flip_rate": permutation["flip_rate"],
        "permutation_n": permutation["n"],
        "none_present_accuracy": variants["none_present"]["acc"],
        "none_present_n": variants["none_present"]["n"],
        "none_absent_accuracy": variants["none_absent"]["acc"],
        "none_absent_n": variants["none_absent"]["n"],
    }


def build_summary(paths, profile, title_suffix):
    kev, jev, paired = (load_json(paths[key]) for key in ("candidate_report", "reference_report", "paired"))
    kev_rows, jev_rows = (load_json(paths[key]) for key in ("candidate_rows", "reference_rows"))
    a, clean_records = validate_report(kev, kev_rows, "kev")
    b, jev_clean_records = validate_report(jev, jev_rows, "Jev")
    require(a.keys() == b.keys(), "Question IDs differ between models")
    for key, row in a.items():
        require(all(row[field] == b[key][field]
                    for field in ("group", "source", "task", "type", "variant", "keys", "label")),
                f"Pair metadata differs: {key}")
    require(clean_records == jev_clean_records, "Clean record counts differ")
    require(kev["suite_sha256"] == jev["suite_sha256"] == paired["suite_sha256"],
            "Suite hashes differ")
    require(set(kev["tasks"]) == set(jev["tasks"]), "Task sets differ between models")
    probe_sizes = {report["permutation"]["n"] for report in (kev, jev)} | {
        report["variants"][variant]["n"] for report in (kev, jev) for variant in ("none_present", "none_absent")}
    require(len(probe_sizes) == 1, "Perturbation probe sizes differ; the shared footnote assumes one size")
    tasks = []
    for task in sorted(kev["tasks"]):
        ka, ja = kev["tasks"][task], jev["tasks"][task]
        require(ka["n"] == ja["n"], f"{task}: model question counts differ")
        tasks.append({"task": task, "label": TASK_LABELS.get(task, task), "questions": ka["n"],
                      "kev_accuracy": ka["acc"], "jev_accuracy": ja["acc"]})
    macro = {name: float(np.mean([row[f"{name}_accuracy"] for row in tasks]))
             for name in ("kev", "jev")}
    interval = paired["paired"]["acc"]
    close(macro["kev"] - macro["jev"], interval["macro_acc_delta"], "Saved macro delta differs")
    require(interval["ci95"][0] <= interval["macro_acc_delta"] <= interval["ci95"][1],
            "Invalid saved confidence interval")
    for name, report in (("candidate", kev), ("reference", jev)):
        require(report["clean"]["n"] == paired["clean"][name]["n"], "Comparison population differs")
        close(report["clean"]["acc"], paired["clean"][name]["acc"], "Comparison accuracy differs")
    spec = PROFILES[profile]
    footnote_values = {"reference_zeros": jev["metric_policy"]["returned_zeros"],
                       "emotion_nll": jev["tasks"].get("emotion", {}).get("nll", float("nan"))}
    return {
        "profile": profile,
        "status": spec["status"],
        "title": "kev vs. Jev" + title_suffix,
        "eyebrow": spec["eyebrow"],
        "suite": spec["suite"],
        "suite_sha256": kev["suite_sha256"],
        "inputs": {key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for key, path in paths.items()},
        "models": {"kev": spec["candidate_description"], "jev": jev["provider"]},
        "candidate_label": spec["candidate_label"],
        "extra_units": spec["extra_units"],
        "coverage_per_model": kev["coverage"],
        "clean_records": clean_records,
        "clean_questions": kev["clean"]["n"],
        "tasks": tasks,
        "tasks_where_jev_higher": sum(row["kev_accuracy"] < row["jev_accuracy"] for row in tasks),
        "macro_accuracy": macro,
        "macro_accuracy_difference": {
            "direction": "kev minus Jev",
            "fraction": interval["macro_acc_delta"],
            "percentage_points": 100 * interval["macro_acc_delta"],
            "ci95_fraction": interval["ci95"],
            "ci95_percentage_points": [100 * value for value in interval["ci95"]],
            "bootstrap_samples": interval["samples"],
            "bootstrap_unit": interval["unit"],
            "interval_source": str(paths["paired"]),
        },
        "micro_accuracy": {"kev": kev["clean"]["acc"], "jev": jev["clean"]["acc"]},
        "micro_accuracy_difference_percentage_points": 100 * (kev["clean"]["acc"] - jev["clean"]["acc"]),
        "behaviour": {"kev": behaviour(kev), "jev": behaviour(jev)},
        "reference_returned_zeros": jev["metric_policy"]["returned_zeros"],
        "nll_floor_sensitivity_not_plotted": paired["nll_floor_sensitivity"],
        "footnotes": [line.format(**footnote_values) for line in spec["footnotes"]],
        "caveats": spec["caveats"] + SHARED_CAVEATS,
        "png_title": spec["png_title"],
    }


def plot(summary, output):
    ink, muted, rule = "#202B30", "#5D6970", "#DCE1E4"
    colors = {"kev": "#355C6B", "jev": "#AFBBC1"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": ink, "axes.labelcolor": muted,
                         "xtick.color": muted, "ytick.color": ink,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(15, 10.8))
    fig.text(.045, .955, summary["eyebrow"], fontsize=10, weight="bold", color=muted)
    fig.text(.045, .911, summary["title"], fontsize=29, weight="bold")
    sdk = summary["models"]["jev"]["sdk"].replace("ai@", "Vercel AI SDK ")
    fig.text(.045, .878, f"{summary['candidate_label']}     |     Jev: {summary['models']['jev']['model']} · {sdk}",
             fontsize=11.5, color=muted)
    coverage = summary["coverage_per_model"]
    fig.text(.045, .848,
             f"Same frozen development suite · {coverage['evaluated_records']:,} requests / "
             f"{coverage['evaluated_questions']:,} questions {summary['extra_units']}",
             fontsize=11, color=muted)
    fig.add_artist(Line2D([.045, .96], [.825, .825], transform=fig.transFigure, color=rule, lw=1))
    fig.text(.045, .792, "Per-task accuracy", fontsize=15, weight="bold")
    fig.text(.045, .767,
             f"Clean subset only · {summary['clean_records']:,} records / {summary['clean_questions']:,} questions · top-1 exact match",
             fontsize=10, color=muted)

    ax = fig.add_axes([.185, .232, .49, .505])
    tasks = summary["tasks"]
    y = np.arange(len(tasks))
    for name, offset in (("kev", -.17), ("jev", .17)):
        values = [100 * task[f"{name}_accuracy"] for task in tasks]
        bars = ax.barh(y + offset, values, height=.28, color=colors[name], label="kev" if name == "kev" else "Jev", zorder=3)
        for bar, value in zip(bars, values):
            inside = value > 90
            ax.text(value - 1 if inside else value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%",
                    va="center", ha="right" if inside else "left", fontsize=9.5,
                    color=("white" if name == "kev" else ink) if inside else ink)
    ax.set_yticks(y, [f"{task['label']}\nn = {task['questions']:,} questions" for task in tasks], fontsize=10)
    ax.set_ylim(len(tasks) - .5, -.5)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Accuracy · higher is better", labelpad=10, fontsize=10)
    ax.grid(axis="x", color=rule, linewidth=.8, zorder=0)
    ax.tick_params(axis="both", length=0, pad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="lower right", bbox_to_anchor=(1.01, 1.03), ncol=2, frameon=False,
              handlelength=1.2, columnspacing=1.5, fontsize=11)

    fig.add_artist(Line2D([.71, .71], [.226, .80], transform=fig.transFigure, color=rule, lw=1))
    left = .744
    fig.text(left, .792, "Macro-task difference", fontsize=15, weight="bold")
    fig.text(left, .766, "kev minus Jev · equal task weight", fontsize=10, color=muted)
    delta = summary["macro_accuracy_difference"]
    value = delta["percentage_points"]
    lo, hi = delta["ci95_percentage_points"]
    fig.text(left, .717, f"{value:+.2f} pp", fontsize=29, weight="bold", color=colors["kev"])
    fig.text(left, .684, f"95% CI  [{lo:+.2f}, {hi:+.2f}] pp", fontsize=12)
    ci_ax = fig.add_axes([left, .612, .21, .050])
    bound = max(8, np.ceil(max(abs(lo), abs(hi)) / 2) * 2)
    ci_ax.axvline(0, color=muted, linestyle=(0, (3, 3)), lw=1)
    ci_ax.errorbar(value, 0, xerr=[[value - lo], [hi - value]], fmt="o", color=colors["kev"],
                   markersize=6, capsize=5, linewidth=2)
    ci_ax.set_xlim(-bound, bound)
    ci_ax.set_ylim(-1, 1)
    ci_ax.set_yticks([])
    ci_ax.set_xticks([-bound, 0, bound], [f"{int(-bound)}", "0", f"+{int(bound)}"])
    ci_ax.tick_params(axis="x", length=0, labelsize=9)
    for spine in ci_ax.spines.values():
        spine.set_visible(False)
    fig.text(left, .579, "← Jev higher          kev higher →", fontsize=9, color=muted)
    fig.text(left, .548,
             f"Saved paired bootstrap · {delta['bootstrap_samples']:,} draws · source-stratified\n"
             "original records; sibling questions stay together.",
             fontsize=9, color=muted, linespacing=1.45, va="top")
    if lo <= 0 <= hi:
        headline, detail = "Interval includes zero.", "Not evidence of equivalence."
    else:
        headline = "Interval excludes zero."
        detail = f"Jev higher on {summary['tasks_where_jev_higher']} of {len(tasks)} tasks here."
    fig.text(left, .493, headline, fontsize=11, weight="bold")
    fig.text(left, .468, detail, fontsize=10, color=muted)
    fig.add_artist(Line2D([left, .96], [.447, .447], transform=fig.transFigure, color=rule, lw=1))
    fig.text(left, .417, "Aggregates and probes", fontsize=13, weight="bold")
    fig.text(left, .392, "Clean subset unless noted · point estimates", fontsize=9, color=muted)
    fig.text(.885, .366, "kev", color=colors["kev"], weight="bold", ha="right", fontsize=10)
    fig.text(.955, .366, "Jev", color=muted, weight="bold", ha="right", fontsize=10)
    probes = {name: summary["behaviour"][name] for name in ("kev", "jev")}
    pct = "{:.1f}%".format
    probe_n = probes["kev"]["permutation_n"]
    rows = [
        (f"Macro acc · {len(tasks)} tasks", lambda n: pct(100 * summary["macro_accuracy"][n])),
        (f"Micro acc · {summary['clean_questions']} Qs", lambda n: pct(100 * summary["micro_accuracy"][n])),
        ("Mean confidence", lambda n: pct(100 * probes[n]["mean_confidence_clean"])),
        ("ECE (lower is better)", lambda n: f"{probes[n]['ece_clean']:.3f}"),
        ("Perm. argmax flip rate*", lambda n: pct(100 * probes[n]["permutation_flip_rate"])),
        ("NOTA present acc*", lambda n: pct(100 * probes[n]["none_present_accuracy"])),
        ("NOTA absent acc*", lambda n: pct(100 * probes[n]["none_absent_accuracy"])),
    ]
    for i, (label, fmt) in enumerate(rows):
        ypos = .338 - .0245 * i
        fig.text(left, ypos, label, fontsize=8.5)
        fig.text(.885, ypos, fmt("kev"), ha="right", fontsize=9.5)
        fig.text(.955, ypos, fmt("jev"), ha="right", fontsize=9.5)
    fig.text(left, .172, f"*Perturbation records, n = {probe_n} each; NOTA = none-of-the-above.",
             fontsize=8, color=muted, va="bottom")

    fig.add_artist(Line2D([.045, .96], [.16, .16], transform=fig.transFigure, color=rule, lw=1))
    footnotes = summary["footnotes"] + [
        f"Source: {summary['suite']} · saved paired comparison: {Path(delta['interval_source']).name} · no per-task error bars are estimated.",
    ]
    for ypos, text in zip((.132, .105, .078, .051), footnotes):
        fig.text(.045, ypos, text, fontsize=9, color=muted)
    fig.savefig(output, dpi=200, metadata={"Title": summary["png_title"],
                                          "Description": "Clean per-task accuracy, saved paired macro-accuracy confidence interval, and behaviour probes."})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot a kev/Jev comparison snapshot using saved development artifacts only.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="baseline",
                        help="Header, suite label, and footnotes; also supplies default paths.")
    parser.add_argument("--candidate-dir", type=Path, help="kev run directory with report.json and rows.json")
    parser.add_argument("--reference-dir", type=Path, help="Jev run directory with report.json and rows.json")
    parser.add_argument("--paired", type=Path, help="Saved paired comparison JSON from kev.compare")
    parser.add_argument("--out", type=Path, help="Output PNG path")
    parser.add_argument("--summary", type=Path, help="Output JSON path; defaults to <out stem>-summary.json")
    parser.add_argument("--title-suffix", default="", help="Appended to the 'kev vs. Jev' title")
    args = parser.parse_args()
    spec = PROFILES[args.profile]
    candidate = (args.candidate_dir or ROOT / spec["candidate_dir"]).resolve()
    reference = (args.reference_dir or ROOT / spec["reference_dir"]).resolve()
    paths = {"candidate_report": candidate / "report.json", "candidate_rows": candidate / "rows.json",
             "reference_report": reference / "report.json", "reference_rows": reference / "rows.json",
             "paired": (args.paired or ROOT / spec["paired"]).resolve()}
    for path in paths.values():
        require(path.is_file(), f"Missing input: {path}")
    out = (args.out or ROOT / spec["out"]).resolve()
    summary_path = (args.summary or out.with_name(out.stem + "-summary.json")).resolve()
    require(out != summary_path, "Image and summary paths must differ")
    for output in (out, summary_path):
        require(output not in paths.values(), "Output must not overwrite an input artifact")
        require(output.parent.is_dir(), f"Output directory does not exist: {output.parent}")
    summary = build_summary(paths, args.profile, args.title_suffix)
    plot(summary, out)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"image": str(out), "summary": str(summary_path),
                      "macro_accuracy": summary["macro_accuracy"],
                      "micro_accuracy": summary["micro_accuracy"],
                      "macro_accuracy_difference": summary["macro_accuracy_difference"],
                      "behaviour": summary["behaviour"]}, indent=2))


if __name__ == "__main__":
    main()
