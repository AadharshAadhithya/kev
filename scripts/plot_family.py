"""README figure: the kev family against Jev on the frozen out-of-domain suite, plus the capacity / learning-rate curve.

    uv run python scripts/plot_family.py            # -> docs/kev-family.png, docs/kev-family-summary.json

Reads saved result.json files only (development partitions; the locked test is not plotted). Every number is checked
against the trial's own report before it is drawn.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
TASKS = [("qnli", "QNLI"), ("sciq", "SciQ"), ("tweet_offensive", "TweetEval · offensive"), ("paws", "PAWS"), ("mmlu", "MMLU · 4-way"),
         ("emotion", "Emotion-6"), ("contrastive_authorization", "Policy · authorization"), ("contrastive_deadline", "Policy · deadline (Score)"),
         ("composition_held_and_or", "Rule · (A or B) and C"), ("composition_held_or_not", "Rule · (A and B) or not C"), ("composition_held_conditional", "Rule · if A then not B else C")]
MODELS = [("kev-0.6b", "runs/v4-06b-hardened/00-trial-0/result.json", "#AFBBC1"),
          ("kev-4b", "runs/v7-rc3/01-trial-1/result.json", "#6C8E9B"),
          ("kev-8b", "runs/v7-final/00-trial-0/result.json", "#355C6B")]
JEV = "runs/jev-transfer-v4/report.json"
# capacity x learning-rate curve: (label, params in B, lr, [transfer acc per seed], source trials)
CURVE = [("0.6B", 0.6, "2e-4", [0.598, 0.595, 0.605], "v4-06b-hardened"),
         ("4B", 4.0, "2e-4", [0.704, 0.735], "v4-4b-baseline"),
         ("4B", 4.0, "5e-5", [0.759, 0.758, 0.759], "lowdrift-4b-v4/01, recipe-4b-v4-s1, recipe-4b-v4-s2"),
         ("4B", 4.0, "5e-5 + v7 data", [0.773, 0.790, 0.770], "v7-rc3, v7-final/02"),
         ("8B", 8.2, "2e-4", [0.765, 0.741], "v3-8b-s0 (transfer-v3 = same bytes)"),
         ("8B", 8.2, "5e-5", [0.774, 0.779, 0.774], "recipe-8b-r1/00, recipe-8b-r2/01, recipe-8b-s2"),
         ("8B", 8.2, "5e-5 + v7 data", [0.796, 0.774], "v7-final/00, v7-final/01")]


def transfer_tasks(path):
    r = json.loads((ROOT / path).read_text())
    t = r["transfer"] if "transfer" in r else r
    return {k: v["acc"] for k, v in t["tasks"].items()}, t["clean"]["acc"], t["clean"]["brier"]


def main():
    ink, muted, rule = "#202B30", "#5D6970", "#DCE1E4"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": ink, "axes.labelcolor": muted,
                         "xtick.color": muted, "ytick.color": ink, "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(16, 10.5))
    fig.text(.04, .955, "OUT OF DOMAIN  /  FROZEN TRANSFER SUITE (evals/v4/transfer-v4, development partition)", fontsize=10, weight="bold", color=muted)
    fig.text(.04, .905, "kev family vs. Jev on sources kev never trained on", fontsize=27, weight="bold")
    fig.text(.04, .872, "Same 764 records for every model · top-1 accuracy · Jev via Vercel AI Gateway (typesafe-ai/jev) · kev checkpoints are the published research previews",
             fontsize=11, color=muted)
    fig.add_artist(Line2D([.04, .965], [.85, .85], transform=fig.transFigure, color=rule, lw=1))

    ax = fig.add_axes([.24, .12, .43, .69])
    data = {name: transfer_tasks(path) for name, path, _ in MODELS}
    jev_tasks, jev_acc, jev_brier = transfer_tasks(JEV)
    y = np.arange(len(TASKS))
    width = .2
    for i, (name, _, color) in enumerate(MODELS):
        vals = [100 * data[name][0][t] for t, _ in TASKS]
        ax.barh(y + (i - 1.5) * width, vals, height=width, color=color, label=name, zorder=3)
    jv = [100 * jev_tasks[t] for t, _ in TASKS]
    ax.barh(y + 1.5 * width, jv, height=width, color="#E0B04A", label="Jev", zorder=3)
    for yi, v in zip(y, jv):
        ax.text(v + 1, yi + 1.5 * width, f"{v:.0f}", va="center", fontsize=8.5, color=ink)
        v8 = 100 * data["kev-8b"][0][TASKS[yi][0]]
        ax.text(v8 + 1, yi + .5 * width, f"{v8:.0f}", va="center", fontsize=8.5, color=ink)
    ax.set_yticks(y, [label for _, label in TASKS], fontsize=10)
    ax.set_ylim(len(TASKS) - .5, -.5); ax.set_xlim(0, 104)
    ax.set_xticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.set_xlabel("Accuracy · higher is better", labelpad=8, fontsize=10)
    ax.grid(axis="x", color=rule, lw=.8, zorder=0); ax.tick_params(axis="both", length=0, pad=8)
    for s in ax.spines.values(): s.set_visible(False)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=4, frameon=False, fontsize=10, handlelength=1.1, columnspacing=1.4)
    fig.text(.24, .835, "Per-source accuracy", fontsize=14, weight="bold")

    # right panel: overall transfer accuracy vs size x lr
    fig.add_artist(Line2D([.70, .70], [.12, .84], transform=fig.transFigure, color=rule, lw=1))
    left = .735
    fig.text(left, .835, "Overall transfer accuracy", fontsize=14, weight="bold")
    fig.text(left, .812, "Clean questions, all sources · one point per seed", fontsize=9, color=muted)
    cx = fig.add_axes([left, .49, .22, .30])
    colors = {"2e-4": "#AFBBC1", "5e-5": "#6C8E9B", "5e-5 + v7 data": "#355C6B"}
    for label, params, lr, accs, _ in CURVE:
        cx.scatter([params] * len(accs), [100 * a for a in accs], color=colors[lr], s=34, zorder=3, label=(f"lr {lr}" if lr != "5e-5 + v7 data" else "lr 5e-5 + random-structure data") if label == "4B" else None)
    for lr in ("2e-4", "5e-5", "5e-5 + v7 data"):
        pts = [(p, 100 * np.mean(a)) for _, p, l, a, _ in CURVE if l == lr]
        cx.plot([p for p, _ in pts], [m for _, m in pts], color=colors[lr], lw=1.4, zorder=2)
    cx.axhline(100 * jev_acc, color="#E0B04A", lw=1.6, zorder=1); cx.text(8.4, 100 * jev_acc + .8, f"Jev {100*jev_acc:.1f}%", ha="right", fontsize=9, color="#8B6A1A")
    cx.set_xscale("log"); cx.set_xticks([0.6, 4, 8.2], ["0.6B", "4B", "8B"]); cx.set_xlim(0.45, 11)
    cx.set_ylim(55, 90); cx.set_yticks(range(55, 91, 5), [f"{v}%" for v in range(55, 91, 5)])
    cx.grid(axis="y", color=rule, lw=.8, zorder=0); cx.tick_params(length=0, pad=6, labelsize=9)
    for s in cx.spines.values(): s.set_visible(False)
    cx.legend(loc="lower right", frameon=False, fontsize=8.5, title="recipe", title_fontsize=9)
    cx.set_xlabel("backbone parameters (Qwen3-Base)", fontsize=9, labelpad=6)

    fig.text(left, .43, "Read-outs", fontsize=13, weight="bold")
    rows = [("Capacity 0.6B → 4B", "+14–19 pp, matched data"), ("Capacity 4B → 8B", "+0.5–2 pp"), ("lr 2e-4 → 5e-5 at 4B", "+4.7 pp  [+0.4, +9.6]"), ("Random rule structures (v7)", "+1.5–3 pp; held-out rules 0.62 → 0.73"),
            ("Brier out of domain", f"kev-8b {data['kev-8b'][2]:.3f} · Jev {jev_brier:.3f}"), ("Held-out rule pairs, both correct", f"8b {json.loads((ROOT / MODELS[2][1]).read_text())['transfer']['paired_flip']['both_correct_rate']:.2f} · Jev {json.loads((ROOT / JEV).read_text())['paired_flip']['both_correct_rate']:.2f} · screen 0.70")]
    for i, (k, v) in enumerate(rows):
        fig.text(left, .405 - .034 * i, k, fontsize=9.5); fig.text(left, .405 - .034 * i - .016, v, fontsize=9, color=muted)
    fig.text(left, .19, "Every point is a development-partition number from a trial\nwith full provenance in runs/leaderboard.md. The locked test\nwas read once per published checkpoint (see model cards) and\nis not plotted here.", fontsize=8.5, color=muted, va="top", linespacing=1.4)
    fig.add_artist(Line2D([.04, .965], [.085, .085], transform=fig.transFigure, color=rule, lw=1))
    fig.text(.04, .055, "Sources never in kev's training: QNLI, SciQ, TweetEval, PAWS, MMLU, Emotion (public), plus programmatic policy pairs whose rule structure was held out. "
             "Exact-match state deduplication only; Jev's exposure to these public sets is unknown.", fontsize=8.8, color=muted)
    fig.text(.04, .03, "Recipes: LoRA r=16 + pointer head; kev-0.6b lr 2e-4 on decision-v4; kev-4b and kev-8b lr 5e-5 on decision-v7 (random rule structures). Regenerate: uv run python scripts/plot_family.py",
             fontsize=8.8, color=muted)
    out = ROOT / "docs/kev-family.png"
    fig.savefig(out, dpi=170, metadata={"Title": "kev family vs Jev, out of domain"}); plt.close(fig)
    (ROOT / "docs/kev-family-summary.json").write_text(json.dumps({"models": {n: {"path": p, "transfer_acc": data[n][1], "transfer_brier": data[n][2], "tasks": data[n][0]} for n, p, _ in MODELS},
                                                                     "jev": {"path": JEV, "transfer_acc": jev_acc, "transfer_brier": jev_brier, "tasks": jev_tasks}, "curve": CURVE}, indent=1))
    print(out, {n: round(data[n][1], 3) for n, _, _ in MODELS}, "jev", round(jev_acc, 3))


if __name__ == "__main__":
    main()
