"""Tweet-sized benchmark card: docs/kev-benchmark.png (1600x900, 16:9).

    uv run python scripts/plot_tweet.py

Reads the same saved result files as scripts/plot_family.py; nothing is typed in by hand except labels.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = [("kev-0.6b\nQwen3-0.6B", "runs/v4-06b-hardened/00-trial-0/result.json", "runs/v4-06b-hardened/00-trial-0/result.json", "#AFBBC1"),
          ("kev-4b\nQwen3-4B", "runs/v7-rc3/01-trial-1/result.json", "runs/v7-rc3/01-trial-1/result.json", "#6C8E9B"),
          ("kev-8b\nQwen3-8B", "runs/v7-final/00-trial-0/result.json", "runs/v7-final/00-trial-0/result.json", "#355C6B")]
JEV_T, JEV_D = "runs/jev-transfer-v4/report.json", "runs/jev-decision-v4/report.json"
BASES = [("Qwen3-8B base\nuntrained", "runs/probes/qwen3-8b-base-transfer-v4/report.json"), ("Qwen3-30B-A3B base\nuntrained", "runs/probes/qwen3-30b-a3b-base-transfer-v4/report.json")]


def ood(path):
    r = json.loads((ROOT / path).read_text())
    if "transfer" in r: r = r["transfer"]
    return r["clean"]["acc"], r["clean"]["brier"]


def indist(path):
    r = json.loads((ROOT / path).read_text()); return r["clean"]["acc"]


def main():
    ink, muted, rule = "#202B30", "#5D6970", "#DCE1E4"
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": ink, "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": ink,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(16, 9))
    fig.text(.05, .93, "kev", fontsize=40, weight="bold")
    fig.text(.05, .875, "Open decision models: typed questions in, calibrated probabilities out, one forward pass. No text generation.", fontsize=15, color=muted)

    # ---- left: out-of-domain accuracy, the honest number
    ax = fig.add_axes([.19, .21, .42, .56])
    names, vals, colors = [], [], []
    for label, path, _, color in MODELS:
        acc, _ = ood(path); names.append(label); vals.append(100 * acc); colors.append(color)
    for label, path in BASES:
        acc, _ = ood(path); names.append(label); vals.append(100 * acc); colors.append("#E9E2D0")
    jev_acc, jev_brier = ood(JEV_T)
    order = np.argsort(vals)
    names, vals, colors = [names[i] for i in order], [vals[i] for i in order], [colors[i] for i in order]
    names.append("Jev\nTypeSafe, hosted"); vals.append(100 * jev_acc); colors.append("#E0B04A")
    y = np.arange(len(names))
    bars = ax.barh(y, vals, color=colors, height=.66, zorder=3)
    for b, v, n in zip(bars, vals, names):
        ax.text(v + .8, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=15, weight="bold" if n.startswith("kev-") else "normal", color=ink)
    ax.set_yticks(y, names, fontsize=12.5); ax.set_xlim(40, 95); ax.set_xticks(range(40, 91, 10), [f"{v}%" for v in range(40, 91, 10)])
    ax.tick_params(axis="both", length=0, pad=10, labelsize=12); ax.grid(axis="x", color=rule, lw=.8, zorder=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title("Accuracy on data kev never trained on", loc="left", fontsize=18, weight="bold", pad=14)
    fig.text(.05, .10, "764 frozen out-of-domain records (QNLI, SciQ, PAWS, MMLU, Emotion, TweetEval + held-out policy rules), same items for every model.\n"
             "Untrained bases read zero-shot from next-token letter logits. Development partition; each model card records one locked-test read.", fontsize=10.5, color=muted, linespacing=1.5)

    # ---- right: three headline numbers
    x = .70
    fig.text(x, .74, "kev-8b out of domain", fontsize=13, color=muted)
    fig.text(x, .655, f"{100 * ood(MODELS[2][1])[0]:.1f}%", fontsize=44, weight="bold", color="#355C6B")
    fig.text(x, .615, f"Jev {100 * jev_acc:.1f}%  ·  untrained Qwen3-30B-A3B {100 * ood(BASES[1][1])[0]:.1f}%", fontsize=11.5, color=muted)
    fig.text(x, .53, "kev-8b in distribution (locked test)", fontsize=13, color=muted)
    locked = json.loads((ROOT / "runs/locked/kev-8b-v7-preview-ungated/summary.json").read_text())["suites"]
    fig.text(x, .445, f"{100 * locked['decision']['clean']['acc']:.1f}%", fontsize=44, weight="bold", color="#355C6B")
    fig.text(x, .405, f"Jev {100 * indist(JEV_D):.1f}% on the same items (development partition)", fontsize=11.5, color=muted)
    fig.text(x, .32, "trained parameters (kev-4b / kev-8b)", fontsize=13, color=muted)
    fig.text(x, .245, "34M / 46M", fontsize=36, weight="bold", color="#355C6B")
    fig.text(x, .185, "LoRA r=16 + pointer head on a frozen Qwen3 base\n40–70 min on one H100 · serves on a 32 GB Mac", fontsize=11.5, color=muted, linespacing=1.5)

    fig.text(.05, .045, "github.com/jaredpalmer/kev  ·  huggingface.co/jaredpalmer  ·  Apache-2.0  ·  every number reproducible from frozen, checksummed suites",
             fontsize=11, color=muted)
    out = ROOT / "docs/kev-benchmark.png"
    fig.savefig(out, dpi=100, metadata={"Title": "kev benchmarks"}); plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
