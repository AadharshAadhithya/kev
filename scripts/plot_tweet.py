"""Benchmark card: docs/kev-benchmark.png (1600x900, 16:9). Style: scripts/chartstyle.py.

    uv run python scripts/plot_tweet.py

Reads saved result files only; nothing is typed in by hand except labels.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from chartstyle import JEV, KEV, NEUTRAL, TEXT, TEXT2, body, display, heading, hbars, rule, stat, use_style

ROOT = Path(__file__).resolve().parents[1]
MODELS = [("kev-8b", "Qwen3-8B", "runs/v7-final/00-trial-0/result.json"),
          ("kev-4b", "Qwen3-4B", "runs/v7-rc3/01-trial-1/result.json"),
          ("kev-0.6b", "Qwen3-0.6B", "runs/v7-06b/02-trial-2/result.json"),
          ("kev-0.5b", "Qwen2.5-0.5B, prototype", "runs/kev-05b-transfer-v4/report.json")]
BASES = [("Qwen3-8B", "untrained base", "runs/probes/qwen3-8b-base-transfer-v4/report.json"),
         ("Qwen3-30B-A3B", "untrained base", "runs/probes/qwen3-30b-a3b-base-transfer-v4/report.json")]
JEV_T, JEV_D = "runs/jev-transfer-v4/report.json", "runs/jev-decision-v4/report.json"


def clean(path):
    r = json.loads((ROOT / path).read_text())
    if "transfer" in r: r = r["transfer"]
    return r["clean"]


def main():
    use_style()
    fig = plt.figure(figsize=(16, 9))
    heading(fig, .05, .905, "Kev: open decision models, measured against Jev", size=30)
    body(fig, .05, .855, "Typed questions in, calibrated probabilities out, one forward pass, no text generation. LoRA + pointer head on Qwen3 bases.", size=14)
    rule(fig, .05, .95, .82)

    rows = [(n, sub, 100 * clean(p)["acc"], KEV[n]) for n, sub, p in MODELS]
    rows += [(n, sub, 100 * clean(p)["acc"], NEUTRAL) for n, sub, p in BASES]
    rows.sort(key=lambda r: -r[2])
    jev = clean(JEV_T)
    rows.insert(0, ("Jev", "TypeSafe, hosted", 100 * jev["acc"], JEV))

    ax = fig.add_axes([.24, .235, .39, .52])
    heading(fig, .05, .77, "Accuracy on data Kev never trained on", size=17)
    hbars(ax, [display(r[0]) for r in rows], [r[2] for r in rows], [r[3] for r in rows], xlim=(0, 100), fmt="{:.1f}%",
          emphasize=[i for i, r in enumerate(rows) if r[0].startswith("kev")], sublabels=[r[1] for r in rows], ticks=range(0, 101, 25), label_size=13, value_size=14)
    body(fig, .05, .115, "764 frozen out-of-domain records: QNLI, SciQ, PAWS, MMLU, Emotion, TweetEval and held-out policy rules; the same items for every row.\n"
         "Untrained bases are read zero-shot from next-token letter logits. Development partition; each model card records one locked-test read.", size=11)

    x = .70
    k8 = clean(MODELS[0][2]); locked = json.loads((ROOT / "runs/locked/kev-8b-v7-preview-ungated/summary.json").read_text())["suites"]
    stat(fig, x, .76, "Kev-8B, out of domain", f"{100 * k8['acc']:.1f}%", f"Jev {100 * jev['acc']:.1f}%  ·  untrained Qwen3-30B-A3B {100 * clean(BASES[1][2])['acc']:.1f}%", color=KEV["kev-8b"])
    stat(fig, x, .565, "Kev-8B, in distribution, locked test", f"{100 * locked['decision']['clean']['acc']:.1f}%",
         f"Jev {100 * json.loads((ROOT / JEV_D).read_text())['clean']['acc']:.1f}% on the same items", color=KEV["kev-8b"])
    k05 = clean(MODELS[3][2])
    stat(fig, x, .37, "Since the Kev-0.5B prototype, out of domain", f"+{100 * (k8['acc'] - k05['acc']):.0f} pp", f"{100 * k05['acc']:.1f}% → {100 * k8['acc']:.1f}% with 46M trained parameters\n(LoRA r=16 + pointer head on a frozen base)", color=KEV["kev-8b"], value_size=34)

    body(fig, .05, .05, "github.com/jaredpalmer/kev  ·  huggingface.co/jaredpalmer  ·  Apache-2.0  ·  every number reproducible from frozen, checksummed suites", size=11)
    out = ROOT / "docs/kev-benchmark.png"
    fig.savefig(out, dpi=100, metadata={"Title": "Kev benchmarks"}); plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
