"""README figure: the Kev family against Jev, per out-of-domain source, plus the capacity/recipe curve.

    uv run python scripts/plot_family.py            # -> docs/kev-family.png, docs/kev-family-summary.json

Reads saved result.json files only (development partitions; the locked test is not plotted). Style: scripts/chartstyle.py.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from chartstyle import GRID, HOLLOW, JEV, KEV, RULE, TEXT, TEXT2, body, display, heading, rule, strip, use_style

ROOT = Path(__file__).resolve().parents[1]
TASKS = [("sciq", "SciQ"), ("qnli", "QNLI"), ("contrastive_authorization", "Policy: authorization"), ("composition_held_and_or", "Rule: (A or B) and C"),
         ("composition_held_or_not", "Rule: (A and B) or not C"), ("tweet_offensive", "TweetEval offensive"), ("paws", "PAWS"),
         ("composition_held_conditional", "Rule: if A then not B else C"), ("mmlu", "MMLU, 4-way"), ("contrastive_deadline", "Policy: deadline (3-level Score)"), ("emotion", "Emotion, 6-way")]
MODELS = [("kev-0.5b", "runs/kev-05b-transfer-v4/report.json"), ("kev-0.6b", "runs/v7-06b/02-trial-2/result.json"), ("kev-4b", "runs/v7-rc3/01-trial-1/result.json"), ("kev-8b", "runs/v7-final/00-trial-0/result.json")]
JEV_PATH = "runs/jev-transfer-v4/report.json"
# capacity x recipe curve: (params in B, recipe, [transfer acc per seed], source trials)
CURVE = [(0.6, "default recipe", [0.598, 0.595, 0.605], "v4-06b-hardened"), (0.6, "low lr + random rule structures", [0.613, 0.605, 0.620], "v7-06b"),
         (4.0, "default recipe", [0.704, 0.735], "v4-4b-baseline"), (4.0, "low lr", [0.759, 0.758, 0.759], "lowdrift-4b-v4/01, recipe-4b-v4-s1, recipe-4b-v4-s2"),
         (4.0, "low lr + random rule structures", [0.773, 0.790, 0.770], "v7-rc3, v7-final/02"),
         (8.2, "default recipe", [0.765, 0.741], "v3-8b-s0 (transfer-v3 = same bytes)"), (8.2, "low lr", [0.774, 0.779, 0.774], "recipe-8b-r1/00, recipe-8b-r2/01, recipe-8b-s2"),
         (8.2, "low lr + random rule structures", [0.796, 0.774], "v7-final/00, v7-final/01")]
RECIPE_COLOR = {"default recipe": GRID, "low lr": KEV["kev-0.6b"], "low lr + random rule structures": KEV["kev-8b"]}
RECIPE_SHORT = {"default recipe": "default recipe", "low lr": "low lr", "low lr + random rule structures": "low lr +\nrandom rules"}


def transfer(path):
    r = json.loads((ROOT / path).read_text())
    t = r["transfer"] if "transfer" in r else r
    return {k: v["acc"] for k, v in t["tasks"].items()}, t["clean"]["acc"], t["clean"]["brier"], t["paired_flip"]["both_correct_rate"]


def main():
    use_style()
    data = {n: transfer(p) for n, p in MODELS}
    jev_tasks, jev_acc, jev_brier, jev_pairs = transfer(JEV_PATH)
    fig = plt.figure(figsize=(16, 11))
    heading(fig, .05, .93, "Where Kev matches Jev and where it does not, on data Kev never trained on", size=24)
    body(fig, .05, .89, "Accuracy per source on the frozen out-of-domain suite (transfer-v4, 764 records, development partition). Same items for every model. Jev via Vercel AI Gateway.", size=12.5)
    rule(fig, .05, .96, .865)

    # ---- left: dot plot per source, one row per source, one shared plot lane
    ax = fig.add_axes([.27, .175, .40, .645])
    heading(fig, .05, .835, "Accuracy by source", size=15)
    y = np.arange(len(TASKS))
    for yi, (key, _) in zip(y, TASKS):
        vals = {n: 100 * data[n][0][key] for n, _ in MODELS}; vals["Jev"] = 100 * jev_tasks[key]
        lo, hi = min(vals.values()), max(vals.values())
        ax.plot([lo, hi], [yi, yi], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
        for n, v in vals.items():
            if n in HOLLOW: ax.scatter([v], [yi], s=95, facecolor="white", edgecolor=KEV["kev-4b"], linewidth=1.8, zorder=3)
            else: ax.scatter([v], [yi], s=95, color=JEV if n == "Jev" else KEV[n], zorder=3, edgecolor="white", linewidth=1.2)
        # direct labels: the best Kev and Jev, offset vertically so they never overlap
        best = max((n for n, _ in MODELS if n != "kev-0.5b"), key=lambda n: vals[n])
        ax.annotate(f"{vals[best]:.0f}", (vals[best], yi), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9.5, color=KEV[best], weight="medium")
        ax.annotate(f"{vals['Jev']:.0f}", (vals["Jev"], yi), xytext=(0, -15), textcoords="offset points", ha="center", fontsize=9.5, color=JEV, weight="medium")
    ax.set_yticks(y); ax.set_yticklabels([label for _, label in TASKS], fontsize=12); ax.set_ylim(len(TASKS) - .5, -.5)
    ax.set_xlim(20, 104); ax.set_xticks(range(20, 101, 20)); ax.set_xticklabels([f"{v}%" for v in range(20, 101, 20)], fontsize=10.5, color=TEXT2)
    ax.grid(axis="x", color=GRID, lw=.8, zorder=0); ax.tick_params(length=0); strip(ax)
    # key as direct text, once
    kx = .27
    for i, (n, c) in enumerate([("Jev", JEV), ("kev-8b", KEV["kev-8b"]), ("kev-4b", KEV["kev-4b"]), ("kev-0.6b", KEV["kev-0.6b"]), ("kev-0.5b prototype", KEV["kev-4b"])]):
        fig.text(kx + .082 * i, .835, "○" if "0.5b" in n else "●", fontsize=13, color=c, ha="left", va="baseline", weight="bold" if "0.5b" in n else "regular")
        fig.text(kx + .082 * i + .014, .835, display(n), fontsize=11.5, color=TEXT, ha="left", va="baseline")
    body(fig, .05, .125, "Notice: on classification-shaped sources and trained rule families the 4B and 8B are within a few points of Jev or above it.\n"
         "The gap is concentrated in knowledge (MMLU), day-precision date arithmetic (deadline) and noisy-label emotion.\n"
         "Sources: QNLI, SciQ, TweetEval, PAWS, MMLU, Emotion; the policy families and rule structures shown were never trained.", size=11, va="top")

    # ---- right: capacity x recipe
    from matplotlib.lines import Line2D
    fig.add_artist(Line2D([.72, .72], [.06, .84], transform=fig.transFigure, color=RULE, lw=0.8))
    x0 = .755
    heading(fig, x0, .835, "What moved the overall number", size=15)
    cx = fig.add_axes([x0, .50, .145, .29])
    for recipe, color in RECIPE_COLOR.items():
        pts = [(p, [100 * a for a in accs]) for p, r, accs, _ in CURVE if r == recipe]
        for p, vals in pts: cx.scatter([p] * len(vals), vals, color=color, s=40, zorder=3, edgecolor="white", linewidth=1)
        cx.plot([p for p, _ in pts], [np.mean(v) for _, v in pts], color=color, lw=1.6, zorder=2)
        cx.annotate(RECIPE_SHORT[recipe], (pts[-1][0], np.mean(pts[-1][1])), xytext=(9, {"default recipe": -9, "low lr": -2, "low lr + random rule structures": 8}[recipe]), textcoords="offset points", fontsize=9.5, color=color if color != GRID else TEXT2, va="center", weight="medium" if color != GRID else "regular", linespacing=1.2)
    cx.axhline(100 * jev_acc, color=JEV, lw=1.6, zorder=1); cx.annotate(f"Jev {100 * jev_acc:.1f}%", (8.2, 100 * jev_acc), xytext=(9, 0), textcoords="offset points", fontsize=9.5, color=JEV, va="center", weight="medium")
    cx.set_xscale("log"); cx.set_xticks([0.6, 4, 8.2]); cx.set_xticklabels(["0.6B", "4B", "8B"], fontsize=10.5, color=TEXT2); cx.set_xlim(0.45, 11)
    cx.set_ylim(55, 90); cx.set_yticks(range(60, 91, 10)); cx.set_yticklabels([f"{v}%" for v in range(60, 91, 10)], fontsize=10.5, color=TEXT2)
    cx.grid(axis="y", color=GRID, lw=.8, zorder=0); cx.tick_params(length=0); strip(cx); cx.minorticks_off()
    body(fig, x0, .455, "Backbone size (Qwen3-Base); one point per seed", size=10.5)

    rows = [("Capacity, 0.6B to 4B", "+14 to +19 pp, matched data"), ("Capacity, 4B to 8B", "+0.5 to +2 pp"),
            ("Learning rate 2e-4 to 5e-5, at 4B", "+4.7 pp, 95% CI [+0.4, +9.6]"), ("60 random rule structures instead of 8 shapes", "+1.5 to +3 pp; held-out rules 0.62 to 0.73"),
            ("Brier out of domain", f"Kev-8B {data['kev-8b'][2]:.3f}, Jev {jev_brier:.3f}"), ("Held-out rule pairs, both siblings correct", f"Kev-8B {data['kev-8b'][3]:.2f}, Jev {jev_pairs:.2f}")]
    for i, (k, v) in enumerate(rows):
        fig.text(x0, .405 - .045 * i, k, fontsize=11, color=TEXT, va="baseline"); fig.text(x0, .405 - .045 * i - .018, v, fontsize=10.5, color=TEXT2, va="baseline")
    body(fig, x0, .115, "Development-partition numbers; provenance in\nruns/leaderboard.md. The locked test was read once\nper checkpoint and is not plotted.", size=10, va="top")
    body(fig, .05, .018, "LoRA r=16 + pointer head; Kev-0.6B/4B/8B on decision-v7 (Kev-0.6B at lr 1e-4, the others at 5e-5); Kev-0.5B is the Qwen2.5 prototype, scored on the same items.  Regenerate: uv run python scripts/plot_family.py", size=10)

    out = ROOT / "docs/kev-family.png"
    fig.savefig(out, dpi=170, metadata={"Title": "Kev family vs Jev, out of domain"}); plt.close(fig)
    (ROOT / "docs/kev-family-summary.json").write_text(json.dumps({"models": {n: {"path": p, "transfer_acc": data[n][1], "transfer_brier": data[n][2], "tasks": data[n][0]} for n, p in MODELS},
                                                                     "jev": {"path": JEV_PATH, "transfer_acc": jev_acc, "transfer_brier": jev_brier, "tasks": jev_tasks}, "curve": CURVE}, indent=1))
    print(out, {n: round(data[n][1], 3) for n, _ in MODELS}, "jev", round(jev_acc, 3))


if __name__ == "__main__":
    main()
