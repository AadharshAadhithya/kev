"""Diagnostic: how much knowledge does the *base* model show on the frozen MMLU/SciQ transfer items with a plain
letter-token readout, versus what our trained pointer head gets on the same items?

    uv run python scripts/base_mmlu_probe.py --base Qwen/Qwen3-4B-Base --suite evals/v4/transfer-v4 --tasks mmlu,sciq

Prints accuracy per task for the base model (zero-shot, next-token logits over the option letters). Compare with the
per-task numbers in runs/<study>/<trial>/result.json -> transfer.tasks. Read-only; touches no suite files.
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kev.suite import load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--suite", default="evals/v4/transfer-v4")
    ap.add_argument("--tasks", default="mmlu,sciq")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.base)
    dtype = torch.bfloat16 if a.device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=dtype).to(a.device).eval()
    letters = "ABCDEFGH"
    letter_ids = [tok.encode(" " + L, add_special_tokens=False)[0] for L in letters]
    records = [r for r in load_split(a.suite, "development") if r["_meta"]["variant"] == "clean" and r["_meta"]["source"] in a.tasks.split(",")]
    hits, n = {}, {}
    with torch.no_grad():
        for r in records:
            qid, q = next(iter(r["questions"].items()))
            if q["type"] == "noul": q = {**q, "criteria": {"false": "No", "true": "Yes"}, "label": str(bool(q["label"])).lower()}
            elif q["type"] != "choice": continue
            keys = list(q["criteria"])
            state = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"]) if not isinstance(r["state"], dict) else " ".join(f"{k}: {v}" for k, v in r["state"].items())
            prompt = f"{state}\n{q['instructions']}\n" + "\n".join(f"{letters[i]}. {q['criteria'][k]}" for i, k in enumerate(keys)) + "\nAnswer:"
            ids = tok(prompt, return_tensors="pt").to(a.device)
            logits = model(**ids).logits[0, -1]
            pred = int(torch.argmax(logits[letter_ids[: len(keys)]]))
            src = r["_meta"]["source"]
            hits[src] = hits.get(src, 0) + int(keys[pred] == q["label"]); n[src] = n.get(src, 0) + 1
    print(json.dumps({"base": a.base, "readout": "zero-shot next-token letter logits", **{k: {"n": n[k], "acc": round(hits[k] / n[k], 3)} for k in n}}))


if __name__ == "__main__":
    main()
