import argparse, json, os, random, time
from collections import Counter
import torch
import torch.nn.functional as F
from .data import build, augment, materialize
from .model import DecisionModel, load_tokenizer, encode


def permuted_copy(rec, rng):
    """Re-shuffle options of every Choice question with K>=3; return (record, perms) with perms[q] = new->old index or None."""
    out, perms = {"state": rec["state"], "questions": []}, []
    for q in rec["questions"]:
        if q["qtype"] == "choice" and len(q["options"]) >= 3:
            perm = list(range(len(q["options"]))); rng.shuffle(perm)
            out["questions"].append({**q, "options": [q["options"][j] for j in perm], "label": perm.index(q["label"])}); perms.append(perm)
        else:
            out["questions"].append(q); perms.append(None)
    return out, perms


def question_loss(z, q, dev, ord_w):
    """CE for all types; Score adds an ordinal term |E[level] - y| / (L-1) so near-misses cost less than far ones."""
    y = torch.tensor([q["label"]], device=dev)
    loss = F.cross_entropy(z[None], y)
    if q["qtype"] == "score" and ord_w > 0:
        p = F.softmax(z, -1); levels = torch.arange(len(p), device=dev, dtype=p.dtype)
        loss = loss + ord_w * ((p * levels).sum() - q["label"]).abs() / (len(p) - 1)
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n_per_source", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--holdout", default="", help="comma-separated sources excluded from training (evaluated as out-of-source)")
    ap.add_argument("--perm_kl", type=float, default=0.5, help="weight of symmetric KL between predictions under two option orders")
    ap.add_argument("--perm_frac", type=float, default=0.3, help="fraction of records that get the second permuted forward pass")
    ap.add_argument("--ord_w", type=float, default=0.5, help="weight of ordinal |E[level]-y| term for Score questions")
    ap.add_argument("--out", default="runs/kev")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = load_tokenizer(a.base)
    model = DecisionModel(a.base, tok, dev, lora=a.lora)
    print(f"device={dev} trainable params={sum(p.numel() for p in model.trainable_parameters())/1e6:.1f}M")

    holdout = [s for s in a.holdout.split(",") if s]
    reqs = build(a.n_per_source, "train", a.seed, exclude=holdout)
    print(f"{len(reqs)} training requests (holdout={holdout}), questions by type "
          f"{dict(Counter(q['qtype'] for r in reqs for q in materialize(r)['questions']))}")

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(reqs) // a.accum
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1), pct_start=0.1)
    model.train(); t0 = time.time(); run = Counter(); step = 0; seen = 0
    for ep in range(a.epochs):
        rng.shuffle(reqs)
        for i, req in enumerate(reqs):
            rec = materialize(augment(req, rng))  # fresh permutation / distractors each epoch
            try: enc = encode(tok, rec)
            except ValueError: continue
            logits = model(enc)
            loss = sum(question_loss(z, q, dev, a.ord_w) for z, q in zip(logits, rec["questions"])) / len(logits)
            run["ce"] += loss.item()
            if a.perm_kl > 0 and rng.random() < a.perm_frac and any(q["qtype"] == "choice" and len(q["options"]) >= 3 for q in rec["questions"]):
                rec2, perms = permuted_copy(rec, rng)
                logits2 = model(encode(tok, rec2)); kl, n = 0.0, 0
                for z1, z2, perm in zip(logits, logits2, perms):
                    if perm is None: continue
                    lp1 = F.log_softmax(z1, -1); lp2 = F.log_softmax(z2, -1)[torch.tensor([perm.index(j) for j in range(len(perm))], device=dev)]
                    kl = kl + 0.5 * (F.kl_div(lp2, lp1, log_target=True, reduction="sum") + F.kl_div(lp1, lp2, log_target=True, reduction="sum")); n += 1
                kl = kl / n; loss = loss + a.perm_kl * kl; run["kl"] += kl.item(); run["kl_n"] += 1
            (loss / a.accum).backward(); run["n"] += 1; seen += 1
            if (i + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if dev == "mps": torch.mps.empty_cache()
                if step % 10 == 0:
                    print(f"ep{ep} step {step}/{steps} loss {run['ce']/run['n']:.3f} kl {run['kl']/max(run['kl_n'],1):.3f} {(time.time()-t0)/seen:.2f}s/rec", flush=True)
                    run = Counter()
    os.makedirs(a.out, exist_ok=True)
    model.lm.save_pretrained(a.out)
    torch.save({"head": model.head.state_dict(), "base": a.base, "lora": a.lora, "holdout": holdout, "args": vars(a)}, f"{a.out}/head.pt")
    tok.save_pretrained(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
