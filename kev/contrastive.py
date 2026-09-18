"""Programmatic contrastive pairs with certain labels (PLAN.md step 2).

Each item is a policy plus a case described by a few sentences. Every sentence carries zero or more explicit facts;
the label is a pure function `evaluate(facts)` of the facts present, returning UNDETERMINED when a required fact is
missing. A pair is the same item with exactly one sentence changed so that the label flips.

Two code-level checks run on every pair before it is kept (no model anywhere):
  ablation   : removing any evidence sentence (one that carries a required fact) makes the label UNDETERMINED,
               so the answer provably lives in the state, not in the instruction, the options, or the policy alone;
  invariance : removing any filler sentence leaves the label unchanged.
Evidence status is derived from the evaluator, not declared. Both siblings share `pair_id`/`family_id`, and
splits are assigned by family so siblings never straddle a split.
"""
import hashlib
import json
import random
from datetime import date, timedelta

UNDETERMINED = "UNDETERMINED"

NAMES = ["Mira", "Noah", "Priya", "Tomas", "Aiko", "Lena", "Omar", "Sana", "Jonas", "Ravi", "Elin", "Kofi"]
ROLES = ["account owner", "billing manager", "support agent", "warehouse lead"]
ITEMS = ["a pair of running shoes", "a desk lamp", "a wireless keyboard", "a rain jacket", "a coffee grinder", "a backpack"]
PROGRAMS = ["the volunteer driver program", "the apprenticeship", "the rental agreement", "the night-shift roster"]


def _day(d):
    return d.strftime("%B %-d, %Y")


def _need(facts, *keys):
    return all(k in facts for k in keys)


# A family returns (item_a, item_b). An item is {policy, sentences: [(text, facts)], evaluate, question}.
# The two items must differ in exactly one sentence, and evaluate() must give different labels.

def family_return_window(rng):
    window = rng.choice([14, 30, 45, 60]); item = rng.choice(ITEMS); name = rng.choice(NAMES)
    bought = date(2026, rng.randint(1, 9), rng.randint(1, 28))
    def evaluate(f):
        return (f["request"] - f["purchase"]).days <= window if _need(f, "request", "purchase") else UNDETERMINED
    def build(days):
        request = bought + timedelta(days=days)
        return {"policy": f"Returns are accepted only if the return request is submitted within {window} days of the purchase date.",
                "sentences": [(f"{name} bought {item} on {_day(bought)}.", {"purchase": bought}),
                              (f"The return request was submitted on {_day(request)}.", {"request": request}),
                              (f"The order was paid by card and shipped to {name}'s home address.", {})],
                "evaluate": evaluate, "question": {"type": "noul", "instructions": "Is this return request within the policy window?"}}
    return build(rng.randint(1, window - 1)), build(window + rng.randint(1, 30))


def family_spend_threshold(rng):
    limit = rng.choice([250, 500, 1000, 2500]); name = rng.choice(NAMES); role = rng.choice(ROLES)
    def evaluate(f):
        return ("auto_approved" if f["amount"] <= limit else "director_signoff") if _need(f, "amount") else UNDETERMINED
    def build(amount):
        return {"policy": f"Expense claims of ${limit:,} or less are approved automatically. Claims above ${limit:,} require director sign-off.",
                "sentences": [(f"{name}, the {role}, submitted an expense claim.", {}),
                              (f"The claim total is ${amount:,}.", {"amount": amount}),
                              ("Receipts were attached for every line item.", {})],
                "evaluate": evaluate, "question": {"type": "choice", "instructions": "How is this claim handled under the policy?",
                                                   "criteria": {"auto_approved": "Approved without further review", "director_signoff": "Requires director sign-off", "rejected": "Rejected outright"}}}
    return build(limit - rng.randint(1, limit // 2)), build(limit + rng.randint(1, limit))


def family_authorization(rng):
    approver, other = rng.sample(NAMES, 2); account = rng.randint(10, 99); amount = rng.choice([40, 120, 350, 900])
    def evaluate(f):
        return f["signer"] == f["approver"] if _need(f, "signer", "approver") else UNDETERMINED
    def build(signer):
        return {"policy": "A refund is authorized only when its sole authorization was signed by someone who may authorize refunds for that account.",
                "sentences": [(f"Only {approver} may authorize refunds for account {account}.", {"approver": approver}),
                              (f"The sole authorization for this refund on account {account} was signed by {signer}.", {"signer": signer}),
                              (f"The refund amount is ${amount}.", {})],
                "evaluate": evaluate, "question": {"type": "noul", "instructions": "Is the refund authorized?"}}
    return build(approver), build(other)


def family_age_eligibility(rng):
    minimum = rng.choice([16, 18, 21, 25]); name = rng.choice(NAMES); program = rng.choice(PROGRAMS)
    def evaluate(f):
        return f["age"] >= minimum if _need(f, "age") else UNDETERMINED
    def build(age):
        return {"policy": f"Applicants must be at least {minimum} years old to be eligible for {program}.",
                "sentences": [(f"{name} applied to join {program}.", {}),
                              (f"{name} is {age} years old.", {"age": age}),
                              ("The application form was complete and signed.", {})],
                "evaluate": evaluate, "question": {"type": "noul", "instructions": "Is the applicant eligible?"}}
    return build(minimum + rng.randint(0, 20)), build(minimum - rng.randint(1, 5))


def family_quantity_limit(rng):
    limit = rng.choice([2, 3, 5, 10]); item = rng.choice(ITEMS); name = rng.choice(NAMES)
    def evaluate(f):
        if not _need(f, "qty"): return UNDETERMINED
        return "within_limit" if f["qty"] <= limit else "slightly_over" if f["qty"] <= 2 * limit else "far_over"
    def build(qty):
        return {"policy": f"Customers may order at most {limit} units of any single item per order. Orders up to double the limit are held for review; larger orders are cancelled.",
                "sentences": [(f"{name} placed an order for {item}.", {}),
                              (f"The order quantity is {qty}.", {"qty": qty}),
                              ("Delivery was requested to a residential address.", {})],
                "evaluate": evaluate, "question": {"type": "choice", "instructions": "What happens to this order?",
                                                   "criteria": {"within_limit": "Processed normally", "slightly_over": "Held for review", "far_over": "Cancelled"}}}
    return build(rng.randint(1, limit)), build(rng.choice([rng.randint(limit + 1, 2 * limit), rng.randint(2 * limit + 1, 4 * limit)]))


def family_deadline(rng):
    name = rng.choice(NAMES); due = date(2026, rng.randint(2, 11), rng.randint(1, 28)); grace = rng.choice([3, 7, 14])
    def evaluate(f):
        if not _need(f, "received", "due"): return UNDETERMINED
        late = (f["received"] - f["due"]).days
        return 0 if late <= 0 else 1 if late <= grace else 2
    def build(offset):
        received = due + timedelta(days=offset)
        return {"policy": f"Reports received by the deadline are on time. Reports received within {grace} days after the deadline are late but accepted. Later reports are refused.",
                "sentences": [(f"The filing deadline for {name}'s report was {_day(due)}.", {"due": due}),
                              (f"The report was received on {_day(received)}.", {"received": received}),
                              ("The report was submitted through the online portal.", {})],
                "evaluate": evaluate, "question": {"type": "score", "instructions": "How late is this report?", "criteria": ["On time", "Late but accepted", "Refused"]}}
    return build(-rng.randint(0, 10)), build(rng.choice([rng.randint(1, grace), grace + rng.randint(1, 20)]))


FAMILIES = {"return_window": family_return_window, "spend_threshold": family_spend_threshold, "authorization": family_authorization,
            "age_eligibility": family_age_eligibility, "quantity_limit": family_quantity_limit, "deadline": family_deadline}


def label_of(item, drop=None):
    facts = {}
    for i, (_, f) in enumerate(item["sentences"]):
        if i != drop: facts.update(f)
    return item["evaluate"](facts)


def check_pair(a, b):
    """None if the pair is valid, else the reason it is rejected."""
    la, lb = label_of(a), label_of(b)
    if UNDETERMINED in (la, lb): return "label_undetermined"
    if la == lb: return "labels_equal"
    if sum(x[0] != y[0] for x, y in zip(a["sentences"], b["sentences"])) != 1 or len(a["sentences"]) != len(b["sentences"]):
        return "not_exactly_one_sentence_differs"
    for item, label in ((a, la), (b, lb)):
        evidence = 0
        for i, (_, facts) in enumerate(item["sentences"]):
            got = label_of(item, drop=i)
            if facts and got != UNDETERMINED: return "ablation_failed"      # evidence removed must make it undeterminable
            if not facts and got != label: return "invariance_failed"          # filler removed must not change the label
            evidence += bool(facts)
        if evidence < 1: return "no_evidence_sentence"
    return None


def to_request(item, family, pair_id, sibling, rng):
    order = list(range(len(item["sentences"]))); rng.shuffle(order)
    sentences = [item["sentences"][i][0] for i in order]
    q = {**item["question"], "label": label_of(item), "src": f"contrastive_{family}"}
    return {"state": {"policy": item["policy"], "case": " ".join(sentences)}, "questions": {"decision": q},
            "_meta": {"source": "contrastive", "family": family, "family_id": f"{family}/{pair_id}", "pair_id": pair_id, "sibling": sibling,
                      "repo": None, "revision": None, "split": "generated", "row": pair_id, "id": f"contrastive/{family}/{pair_id}/{sibling}",
                      "text_sha256": hashlib.sha256(" ".join(sentences).casefold().encode()).hexdigest(),
                      "row_sha256": hashlib.sha256(json.dumps([t for t, _ in item["sentences"]]).encode()).hexdigest()}}


def generate(n_pairs_per_family, seed, families=None):
    """Sibling-adjacent records plus a per-family report of rejections."""
    records, report = [], {}
    for family in (families or FAMILIES):
        rng = random.Random(f"{seed}:{family}")
        kept, rejected, reasons, attempts = 0, 0, {}, 0
        while kept < n_pairs_per_family and attempts < 50 * n_pairs_per_family:
            attempts += 1
            a, b = FAMILIES[family](rng)
            why = check_pair(a, b)
            if why:
                rejected += 1; reasons[why] = reasons.get(why, 0) + 1; continue
            pair_id = f"{seed}-{family}-{kept:04d}"
            order_seed = rng.getrandbits(64)
            records += [to_request(a, family, pair_id, "a", random.Random(order_seed)),
                        to_request(b, family, pair_id, "b", random.Random(order_seed))]
            kept += 1
        if kept < n_pairs_per_family:
            raise ValueError(f"{family}: only {kept}/{n_pairs_per_family} pairs passed checks ({reasons})")
        report[family] = {"pairs": kept, "rejected": rejected, "reasons": reasons}
    return records, report


def paired_flip(rows):
    """Pair-level metrics from benchmark rows carrying pair_id/sibling. A model that ignores the state cannot flip."""
    by_pair = {}
    for r in rows:
        if r.get("pair_id"):
            key = (r["pair_id"], r.get("question", "decision"))
            pair = by_pair.setdefault(key, {})
            if r["sibling"] in pair:
                raise ValueError("duplicate contrastive sibling")
            pair[r["sibling"]] = r
    if not by_pair:
        return None
    if any(set(p) != {"a", "b"} for p in by_pair.values()):
        raise ValueError("incomplete contrastive pair")
    prediction = lambda r: r["keys"][max(range(len(r["p"])), key=r["p"].__getitem__)]
    truth = lambda r: r["keys"][r["label"]]
    relevant = [p for p in by_pair.values() if truth(p["a"]) != truth(p["b"])]
    invariant = [p for p in by_pair.values() if truth(p["a"]) == truth(p["b"])]
    both = lambda ps: sum(all(prediction(r) == truth(r) for r in p.values()) for p in ps) / len(ps) if ps else None
    result = {"pairs": len(relevant),
              "flip_rate": sum(prediction(p["a"]) != prediction(p["b"]) for p in relevant) / len(relevant) if relevant else None,
              "both_correct_rate": both(relevant)}
    if invariant:
        result.update(invariant_pairs=len(invariant), invariance_rate=sum(prediction(p["a"]) == prediction(p["b"]) for p in invariant) / len(invariant),
                      invariant_both_correct_rate=both(invariant))
    return result
