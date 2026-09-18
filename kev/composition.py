import copy
import hashlib
import json
import random
from datetime import date, timedelta

UNKNOWN = None
SHAPES = {
    "atom": 0,
    "negation": ("not", 0),
    "conjunction": ("and", 0, 1),
    "disjunction": ("or", 0, 1),
    "exception": ("unless", 0, 1),
    "conditional": ("if", 0, 1, 2),
    "nested_and": ("and", ("and", 0, 1), 2),
    "nested_or": ("or", 0, ("or", 1, 2)),
    "held_and_or": ("and", ("or", 0, 1), 2),
    "held_or_not": ("or", ("and", 0, 1), ("not", 2)),
    "held_conditional": ("if", 0, ("not", 1), 2),
    "final_combination": ("and", ("if", 0, 1, 2), 3),
    "final_negation": ("not", ("or", ("and", 0, 1), 2)),
    "final_exception": ("or", ("unless", 0, 1), 2),
}
TRAIN_SHAPES = tuple(list(SHAPES)[:8])
DEV_SHAPES = tuple(list(SHAPES)[8:11])
TEST_SHAPES = tuple(list(SHAPES)[11:])
KINDS = ("lt", "le", "gt", "ge", "eq", "range", "match", "elapsed", "flag")


def atom_value(atom, facts):
    if any(key not in facts for key in atom["fields"]):
        return UNKNOWN
    values = [facts[k] for k in atom["fields"]]
    kind, threshold = atom["kind"], atom["threshold"]
    x = values[0]
    if kind == "lt": return x < threshold
    if kind == "le": return x <= threshold
    if kind == "gt": return x > threshold
    if kind == "ge": return x >= threshold
    if kind == "eq": return x == threshold
    if kind == "range": return threshold <= x <= threshold + 10
    if kind == "match": return x == values[1]
    if kind == "elapsed": return (date.fromisoformat(values[1]) - date.fromisoformat(x)).days <= threshold
    if kind == "flag": return x
    raise ValueError(f"unknown atom {kind}")


def evaluate_rule(tree, atoms, facts):
    if isinstance(tree, int):
        return atom_value(atoms[tree], facts)
    op, *children = tree
    values = [evaluate_rule(t, atoms, facts) for t in children]
    if op == "not": return None if values[0] is None else not values[0]
    if op == "unless":
        a, exception = values
        values = [a, None if exception is None else not exception]
        op = "and"
    if op == "and":
        return False if False in values else None if None in values else True
    if op == "or":
        return True if True in values else None if None in values else False
    if op == "if":
        condition, yes, no = values
        return yes if condition is True else no if condition is False else yes if yes == no else None
    raise ValueError(f"unknown operation {op}")


def atom_text(atom):
    kind, fields, t = atom["kind"], atom["fields"], atom["threshold"]
    key = fields[0]
    words = {"lt": "less than", "le": "at most", "gt": "greater than", "ge": "at least", "eq": "equal to"}
    if kind in words: return f"{key} is {words[kind]} {t}"
    if kind == "range": return f"{key} is between {t} and {t + 10}, including both endpoints"
    if kind == "match": return f"{fields[0]} is the same person as {fields[1]}"
    if kind == "elapsed": return f"the elapsed days from {fields[0]} to {fields[1]} are at most {t}"
    return f"{key} is yes"


def render_rule(tree, atoms, style):
    if isinstance(tree, int): return atom_text(atoms[tree])
    op, *children = tree
    parts = [render_rule(c, atoms, style) for c in children]
    if op == "not": return f"NOT ({parts[0]})" if style == 0 else f"it is not the case that ({parts[0]})"
    if op in ("and", "or"):
        if style == 0: return f"({parts[0]}) {op.upper()} ({parts[1]})"
        connector = "both" if op == "and" else "at least one of"
        return f"{connector} these conditions hold: [({parts[0]}); ({parts[1]})]"
    if op == "unless": return f"({parts[0]}) holds and the exception ({parts[1]}) does not hold"
    return f"if ({parts[0]}), use ({parts[1]}); otherwise use ({parts[2]})"


def leaf_indices(tree):
    if isinstance(tree, int): return {tree}
    return set().union(*(leaf_indices(t) for t in tree[1:]))


def make_atoms(tree, rng):
    nouns = rng.sample(["request", "account", "package", "review", "member", "shipment", "entry", "case"], 4)
    atoms = []
    for i in range(max(leaf_indices(tree)) + 1):
        kind = rng.choice(KINDS)
        prefix = nouns[i]
        fields = [f"{prefix} value"]
        if kind == "match": fields = [f"{prefix} signer", f"{prefix} designated approver"]
        elif kind == "elapsed": fields = [f"{prefix} start date", f"{prefix} end date"]
        elif kind == "flag": fields = [f"{prefix} verified"]
        atoms.append({"kind": kind, "fields": fields, "threshold": rng.randint(5, 60)})
    return atoms


def fact_domains(atoms, rng):
    domains = {}
    for a in atoms:
        kind, fs, t = a["kind"], a["fields"], a["threshold"]
        if kind == "match":
            people = rng.sample(["Mira", "Noah", "Aiko", "Ravi", "Sana", "Elin", "Tomas", "Kofi"], 3)
            domains.update({k: people for k in fs})
        elif kind == "elapsed":
            day = date(2027, rng.randint(1, 8), rng.randint(1, 28))
            domains[fs[0]] = [day.isoformat()]
            domains[fs[1]] = [(day + timedelta(days=n)).isoformat() for n in (max(0, t - 1), t, t + 1, t + 10)]
        elif kind == "flag": domains[fs[0]] = [False, True]
        else: domains[fs[0]] = [t - 1, t, t + 1, t + 10, t + 11]
    domains["routing reference"] = [rng.randint(100, 500), rng.randint(501, 999)]
    return domains


def rendered_facts(facts, order):
    def value(v):
        return "yes" if v is True else "no" if v is False else str(v)
    return [f"The {k} is {value(facts[k])}." for k in order]


def generate(groups_per_shape, seed, shapes=TRAIN_SHAPES, styles=(0, 1), source="compositional"):
    records = []
    for shape in shapes:
        tree = SHAPES[shape]
        rng = random.Random(f"{seed}:{shape}")
        for i in range(groups_per_shape):
            atoms = make_atoms(tree, rng)
            domains = fact_domains(atoms, rng)
            for attempt in range(1000):
                facts = {k: rng.choice(v) for k, v in domains.items()}
                label = evaluate_rule(tree, atoms, facts)
                changed = None
                keys = list(domains); rng.shuffle(keys)
                for key in keys:
                    for value in domains[key]:
                        edited = {**facts, key: value}
                        if evaluate_rule(tree, atoms, edited) != label:
                            missing = {k: v for k, v in facts.items() if k != key}
                            if evaluate_rule(tree, atoms, missing) is None:
                                changed = (edited, key)
                                break
                    if changed: break
                if changed: break
            if changed is None:
                raise ValueError(f"cannot create a decisive edit for {shape}")
            edited, deciding = changed
            nuisance = {**facts, "routing reference": next(v for v in domains["routing reference"] if v != facts["routing reference"])}
            order = list(facts); rng.shuffle(order)
            style = rng.choice(styles)
            rule = render_rule(tree, atoms, style)
            policy = (f"Approve exactly when {rule}. Otherwise deny. The routing reference does not affect eligibility." if style != 2
                      else f"Approval requires the following rule to be true: {rule}. A false rule means denial. Routing references are irrelevant.")
            keys = ["accept", "reject"]; rng.shuffle(keys)
            criteria = {k: "The policy permits this case" if k == "accept" else "The policy does not permit this case" for k in keys}
            group = f"composition/{seed}/{shape}/{i}"
            for kind, a, b in (("relevant", facts, edited), ("irrelevant", facts, nuisance)):
                for sibling, values in (("a", a), ("b", b)):
                    result = evaluate_rule(tree, atoms, values)
                    state = {"policy": policy, "case": " ".join(rendered_facts(values, order))}
                    identifier = f"{group}/{kind}/{sibling}"
                    records.append({"state": state, "questions": {"decision": {"type": "choice",
                        "instructions": "Apply the policy to this case.", "criteria": dict(criteria),
                        "label": "accept" if result else "reject", "src": f"composition_{shape}"}},
                        "_meta": {"id": identifier, "group_id": group, "source": source, "variant": "clean",
                                  "pair_id": f"{group}/{kind}", "sibling": sibling, "pair_kind": kind,
                                  "family": shape, "family_id": shape, "render_style": style,
                                  "text_sha256": hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
                                  "certificate": {"tree": tree, "atoms": atoms, "facts": values, "order": order,
                                                  "deciding_field": deciding, "label": result}}})
    return records


def check_group(records):
    if len(records) != 4: raise ValueError("a composition group requires four records")
    for r in records:
        c = r["_meta"]["certificate"]
        if evaluate_rule(c["tree"], c["atoms"], c["facts"]) != c["label"]:
            raise ValueError("incorrect certificate label")
        if r["state"]["case"] != " ".join(rendered_facts(c["facts"], c["order"])):
            raise ValueError("rendered facts differ from certificate")
        q = r["questions"]["decision"]
        if q["label"] != ("accept" if c["label"] else "reject"):
            raise ValueError("answer differs from certificate")
    for a, b in (records[:2], records[2:]):
        ca, cb = a["_meta"]["certificate"], b["_meta"]["certificate"]
        if ca["order"] != cb["order"] or a["state"]["policy"] != b["state"]["policy"]:
            raise ValueError("pair changes more than one fact")
        if sum(ca["facts"][k] != cb["facts"][k] for k in ca["facts"]) != 1:
            raise ValueError("pair must change exactly one fact")
        expected_flip = a["_meta"]["pair_kind"] == "relevant"
        if (ca["label"] != cb["label"]) != expected_flip:
            raise ValueError("incorrect intervention label")
        missing = {k: v for k, v in ca["facts"].items() if k != ca["deciding_field"]}
        if expected_flip and evaluate_rule(ca["tree"], ca["atoms"], missing) is not None:
            raise ValueError("deciding evidence ablation did not remove the answer")
    return True
