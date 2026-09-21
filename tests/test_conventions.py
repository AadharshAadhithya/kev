"""Source conventions: facts that have one canonical home must not be re-derived elsewhere.

Each rule is (what it guards, regex, files allowed to match). A failure means a second copy of a rule that already has
a home; call the canonical helper instead (the table in .devin/skills/thermonuclear-code-review/SKILL.md lists them).
Run: uv run python -m pytest tests/test_conventions.py -q
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("kev", "scripts", "space", "tests", "modal_app.py", "modal_probe35.py")

RULES = [
    ("head.pt is read and written through kev.checkpoint (Meta, read_meta, write_meta)",
     r"torch\.(load|save)\([^\n]*head\.pt", {"kev/checkpoint.py"}),
    ("KEV_DTYPE/KEV_MERGE/KEV_ATTN/KEV_LORA_SCALE/KEV_TEMPERATURE are read only by LoadOptions.from_env",
     r"environ(\.get)?\(?\[?\s*\"KEV_(DTYPE|MERGE|ATTN|LORA_SCALE|TEMPERATURE)\"", {"kev/checkpoint.py"}),
    ("option keys come from kev.api.question_keys",
     r"\[\s*\"false\"\s*,\s*\"true\"\s*\]|\[str\(i\) for i in range\(len\(", {"kev/api.py", "tests/test_unit.py"}),   # the unit test pins the contract
    ("the training context is kev.model.MAX_STATE/MAX_BRANCH/MAX_PACKED (kev.suite.CONTEXT in manifests) and kev.model.fits",
     r"(?<![\w.])(>|<=|>=|<)\s*2048\b|\b2048\s*(<|>)|max_(branch|state|packed)\"?\s*[=:]\s*\d{3,}", {"kev/model.py"}),
]


def sources():
    for entry in SCANNED:
        path = ROOT / entry
        yield from (p for p in ([path] if path.is_file() else sorted(path.rglob("*.py"))) if "__pycache__" not in p.parts and p != Path(__file__))


@pytest.mark.parametrize("what,pattern,allowed", RULES, ids=[r[0][:60] for r in RULES])
def test_single_home(what, pattern, allowed):
    regex = re.compile(pattern)
    offenders = []
    for path in sources():
        rel = str(path.relative_to(ROOT))
        if rel in allowed:
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if regex.search(code):
                offenders.append(f"{rel}:{n}: {line.strip()}")
    assert not offenders, f"{what}\n" + "\n".join(offenders)
