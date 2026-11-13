"""Read patient-ID lists from gitignored *.local.txt files next to this script.

misaligned_cases.local.txt: cases whose doctor label is offset in z (skipped / copied as-is).
One case ID per line; '#' starts a comment. Missing file -> empty list.
"""
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def load_cases(name="misaligned_cases.local.txt"):
    p = _HERE / name
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out
