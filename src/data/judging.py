from __future__ import annotations

import re
from typing import Optional

from .load_datasets import normalize_answer


NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")

def is_correct(pred: Optional[str], gold: str) -> bool:
    """
    GSM8K correctness is usually exact after normalization.
    (You can extend this to handle fractions, units, etc.)
    """
    if pred is None:
        return False
    p = normalize_answer(pred)
    g = normalize_answer(gold)

    # exact string match
    if p == g:
        return True

    # numeric match tolerant to "42.0" vs "42"
    if NUM_RE.match(p) and NUM_RE.match(g):
        try:
            return float(p) == float(g)
        except ValueError:
            return False

    return False
