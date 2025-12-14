from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from datasets import load_dataset


# GSM8K answers are often like: "... #### 42"
GSM8K_FINAL_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)

# Very lightweight normalization:
# - remove commas in numbers
# - strip whitespace
def normalize_answer(ans: str) -> str:
    ans = ans.strip()
    ans = ans.replace(",", "")
    # common trailing punctuation
    ans = ans.rstrip(".")
    return ans


@dataclass
class Example:
    uid: str
    problem: str
    gold: str
    meta: Dict[str, Any]


def load_gsm8k(split: str = "train") -> List[Example]:
    """
    Loads GSM8K from HF: dataset 'gsm8k', config 'main'
    Each row has: question, answer
    """
    ds = load_dataset("gsm8k", "main", split=split)
    out: List[Example] = []

    for i, row in enumerate(ds):
        question = row["question"]
        answer = row["answer"]
        m = GSM8K_FINAL_RE.search(answer)
        if not m:
            # fallback: use full answer text (rare)
            gold = normalize_answer(answer)
        else:
            gold = normalize_answer(m.group(1))

        out.append(
            Example(
                uid=f"gsm8k_{split}_{i}",
                problem=question,
                gold=gold,
                meta={"dataset": "gsm8k", "split": split},
            )
        )

    return out


def load_math(split: str = "train") -> List[Example]:
    """
    Loads MATH from HF: dataset 'competition_math' (common mirror) OR 'math_dataset' variants.
    We try the most common one first.

    NOTE: depending on your HF environment, the dataset name can differ.
    Adjust if needed.
    """
    # Try a couple of common dataset IDs
    candidates = [
        ("competition_math", None),
        ("math_dataset", "algebra__linear_1d"),  # fallback style
    ]

    last_err: Optional[Exception] = None
    ds = None
    used = None

    for name, config in candidates:
        try:
            if config is None:
                ds = load_dataset(name, split=split)
            else:
                ds = load_dataset(name, config, split=split)
            used = (name, config)
            break
        except Exception as e:
            last_err = e
            continue

    if ds is None:
        raise RuntimeError(
            f"Failed to load MATH dataset from candidates {candidates}. "
            f"Last error: {last_err}"
        )

    out: List[Example] = []
    for i, row in enumerate(ds):
        # schema differs by mirror; try common keys
        problem = row.get("problem") or row.get("question") or row.get("prompt")
        solution = row.get("solution") or row.get("answer")

        if problem is None or solution is None:
            # skip malformed
            continue

        # MATH mirrors often store final in \boxed{} or in text.
        gold = extract_math_final(solution)
        out.append(
            Example(
                uid=f"math_{split}_{i}",
                problem=problem,
                gold=gold,
                meta={"dataset": used[0], "config": used[1], "split": split},
            )
        )
    return out


BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
def extract_math_final(solution_text: str) -> str:
    """
    Best-effort final answer extraction for MATH-style solutions.
    """
    m = BOXED_RE.search(solution_text)
    if m:
        return normalize_answer(m.group(1))

    # fallback: try last line / last number-ish token
    lines = [ln.strip() for ln in solution_text.splitlines() if ln.strip()]
    if lines:
        return normalize_answer(lines[-1])
    return normalize_answer(solution_text)
