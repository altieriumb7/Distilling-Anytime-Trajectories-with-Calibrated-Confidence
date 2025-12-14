from __future__ import annotations
from typing import Optional


def _common_header() -> str:
    return (
        "You are a careful math solver.\n"
        "Always follow the required output format.\n"
        "Do NOT output multiple final answers.\n"
    )


def prompt_t1_draft(problem: str) -> str:
    return (
        _common_header()
        + "BUDGET=1 (draft): Solve quickly using very few steps.\n"
          "Output format MUST be:\n"
          "1) brief reasoning (1-4 lines max)\n"
          "2) final line: #### <final_answer>\n"
          "3) confidence line: CONF: <number between 0 and 1>\n\n"
          f"PROBLEM:\n{problem}\n\nRESPONSE:\n"
    )


def prompt_t2_resolve(problem: str, draft_text: str) -> str:
    return (
        _common_header()
        + "BUDGET=2 (re-solve): Solve again using a different approach.\n"
          "If your answer differs from the draft, explain why the draft was wrong.\n"
          "Output format MUST be:\n"
          "1) brief reasoning\n"
          "2) final line: #### <final_answer>\n"
          "3) confidence line: CONF: <number between 0 and 1>\n\n"
          f"PROBLEM:\n{problem}\n\n"
          "DRAFT OUTPUT (for reference):\n"
          f"{draft_text}\n\nRESPONSE:\n"
    )


def prompt_t3_verify_code(problem: str, candidate_answer: Optional[str]) -> str:
    cand = candidate_answer if candidate_answer is not None else "(unknown)"
    return (
        _common_header()
        + "INTERNAL VERIFIER: Write Python code to compute the final numeric answer.\n"
          "Rules:\n"
          "- Use ONLY Python standard library.\n"
          "- No file I/O, no networking, no subprocess.\n"
          "- The program MUST print only the numeric result (single line).\n"
          "- Keep the code short.\n\n"
          "Return format MUST be exactly:\n"
          "```python\n"
          "<code>\n"
          "```\n"
          "CONF: <number between 0 and 1>\n\n"
          f"PROBLEM:\n{problem}\n\n"
          f"CANDIDATE ANSWER TO CHECK: {cand}\n\nRESPONSE:\n"
    )


def prompt_t3_verify_text(problem: str, draft_text: str, resolve_text: str, verifier_stdout: str) -> str:
    """
    Student-facing checkpoint t=3: NL verification + possibly corrected answer.
    We allow using verifier_stdout as a 'computed check' signal (teachers sees it; student will learn the style).
    """
    return (
        _common_header()
        + "BUDGET=3 (verify): Verify the solution carefully.\n"
          "Do a quick consistency check and recompute key quantities.\n"
          "If the earlier answer is wrong, correct it.\n"
          "Output format MUST be:\n"
          "1) verification notes (short)\n"
          "2) final line: #### <final_answer>\n"
          "3) confidence line: CONF: <number between 0 and 1>\n\n"
          f"PROBLEM:\n{problem}\n\n"
          "ATTEMPT t1 (draft):\n"
          f"{draft_text}\n\n"
          "ATTEMPT t2 (re-solve):\n"
          f"{resolve_text}\n\n"
          "CHECK (computed result, if available):\n"
          f"{verifier_stdout}\n\n"
          "RESPONSE:\n"
    )


def prompt_t4_repair(problem: str, draft_text: str, resolve_text: str, verify_text: str, verifier_stdout: str, verifier_stderr: str) -> str:
    return (
        _common_header()
        + "BUDGET=4 (repair/final): Produce the best final solution.\n"
          "If the computed check is valid, your final answer MUST match it.\n"
          "Output format MUST be:\n"
          "1) clear reasoning (can be concise)\n"
          "2) final line: #### <final_answer>\n"
          "3) confidence line: CONF: <number between 0 and 1>\n\n"
          f"PROBLEM:\n{problem}\n\n"
          "ATTEMPT t1 (draft):\n"
          f"{draft_text}\n\n"
          "ATTEMPT t2 (re-solve):\n"
          f"{resolve_text}\n\n"
          "ATTEMPT t3 (verify):\n"
          f"{verify_text}\n\n"
          "VERIFIER (python) STDOUT:\n"
          f"{verifier_stdout}\n\n"
          "VERIFIER (python) STDERR (if any):\n"
          f"{verifier_stderr}\n\n"
          "RESPONSE:\n"
    )
