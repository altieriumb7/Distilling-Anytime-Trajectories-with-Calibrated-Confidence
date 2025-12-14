from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple


# --- basic guardrails (best-effort, not a perfect sandbox) ---

DISALLOWED_PATTERNS = [
    r"\bimport\s+os\b",
    r"\bimport\s+sys\b",
    r"\bimport\s+subprocess\b",
    r"\bimport\s+socket\b",
    r"\bimport\s+requests\b",
    r"\bimport\s+urllib\b",
    r"\bimport\s+pathlib\b",
    r"\bfrom\s+os\b",
    r"\bfrom\s+sys\b",
    r"\bopen\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"__\w+__",
]

ALLOWED_IMPORTS = {"math", "fractions", "decimal", "itertools", "statistics", "random"}


CODE_FENCE_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class ExecResult:
    ok: bool
    stdout: str
    stderr: str
    timeout: bool
    rejected: bool
    reason: Optional[str] = None


def extract_python_code(text: str) -> Optional[str]:
    m = CODE_FENCE_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()


def _imports_are_safe(code: str) -> Tuple[bool, Optional[str]]:
    # allow "import math" or "from math import ..."
    imports = re.findall(r"^\s*(?:from|import)\s+([a-zA-Z0-9_]+)", code, flags=re.MULTILINE)
    for mod in imports:
        if mod not in ALLOWED_IMPORTS:
            return False, f"Disallowed import: {mod}"
    return True, None


def is_code_safe(code: str) -> Tuple[bool, Optional[str]]:
    for pat in DISALLOWED_PATTERNS:
        if re.search(pat, code):
            return False, f"Matched disallowed pattern: {pat}"
    ok, why = _imports_are_safe(code)
    if not ok:
        return False, why
    return True, None


def run_python_code(
    code: str,
    *,
    timeout_s: int = 2,
    max_output_chars: int = 2000,
) -> ExecResult:
    """
    Executes code in an isolated temp directory via `python -I -S`.
    - `-I` isolates environment (no user site, ignore PYTHON* env)
    - `-S` avoids importing site
    This is still not a perfect sandbox.
    """
    ok, why = is_code_safe(code)
    if not ok:
        return ExecResult(ok=False, stdout="", stderr="", timeout=False, rejected=True, reason=why)

    with tempfile.TemporaryDirectory() as td:
        script_path = os.path.join(td, "prog.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code + "\n")

        try:
            cp = subprocess.run(
                ["python", "-I", "-S", script_path],
                cwd=td,
                text=True,
                capture_output=True,
                timeout=timeout_s,
            )
            stdout = (cp.stdout or "").strip()
            stderr = (cp.stderr or "").strip()

            if len(stdout) > max_output_chars:
                stdout = stdout[:max_output_chars] + "…(truncated)"
            if len(stderr) > max_output_chars:
                stderr = stderr[:max_output_chars] + "…(truncated)"

            return ExecResult(
                ok=(cp.returncode == 0),
                stdout=stdout,
                stderr=stderr,
                timeout=False,
                rejected=False,
            )
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "").strip() if e.stdout else ""
            err = (e.stderr or "").strip() if e.stderr else ""
            return ExecResult(ok=False, stdout=out, stderr=err, timeout=True, rejected=False, reason="timeout")
