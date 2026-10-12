"""Where a result came from: the code version and the machine's peak memory.

Every result file records the git commit of the code that produced it, and
whether the code (src, scripts, configs, dependency files) had uncommitted
changes at the time, so each number in the report can be traced back to the
exact code. Result files themselves are left out of the check: they change
while runs are in progress.
"""

from __future__ import annotations

import resource
import subprocess
import sys
from typing import Any

CODE_PATHS = ("src", "scripts", "configs", "pyproject.toml", "uv.lock")


def code_version() -> dict[str, Any]:
    """{"commit": the HEAD commit or None, "code_changed": True if code differs from it}."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--", *CODE_PATHS],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "code_changed": None}
    return {"commit": commit, "code_changed": bool(changes)}


def peak_process_memory() -> int:
    """The most memory this process has held so far, in bytes (resident set size)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)  # macOS reports bytes
