"""Where a result came from: the code version, the machine and its peak memory.

Every result file records the git commit of the code that produced it, and
whether the code (src, scripts, configs, dependency files) had uncommitted
changes at the time, so each number in the report can be traced back to the
exact code. Result files themselves are left out of the check: they change
while runs are in progress.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import platform
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


def machine_info(device: str | None = None) -> dict[str, Any]:
    """The machine and library versions a result was produced with (D0.2).

    On macOS this includes the chip and the memory size, read with sysctl.
    """
    info: dict[str, Any] = {
        "device": device,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in ("torch", "numpy", "gensim"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            info[package] = importlib.metadata.version(package)
    if sys.platform == "darwin":
        for key, name in (("chip", "machdep.cpu.brand_string"), ("memory_bytes", "hw.memsize")):
            with contextlib.suppress(OSError, subprocess.CalledProcessError, ValueError):
                value = subprocess.run(
                    ["sysctl", "-n", name], capture_output=True, text=True, check=True
                ).stdout.strip()
                info[key] = int(value) if key == "memory_bytes" else value
    return info
