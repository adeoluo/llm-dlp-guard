"""Shared helpers for the benchmark scripts."""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
DATA = Path(os.environ.get("DLP_BENCH_DATA", ROOT.parent / ".bench-data" / "llm-dlp-guard"))


def peak_rss_mb() -> float:
    """Peak resident memory of this process in MB (macOS reports bytes, Linux kilobytes)."""

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def machine() -> dict[str, Any]:
    chip = ""
    if sys.platform == "darwin":
        try:
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        except OSError:
            chip = ""
    return {
        "platform": platform.platform(),
        "processor": chip or platform.processor(),
        "cpus": os.cpu_count(),
        "python": platform.python_version(),
    }


def write(name: str, payload: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    payload = {"machine": machine(), "written": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")
    return path


class Timer:
    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self.start
