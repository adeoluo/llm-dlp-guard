"""Thread-safe metrics in the Prometheus text format."""

from __future__ import annotations

import threading
from collections import defaultdict


def _labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    inner = ",".join(
        f'{k}="{v.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
        for k, v in sorted(labels.items())
    )
    return "{" + inner + "}"


class Metrics:
    """Counters with labels and per-route latency summaries."""

    def __init__(self, prefix: str = "dlp") -> None:
        self.prefix = prefix
        self._lock = threading.Lock()
        self._counters: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._help: dict[str, str] = {}
        self._latency_sum: dict[str, float] = defaultdict(float)
        self._latency_count: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}

    def inc(self, name: str, value: float = 1.0, help: str = "", **labels: str) -> None:
        with self._lock:
            self._counters[name][_labels(labels)] += value
            if help:
                self._help[name] = help

    def observe_request(self, route: str, status: int, seconds: float) -> None:
        self.inc(
            "requests", help="HTTP requests by route and status.", route=route, status=str(status)
        )
        with self._lock:
            self._latency_sum[route] += seconds
            self._latency_count[route] += 1

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def value(self, name: str, **labels: str) -> float:
        with self._lock:
            return self._counters.get(name, {}).get(_labels(labels), 0.0)

    def render(self) -> str:
        p = self.prefix
        lines: list[str] = []
        with self._lock:
            for name in sorted(self._counters):
                if name in self._help:
                    lines.append(f"# HELP {p}_{name}_total {self._help[name]}")
                lines.append(f"# TYPE {p}_{name}_total counter")
                for labels, v in sorted(self._counters[name].items()):
                    lines.append(f"{p}_{name}_total{labels} {v:g}")
            lines += [
                f"# HELP {p}_request_seconds HTTP request latency.",
                f"# TYPE {p}_request_seconds summary",
            ]
            for route in sorted(self._latency_count):
                lab = _labels({"route": route})
                lines.append(f"{p}_request_seconds_sum{lab} {self._latency_sum[route]:.6f}")
                lines.append(f"{p}_request_seconds_count{lab} {self._latency_count[route]}")
            for name, value in sorted(self._gauges.items()):
                lines += [f"# TYPE {p}_{name} gauge", f"{p}_{name} {value:g}"]
        return "\n".join(lines) + "\n"
