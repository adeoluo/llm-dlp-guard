"""The Rust backend, compiled into the package on install and used by default.

The Python implementation in :mod:`llm_dlp_guard.reference` is always available and is the
reference; ``tests/test_parity.py`` checks that both return identical findings.
"""

from __future__ import annotations

import importlib
from typing import Any, Literal

Backend = Literal["python", "rust"]
BACKENDS: tuple[str, ...] = ("auto", "python", "rust")


def _load_extension() -> Any:
    try:
        return importlib.import_module("llm_dlp_guard.llm_dlp_guard_rs")
    except ImportError:  # pragma: no cover - the extension is built on install
        return None


_EXTENSION: Any = _load_extension()


def rust_available() -> bool:
    """Return whether the Rust extension is importable."""

    return _EXTENSION is not None


def resolve_backend(backend: str) -> Backend:
    """Map ``auto`` to the fastest available backend and validate explicit choices."""

    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    if backend == "auto":
        return "rust" if rust_available() else "python"
    if backend == "rust" and not rust_available():  # pragma: no cover
        raise RuntimeError(
            "the Rust backend is not built; run `uv sync` or `make rust-develop`, "
            "or use backend='python'"
        )
    return "rust" if backend == "rust" else "python"


def extension() -> Any:
    """Return the loaded extension module (only valid when ``rust_available()``)."""

    return _EXTENSION
