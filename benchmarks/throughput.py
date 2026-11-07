"""Scanning throughput in MB/s of UTF-8 text, and streaming overhead.

    python benchmarks/throughput.py

Text: the 81,379 ai4privacy validation records (six languages) plus 20,000 samples of the
secrets corpus, about 14 MB. Every engine scans the same texts one document at a time; the Rust
engines also run as a batch on 8 threads. Slow engines (the Python reference, Presidio and
detect-secrets) are timed on the first documents of the same list and reported per MB.
Each Rust row is the median of three runs (min and max kept).

Scaling: thread count (1, 2, 4, 8) for the batch API, document size (the same text joined
into documents of about 1 KB to 4 MB) for single-threaded scans, and the chunk size of the
stream redactor (1 to 64 characters per chunk).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DATA, Timer, peak_rss_mb, write

from llm_dlp_guard.datasets import load_ai4privacy
from llm_dlp_guard.detector import Detector, Tagger
from llm_dlp_guard.guard import Guard
from llm_dlp_guard.secrets_corpus import generate
from llm_dlp_guard.streaming import StreamRedactor
from llm_dlp_guard.vault import Vault


def mb(texts: list[str]) -> float:
    return sum(len(t.encode()) for t in texts) / 1e6


def time_each(fn: Callable[[str], object], texts: list[str]) -> dict[str, float]:
    with Timer() as t:
        for text in texts:
            fn(text)
    size = mb(texts)
    return {
        "documents": len(texts),
        "mb": round(size, 3),
        "seconds": round(t.seconds, 2),
        "mb_per_second": round(size / t.seconds, 3),
    }


def time_batch(det: Detector, texts: list[str], threads: int) -> dict[str, float]:
    with Timer() as t:
        det.scan_batch(texts, threads=threads)
    size = mb(texts)
    return {
        "documents": len(texts),
        "mb": round(size, 3),
        "seconds": round(t.seconds, 2),
        "mb_per_second": round(size / t.seconds, 3),
    }


def median_of(runs: int, fn: Callable[[], dict[str, float]]) -> dict[str, float]:
    """The run with the median speed, plus the min and max MB/s over ``runs`` runs."""

    rows = sorted((fn() for _ in range(runs)), key=lambda r: r["mb_per_second"])
    mid = dict(rows[len(rows) // 2])
    mid["runs"] = runs
    mid["mb_per_second_min"] = rows[0]["mb_per_second"]
    mid["mb_per_second_max"] = rows[-1]["mb_per_second"]
    return mid


def join_to(texts: list[str], target_bytes: int) -> list[str]:
    """Concatenate consecutive texts (newline separated) into documents of about ``target``."""

    out: list[str] = []
    buf: list[str] = []
    size = 0
    for t in texts:
        buf.append(t)
        size += len(t.encode()) + 1
        if size >= target_bytes:
            out.append("\n".join(buf))
            buf, size = [], 0
    if buf:
        out.append("\n".join(buf))
    return out


def streaming_overhead(guard: Guard, texts: list[str], chunk: int = 4) -> dict[str, float]:
    """Per-chunk latency of the stream redactor (4-character chunks are about one token)."""

    latencies = []
    held = []
    chars = 0
    with Timer() as t:
        for text in texts:
            redactor = StreamRedactor(guard, "response", Vault())
            emitted = 0
            for i in range(0, len(text), chunk):
                start = time.perf_counter()
                emitted += len(redactor.feed(text[i : i + chunk]))
                latencies.append(time.perf_counter() - start)
                held.append(min(i + chunk, len(text)) - emitted)
            redactor.flush()
            chars += len(text)
    latencies.sort()
    return {
        "chunk_chars": chunk,
        "chunks": len(latencies),
        "median_ms": round(statistics.median(latencies) * 1000, 4),
        "p99_ms": round(latencies[int(0.99 * len(latencies))] * 1000, 4),
        "max_ms": round(latencies[-1] * 1000, 3),
        "chars_per_second": round(chars / t.seconds),
        "window_chars": StreamRedactor(guard, "response", Vault()).window,
        "max_held_chars": max(held),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA / "ai4privacy"))
    parser.add_argument("--model", default=str(DATA / "models" / "tagger.crf"))
    parser.add_argument("--slow-docs", type=int, default=3000)
    args = parser.parse_args()
    texts = [r.text for r in load_ai4privacy(args.data, "validation")]
    secrets = [s.text for s in generate(10_000, 10_000, seed=1)]
    texts += secrets
    # Interleave so every prefix has both kinds of text.
    texts = [t for pair in zip(texts[::2], texts[1::2], strict=False) for t in pair]
    slow = texts[: args.slow_docs]
    print(f"{len(texts):,} documents, {mb(texts):.1f} MB", flush=True)

    tagger = Tagger.load(args.model)
    patterns_rust = Detector(backend="rust")
    combined_rust = Detector(tagger=tagger)
    patterns_py = Detector(backend="python")
    rows: dict[str, dict[str, float]] = {}
    rows["llm-dlp-guard patterns, Rust, 1 thread"] = median_of(
        3, lambda: time_each(patterns_rust.scan, texts)
    )
    rows["llm-dlp-guard patterns, Rust, batch on 8 threads"] = median_of(
        3, lambda: time_batch(patterns_rust, texts, 8)
    )
    rows["llm-dlp-guard patterns + tagger, Rust, 1 thread"] = median_of(
        3, lambda: time_each(combined_rust.scan, texts)
    )
    rows["llm-dlp-guard patterns + tagger, Rust, batch on 8 threads"] = median_of(
        3, lambda: time_batch(combined_rust, texts, 8)
    )
    rows["llm-dlp-guard patterns, Python reference"] = time_each(patterns_py.scan, slow)
    for name, row in rows.items():
        print(name, row, flush=True)

    from presidio_analyzer import AnalyzerEngine

    presidio = AnalyzerEngine()
    english = [t for t in slow if t.isascii()][: args.slow_docs // 2]
    rows["Presidio 2.2 analyze (en_core_web_lg)"] = time_each(
        lambda t: presidio.analyze(t, language="en"), english
    )
    print("presidio", rows["Presidio 2.2 analyze (en_core_web_lg)"], flush=True)

    from detect_secrets.core.scan import _process_line_based_plugins
    from detect_secrets.settings import default_settings

    def ds(text: str) -> object:
        return list(
            _process_line_based_plugins(list(enumerate(text.split("\n"), start=1)), "snippet.txt")
        )

    with default_settings():
        rows["detect-secrets 1.5 line scan"] = time_each(ds, slow)
    print("detect-secrets", rows["detect-secrets 1.5 line scan"], flush=True)

    threads: dict[str, dict[str, dict[str, float]]] = {"patterns": {}, "patterns + tagger": {}}
    for n in (1, 2, 4, 8):
        threads["patterns"][str(n)] = median_of(3, lambda n=n: time_batch(patterns_rust, texts, n))
        threads["patterns + tagger"][str(n)] = median_of(
            3, lambda n=n: time_batch(combined_rust, texts, n)
        )
        print("threads", n, threads["patterns"][str(n)]["mb_per_second"],
              threads["patterns + tagger"][str(n)]["mb_per_second"], flush=True)  # fmt: skip

    doc_size: dict[str, dict[str, dict[str, float]]] = {"patterns": {}, "patterns + tagger": {}}
    for target in (1_000, 16_000, 256_000, 4_000_000):
        docs = join_to(texts, target)
        key = f"{target // 1000} KB"
        doc_size["patterns"][key] = median_of(3, lambda d=docs: time_each(patterns_rust.scan, d))
        doc_size["patterns + tagger"][key] = time_each(combined_rust.scan, docs)
        print("doc size", key, doc_size["patterns"][key]["mb_per_second"],
              doc_size["patterns + tagger"][key]["mb_per_second"], flush=True)  # fmt: skip

    guard = Guard(tagger=None)
    long_answers = ["\n".join(texts[i : i + 12]) for i in range(0, 2400, 12)]
    stream = streaming_overhead(guard, long_answers)
    print("stream", stream, flush=True)
    chunk_sweep = {str(c): streaming_overhead(guard, long_answers, c) for c in (1, 4, 16, 64)}
    stream_tagger = streaming_overhead(Guard(tagger=tagger), long_answers[:50], 4)
    print("stream with tagger", stream_tagger, flush=True)
    write(
        "throughput.json",
        {
            "documents": len(texts),
            "megabytes": round(mb(texts), 2),
            "slow_engine_documents": len(slow),
            "rows": rows,
            "thread_scaling": threads,
            "document_size_scaling": doc_size,
            "streaming_patterns_only": stream,
            "streaming_chunk_sweep": chunk_sweep,
            "streaming_with_tagger": stream_tagger,
            "peak_rss_mb": round(peak_rss_mb(), 1),
        },
    )


if __name__ == "__main__":
    main()
