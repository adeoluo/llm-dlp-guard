"""This package against Microsoft Presidio on shared coarse entity types.

    python benchmarks/presidio_compare.py [--nemotron-sample 5000]

Both systems run on the same texts; labels are mapped to shared coarse types (PERSON, EMAIL,
PHONE, CREDIT_CARD, ADDRESS, DATE_OF_BIRTH, SSN, ID_NUMBER, ACCOUNT_NUMBER, IP_ADDRESS, USERNAME,
PASSWORD, API_KEY; see ``llm_dlp_guard.datasets``). Adjacent spans of one type separated only by
whitespace are merged on both sides (``John`` + ``Smith``). Predictions of types a dataset does
not annotate are dropped, and on Nemotron-PII predictions that only overlap unmapped annotations
(company names, dates, URLs...) are ignored. Scores are exact (same start, end and type) and
overlap (a prediction overlapping a gold span of its type).

* ai4privacy validation, English: all 17,046 records.
* Nemotron-PII test: a seeded sample of 5,000 documents for both systems, plus this package on
  all 100,000.

Presidio 2.2 runs with its default recognizers and spaCy ``en_core_web_lg``, default thresholds.

Every micro score has a 95% percentile bootstrap interval over documents (1,000 resamples,
seed 0), and each llm-dlp-guard row has a paired bootstrap interval of its F1 difference to
Presidio on the same resampled documents.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DATA, Timer, peak_rss_mb, write

from llm_dlp_guard import metrics
from llm_dlp_guard.datasets import (
    AI4PRIVACY_TO_COARSE,
    NEMOTRON_TO_COARSE,
    OURS_TO_COARSE,
    PRESIDIO_TO_COARSE,
    Record,
    load_ai4privacy,
    load_nemotron,
)
from llm_dlp_guard.detector import Detector
from llm_dlp_guard.findings import Finding, resolve

Spans = list[tuple[int, int, str]]


def ours(detector: Detector) -> Callable[[str], Spans]:
    def run(text: str) -> Spans:
        return metrics.map_labels(
            [(f.start, f.end, f.label) for f in detector.scan(text)], OURS_TO_COARSE
        )

    return run


def presidio() -> Callable[[str], Spans]:
    from presidio_analyzer import AnalyzerEngine

    engine = AnalyzerEngine()

    def run(text: str) -> Spans:
        found = [
            Finding(
                r.start, r.end, PRESIDIO_TO_COARSE[r.entity_type], r.entity_type, float(r.score)
            )
            for r in engine.analyze(text, language="en")
            if r.entity_type in PRESIDIO_TO_COARSE
        ]
        return [(f.start, f.end, f.label) for f in resolve(found)]

    return run


PerDoc = dict[str, list[metrics.Counts]]


def evaluate(
    records: list[Record], run: Callable[[str], Spans], scope: set[str], per_doc: PerDoc
) -> dict[str, object]:
    """Exact and overlap scores; per-document micro counts are appended to ``per_doc``."""

    exact: dict[str, metrics.Counts] = {}
    overlap: dict[str, metrics.Counts] = {}
    per_doc.setdefault("exact", [])
    per_doc.setdefault("overlap", [])
    chars = 0
    with Timer() as t:
        preds = [run(r.text) for r in records]
    for r, pred in zip(records, preds, strict=True):
        chars += len(r.text.encode())
        pred = metrics.merge_adjacent(r.text, [p for p in pred if p[2] in scope])
        gold = metrics.merge_adjacent(r.text, r.spans)
        doc_exact = metrics.score_document(gold, pred, "exact", r.ignore)
        doc_overlap = metrics.score_document(gold, pred, "overlap", r.ignore)
        metrics.accumulate(exact, doc_exact)
        metrics.accumulate(overlap, doc_overlap)
        per_doc["exact"].append(metrics.micro(doc_exact))
        per_doc["overlap"].append(metrics.micro(doc_overlap))
    return {
        "documents": len(records),
        "seconds": round(t.seconds, 1),
        "mb_per_second": round(chars / 1e6 / t.seconds, 3),
        "exact": {
            "micro": metrics.micro(exact).to_dict(),
            "types": {k: v.to_dict() for k, v in sorted(exact.items())},
        },
        "overlap": {
            "micro": metrics.micro(overlap).to_dict(),
            "types": {k: v.to_dict() for k, v in sorted(overlap.items())},
        },
    }


PRESIDIO = "Presidio 2.2 (default recognizers, en_core_web_lg)"


def add_intervals(results: dict[str, dict[str, object]], per_doc: dict[str, PerDoc]) -> None:
    """Bootstrap intervals for every system, paired against Presidio when it ran."""

    for name, res in results.items():
        for mode in ("exact", "overlap"):
            other = per_doc[PRESIDIO][mode] if PRESIDIO in per_doc and name != PRESIDIO else None
            res[mode]["ci95"] = metrics.bootstrap(per_doc[name][mode], other=other)  # type: ignore[index]


def coarse_ai4privacy(records: list[Record]) -> list[Record]:
    """Gold spans (tagger label names) mapped to coarse types."""

    return [
        Record(r.text, metrics.map_labels(r.spans, OURS_TO_COARSE), r.language, r.uid)
        for r in records
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--model", default=str(DATA / "models" / "tagger.crf"))
    parser.add_argument("--nemotron-sample", type=int, default=5000)
    parser.add_argument("--skip-presidio", action="store_true")
    args = parser.parse_args()

    combined = Detector.from_model(args.model)
    patterns = Detector()
    systems: list[tuple[str, Callable[[str], Spans]]] = [
        ("llm-dlp-guard (patterns + tagger)", ours(combined)),
        ("llm-dlp-guard (patterns only)", ours(patterns)),
    ]
    if not args.skip_presidio:
        systems.append((PRESIDIO, presidio()))

    out: dict[str, object] = {}
    english = coarse_ai4privacy(
        list(load_ai4privacy(Path(args.data) / "ai4privacy", "validation", ("en",)))
    )
    scope = set(AI4PRIVACY_TO_COARSE.values())
    out["ai4privacy_en"] = {"records": len(english), "results": {}}
    per_doc: dict[str, PerDoc] = {}
    for name, run in systems:
        res = evaluate(english, run, scope, per_doc.setdefault(name, {}))
        out["ai4privacy_en"]["results"][name] = res  # type: ignore[index]
        print(
            "ai4privacy",
            name,
            res["exact"]["micro"],
            res["overlap"]["micro"],
            res["seconds"],
            flush=True,
        )  # type: ignore[index]
    add_intervals(out["ai4privacy_en"]["results"], per_doc)  # type: ignore[index]

    nemotron = list(load_nemotron(Path(args.data) / "nemotron" / "test.parquet"))
    sample = random.Random(0).sample(nemotron, min(args.nemotron_sample, len(nemotron)))
    scope = set(NEMOTRON_TO_COARSE.values())
    out["nemotron_sample"] = {
        "documents": len(sample),
        "of": len(nemotron),
        "seed": 0,
        "results": {},
    }
    per_doc = {}
    for name, run in systems:
        res = evaluate(sample, run, scope, per_doc.setdefault(name, {}))
        out["nemotron_sample"]["results"][name] = res  # type: ignore[index]
        print(
            "nemotron",
            name,
            res["exact"]["micro"],
            res["overlap"]["micro"],
            res["seconds"],
            flush=True,
        )  # type: ignore[index]
    add_intervals(out["nemotron_sample"]["results"], per_doc)  # type: ignore[index]
    full_docs: PerDoc = {}
    full = evaluate(nemotron, ours(combined), scope, full_docs)
    full["exact"]["ci95"] = metrics.bootstrap(full_docs["exact"])  # type: ignore[index]
    full["overlap"]["ci95"] = metrics.bootstrap(full_docs["overlap"])  # type: ignore[index]
    out["nemotron_full"] = {"documents": len(nemotron), "result": full}
    print("nemotron full", full["exact"]["micro"], full["overlap"]["micro"], flush=True)
    out["peak_rss_mb"] = round(peak_rss_mb(), 1)
    write("presidio_compare.json", out)


if __name__ == "__main__":
    main()
