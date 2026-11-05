"""Secrets detection on the generated corpus: this package against detect-secrets.

    python benchmarks/secrets_detection.py [--positives 46000 --negatives 46000 --seed 0]

The corpus (``llm_dlp_guard.secrets_corpus``) holds one secret per positive sample, in the real
format of 21 credential types with random values, and hard negatives in the same contexts.
detect-secrets 1.5 runs its line-based scan with all default plugins and filters (the code path
of ``detect-secrets scan``), once with a ``.txt`` and once with a ``.yaml`` file name, since its
keyword rules depend on the file type. A detection counts if it overlaps the secret. Precision
counts every reported span; the false-alarm rate is the share of negative samples with any
report. Only this package's secret categories are counted (its PII findings, such as private IP
addresses in connection strings, are left out).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import Timer, peak_rss_mb, write

from llm_dlp_guard.catalog import default_catalog
from llm_dlp_guard.detector import Detector
from llm_dlp_guard.secrets_corpus import Sample, generate

Spans = list[tuple[int, int, str]]


def ours() -> Callable[[list[str]], list[Spans]]:
    det = Detector()
    secret_labels = {p.label for p in default_catalog().patterns if p.category == "secret"}

    def run(texts: list[str]) -> list[Spans]:
        return [
            [(f.start, f.end, f.label) for f in found if f.label in secret_labels]
            for found in det.scan_batch(texts)
        ]

    return run


def detect_secrets(filename: str) -> Callable[[list[str]], list[Spans]]:
    from detect_secrets.core.scan import _process_line_based_plugins
    from detect_secrets.settings import default_settings

    def one(text: str) -> Spans:
        lines = text.split("\n")
        offsets = [0]
        for line in lines[:-1]:
            offsets.append(offsets[-1] + len(line) + 1)
        out: Spans = []
        for secret in _process_line_based_plugins(list(enumerate(lines, start=1)), filename):
            line_no = secret.line_number
            line = lines[line_no - 1]
            value = secret.secret_value or ""
            pos = line.find(value) if value else -1
            start = offsets[line_no - 1] + (pos if pos >= 0 else 0)
            end = start + (len(value) if pos >= 0 else len(line))
            out.append((start, end, secret.type))
        return out

    def run(texts: list[str]) -> list[Spans]:
        with default_settings():
            return [one(t) for t in texts]

    return run


def score(samples: list[Sample], preds: list[Spans]) -> dict[str, object]:
    found: Counter[str] = Counter()
    labelled: Counter[str] = Counter()
    total: Counter[str] = Counter()
    reported = correct = 0
    alarms = negatives = 0
    false_alarm_examples: Counter[str] = Counter()
    for s, pred in zip(samples, preds, strict=True):
        reported += len(pred)
        if not s.spans:
            negatives += 1
            alarms += bool(pred)
            for p in pred:
                false_alarm_examples[p[2]] += 1
            continue
        ((gs, ge, label),) = s.spans
        total[label] += 1
        hits = [p for p in pred if p[0] < ge and gs < p[1]]
        correct += len(hits)
        if hits:
            found[label] += 1
            labelled[label] += any(p[2] == label for p in hits)
    positives = sum(total.values())
    return {
        "recall": round(sum(found.values()) / positives, 4),
        "precision": round(correct / reported, 4) if reported else 0.0,
        "false_alarm_rate": round(alarms / negatives, 4),
        "reported": reported,
        "recall_by_type": {k: round(found[k] / total[k], 4) for k in sorted(total)},
        "label_accuracy": round(sum(labelled.values()) / positives, 4),
        "false_alarms_by_type": dict(false_alarm_examples.most_common(10)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positives", type=int, default=46000)
    parser.add_argument("--negatives", type=int, default=46000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    samples = list(generate(args.positives, args.negatives, args.seed))
    texts = [s.text for s in samples]
    by_kind: dict[str, int] = defaultdict(int)
    for s in samples:
        by_kind[s.kind] += 1
    results = {}
    for name, run in (
        ("llm-dlp-guard", ours()),
        ("detect-secrets 1.5 (.txt)", detect_secrets("snippet.txt")),
        ("detect-secrets 1.5 (.yaml)", detect_secrets("snippet.yaml")),
    ):
        with Timer() as t:
            preds = run(texts)
        results[name] = {**score(samples, preds), "seconds": round(t.seconds, 2)}
        print(
            name,
            {
                k: v
                for k, v in results[name].items()
                if k not in ("recall_by_type", "false_alarms_by_type")
            },
            flush=True,
        )
    write(
        "secrets.json",
        {
            "corpus": {
                "positives": args.positives,
                "negatives": args.negatives,
                "seed": args.seed,
                "characters": sum(len(t) for t in texts),
                "samples_by_kind": dict(sorted(by_kind.items())),
            },
            "peak_rss_mb": round(peak_rss_mb(), 1),
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
