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

The corpus is generated three times (seeds 0, 1, 2); the headline is seed 0 with 95% Wilson
intervals, and the per-seed rows show the spread. The ablation reruns this package with every
pattern validator removed (checksums, entropy and placeholder filters).

External check: the Nemotron-PII test split (``nvidia/Nemotron-PII``, written independently of
this package) labels ``api_key`` and ``password`` values. Every document with such a label plus
an equal seeded sample of documents without one is scanned by both tools; a gold value is found
if any secret report overlaps it, and a report is correct if it overlaps a gold value.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import random
from dataclasses import replace

from _common import DATA, Timer, peak_rss_mb, write

from llm_dlp_guard.catalog import Catalog, default_catalog
from llm_dlp_guard.datasets import load_nemotron
from llm_dlp_guard.detector import Detector
from llm_dlp_guard.metrics import wilson
from llm_dlp_guard.secrets_corpus import Sample, generate

Spans = list[tuple[int, int, str]]


def ours(validators: bool = True) -> Callable[[list[str]], list[Spans]]:
    catalog = default_catalog()
    if not validators:
        catalog = Catalog([replace(p, validator=None) for p in catalog.patterns])
    det = Detector(catalog)
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
    hits = sum(found.values())
    return {
        "recall": round(hits / positives, 4),
        "recall_ci95": wilson(hits, positives),
        "precision": round(correct / reported, 4) if reported else 0.0,
        "precision_ci95": wilson(correct, reported),
        "false_alarm_rate": round(alarms / negatives, 4),
        "false_alarm_ci95": wilson(alarms, negatives),
        "reported": reported,
        "recall_by_type": {k: round(found[k] / total[k], 4) for k in sorted(total)},
        "label_accuracy": round(sum(labelled.values()) / positives, 4),
        "false_alarms_by_type": dict(false_alarm_examples.most_common(10)),
    }


SUMMARY = ("recall", "precision", "false_alarm_rate", "label_accuracy", "seconds")


def systems() -> list[tuple[str, Callable[[list[str]], list[Spans]]]]:
    return [
        ("llm-dlp-guard", ours()),
        ("detect-secrets 1.5 (.txt)", detect_secrets("snippet.txt")),
        ("detect-secrets 1.5 (.yaml)", detect_secrets("snippet.yaml")),
    ]


def nemotron_secrets(path: Path, seed: int = 0) -> dict[str, object]:
    """Both tools on Nemotron-PII documents with ``api_key``/``password`` labels."""

    wanted = {"API_KEY", "PASSWORD"}
    records = list(load_nemotron(path))
    with_secret = [r for r in records if any(s[2] in wanted for s in r.spans)]
    without = [r for r in records if not any(s[2] in wanted for s in r.spans)]
    docs = with_secret + random.Random(seed).sample(without, min(len(with_secret), len(without)))
    texts = [r.text for r in docs]
    out: dict[str, object] = {
        "documents_with_secret": len(with_secret),
        "documents_without": len(docs) - len(with_secret),
        "gold": {k: sum(s[2] == k for r in with_secret for s in r.spans) for k in sorted(wanted)},
        "results": {},
    }
    for name, run in systems():
        with Timer() as t:
            preds = run(texts)
        found: Counter[str] = Counter()
        total: Counter[str] = Counter()
        reported = correct = alarms = 0
        for r, pred in zip(docs, preds, strict=True):
            gold = [s for s in r.spans if s[2] in wanted]
            reported += len(pred)
            correct += sum(any(p[0] < g[1] and g[0] < p[1] for g in gold) for p in pred)
            alarms += bool(pred) and not gold
            for g in gold:
                total[g[2]] += 1
                found[g[2]] += any(p[0] < g[1] and g[0] < p[1] for p in pred)
        n_gold = sum(total.values())
        res = {
            "recall": round(sum(found.values()) / n_gold, 4),
            "recall_ci95": wilson(sum(found.values()), n_gold),
            "recall_by_type": {k: round(found[k] / total[k], 4) for k in sorted(total)},
            "precision": round(correct / reported, 4) if reported else 0.0,
            "precision_ci95": wilson(correct, reported),
            "reported": reported,
            "false_alarm_rate": round(alarms / (len(docs) - len(with_secret)), 4),
            "seconds": round(t.seconds, 1),
        }
        out["results"][name] = res  # type: ignore[index]
        print("nemotron", name, res, flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positives", type=int, default=46000)
    parser.add_argument("--negatives", type=int, default=46000)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--nemotron", default=str(DATA / "nemotron" / "test.parquet"))
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    results: dict[str, dict[str, object]] = {}
    per_seed: dict[str, dict[str, dict[str, object]]] = {}
    corpus: dict[str, object] = {}
    for seed in seeds:
        samples = list(generate(args.positives, args.negatives, seed))
        texts = [s.text for s in samples]
        runs = systems()
        if seed == seeds[0]:
            runs.append(("llm-dlp-guard without validators", ours(validators=False)))
            by_kind: dict[str, int] = defaultdict(int)
            for s in samples:
                by_kind[s.kind] += 1
            corpus = {
                "positives": args.positives,
                "negatives": args.negatives,
                "seed": seed,
                "characters": sum(len(t) for t in texts),
                "samples_by_kind": dict(sorted(by_kind.items())),
            }
        for name, run in runs:
            with Timer() as t:
                preds = run(texts)
            res = {**score(samples, preds), "seconds": round(t.seconds, 2)}
            if seed == seeds[0]:
                results[name] = res
            per_seed.setdefault(name, {})[str(seed)] = {k: res[k] for k in SUMMARY}
            print(seed, name, per_seed[name][str(seed)], flush=True)
    external = nemotron_secrets(Path(args.nemotron)) if Path(args.nemotron).exists() else None
    write(
        "secrets.json",
        {
            "corpus": corpus,
            "seeds": seeds,
            "peak_rss_mb": round(peak_rss_mb(), 1),
            "results": results,
            "per_seed": per_seed,
            "nemotron": external,
        },
    )


if __name__ == "__main__":
    main()
