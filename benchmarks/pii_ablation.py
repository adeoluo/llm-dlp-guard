"""Ablations, score-floor sweep, learning curve and seed spread on ai4privacy validation.

    python benchmarks/pii_ablation.py [--curve 10000,40000,160000 --seeds 0,1,2]

Uses the saved tagger from ``benchmarks/pii_ai4privacy.py`` (trained on all 325,517 training
records) and the chosen hyperparameters (learning rate 0.1, 6 epochs, 2^21 feature buckets).

1. Component ablation on all 81,379 validation records, exact entity match in the ai4privacy
   label space, each with a 95% document-level bootstrap interval (1,000 resamples) and a paired
   interval of the F1 difference to the full detector: full (patterns + tagger), tagger alone,
   patterns alone, full without pattern validators, full without the rule that validated
   structured matches win overlaps with tagger spans.
2. Score floor sweep: the full detector with tagger floors 0.0 to 0.9.
3. Learning curve: taggers trained on seeded random subsets of the training split, evaluated
   alone on the full validation split; the smallest size is trained with several seeds to show
   the spread due to the training order.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DATA, RESULTS, Timer, peak_rss_mb, write

from llm_dlp_guard import metrics, validators
from llm_dlp_guard.catalog import Catalog, default_catalog
from llm_dlp_guard.datasets import PATTERN_TO_AI4PRIVACY, TAGGER_LABELS, Record, load_ai4privacy
from llm_dlp_guard.detector import Detector, Tagger
from llm_dlp_guard.findings import Finding, resolve

COMBINED_MAP = {label: label for label in TAGGER_LABELS} | PATTERN_TO_AI4PRIVACY


def per_doc(records: list[Record], predict: object, mapping: dict[str, str]) -> dict[str, object]:
    """Micro scores, a bootstrap interval and the per-document counts (for paired tests)."""

    scope = set(mapping.values())
    docs: list[metrics.Counts] = []
    total = metrics.Counts()
    with Timer() as t:
        for r in records:
            spans = metrics.map_labels(
                [(f.start, f.end, f.label) for f in predict(r.text)],  # type: ignore[operator]
                mapping,
            )
            gold = [s for s in r.spans if s[2] in scope]
            doc = metrics.micro(metrics.score_document(gold, spans, "exact"))
            docs.append(doc)
            total.add(doc)
    return {"micro": total.to_dict(), "seconds": round(t.seconds, 1), "_docs": docs}


def no_validators() -> Catalog:
    return Catalog([replace(p, validator=None) for p in default_catalog().patterns])


def without_structured_rule(det: Detector, tagger: Tagger, floor: float) -> object:
    """Patterns and tagger spans resolved by score alone (tagger spans may cover validated
    matches)."""

    def run(text: str) -> list[Finding]:
        found = det.candidates(text)
        tagged = [f for f in tagger.predict(text) if f.score >= floor]
        return resolve(found + tagged)

    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA / "ai4privacy"))
    parser.add_argument("--model", default=str(DATA / "models" / "tagger.crf"))
    parser.add_argument("--curve", default="10000,40000,160000")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args()
    chosen = json.loads((RESULTS / "pii_ai4privacy.json").read_text(encoding="utf-8"))["chosen"]

    validation = list(load_ai4privacy(args.data, "validation"))
    tagger = Tagger.load(args.model)
    floor = 0.5
    full = Detector(tagger=tagger, ner_min_score=floor)
    patterns = Detector()
    tagger_map = {label: label for label in TAGGER_LABELS}

    def tagger_only(text: str) -> list[Finding]:
        return [f for f in tagger.predict(text) if f.score >= floor]

    variants: dict[str, tuple[object, dict[str, str]]] = {
        "full (patterns + tagger)": (full.scan, COMBINED_MAP),
        "tagger only": (tagger_only, tagger_map),
        "patterns only": (patterns.scan, PATTERN_TO_AI4PRIVACY),
        "full without pattern validators": (
            Detector(no_validators(), tagger, ner_min_score=floor).scan,
            COMBINED_MAP,
        ),
        "full without the structured-match rule": (
            without_structured_rule(patterns, tagger, floor),
            COMBINED_MAP,
        ),
    }
    ablation: dict[str, dict[str, object]] = {}
    for name, (predict, mapping) in variants.items():
        ablation[name] = per_doc(validation, predict, mapping)
        print(name, ablation[name]["micro"], ablation[name]["seconds"], flush=True)
    reference = ablation["full (patterns + tagger)"]["_docs"]
    for name, res in ablation.items():
        other = None if name.startswith("full (") else reference
        res["ci95"] = metrics.bootstrap(
            res.pop("_docs"),  # type: ignore[arg-type]
            other=other,  # type: ignore[arg-type]
            resamples=args.resamples,
        )

    sweep: dict[str, dict[str, object]] = {}
    for f in (0.0, 0.3, 0.5, 0.7, 0.9):
        det = Detector(tagger=tagger, ner_min_score=f)
        res = per_doc(validation, det.scan, COMBINED_MAP)
        res.pop("_docs")
        sweep[f"{f:.1f}"] = res
        print("floor", f, res["micro"], flush=True)
    del full

    train = list(load_ai4privacy(args.data, "train"))
    sizes = [int(s) for s in args.curve.split(",") if s]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    curve: list[dict[str, object]] = []
    for size in sizes:
        for seed in seeds if size == sizes[0] else seeds[:1]:
            subset = random.Random(seed).sample(train, size)
            with Timer() as t:
                small = Tagger.train(
                    [r.text for r in subset], [r.spans for r in subset], TAGGER_LABELS,
                    n_buckets=1 << chosen["bucket_bits"], epochs=chosen["epochs"],
                    learning_rate=chosen["learning_rate"], seed=seed,
                )  # fmt: skip
            res = per_doc(
                validation,
                lambda text, m=small: [f for f in m.predict(text) if f.score >= floor],
                tagger_map,
            )
            row = {
                "train_records": size,
                "seed": seed,
                "train_seconds": round(t.seconds, 1),
                "micro": res["micro"],
                "ci95": metrics.bootstrap(res["_docs"], resamples=args.resamples),  # type: ignore[arg-type]
            }
            curve.append(row)
            print("curve", {k: v for k, v in row.items() if k != "ci95"}, flush=True)
            del small
    pii = json.loads((RESULTS / "pii_ai4privacy.json").read_text(encoding="utf-8"))
    curve.append(
        {
            "train_records": pii["train_records"],
            "seed": 0,
            "train_seconds": pii["train_seconds"],
            "micro": ablation["tagger only"]["micro"],
            "ci95": ablation["tagger only"]["ci95"],
        }
    )

    write(
        "pii_ablation.json",
        {
            "dataset": "ai4privacy/pii-masking-400k validation",
            "records": len(validation),
            "entities": sum(len(r.spans) for r in validation),
            "ner_min_score": floor,
            "resamples": args.resamples,
            "structured_validators": sorted(validators.STRUCTURED),
            "ablation": ablation,
            "score_floor_sweep": sweep,
            "learning_curve": curve,
            "peak_rss_mb": round(peak_rss_mb(), 1),
        },
    )


if __name__ == "__main__":
    main()
