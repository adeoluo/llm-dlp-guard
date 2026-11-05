"""Tagger training and entity-level evaluation on ai4privacy/pii-masking-400k.

    python benchmarks/pii_ai4privacy.py            # grid on a dev split, final model, evaluation

1. Hyperparameters are chosen on a dev split carved from the training data (records whose uid
   hash falls in the last tenth); the validation split is not used for any choice.
2. The final tagger is trained on all 325,517 training records (six languages) and saved to
   ``$DLP_BENCH_DATA/models/tagger.crf``.
3. The validation split (81,379 records) is scored with exact entity matching (same start, end
   and label) for the tagger alone (with the detector's default 0.5 score floor, and without
   it), the pattern catalog alone (pattern labels mapped into the ai4privacy label space) and
   both combined, per label and per language.

``--skip-train`` re-evaluates the saved model and keeps the grid and training figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DATA, RESULTS, Timer, peak_rss_mb, write

from llm_dlp_guard import metrics
from llm_dlp_guard.datasets import PATTERN_TO_AI4PRIVACY, TAGGER_LABELS, Record, load_ai4privacy
from llm_dlp_guard.detector import Detector, Tagger
from llm_dlp_guard.findings import Finding

GRID = [(0.05, 3), (0.1, 3), (0.2, 3), (0.05, 6), (0.1, 6)]
BUCKET_BITS = 21


def is_dev(r: Record) -> bool:
    return int(hashlib.sha1(r.uid.encode()).hexdigest(), 16) % 10 == 9


def evaluate(
    records: list[Record], predict: object, mapping: dict[str, str] | None = None
) -> dict[str, object]:
    total: dict[str, metrics.Counts] = {}
    by_lang: dict[str, dict[str, metrics.Counts]] = defaultdict(dict)
    labels_in_scope = set(TAGGER_LABELS) if mapping is None else set(mapping.values())
    for r in records:
        pred = predict(r.text)  # type: ignore[operator]
        spans = [(f.start, f.end, f.label) for f in pred]
        if mapping is not None:
            spans = metrics.map_labels(spans, mapping)
        gold = [s for s in r.spans if s[2] in labels_in_scope]
        doc = metrics.score_document(gold, spans, "exact")
        metrics.accumulate(total, doc)
        metrics.accumulate(by_lang[r.language], doc)
    micro = metrics.micro(total)
    return {
        "micro": micro.to_dict(),
        "macro_f1": round(metrics.macro_f1(total), 4),
        "labels": {k: {**v.to_dict(), "support": v.support} for k, v in sorted(total.items())},
        "languages": {k: metrics.micro(v).to_dict() for k, v in sorted(by_lang.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA / "ai4privacy"))
    parser.add_argument("--model", default=str(DATA / "models" / "tagger.crf"))
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="evaluate the saved model; keep the grid and training figures of the last full run",
    )
    args = parser.parse_args()
    previous = RESULTS / "pii_ai4privacy.json"

    train = list(load_ai4privacy(args.data, "train"))
    validation = list(load_ai4privacy(args.data, "validation"))
    print(f"train {len(train):,} records, validation {len(validation):,}", flush=True)

    grid: list[dict[str, object]] = []
    best = (args.learning_rate, args.epochs)
    if args.skip_train:
        last = json.loads(previous.read_text(encoding="utf-8"))
        grid = last["grid"]
        best = (last["chosen"]["learning_rate"], last["chosen"]["epochs"])
    elif not args.skip_grid:
        fit = [r for r in train if not is_dev(r)]
        dev = [r for r in train if is_dev(r)]
        print(f"grid: fit {len(fit):,}, dev {len(dev):,}", flush=True)
        best_f1 = -1.0
        for lr, epochs in GRID:
            with Timer() as t:
                tagger = Tagger.train(
                    [r.text for r in fit], [r.spans for r in fit], TAGGER_LABELS,
                    n_buckets=1 << BUCKET_BITS, epochs=epochs, learning_rate=lr, seed=0,
                )  # fmt: skip
            result = evaluate(dev, tagger.predict)
            f1 = float(result["micro"]["f1"])  # type: ignore[index]
            grid.append(
                {
                    "learning_rate": lr,
                    "epochs": epochs,
                    "dev_micro_f1": f1,
                    "train_seconds": round(t.seconds, 1),
                }
            )
            print(grid[-1], flush=True)
            if f1 > best_f1:
                best_f1, best = f1, (lr, epochs)
            del tagger

    lr, epochs = best
    if args.skip_train:
        tagger = Tagger.load(args.model)
        train_seconds = float(last["train_seconds"])
    else:
        with Timer() as t:
            tagger = Tagger.train(
                [r.text for r in train], [r.spans for r in train], TAGGER_LABELS,
                n_buckets=1 << BUCKET_BITS, epochs=epochs, learning_rate=lr, seed=0, verbose=True,
            )  # fmt: skip
        train_seconds = t.seconds
        Path(args.model).parent.mkdir(parents=True, exist_ok=True)
        tagger.save(args.model)
    print(f"final model: lr {lr}, {epochs} epochs, {train_seconds:.1f} s", flush=True)

    floor = 0.5
    patterns = Detector()
    combined = Detector(tagger=tagger, ner_min_score=floor)
    combined_map = {label: label for label in TAGGER_LABELS} | PATTERN_TO_AI4PRIVACY

    def tagger_floor(text: str) -> list[Finding]:
        return [f for f in tagger.predict(text) if f.score >= floor]

    results = {}
    for name, predict, mapping in (
        ("tagger", tagger_floor, None),
        ("tagger_no_floor", tagger.predict, None),
        ("patterns", patterns.scan, PATTERN_TO_AI4PRIVACY),
        ("combined", combined.scan, combined_map),
    ):
        with Timer() as t:
            results[name] = evaluate(validation, predict, mapping)
        results[name]["seconds"] = round(t.seconds, 1)  # type: ignore[index]
        print(name, results[name]["micro"], flush=True)

    model_size = Path(args.model).stat().st_size
    write(
        "pii_ai4privacy.json",
        {
            "dataset": "ai4privacy/pii-masking-400k",
            "train_records": len(train),
            "validation_records": len(validation),
            "validation_entities": sum(len(r.spans) for r in validation),
            "validation_characters": sum(len(r.text) for r in validation),
            "grid": grid,
            "chosen": {"learning_rate": lr, "epochs": epochs, "bucket_bits": BUCKET_BITS},
            "ner_min_score": floor,
            "train_seconds": round(train_seconds, 1),
            "model_bytes": model_size,
            "peak_rss_mb": round(peak_rss_mb(), 1),
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
