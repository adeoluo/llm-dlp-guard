"""Entity-level precision, recall and F1 for character spans."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

Span = tuple[int, int, str]


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    def add(self, other: Counts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def to_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def map_labels(spans: Iterable[Span], mapping: Mapping[str, str]) -> list[Span]:
    """Rename labels with ``mapping``; spans whose label is not mapped are dropped."""

    return [(s, e, mapping[label]) for s, e, label in spans if label in mapping]


def merge_adjacent(text: str, spans: Sequence[Span]) -> list[Span]:
    """Join same-label spans separated only by whitespace (``John`` + ``Smith``)."""

    out: list[Span] = []
    for s, e, label in sorted(spans):
        if out and out[-1][2] == label and out[-1][1] <= s and text[out[-1][1] : s].isspace():
            out[-1] = (out[-1][0], e, label)
        else:
            out.append((s, e, label))
    return out


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def score_document(
    gold: Sequence[Span],
    pred: Sequence[Span],
    mode: str = "exact",
    ignore: Sequence[tuple[int, int]] = (),
) -> dict[str, Counts]:
    """Per-label counts for one document.

    ``exact``: a prediction is correct if a gold span has the same start, end and label.
    ``overlap``: a prediction is correct if it overlaps a gold span of its label, and a gold
    span is found if any prediction of its label overlaps it. Predictions that overlap an
    ``ignore`` region and no gold span are skipped.
    """

    if mode not in ("exact", "overlap"):
        raise ValueError("mode must be 'exact' or 'overlap'")
    counts: dict[str, Counts] = defaultdict(Counts)
    kept = [
        p
        for p in pred
        if not ignore
        or any(_overlaps((p[0], p[1]), (g[0], g[1])) for g in gold)
        or not any(_overlaps((p[0], p[1]), r) for r in ignore)
    ]
    if mode == "exact":
        gold_set = set(gold)
        pred_set = set(kept)
        for g in gold_set:
            counts[g[2]].tp += g in pred_set
            counts[g[2]].fn += g not in pred_set
        for p in pred_set - gold_set:
            counts[p[2]].fp += 1
        return counts
    for p in kept:
        hit = any(g[2] == p[2] and _overlaps((p[0], p[1]), (g[0], g[1])) for g in gold)
        if not hit:
            counts[p[2]].fp += 1
    for g in gold:
        found = any(p[2] == g[2] and _overlaps((p[0], p[1]), (g[0], g[1])) for p in kept)
        counts[g[2]].tp += found
        counts[g[2]].fn += not found
    return counts


def accumulate(total: dict[str, Counts], doc: Mapping[str, Counts]) -> None:
    for label, c in doc.items():
        total.setdefault(label, Counts()).add(c)


def micro(total: Mapping[str, Counts]) -> Counts:
    out = Counts()
    for c in total.values():
        out.add(c)
    return out


def macro_f1(total: Mapping[str, Counts]) -> float:
    labels = [c for c in total.values() if c.support > 0]
    return sum(c.f1 for c in labels) / len(labels) if labels else 0.0
