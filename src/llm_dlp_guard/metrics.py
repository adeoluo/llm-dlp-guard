"""Entity-level precision, recall and F1 for character spans."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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


def _prf(
    tp: NDArray[np.float64], fp: NDArray[np.float64], fn: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Micro precision, recall and F1 for arrays of summed counts (0 where undefined)."""

    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return np.stack([p, r, f])


def bootstrap(
    docs: Sequence[Counts],
    *,
    resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
    other: Sequence[Counts] | None = None,
) -> dict[str, list[float]]:
    """Percentile bootstrap intervals for micro precision, recall and F1 over documents.

    Documents are the resampling unit (entities in one document are not independent). Each
    resample draws ``len(docs)`` documents with replacement, sums their counts and recomputes
    the micro scores; the interval is the ``(1 - level) / 2`` and ``(1 + level) / 2``
    percentiles. With ``other`` (the same documents scored by a second system) the same
    resampled indices are applied to both and the result also holds the paired interval of the
    F1 difference ``docs - other`` and the share of resamples where it is not positive.
    """

    counts = np.array([(c.tp, c.fp, c.fn) for c in docs], dtype=np.float64).reshape(-1, 3)
    n = len(counts)
    if n == 0:
        raise ValueError("bootstrap needs at least one document")
    paired = None
    if other is not None:
        if len(other) != n:
            raise ValueError("paired bootstrap needs the same documents for both systems")
        paired = np.array([(c.tp, c.fp, c.fn) for c in other], dtype=np.float64).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    scores = np.empty((3, resamples))
    diffs = np.empty(resamples)
    step = max(1, 2_000_000 // n)
    for lo in range(0, resamples, step):
        hi = min(resamples, lo + step)
        # Multinomial weights equal the multiplicity of each document in a resample.
        weights = rng.multinomial(n, np.full(n, 1.0 / n), size=hi - lo).astype(np.float64)
        sums = weights @ counts
        scores[:, lo:hi] = _prf(sums[:, 0], sums[:, 1], sums[:, 2])
        if paired is not None:
            sums_b = weights @ paired
            diffs[lo:hi] = scores[2, lo:hi] - _prf(sums_b[:, 0], sums_b[:, 1], sums_b[:, 2])[2]
    q = [(1 - level) / 2 * 100, (1 + level) / 2 * 100]
    out = {
        name: [round(float(x), 4) for x in np.percentile(scores[i], q)]
        for i, name in enumerate(("precision", "recall", "f1"))
    }
    if paired is not None:
        out["f1_difference"] = [round(float(x), 4) for x in np.percentile(diffs, q)]
        out["share_not_better"] = [round(float(np.mean(diffs <= 0)), 4)]
    return out


def wilson(successes: int, total: int, z: float = 1.959964) -> list[float]:
    """Wilson score interval for a binomial proportion (95% by default)."""

    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [round(float(max(0.0, centre - half)), 4), round(float(min(1.0, centre + half)), 4)]
