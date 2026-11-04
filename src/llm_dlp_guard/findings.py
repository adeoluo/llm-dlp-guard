"""The finding record and overlap resolution."""

from __future__ import annotations

import bisect
from dataclasses import dataclass

NER_RANK = 1_000_000


@dataclass(frozen=True)
class Finding:
    """One detected span; ``start`` and ``end`` are ``str`` indices, ``end`` exclusive."""

    start: int
    end: int
    label: str
    source: str
    score: float
    rank: int = 0
    # Extent of the whole pattern match when the pattern reports a capture group (-1: the span).
    match_start: int = -1
    match_end: int = -1

    @property
    def extent(self) -> tuple[int, int]:
        """The whole match; streaming never cuts inside it."""

        if self.match_start < 0:
            return self.start, self.end
        return self.match_start, self.match_end

    def text(self, document: str) -> str:
        return document[self.start : self.end]

    def to_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "source": self.source,
            "score": round(self.score, 4),
        }


def resolve(candidates: list[Finding]) -> list[Finding]:
    """Keep a non-overlapping subset: highest score, then longest, then earliest, then rank."""

    if len(candidates) <= 1:
        return list(candidates)
    ordered = sorted(candidates, key=lambda f: (-f.score, -(f.end - f.start), f.start, f.rank))
    kept: list[Finding] = []
    starts: list[int] = []

    for cand in ordered:
        idx = bisect.bisect_left(starts, cand.start)
        if idx > 0 and kept[idx - 1].end > cand.start:
            continue
        if idx < len(kept) and kept[idx].start < cand.end:
            continue
        kept.insert(idx, cand)
        starts.insert(idx, cand.start)
    return kept
