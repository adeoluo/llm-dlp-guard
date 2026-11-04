"""Python reference implementation of the Rust core.

It follows ``rust/src`` line for line: the anchor prefilter, pattern matching, validators and
overlap resolution in :class:`ReferenceEngine`, and the tokenizer, hashed features, Viterbi
decoding and forward-backward marginals of the CRF tagger in :class:`ReferenceCrf` (which reads
models written by the Rust trainer). It is slower and exists for parity tests and as readable
documentation of the algorithms.
"""

from __future__ import annotations

import json
import re
import struct
import unicodedata
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from llm_dlp_guard import validators
from llm_dlp_guard.catalog import Catalog
from llm_dlp_guard.findings import NER_RANK, Finding, resolve

_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1
SHAPE_LIMIT = 16
FEATURES_PER_TOKEN = 19
MAGIC = b"DLPCRF01"
# Rust's char::is_numeric is the N* general categories (str.isnumeric also accepts CJK numerals).
_NUMERIC = frozenset({"Nd", "Nl", "No"})


class ReferenceEngine:
    """The pattern engine in Python (``re``)."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self._compiled: list[tuple[re.Pattern[str], int | str]] = []
        for p in catalog.patterns:
            if p.validator is not None and p.validator not in validators.VALIDATORS:
                raise ValueError(f"{p.name}: unknown validator {p.validator}")
            rx = re.compile(p.regex)
            if p.group is not None and p.group not in rx.groupindex:
                raise ValueError(f"{p.name}: no group named {p.group}")
            self._compiled.append((rx, p.group if p.group is not None else 0))

    def active(self, text: str) -> list[int]:
        lowered = text.translate(_ASCII_LOWER)
        out = []
        for i, p in enumerate(self.catalog.patterns):
            if not p.anchors or any(a.translate(_ASCII_LOWER) in lowered for a in p.anchors):
                out.append(i)
        return out

    def candidates(self, text: str, start: int = 0) -> list[Finding]:
        """Validated matches starting at or after ``start`` (earlier text is context)."""

        out: list[Finding] = []
        for idx in self.active(text[start:]):
            p = self.catalog.patterns[idx]
            rx, group = self._compiled[idx]
            for m in rx.finditer(text, start):
                s, e = m.span(group)
                if s < 0 or s >= e:
                    continue
                factor = 1.0 if p.validator is None else validators.check(p.validator, text[s:e])
                if factor is None:
                    continue
                out.append(
                    Finding(s, e, p.label, p.name, p.score * factor, idx, m.start(), m.end())
                )
        return out

    def scan(self, text: str) -> list[Finding]:
        return resolve(self.candidates(text))


def _is_space(c: str) -> bool:
    return c.isspace()


def _is_word(c: str) -> bool:
    return c.isalnum() or c == "_"


def tokenize(text: str) -> list[tuple[int, int]]:
    """Runs of word characters, single other non-space characters."""

    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, c in enumerate(text):
        if _is_word(c):
            if start is None:
                start = i
            continue
        if start is not None:
            out.append((start, i))
            start = None
        if not _is_space(c):
            out.append((i, i + 1))
    if start is not None:
        out.append((start, len(text)))
    return out


def fnv1a(data: bytes, h: int = FNV_OFFSET) -> int:
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & MASK64
    return h


def _feature(template: str, *parts: str) -> int:
    return fnv1a((template + "=" + "".join(parts)).encode("utf-8"))


def _shape(word: str) -> str:
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif unicodedata.category(c) in _NUMERIC:
            out.append("d")
        else:
            out.append(c)
    return "".join(out)


def _short_shape(full: str) -> str:
    out: list[str] = []
    for c in full:
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)[:SHAPE_LIMIT]


def token_features(text: str, tokens: list[tuple[int, int]]) -> list[list[int]]:
    """Hashed (unreduced) features per token, as ``crf::token_features``."""

    n = len(tokens)
    lower = [text[s:e].lower() for s, e in tokens]
    shapes = [_shape(text[s:e]) for s, e in tokens]
    shorts = [_short_shape(s) for s in shapes]

    def gap(i: int) -> str:
        if i == 0:
            return "s"
        if i == n:
            return "e"
        return "1" if tokens[i - 1][1] < tokens[i][0] else "0"

    def word(i: int) -> str:
        return "<s>" if i < 0 else "</s>" if i >= n else lower[i]

    def short(i: int) -> str:
        return "<s>" if i < 0 else "</s>" if i >= n else shorts[i]

    out = []
    for i in range(n):
        w = lower[i]
        out.append(
            [
                fnv1a(b"b"),
                _feature("w", w),
                _feature("p2", w[:2]),
                _feature("p3", w[:3]),
                _feature("s2", w[-2:]),
                _feature("s3", w[-3:]),
                _feature("sh", shapes[i][:SHAPE_LIMIT]),
                _feature("ss", shorts[i]),
                _feature("ln", str(min(len(w), 12))),
                _feature("g", gap(i)),
                _feature("ga", gap(i + 1)),
                _feature("w-1", word(i - 1)),
                _feature("w+1", word(i + 1)),
                _feature("w-2", word(i - 2)),
                _feature("w+2", word(i + 2)),
                _feature("ss-1", short(i - 1)),
                _feature("ss+1", short(i + 1)),
                _feature("w-1w", word(i - 1), "|", w),
                _feature("ss-1ss", short(i - 1), "|", shorts[i], "|", short(i + 1)),
            ]
        )
    return out


def tag_names(labels: list[str]) -> list[str]:
    tags = ["O"]
    for label in labels:
        tags += [f"B-{label}", f"I-{label}"]
    return tags


def gold_tags(text: str, spans: list[tuple[int, int, str]], labels: list[str]) -> list[int]:
    """BIO tag ids: a token belongs to the first span (by start) it overlaps."""

    tokens = tokenize(text)
    tags = [0] * len(tokens)
    for start, end, label in sorted(spans, key=lambda s: (s[0], s[1])):
        if label not in labels:
            continue
        li = labels.index(label)
        first = True
        for t, (ts, te) in enumerate(tokens):
            if ts < end and te > start and tags[t] == 0:
                tags[t] = 1 + 2 * li if first else 2 + 2 * li
                first = False
    return tags


class ReferenceCrf:
    """CRF inference in numpy, reading the Rust model format."""

    def __init__(
        self,
        labels: list[str],
        n_buckets: int,
        weights: NDArray[np.float32],
        transitions: NDArray[np.float32],
        start: NDArray[np.float32],
        end: NDArray[np.float32],
    ) -> None:
        self.labels = labels
        self.tags = tag_names(labels)
        self.n_buckets = n_buckets
        self.weights = weights
        self.transitions = transitions
        self.start = start
        self.end = end

    @classmethod
    def load(cls, path: str | Path) -> ReferenceCrf:
        raw = Path(path).read_bytes()
        if raw[:8] != MAGIC:
            raise ValueError("not an llm-dlp-guard CRF model")
        (hlen,) = struct.unpack_from("<I", raw, 8)
        header = json.loads(raw[12 : 12 + hlen])
        labels = list(header["labels"])
        n_buckets = int(header["n_buckets"])
        t = 1 + 2 * len(labels)
        pos = 12 + hlen
        params = np.frombuffer(raw, dtype="<f4", count=2 * t + t * t, offset=pos)
        pos += 4 * (2 * t + t * t)
        (rows,) = struct.unpack_from("<Q", raw, pos)
        pos += 8
        weights = np.zeros((n_buckets, t), dtype=np.float32)
        record = np.dtype([("row", "<u4"), ("w", "<f4", (t,))])
        table = np.frombuffer(raw, dtype=record, count=rows, offset=pos)
        weights[table["row"]] = table["w"]
        return cls(
            labels,
            n_buckets,
            weights,
            params[2 * t :].reshape(t, t).copy(),
            params[:t].copy(),
            params[t : 2 * t].copy(),
        )

    def feature_ids(self, text: str) -> list[list[int]]:
        return [[h % self.n_buckets for h in fs] for fs in token_features(text, tokenize(text))]

    def _emissions(self, text: str) -> NDArray[np.float64]:
        ids = np.asarray(self.feature_ids(text), dtype=np.int64)
        result: NDArray[np.float64] = self.weights[ids].astype(np.float64).sum(axis=1)
        return result

    def tag_path(self, text: str) -> list[int]:
        emit = self._emissions(text) if tokenize(text) else np.zeros((0, len(self.tags)))
        return self._viterbi(emit)

    def _viterbi(self, emit: NDArray[np.float64]) -> list[int]:
        n = emit.shape[0]
        if n == 0:
            return []
        trans = self.transitions.astype(np.float64)
        score = self.start.astype(np.float64) + emit[0]
        back = np.zeros((n, emit.shape[1]), dtype=np.int64)
        for i in range(1, n):
            cand = score[:, None] + trans
            back[i] = cand.argmax(axis=0)
            score = cand.max(axis=0) + emit[i]
        path = [int((score + self.end.astype(np.float64)).argmax())]
        for i in range(n - 1, 0, -1):
            path.append(int(back[i, path[-1]]))
        return path[::-1]

    def marginals(self, emit: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Log partition and token marginals by log-space forward-backward."""

        def lse(x: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
            m = x.max(axis=axis, keepdims=True)
            out: NDArray[np.float64] = (
                m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))
            ).squeeze(axis)
            return out

        n, t = emit.shape
        trans = self.transitions.astype(np.float64)
        alpha = np.zeros((n, t))
        beta = np.zeros((n, t))
        alpha[0] = self.start + emit[0]
        for i in range(1, n):
            alpha[i] = lse(alpha[i - 1][:, None] + trans, 0) + emit[i]
        beta[n - 1] = self.end
        for i in range(n - 2, -1, -1):
            beta[i] = lse(trans + (emit[i + 1] + beta[i + 1])[None, :], 1)
        log_z = float(lse((alpha[n - 1] + self.end)[None, :], 1)[0])
        return log_z, np.exp(alpha + beta - log_z)

    def nll(self, text: str, gold: list[int]) -> float:
        emit = self._emissions(text)
        log_z, _ = self.marginals(emit)
        score = float(self.start[gold[0]] + self.end[gold[-1]])
        score += float(sum(emit[i, g] for i, g in enumerate(gold)))
        score += float(sum(self.transitions[gold[i - 1], gold[i]] for i in range(1, len(gold))))
        return log_z - score

    def predict(self, text: str) -> list[Finding]:
        tokens = tokenize(text)
        if not tokens:
            return []
        emit = self._emissions(text)
        path = self._viterbi(emit)
        _, marg = self.marginals(emit)
        spans: list[list[float]] = []  # tok start, tok end, label, sum
        for i, tag in enumerate(path):
            if tag == 0:
                continue
            label = (tag - 1) // 2
            inside = tag % 2 == 0
            p = float(marg[i, tag])
            if spans and inside and spans[-1][1] == i and spans[-1][2] == label:
                spans[-1][1] = i + 1
                spans[-1][3] += p
            else:
                spans.append([i, i + 1, label, p])
        return [
            Finding(
                tokens[int(s)][0],
                tokens[int(e) - 1][1],
                self.labels[int(lab)],
                "ner",
                total / (e - s),
                NER_RANK,
                tokens[int(s)][0],
                tokens[int(e) - 1][1],
            )
            for s, e, lab, total in spans
        ]
