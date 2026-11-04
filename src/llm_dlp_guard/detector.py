"""The detector: the pattern catalog plus an optional CRF tagger, on the Rust or Python backend."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llm_dlp_guard.backend import extension, resolve_backend
from llm_dlp_guard.catalog import Catalog, default_catalog
from llm_dlp_guard.findings import Finding, resolve
from llm_dlp_guard.reference import ReferenceCrf, ReferenceEngine

Span = tuple[int, int, str]


Row = tuple[int, int, str, str, float, int, int, int]


def _rows(rows: Sequence[Row]) -> list[Finding]:
    return [Finding(*row) for row in rows]


def _shift(f: Finding, delta: int) -> Finding:
    ms, me = f.extent
    return Finding(
        f.start + delta, f.end + delta, f.label, f.source, f.score, f.rank, ms + delta, me + delta
    )


class Tagger:
    """The CRF tagger for names, addresses and other free-form PII."""

    def __init__(self, handle: Any, backend: str) -> None:
        self._handle = handle
        self.backend = backend

    @classmethod
    def load(cls, path: str | Path, backend: str = "auto") -> Tagger:
        chosen = resolve_backend(backend)
        if chosen == "rust":
            return cls(extension().RustCrf.load(str(path)), chosen)
        return cls(ReferenceCrf.load(path), chosen)

    @classmethod
    def train(
        cls,
        texts: Sequence[str],
        spans: Sequence[Sequence[Span]],
        labels: Sequence[str],
        *,
        n_buckets: int = 1 << 20,
        epochs: int = 3,
        learning_rate: float = 0.1,
        batch_size: int = 64,
        seed: int = 0,
        verbose: bool = False,
        threads: int = 0,
    ) -> Tagger:
        """Train with the Rust trainer (AdaGrad on the CRF log-likelihood)."""

        resolve_backend("rust")
        handle = extension().train_crf(
            list(texts),
            [list(s) for s in spans],
            list(labels),
            n_buckets=n_buckets,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=seed,
            verbose=verbose,
            threads=threads,
        )
        return cls(handle, "rust")

    @property
    def labels(self) -> list[str]:
        return list(self._handle.labels)

    @property
    def handle(self) -> Any:
        return self._handle

    def save(self, path: str | Path) -> None:
        if self.backend != "rust":
            raise RuntimeError("saving is only supported for Rust-trained models")
        self._handle.save(str(path))

    def predict(self, text: str) -> list[Finding]:
        if self.backend == "rust":
            return _rows(self._handle.predict(text))
        result: list[Finding] = self._handle.predict(text)
        return result

    def tag_path(self, text: str) -> list[int]:
        return list(self._handle.tag_path(text))


class Detector:
    """Patterns and the optional tagger, resolved to non-overlapping findings."""

    def __init__(
        self,
        catalog: Catalog | None = None,
        tagger: Tagger | None = None,
        *,
        ner_min_score: float = 0.5,
        backend: str = "auto",
    ) -> None:
        self.catalog = catalog if catalog is not None else default_catalog()
        self.backend = resolve_backend(backend)
        self.tagger = tagger
        self.ner_min_score = ner_min_score
        if tagger is not None and tagger.backend != self.backend:
            raise ValueError("the tagger and the detector must use the same backend")
        self._rust: Any = None
        self._engine: ReferenceEngine | None = None
        if self.backend == "rust":
            self._rust = extension().RustDetector(
                self.catalog.to_json(),
                tagger.handle if tagger is not None else None,
                ner_min_score,
            )
        else:
            self._engine = ReferenceEngine(self.catalog)

    @classmethod
    def from_model(
        cls,
        model: str | Path | None,
        catalog: Catalog | None = None,
        *,
        ner_min_score: float = 0.5,
        backend: str = "auto",
    ) -> Detector:
        tagger = Tagger.load(model, backend) if model else None
        return cls(catalog, tagger, ner_min_score=ner_min_score, backend=backend)

    def candidates(self, text: str, start: int = 0) -> list[Finding]:
        """Validated pattern matches and tagger spans starting at or after ``start``, before
        overlap resolution. Text before ``start`` is context for assertions such as ``\\b``."""

        if self._rust is not None:
            return _rows(self._rust.candidates(text, start))
        assert self._engine is not None
        found = self._engine.candidates(text, start)
        if self.tagger is not None:
            found += [
                _shift(f, start)
                for f in self.tagger.predict(text[start:])
                if f.score >= self.ner_min_score
            ]
        return found

    def scan(self, text: str) -> list[Finding]:
        if self._rust is not None:
            return _rows(self._rust.scan(text))
        return resolve(self.candidates(text))

    def scan_batch(self, texts: Sequence[str], threads: int = 0) -> list[list[Finding]]:
        """Scan many texts; the Rust backend spreads them over ``threads`` (0 means all cores)."""

        if self._rust is not None:
            return [_rows(r) for r in self._rust.scan_batch(list(texts), threads)]
        return [self.scan(t) for t in texts]
