from __future__ import annotations

from pathlib import Path

import pytest

from llm_dlp_guard.detector import Detector, Tagger
from llm_dlp_guard.reference import ReferenceCrf

from .conftest import TAGGER_TEST_LABELS, make_records, requires_rust

pytestmark = requires_rust


def test_tagger_learns_held_out_sentences(tagger_path: Path) -> None:
    tagger = Tagger.load(tagger_path)
    assert tagger.labels == TAGGER_TEST_LABELS
    texts, spans = make_records(200, seed=42)
    hit = total = 0
    for text, gold in zip(texts, spans, strict=True):
        pred = {(f.start, f.end, f.label) for f in tagger.predict(text)}
        hit += len(pred & set(gold))
        total += len(gold)
    assert hit / total > 0.95


def test_scores_are_probabilities(tagger_path: Path) -> None:
    found = Tagger.load(tagger_path).predict("Dear Hana, your appointment in Cork is confirmed.")
    assert [f.label for f in found] == ["GIVEN_NAME", "CITY"]
    assert all(0.5 < f.score <= 1.0 + 1e-9 for f in found)
    assert all(f.source == "ner" for f in found)


def test_save_requires_rust_backend(tagger_path: Path, tmp_path: Path) -> None:
    tagger = Tagger.load(tagger_path, backend="python")
    with pytest.raises(RuntimeError):
        tagger.save(tmp_path / "x.crf")
    assert tagger.backend == "python"
    assert isinstance(tagger.handle, ReferenceCrf)


def test_load_rejects_other_files(tmp_path: Path) -> None:
    bad = tmp_path / "bad.crf"
    bad.write_bytes(b"not a model at all")
    with pytest.raises(OSError):
        Tagger.load(bad, backend="rust")
    with pytest.raises(ValueError):
        Tagger.load(bad, backend="python")


def test_training_argument_checks() -> None:
    with pytest.raises(ValueError):
        Tagger.train(["a"], [], ["X"])
    with pytest.raises(ValueError):
        Tagger.train(["a"], [[]], [])


def test_training_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    texts, spans = make_records(80, seed=5)
    a = Tagger.train(
        texts, spans, TAGGER_TEST_LABELS, n_buckets=1 << 10, epochs=1, seed=3, threads=1
    )
    b = Tagger.train(
        texts, spans, TAGGER_TEST_LABELS, n_buckets=1 << 10, epochs=1, seed=3, threads=1
    )
    a.save(tmp_path / "a.crf")
    b.save(tmp_path / "b.crf")
    assert (tmp_path / "a.crf").read_bytes() == (tmp_path / "b.crf").read_bytes()


def test_detector_merges_tagger_and_patterns(tagger_path: Path) -> None:
    det = Detector.from_model(tagger_path)
    text = "Ship the parcel to Ivo Novak, 12 Mill Lane, Brno. Mail ivo@example.org"
    labels = [f.label for f in det.scan(text)]
    assert labels[:2] == ["GIVEN_NAME", "SURNAME"]
    assert "EMAIL" in labels
    with pytest.raises(ValueError):
        Detector(tagger=Tagger.load(tagger_path, backend="python"), backend="rust")


@pytest.mark.parametrize("backend", ["rust", "python"])
def test_structured_matches_win_over_tagger_spans(tagger_path: Path, backend: str) -> None:
    det = Detector.from_model(tagger_path, backend=backend)
    structured = {"credit_card", "iban", "ipv4", "ipv6"}
    for value in ["81.2.69.160", "4111 1111 1111 1111", "GB82 WEST 1234 5698 7654 32", "fe80::1"]:
        text = f"Ship the parcel to Ivo Novak, {value} Mill Lane, Brno."
        found = det.candidates(text)
        spans = [(f.start, f.end) for f in found if f.source in structured]
        assert spans, value
        ner = [f for f in found if f.source == "ner"]
        assert not [f for f in ner for s, e in spans if s < f.end and f.start < e]
        start = text.index(value)
        assert (start, start + len(value)) in [(f.start, f.end) for f in det.scan(text)]
