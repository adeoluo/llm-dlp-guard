from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_dlp_guard import datasets, metrics
from llm_dlp_guard import secrets_corpus as sc
from llm_dlp_guard.detector import Detector
from llm_dlp_guard.validators import github_checksum_valid, jwt_valid


def test_exact_and_overlap_scoring() -> None:
    gold = [(0, 4, "A"), (10, 15, "B")]
    pred = [(0, 4, "A"), (9, 15, "B"), (20, 22, "A")]
    exact = metrics.score_document(gold, pred, "exact")
    assert (exact["A"].tp, exact["A"].fp, exact["B"].fn, exact["B"].fp) == (1, 1, 1, 1)
    overlap = metrics.score_document(gold, pred, "overlap")
    assert (overlap["B"].tp, overlap["B"].fp, overlap["A"].fp) == (1, 0, 1)
    ignored = metrics.score_document(gold, pred, "overlap", ignore=[(19, 23)])
    assert ignored["A"].fp == 0
    with pytest.raises(ValueError):
        metrics.score_document(gold, pred, "fuzzy")
    total: dict[str, metrics.Counts] = {}
    metrics.accumulate(total, exact)
    metrics.accumulate(total, exact)
    m = metrics.micro(total)
    assert (m.tp, m.fp, m.fn) == (2, 4, 2)
    assert m.precision == pytest.approx(1 / 3) and m.recall == pytest.approx(0.5)
    assert m.f1 == pytest.approx(0.4) and m.to_dict()["f1"] == 0.4
    assert metrics.macro_f1(total) == pytest.approx((2 / 3 + 0) / 2)
    assert metrics.Counts().f1 == 0.0 and metrics.macro_f1({}) == 0.0


def test_label_mapping_and_merging() -> None:
    text = "John  Smith, Bob"
    spans = [(0, 4, "PERSON"), (6, 11, "PERSON"), (13, 16, "PERSON")]
    assert metrics.merge_adjacent(text, spans) == [(0, 11, "PERSON"), (13, 16, "PERSON")]
    assert metrics.map_labels([(0, 1, "GIVEN_NAME"), (1, 2, "X")], datasets.OURS_TO_COARSE) == [
        (0, 1, "PERSON")
    ]
    assert set(datasets.AI4PRIVACY_TO_COARSE.values()) <= set(datasets.COARSE_TYPES)
    assert set(datasets.NEMOTRON_TO_COARSE.values()) <= set(datasets.COARSE_TYPES)
    assert set(datasets.PRESIDIO_TO_COARSE.values()) <= set(datasets.COARSE_TYPES)
    assert set(datasets.TAGGER_LABELS) <= set(datasets.OURS_TO_COARSE)


def test_load_ai4privacy(tmp_path: Path) -> None:
    root = tmp_path / "validation"
    root.mkdir()
    rows = [
        {
            "source_text": "Hi Ana",
            "language": "en",
            "uid": 1,
            "privacy_mask": [
                {"start": 3, "end": 6, "label": "GIVENNAME"},
                {"start": 0, "end": 2, "label": "OTHER"},
            ],
        },
        {"source_text": "Hallo", "language": "en", "uid": 2, "privacy_mask": []},
    ]
    (root / "1en.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    recs = list(datasets.load_ai4privacy(tmp_path, "validation", ("en",)))
    assert recs[0].spans == [(3, 6, "GIVEN_NAME")] and recs[0].uid == "1"
    assert len(list(datasets.load_ai4privacy(tmp_path, "validation", ("en",), limit=1))) == 1
    assert datasets.ai4privacy_files(tmp_path, "train")[1].name == "de.jsonl"


def test_load_nemotron(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    spans = str(
        [
            {"start": 0, "end": 4, "label": "first_name"},
            {"start": 5, "end": 9, "label": "company_name"},
        ]
    )
    table = pa.table(
        {
            "uid": ["a", "a", "b"],
            "text": ["Lena Acme", "Lena Acme", "x"],
            "spans": [spans, spans, "[]"],
            "locale": ["us", "us", "intl"],
        }
    )
    path = tmp_path / "t.parquet"
    pq.write_table(table, path)
    recs = list(datasets.load_nemotron(path))
    assert len(recs) == 2
    assert recs[0].spans == [(0, 4, "PERSON")] and recs[0].ignore == [(5, 9)]
    assert len(list(datasets.load_nemotron(path, limit=1))) == 1


def test_corpus_formats_and_determinism() -> None:
    a = list(sc.generate(len(sc._KINDS) * 4, 50, seed=1))
    b = list(sc.generate(len(sc._KINDS) * 4, 50, seed=1))
    assert a == b
    for s in a:
        for start, end, label in s.spans:
            value = s.text[start:end]
            if label == "GITHUB_TOKEN" and value.startswith("gh") and "_pat_" not in value:
                assert github_checksum_valid(value)
            if label == "JWT":
                assert jwt_valid(value)
    assert sum(1 for s in a if s.kind == "negative") == 50
    assert "PRIVATE_KEY" in sc.kinds()
    assert json.loads(a[0].to_json())["kind"] == a[0].kind


def test_detector_finds_every_generated_secret() -> None:
    det = Detector()
    missed = []
    for s in sc.generate(len(sc._KINDS) * 20, 0, seed=8):
        ((start, end, _),) = s.spans
        if not any(f.start < end and start < f.end for f in det.scan(s.text)):
            missed.append(s.text)
    assert len(missed) <= 2, missed
