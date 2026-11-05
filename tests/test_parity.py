"""The Rust core and the Python reference return identical results."""

from __future__ import annotations

import random
import string
import unicodedata
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llm_dlp_guard import reference
from llm_dlp_guard import secrets_corpus as sc
from llm_dlp_guard.backend import extension
from llm_dlp_guard.detector import Detector, Tagger

from .conftest import TAGGER_TEST_LABELS, make_records, requires_rust

pytestmark = requires_rust

# Python (str.isalnum, re's \\w) and Rust (char::is_alphanumeric, regex's \\w) agree on which
# characters are word characters except for combining marks (M*), other numbers such as '¼' (No),
# connector punctuation other than '_' (Pc), the zero-width joiners and the circled and squared
# Latin letters (Other_Alphabetic symbols), and on code points newer than Python's Unicode
# database (Cn to Python). Texts here leave those out; see docs/ARCHITECTURE.md.
_OTHER_ALPHABETIC_SYMBOLS = "".join(
    chr(c)
    for r in ((0x24B6, 0x24EA), (0x1F130, 0x1F14A), (0x1F150, 0x1F16A), (0x1F170, 0x1F18A))
    for c in range(*r)
)
ALPHABET = st.characters(
    blacklist_categories=("Mn", "Mc", "Me", "No", "Pc", "Cs", "Cn"),
    blacklist_characters="\x1c\x1d\x1e\x1f\u200c\u200d" + _OTHER_ALPHABETIC_SYMBOLS,
)


def _fragments(seed: int) -> list[str]:
    rng = random.Random(seed)
    gens = [
        sc.aws_access_key, sc.github_token, sc.slack_token, sc.stripe_key, sc.google_api_key,
        sc.openai_key, sc.jwt, sc.sendgrid_key, sc.twilio_key, sc.huggingface_token,
    ]  # fmt: skip
    out = [g(rng) for g in gens]
    out += [
        "jane.doe@example.org", "4111 1111 1111 1111", "GB82 WEST 1234 5698 7654 32", "123-45-6789",
        "+44 20 7946 0958", "(415) 555-0100", "8.8.8.8", "10.0.0.1", "2001:db8::1", "fe80::1",
        "password=q8Zr2LmP0xTv9KcW", "token: abc", "postgres://u:p4ssW0rd@h/db", "AKIA", "eyJ", "@", ":",
        "-", " ", "\n", "Zoë", "東京", "١٢٣", "Ⅻ", "\u017fecret", "İstanbul",
    ]  # fmt: skip
    return out


TEXTS = st.lists(
    st.one_of(st.sampled_from(_fragments(0)), st.text(ALPHABET, max_size=12)), max_size=12
).map("".join)


@settings(max_examples=400, deadline=None)
@given(TEXTS)
def test_pattern_candidates_identical(text: str) -> None:
    rust = Detector(backend="rust")
    py = Detector(backend="python")
    assert sorted(rust.candidates(text), key=repr) == sorted(py.candidates(text), key=repr)
    assert rust.scan(text) == py.scan(text)


@settings(max_examples=200, deadline=None)
@given(TEXTS, st.integers(0, 40))
def test_candidates_from_offset_identical(text: str, start: int) -> None:
    start = min(start, len(text))
    rust = Detector(backend="rust").candidates(text, start)
    py = Detector(backend="python").candidates(text, start)
    assert sorted(rust, key=repr) == sorted(py, key=repr)
    assert all(f.start >= start for f in py)


def test_secrets_corpus_identical() -> None:
    rust = Detector(backend="rust")
    py = Detector(backend="python")
    for sample in sc.generate(400, 400, seed=11):
        assert rust.scan(sample.text) == py.scan(sample.text), sample.text


@settings(max_examples=300, deadline=None)
@given(st.text(ALPHABET, max_size=40))
def test_tokenizer_identical(text: str) -> None:
    assert [tuple(t) for t in extension().tokenize(text)] == reference.tokenize(text)


def test_tokenizer_examples() -> None:
    text = "Hi, Zoë! a_b 42.5\u00a0x"
    assert [text[s:e] for s, e in reference.tokenize(text)] == [
        "Hi",
        ",",
        "Zoë",
        "!",
        "a_b",
        "42",
        ".",
        "5",
        "x",
    ]


@settings(max_examples=150, deadline=None)
@given(st.text(ALPHABET, max_size=40), st.data())
def test_gold_tags_identical(text: str, data: st.DataObject) -> None:
    labels = ["A", "B"]
    n = len(text)
    spans = data.draw(
        st.lists(
            st.tuples(st.integers(0, n), st.integers(0, n), st.sampled_from(["A", "B", "Z"])).map(
                lambda t: (min(t[0], t[1]), max(t[0], t[1]), t[2])
            ),
            max_size=4,
        )
    )
    assert list(extension().gold_tags(text, spans, labels)) == reference.gold_tags(
        text, spans, labels
    )


@pytest.fixture(scope="module")
def models(tagger_path: Path) -> tuple[Tagger, reference.ReferenceCrf]:
    return Tagger.load(tagger_path, "rust"), reference.ReferenceCrf.load(tagger_path)


@settings(max_examples=150, deadline=None)
@given(text=st.text(ALPHABET, max_size=30))
def test_feature_ids_identical(models: tuple[Tagger, reference.ReferenceCrf], text: str) -> None:
    rust, py = models
    assert [list(f) for f in rust.handle.feature_ids(text)] == py.feature_ids(text)


def test_tagger_paths_and_predictions_identical(
    models: tuple[Tagger, reference.ReferenceCrf],
) -> None:
    rust, py = models
    texts, _ = make_records(200, seed=99)
    texts += ["", "   ", "Zoë Ünal lives in Graz.", "Bram", "\u00e9" * 5, "日本語 テキスト"]
    for text in texts:
        assert rust.tag_path(text) == py.tag_path(text), text
        a, b = rust.predict(text), py.predict(text)
        assert [(f.start, f.end, f.label) for f in a] == [(f.start, f.end, f.label) for f in b]
        for x, y in zip(a, b, strict=True):
            assert x.score == pytest.approx(y.score, abs=1e-5)


def test_nll_matches_reference_and_brute_force(
    models: tuple[Tagger, reference.ReferenceCrf],
) -> None:
    rust, py = models
    text = "Ask Lena Sato"
    gold = extension().gold_tags(
        text, [(4, 8, "GIVEN_NAME"), (9, 13, "SURNAME")], TAGGER_TEST_LABELS
    )
    a = rust.handle.nll(text, list(gold))
    b = py.nll(text, list(gold))
    assert a == pytest.approx(b, rel=1e-6)
    # Brute force: log Z over every tag sequence of the 3 tokens.
    import itertools

    import numpy as np

    emit = py._emissions(text)
    t = emit.shape[1]
    scores = []
    for seq in itertools.product(range(t), repeat=3):
        s = py.start[seq[0]] + py.end[seq[-1]] + sum(emit[i, y] for i, y in enumerate(seq))
        s += sum(py.transitions[seq[i - 1], seq[i]] for i in range(1, 3))
        scores.append(float(s))
    log_z = float(np.logaddexp.reduce(scores))
    gold_score = float(
        py.start[gold[0]]
        + py.end[gold[-1]]
        + sum(emit[i, y] for i, y in enumerate(gold))
        + sum(py.transitions[gold[i - 1], gold[i]] for i in range(1, 3))
    )
    assert a == pytest.approx(log_z - gold_score, rel=1e-6)
    with pytest.raises(ValueError):
        rust.handle.nll(text, [0])


def test_combined_detector_identical(tagger_path: Path) -> None:
    rust = Detector.from_model(tagger_path, backend="rust")
    py = Detector.from_model(tagger_path, backend="python")
    texts, _ = make_records(50, seed=3)
    texts = [t + " mail dana.ruiz@example.org card 4111 1111 1111 1111" for t in texts]
    for text in texts:
        a, b = rust.scan(text), py.scan(text)
        assert [(f.start, f.end, f.label, f.source) for f in a] == [
            (f.start, f.end, f.label, f.source) for f in b
        ]


def test_unicode_categories_assumption() -> None:
    # The excluded categories are exactly where the engines' word definitions can differ.
    assert unicodedata.category("\u0301") == "Mn"
    assert string.ascii_letters.isalpha()


@pytest.mark.parametrize(
    "text",
    [
        "İstanbulpassword=q8Zr2LmP0xTv9KcW",
        "ap\u0131_key=q8Zr2LmP0xTv9KcW",
        "Authorİzation: Bearer abc",
    ],
)
def test_turkic_i_is_not_a_case_variant_of_i(text: str) -> None:
    rust = Detector(backend="rust").candidates(text)
    assert sorted(rust, key=repr) == sorted(Detector(backend="python").candidates(text), key=repr)
