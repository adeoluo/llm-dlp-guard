from __future__ import annotations

import random

import pytest

from llm_dlp_guard import secrets_corpus as sc
from llm_dlp_guard.backend import resolve_backend
from llm_dlp_guard.catalog import Catalog, Pattern, default_catalog
from llm_dlp_guard.detector import Detector
from llm_dlp_guard.findings import Finding, resolve
from llm_dlp_guard.reference import ReferenceEngine

from .conftest import requires_rust

BACKENDS = ["python", pytest.param("rust", marks=requires_rust)]
RNG = random.Random(7)

POSITIVES = [
    ("write to jane.doe@example.org today", "EMAIL", "jane.doe@example.org"),
    ("card 4111 1111 1111 1111 exp 09/27", "CREDIT_CARD", "4111 1111 1111 1111"),
    ("amex 378282246310005.", "CREDIT_CARD", "378282246310005"),
    ("IBAN GB82 WEST 1234 5698 7654 32 please", "IBAN", "GB82 WEST 1234 5698 7654 32"),
    ("ssn 123-45-6789", "US_SSN", "123-45-6789"),
    ("call +44 20 7946 0958 now", "PHONE", "+44 20 7946 0958"),
    ("call (415) 555-0100 now", "PHONE", "(415) 555-0100"),
    ("server 8.8.4.4 is down", "IP_ADDRESS", "8.8.4.4"),
    ("v6 2001:4860:4860::8888 ok", "IP_ADDRESS", "2001:4860:4860::8888"),
    ("key " + sc.aws_access_key(RNG), "AWS_ACCESS_KEY", None),
    ("token " + sc.github_token(RNG) + " end", "GITHUB_TOKEN", None),
    ("x " + sc.github_pat(RNG), "GITHUB_TOKEN", None),
    ("t=" + sc.gitlab_token(RNG) + ";", "GITLAB_TOKEN", None),
    ("slack " + sc.slack_token(RNG), "SLACK_TOKEN", None),
    ("hook " + sc.slack_webhook(RNG), "SLACK_WEBHOOK", None),
    ("stripe " + sc.stripe_key(RNG), "STRIPE_KEY", None),
    ("g " + sc.google_api_key(RNG), "GOOGLE_API_KEY", None),
    ("o " + sc.openai_key(RNG), "OPENAI_API_KEY", None),
    ("a " + sc.anthropic_key(RNG), "ANTHROPIC_API_KEY", None),
    ("hf " + sc.huggingface_token(RNG), "HUGGINGFACE_TOKEN", None),
    ("npm " + sc.npm_token(RNG), "NPM_TOKEN", None),
    ("sg " + sc.sendgrid_key(RNG), "SENDGRID_API_KEY", None),
    ("tw " + sc.twilio_key(RNG), "TWILIO_API_KEY", None),
    ("AccountKey=" + sc.azure_key(RNG) + ";", "AZURE_STORAGE_KEY", None),
    ('{"private_key_id": "' + sc.gcp_key_id(RNG) + '"}', "GCP_KEY_ID", None),
    ("jwt " + sc.jwt(RNG), "JWT", None),
    (sc.private_key(RNG), "PRIVATE_KEY", None),
    ("aws_secret_access_key = " + sc.aws_secret_key(RNG), "AWS_SECRET_KEY", None),
    ("Authorization: Bearer " + sc.bearer(RNG), "BEARER_TOKEN", None),
    ("postgres://app:Zq8rT2mPx@db.internal:5432/x", "PASSWORD", "Zq8rT2mPx"),
    ("DB_PASSWORD=q8Zr2LmP0xTv9KcW", "GENERIC_SECRET", "q8Zr2LmP0xTv9KcW"),
    ("My password is: Tr0ub4dor&3", "GENERIC_SECRET", "Tr0ub4dor&3"),
]

NEGATIVES = [
    "card 4111 1111 1111 1112",
    "account 883442237549304148",
    "ssn 666-45-6789",
    "version 1.02.3.4 and 999.1.1.1",
    "time 12:30:45",
    "commit 3f786850e387550fdab836ed7e6dc881de23001b",
    "password: ${DB_PASSWORD}",
    "api_key = YOUR_API_KEY",
    "The password must be at least 12 characters long.",
    "jwt eyJhbGciOiJIUzI1NiJ9.bm90IGpzb24.c2ln",
    "IBAN GB82 WEST 1234 5698 7654 33",
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("text", "label", "value"), POSITIVES)
def test_positive(backend: str, text: str, label: str, value: str | None) -> None:
    found = Detector(backend=backend).scan(text)
    labels = [f.label for f in found]
    assert label in labels, (text, found)
    if value is not None:
        hit = next(f for f in found if f.label == label)
        assert hit.text(text) == value


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("text", NEGATIVES)
def test_negative(backend: str, text: str) -> None:
    found = [f for f in Detector(backend=backend).scan(text) if f.score >= 0.5]
    assert found == [], (text, found)


def test_github_checksum_affects_score() -> None:
    token = sc.github_token(random.Random(1))
    bad = token[:-1] + ("A" if token[-1] != "A" else "B")
    good_score = Detector(backend="python").scan(token)[0].score
    bad_score = Detector(backend="python").scan(bad)[0].score
    assert good_score == pytest.approx(0.99)
    assert bad_score == pytest.approx(0.99 * 0.8)


def test_private_ip_scores_lower() -> None:
    det = Detector(backend="python")
    assert det.scan("at 10.0.0.7")[0].score == pytest.approx(0.48)
    assert det.scan("at 8.8.8.8")[0].score == pytest.approx(0.8)


def test_prefilter_selects_only_anchored_patterns() -> None:
    engine = ReferenceEngine(default_catalog())
    active = {engine.catalog.patterns[i].name for i in engine.active("plain words only")}
    assert "aws_access_key_id" not in active
    assert "credit_card" in active
    assert "aws_access_key_id" in {engine.catalog.patterns[i].name for i in engine.active("akia")}


def test_resolve_rules() -> None:
    a = Finding(0, 10, "A", "a", 0.5, 0)
    b = Finding(5, 12, "B", "b", 0.9, 1)
    c = Finding(12, 14, "C", "c", 0.1, 2)
    assert resolve([a, b, c]) == [b, c]
    d = Finding(0, 4, "D", "d", 0.5, 3)
    e = Finding(0, 8, "E", "e", 0.5, 4)
    assert resolve([d, e]) == [e]
    assert resolve([]) == []
    assert Finding(1, 3, "X", "x", 1.0).text("abcd") == "bc"
    assert Finding(1, 3, "X", "x", 0.123456).to_dict()["score"] == 0.1235


def test_catalog_api_and_errors() -> None:
    cat = default_catalog()
    assert cat.by_name("email").label == "EMAIL"
    with pytest.raises(KeyError):
        cat.by_name("missing")
    assert "EMAIL" in cat.labels()
    assert cat.categories()["JWT"] == "secret"
    with pytest.raises(ValueError, match="duplicate"):
        cat.extended([cat.patterns[0]])
    assert cat.by_name("private_key").max_length() is not None
    assert Pattern("u", "U", "pii", "a+", 0.5).max_length() is None
    with pytest.raises(ValueError, match="unknown validator"):
        ReferenceEngine(Catalog([Pattern("x", "X", "pii", "a", 0.5, validator="nope")]))
    with pytest.raises(ValueError, match="no group"):
        ReferenceEngine(Catalog([Pattern("x", "X", "pii", "a", 0.5, group="g")]))


@requires_rust
def test_rust_rejects_bad_catalogs() -> None:
    for bad in (
        Catalog([Pattern("x", "X", "pii", "(", 0.5)]),
        Catalog([Pattern("x", "X", "pii", "a", 0.5, validator="nope")]),
        Catalog([Pattern("x", "X", "pii", "a", 0.5, group="g")]),
        Catalog([Pattern("x", "X", "pii", "a", 1.5)]),
        Catalog([Pattern("x", "X", "pii", "a", 0.5, anchors=("",))]),
    ):
        with pytest.raises(ValueError):
            Detector(bad, backend="rust")


def test_backend_resolution() -> None:
    assert resolve_backend("python") == "python"
    assert resolve_backend("auto") in ("python", "rust")
    with pytest.raises(ValueError):
        resolve_backend("gpu")


@pytest.mark.parametrize("backend", BACKENDS)
def test_scan_batch_matches_scan(backend: str) -> None:
    det = Detector(backend=backend)
    texts = [s.text for s in sc.generate(60, 20, seed=5)]
    assert det.scan_batch(texts, threads=2) == [det.scan(t) for t in texts]


@pytest.mark.parametrize("backend", BACKENDS)
def test_offsets_are_characters(backend: str) -> None:
    text = "Zoë Ünal → mail zoe.u@example.com and 8.8.8.8"
    found = Detector(backend=backend).scan(text)
    assert [f.text(text) for f in found] == ["zoe.u@example.com", "8.8.8.8"]
