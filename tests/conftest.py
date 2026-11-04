from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from llm_dlp_guard.backend import rust_available
from llm_dlp_guard.detector import Tagger

if os.environ.get("DLP_REQUIRE_RUST") == "1" and not rust_available():  # pragma: no cover
    raise RuntimeError("DLP_REQUIRE_RUST=1 but the Rust extension is not importable")

requires_rust = pytest.mark.skipif(not rust_available(), reason="Rust extension not built")

GIVEN = [
    "Alice",
    "Bram",
    "Chen",
    "Dana",
    "Emil",
    "Farah",
    "Goran",
    "Hana",
    "Ivo",
    "Jun",
    "Kemal",
    "Lena",
]
SURNAMES = [
    "Moreau",
    "Okafor",
    "Liang",
    "Ruiz",
    "Varga",
    "Haddad",
    "Novak",
    "Sato",
    "Berg",
    "Kowalski",
]
CITIES = ["Lyon", "Utrecht", "Porto", "Graz", "Turku", "Brno", "Cork", "Gent", "Bergen", "Split"]
STREETS = ["Elm Street", "Harbour Road", "Mill Lane", "Station Avenue", "Kingsway"]
TEMPLATES = [
    "Please ask {g} {s} to visit {c} next week.",
    "Ship the parcel to {g} {s}, {n} {st}, {c}.",
    "Dear {g}, your appointment in {c} is confirmed.",
    "{g} {s} moved to {n} {st} in {c} last year.",
    "Contact {g} about the {c} office.",
]


def make_records(n: int, seed: int = 0) -> tuple[list[str], list[list[tuple[int, int, str]]]]:
    """Fictional sentences with GIVEN_NAME, SURNAME, CITY, STREET and BUILDING_NUMBER spans."""

    rng = random.Random(seed)
    texts, spans = [], []
    for _ in range(n):
        template = rng.choice(TEMPLATES)
        values = {
            "g": (rng.choice(GIVEN), "GIVEN_NAME"),
            "s": (rng.choice(SURNAMES), "SURNAME"),
            "c": (rng.choice(CITIES), "CITY"),
            "st": (rng.choice(STREETS), "STREET"),
            "n": (str(rng.randint(1, 250)), "BUILDING_NUMBER"),
        }
        text, sp = "", []
        rest = template
        while "{" in rest:
            before, after = rest.split("{", 1)
            key, rest = after.split("}", 1)
            text += before
            value, label = values[key]
            sp.append((len(text), len(text) + len(value), label))
            text += value
        text += rest
        texts.append(text)
        spans.append(sp)
    return texts, spans


TAGGER_TEST_LABELS = ["GIVEN_NAME", "SURNAME", "CITY", "STREET", "BUILDING_NUMBER"]


@pytest.fixture(scope="session")
def tagger_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not rust_available():  # pragma: no cover
        pytest.skip("training needs the Rust extension")
    texts, spans = make_records(600, seed=1)
    tagger = Tagger.train(
        texts,
        spans,
        TAGGER_TEST_LABELS,
        n_buckets=1 << 14,
        epochs=4,
        learning_rate=0.2,
        batch_size=16,
        seed=0,
    )
    path = tmp_path_factory.mktemp("model") / "tagger.crf"
    tagger.save(path)
    return path
