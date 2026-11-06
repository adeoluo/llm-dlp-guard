"""Bootstrap and Wilson intervals used by the benchmark scripts."""

from __future__ import annotations

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llm_dlp_guard.metrics import Counts, bootstrap, micro, wilson

COUNTS = st.builds(Counts, st.integers(0, 20), st.integers(0, 20), st.integers(0, 20))


def test_identical_documents_give_a_point_interval() -> None:
    docs = [Counts(3, 1, 2)] * 50
    ci = bootstrap(docs, resamples=200)
    point = micro({str(i): d for i, d in enumerate(docs)})
    for name in ("precision", "recall", "f1"):
        lo, hi = ci[name]
        assert lo == hi == pytest.approx(round(getattr(point, name), 4))


@settings(max_examples=40, deadline=None)
@given(st.lists(COUNTS, min_size=2, max_size=60), st.integers(0, 2**31))
def test_interval_is_ordered_and_in_unit_range(docs: list[Counts], seed: int) -> None:
    ci = bootstrap(docs, resamples=100, seed=seed)
    for name in ("precision", "recall", "f1"):
        lo, hi = ci[name]
        assert 0.0 <= lo <= hi <= 1.0


def test_interval_contains_the_point_estimate_and_narrows_with_more_documents() -> None:
    rng = random.Random(0)

    def docs(n: int) -> list[Counts]:
        return [Counts(rng.randint(0, 5), rng.randint(0, 2), rng.randint(0, 2)) for _ in range(n)]

    small, large = docs(100), docs(5000)
    for sample in (small, large):
        total = Counts()
        for d in sample:
            total.add(d)
        lo, hi = bootstrap(sample, resamples=500)["f1"]
        assert lo <= total.f1 <= hi
    width = lambda d: bootstrap(d, resamples=500)["f1"][1] - bootstrap(d, resamples=500)["f1"][0]  # noqa: E731
    assert width(large) < width(small)


def test_same_seed_same_interval() -> None:
    docs = [Counts(i % 4, i % 3, i % 2) for i in range(300)]
    assert bootstrap(docs, seed=7) == bootstrap(docs, seed=7)


def test_paired_against_itself_has_zero_difference() -> None:
    docs = [Counts(i % 5, i % 2, i % 3) for i in range(200)]
    ci = bootstrap(docs, resamples=300, other=docs)
    assert ci["f1_difference"] == [0.0, 0.0]
    assert ci["share_not_better"] == [1.0]


def test_paired_clear_winner() -> None:
    good = [Counts(5, 0, 0) for _ in range(200)]
    bad = [Counts(2, 3, 3) for _ in range(200)]
    ci = bootstrap(good, resamples=300, other=bad)
    assert ci["f1_difference"][0] > 0.5
    assert ci["share_not_better"] == [0.0]


def test_bootstrap_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        bootstrap([])
    with pytest.raises(ValueError):
        bootstrap([Counts(1, 0, 0)], other=[Counts(1, 0, 0), Counts(0, 1, 0)])


@pytest.mark.parametrize(
    ("k", "n", "expected"),
    [
        # Reference values from the closed form (Wilson 1927), as in statsmodels proportion_confint.
        (0, 10, (0.0, 0.2775)),
        (5, 10, (0.2366, 0.7634)),
        (10, 10, (0.7225, 1.0)),
        (81, 263, (0.2553, 0.3662)),
    ],
)
def test_wilson_reference_values(k: int, n: int, expected: tuple[float, float]) -> None:
    lo, hi = wilson(k, n)
    assert lo == pytest.approx(expected[0], abs=1e-4)
    assert hi == pytest.approx(expected[1], abs=1e-4)


@given(st.integers(0, 10_000), st.integers(1, 10_000))
def test_wilson_contains_the_proportion(k: int, n: int) -> None:
    k = min(k, n)
    lo, hi = wilson(k, n)
    assert lo <= k / n + 1e-4 and k / n - 1e-4 <= hi
    assert wilson(0, 0) == [0.0, 0.0]
