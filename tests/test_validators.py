from __future__ import annotations

import random
import string

import pytest
from hypothesis import given
from hypothesis import strategies as st

from llm_dlp_guard import validators as v
from llm_dlp_guard.backend import extension, rust_available

from .conftest import requires_rust


def luhn_complete(prefix: str) -> str:
    total = 0
    for i, c in enumerate(reversed(prefix)):
        d = int(c)
        if i % 2 == 0:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return prefix + str((10 - total % 10) % 10)


def make_iban(country: str, body: str) -> str:
    numeric = "".join(str(int(c, 36)) for c in (body + country + "00"))
    return f"{country}{98 - int(numeric) % 97:02d}{body}"


def test_luhn_known_values() -> None:
    assert v.luhn_valid("4111 1111 1111 1111")
    assert v.luhn_valid("378282246310005")
    assert not v.luhn_valid("4111 1111 1111 1112")
    assert not v.luhn_valid("1111111111111111")
    assert not v.luhn_valid("12345")
    assert not v.luhn_valid("4111/1111/1111/1111")


@given(st.text(alphabet="0123456789", min_size=11, max_size=18))
def test_luhn_completion_is_valid_and_single_digit_change_breaks_it(prefix: str) -> None:
    number = luhn_complete(prefix)
    if len(set(number)) == 1:
        return
    assert v.luhn_valid(number)
    last = int(number[-1])
    assert not v.luhn_valid(number[:-1] + str((last + 1) % 10))


def test_iban() -> None:
    assert v.iban_valid("GB82 WEST 1234 5698 7654 32")
    assert v.iban_valid("DE89370400440532013000")
    assert not v.iban_valid("DE89370400440532013001")
    assert not v.iban_valid("XX89370400440532013000")
    assert not v.iban_valid("DE8937040044053201300")
    assert not v.iban_valid("DE893704004405320130é0")


@given(st.sampled_from(sorted(v.IBAN_LENGTHS)), st.randoms())
def test_constructed_ibans_validate(country: str, rnd: random.Random) -> None:
    length = v.IBAN_LENGTHS[country]
    body = "".join(rnd.choice(string.digits + string.ascii_uppercase) for _ in range(length - 4))
    iban = make_iban(country, body)
    assert v.iban_valid(iban)
    assert v.iban_valid(" ".join(iban[i : i + 4] for i in range(0, len(iban), 4)))


def test_ip_factors() -> None:
    assert v.ipv4_factor("8.8.8.8") == 1.0
    assert v.ipv4_factor("10.1.2.3") == 0.6
    assert v.ipv4_factor("172.20.0.1") == 0.6
    assert v.ipv4_factor("203.0.113.9") == 0.6
    assert v.ipv4_factor("0.0.0.0") is None
    assert v.ipv4_factor("1.2.3") is None
    assert v.ipv4_factor("1.2.3.04") is None
    assert v.ipv4_factor("1.2.3.256") is None
    assert v.ipv4_factor("1.2.3.4.5") is None
    assert v.ipv6_factor("2001:db8::1") == 1.0
    assert v.ipv6_factor("::1") == 0.6
    assert v.ipv6_factor("fd12:3456::1") == 0.6
    assert v.ipv6_factor("::") is None
    assert v.ipv6_factor("1:2") is None
    assert v.ipv6_factor("12:30:45") is None
    assert v.ipv6_factor("g::1") is None


def test_jwt_structure() -> None:
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJl"
    assert v.jwt_valid(token)
    assert v.jwt_valid(token.rsplit(".", 1)[0] + ".")
    assert not v.jwt_valid("eyJ0eXAiOiJKV1QifQ.eyJzdWIiOiIxIn0.c2ln")  # no alg
    assert not v.jwt_valid("a.b")
    assert not v.jwt_valid("eyJhbGciOiJIUzI1NiJ9.e!.c2ln")
    assert not v.jwt_valid("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c!")
    assert not v.jwt_valid("eyJhbGciOiJIUzI1NiJ9.bm90IGpzb24.c2ln")
    assert not v.jwt_valid("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0X.c2ln")


@given(st.text(alphabet=string.ascii_letters + string.digits, min_size=30, max_size=30))
def test_github_checksum_round_trip(random_part: str) -> None:
    token = "ghp_" + random_part + v.github_checksum(random_part)
    assert v.github_checksum_valid(token)
    flipped = "ghp_" + ("A" if random_part[0] != "A" else "B") + random_part[1:] + token[-6:]
    assert not v.github_checksum_valid(flipped)
    assert not v.github_checksum_valid("ghp_short")


def test_ssn_phone_entropy_secret_values() -> None:
    assert v.ssn_valid("123-45-6789")
    assert not v.ssn_valid("000-12-3456")
    assert not v.ssn_valid("923-12-3456")
    assert not v.ssn_valid("123-45-0000")
    assert not v.ssn_valid("12-345-678")
    assert v.phone_factor("+44 20 7946 0958") == 1.0
    assert v.phone_factor("+1 23") is None
    assert v.shannon_entropy("") == 0.0
    assert v.shannon_entropy("aaaa") == 0.0
    assert abs(v.shannon_entropy("abcd") - 2.0) < 1e-12
    assert v.secret_value_factor("") is None
    assert v.secret_value_factor("${TOKEN}") is None
    assert v.secret_value_factor("changeme") is None
    assert v.secret_value_factor("os.environ[") is None
    assert v.secret_value_factor("process.env.TOKEN") is None
    assert v.secret_value_factor("hunter2!") == 0.7
    assert v.secret_value_factor("q8Zr2LmP0xTv9KcW") == 1.0
    assert v.secret_value_factor("aaaaaaab") is None
    assert v.check("nope", "x") is None


VALUES = [
    "4111 1111 1111 1111",
    "4111-1111-1111-1112",
    "GB82 WEST 1234 5698 7654 32",
    "DE89370400440532013000",
    "8.8.8.8",
    "192.168.0.1",
    "300.1.1.1",
    "2001:db8::1",
    "fe80::1",
    "::",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJl",
    "123-45-6789",
    "666-45-6789",
    "+1 415 555 0100",
    "hunter2!",
    "${X}",
    "config.token",
    "q8Zr2LmP0xTv9KcW",
    "",
]


@requires_rust
@pytest.mark.parametrize("name", v.VALIDATORS)
def test_rust_validators_match_python(name: str) -> None:
    ext = extension()
    rng = random.Random(name)
    values = [
        *VALUES,
        *(
            "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 40)))
            for _ in range(300)
        ),
    ]
    for value in values:
        assert ext.validate(name, value) == v.check(name, value), (name, value)


@requires_rust
@given(st.text(max_size=60))
def test_rust_entropy_matches_python(value: str) -> None:
    assert extension().shannon_entropy(value) == pytest.approx(v.shannon_entropy(value), abs=1e-12)


def test_rust_available_flag() -> None:
    assert isinstance(rust_available(), bool)
