from __future__ import annotations

import re

from hypothesis import given
from hypothesis import strategies as st

from llm_dlp_guard.validators import iban_valid, luhn_valid
from llm_dlp_guard.vault import Vault


def test_tokens_keep_shape_and_validity() -> None:
    vault = Vault(b"k" * 32)
    card = vault.tokenize("4111 1111 1111 1111", "CREDIT_CARD")
    assert re.fullmatch(r"\d{4} \d{4} \d{4} \d{4}", card)
    assert luhn_valid(card) and card != "4111 1111 1111 1111"
    iban = vault.tokenize("GB82 WEST 1234 5698 7654 32", "IBAN")
    assert iban.startswith("GB") and iban_valid(iban)
    email = vault.tokenize("jane.doe@example.org", "EMAIL")
    assert re.fullmatch(r"[a-z]{4}\.[a-z]{3}@[a-z]{7}\.org", email)
    name = vault.tokenize("Zoë", "GIVEN_NAME")
    assert re.fullmatch(r"[A-Z][a-z]{2}", name)
    assert vault.tokenize("Zoë", "GIVEN_NAME") == name
    assert len(vault) == 4
    assert vault.max_token_length == len("GB82 WEST 1234 5698 7654 32")
    assert name in vault and vault.original(name) == "Zoë"
    assert set(vault.tokens()) == {card, iban, email, name}


def test_same_key_same_tokens_different_key_differs() -> None:
    a, b, c = Vault(b"1" * 32), Vault(b"1" * 32), Vault(b"2" * 32)
    assert a.tokenize("Lena", "GIVEN_NAME") == b.tokenize("Lena", "GIVEN_NAME")
    assert a.tokenize("Lena Sato", "X") != c.tokenize("Lena Sato", "X")


def test_values_without_letters_or_digits_get_labels() -> None:
    vault = Vault()
    assert vault.tokenize("--", "X") == "[X_1]"
    assert vault.original("[X_1]") == "--"


def test_restore_whole_tokens_only() -> None:
    vault = Vault(b"z" * 32)
    token = vault.tokenize("Lena", "GIVEN_NAME")
    text = f"Hello {token}! x{token} {token}s"
    assert vault.restore(text) == f"Hello Lena! x{token} {token}s"
    assert vault.find(text) == [(6, 6 + len(token))]
    assert Vault().restore("nothing") == "nothing"
    assert Vault().find("nothing") == []


@given(
    st.lists(st.text(alphabet="abcXYZ0123 .@-", min_size=1, max_size=20), min_size=1, max_size=8)
)
def test_restore_inverts_tokenize(values: list[str]) -> None:
    vault = Vault(b"p" * 32)
    tokens = [vault.tokenize(v, "X") for v in values]
    assert len(set(tokens)) == len(set(values))
    for value, token in zip(values, tokens, strict=True):
        assert vault.original(token) == value
