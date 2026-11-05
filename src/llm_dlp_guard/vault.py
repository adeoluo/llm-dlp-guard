"""Reversible redaction: a vault of format-preserving tokens.

Each sensitive value is replaced by a token of the same shape (digits stay digits, letters keep
their case, separators stay), drawn from HMAC-SHA256 under a per-vault random key. Card numbers
keep their leading network digit and a valid Luhn digit, IBANs keep their country and valid check
digits, IPv4 addresses keep four in-range octets, and email addresses keep the ``@`` and the
top-level domain, so the model still sees well-formed data. The vault maps
tokens back to the original values so responses can be restored for the user. A vault lives for
one conversation or one request; the key never leaves the process.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from collections.abc import Iterator

from llm_dlp_guard.validators import luhn_valid

_DIGITS = "0123456789"
_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LOWER = "abcdefghijklmnopqrstuvwxyz"


def _stream(key: bytes, label: str, value: str, counter: int) -> Iterator[int]:
    block = 0
    while True:
        msg = f"{label}\x00{counter}\x00{block}\x00{value}".encode()
        yield from hmac.new(key, msg, hashlib.sha256).digest()
        block += 1


def _shape_preserving(value: str, rand: Iterator[int]) -> str:
    out = []
    for c in value:
        if c.isdigit() and c.isascii():
            out.append(_DIGITS[next(rand) % 10])
        elif c.isalpha():
            alphabet = _UPPER if c.isupper() else _LOWER
            out.append(alphabet[next(rand) % 26])
        else:
            out.append(c)
    return "".join(out)


def _fix_luhn(token: str) -> str:
    positions = [i for i, c in enumerate(token) if c.isdigit()]
    if len(positions) < 12:
        return token
    chars = list(token)
    for d in _DIGITS:
        chars[positions[-1]] = d
        if luhn_valid("".join(chars)):
            break
    return "".join(chars)


def _keep_first_digit(original: str, token: str) -> str:
    i = next((k for k, c in enumerate(original) if c.isdigit()), None)
    if i is None or i >= len(token):
        return token
    return token[:i] + original[i] + token[i + 1 :]


_OCTET_RANGES = {1: (0, 9), 2: (10, 99), 3: (100, 255)}


def _fix_ipv4(token: str) -> str:
    """Map every dotted octet into the valid range for its digit count (so 992 becomes 1xx/2xx)."""

    parts = token.split(".")
    if len(parts) != 4 or not all(p.isdigit() and 1 <= len(p) <= 3 for p in parts):
        return token
    fixed = []
    for p in parts:
        lo, hi = _OCTET_RANGES[len(p)]
        fixed.append(str(lo + int(p) % (hi - lo + 1)))
    return ".".join(fixed)


def _fix_iban(original: str, token: str) -> str:
    compact = token.replace(" ", "")
    if len(compact) < 5:
        return token
    country = original.replace(" ", "")[:2].upper()
    body = compact[4:]
    numeric = "".join(str(int(c, 36)) for c in (body + country + "00").upper())
    check = f"{98 - int(numeric) % 97:02d}"
    fixed = country + check + body
    # Re-insert the original spacing.
    out, k = [], 0
    for c in token:
        if c == " ":
            out.append(" ")
        else:
            out.append(fixed[k])
            k += 1
    return "".join(out)


def _fix_email(original: str, token: str) -> str:
    if "@" not in original:
        return token
    tld = original.rsplit(".", 1)[-1]
    head = token.rsplit(".", 1)[0]
    return f"{head}.{tld}"


class Vault:
    """Token <-> value store with format-preserving tokens."""

    def __init__(self, key: bytes | None = None) -> None:
        self._key = key if key is not None else os.urandom(32)
        self._forward: dict[tuple[str, str], str] = {}
        self._reverse: dict[str, str] = {}
        self._pattern: re.Pattern[str] | None = None

    def __len__(self) -> int:
        return len(self._reverse)

    def __contains__(self, token: object) -> bool:
        return token in self._reverse

    @property
    def max_token_length(self) -> int:
        return max((len(t) for t in self._reverse), default=0)

    def tokens(self) -> list[str]:
        return list(self._reverse)

    def original(self, token: str) -> str | None:
        return self._reverse.get(token)

    def tokenize(self, value: str, label: str) -> str:
        """The token for ``value`` (stable within this vault)."""

        existing = self._forward.get((label, value))
        if existing is not None:
            return existing
        counter = 0
        while True:
            token = _shape_preserving(value, _stream(self._key, label, value, counter))
            if label == "CREDIT_CARD":
                token = _fix_luhn(_keep_first_digit(value, token))
            elif label == "IP_ADDRESS":
                token = _fix_ipv4(token)
            elif label == "IBAN":
                token = _fix_iban(value, token)
            elif label == "EMAIL":
                token = _fix_email(value, token)
            counter += 1
            taken = token in self._reverse or any(token == v for (_, v) in self._forward)
            if token != value and not taken:
                break
            if counter > 64:
                # Values without letters or digits cannot be reshaped; fall back to a label.
                token = f"[{label}_{len(self._reverse) + 1}]"
                break
        self._forward[(label, value)] = token
        self._reverse[token] = value
        self._pattern = None
        return token

    def _compiled(self) -> re.Pattern[str] | None:
        if not self._reverse:
            return None
        if self._pattern is None:
            alts = "|".join(re.escape(t) for t in sorted(self._reverse, key=len, reverse=True))
            self._pattern = re.compile(rf"(?<!\w)(?:{alts})(?!\w)")
        return self._pattern

    def find(self, text: str, start: int = 0) -> list[tuple[int, int]]:
        """Spans of whole-token occurrences in ``text`` at or after ``start``."""

        rx = self._compiled()
        return [] if rx is None else [m.span() for m in rx.finditer(text, start)]

    def restore(self, text: str) -> str:
        """Replace every whole-token occurrence with its original value."""

        rx = self._compiled()
        if rx is None:
            return text
        return rx.sub(lambda m: self._reverse[m.group(0)], text)
