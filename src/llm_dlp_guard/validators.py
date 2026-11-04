"""Validators, the Python reference for ``rust/src/validators.rs``.

Each returns ``None`` to reject a candidate or a factor in (0, 1] that multiplies the pattern's
base score. The Rust core runs the same logic; ``tests/test_parity.py`` checks agreement.
"""

from __future__ import annotations

import ipaddress
import math
import zlib
from collections import Counter
from collections.abc import Callable

IBAN_LENGTHS: dict[str, int] = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16, "BG": 22, "BH": 22,
    "BI": 27, "BR": 29, "BY": 28, "CH": 21, "CR": 22, "CY": 28, "CZ": 24, "DE": 22, "DJ": 27,
    "DK": 18, "DO": 28, "EE": 20, "EG": 29, "ES": 24, "FI": 18, "FK": 18, "FO": 18, "FR": 27,
    "GB": 22, "GE": 22, "GI": 23, "GL": 18, "GR": 27, "GT": 28, "HN": 28, "HR": 21, "HU": 28,
    "IE": 22, "IL": 23, "IQ": 23, "IS": 26, "IT": 27, "JO": 30, "KW": 30, "KZ": 20, "LB": 28,
    "LC": 32, "LI": 21, "LT": 20, "LU": 20, "LV": 21, "LY": 25, "MC": 27, "MD": 24, "ME": 22,
    "MK": 19, "MN": 20, "MR": 27, "MT": 31, "MU": 30, "NI": 28, "NL": 18, "NO": 15, "OM": 23,
    "PK": 24, "PL": 28, "PS": 29, "PT": 25, "QA": 29, "RO": 24, "RS": 22, "RU": 33, "SA": 24,
    "SC": 31, "SD": 18, "SE": 24, "SI": 19, "SK": 24, "SM": 27, "SO": 23, "ST": 25, "SV": 28,
    "TL": 23, "TN": 24, "TR": 26, "UA": 29, "VA": 22, "VG": 24, "XK": 20, "YE": 30,
}  # fmt: skip

BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

_PLACEHOLDERS = frozenset(
    {
        "null", "none", "nil", "true", "false", "undefined", "password", "passwd", "secret",
        "changeme", "example", "redacted", "placeholder", "your_password", "your_api_key",
    }
)  # fmt: skip


def luhn_valid(value: str) -> bool:
    """Luhn checksum over a card-like string (digits, spaces and dashes)."""

    digits: list[int] = []
    for c in value:
        if "0" <= c <= "9":
            digits.append(ord(c) - 48)
        elif c not in " -":
            return False
    if not 12 <= len(digits) <= 19 or all(d == digits[0] for d in digits):
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            total += d - 9 if d > 9 else d
        else:
            total += d
    return total % 10 == 0


def iban_valid(value: str) -> bool:
    """Known country, registry length, alphanumeric body and mod-97 == 1."""

    compact = value.replace(" ", "")
    if len(compact) < 15 or not all(c.isascii() and c.isalnum() for c in compact):
        return False
    length = IBAN_LENGTHS.get(compact[:2].upper())
    if length is None or len(compact) != length or not compact[2:4].isdigit():
        return False
    remainder = 0
    for c in (compact[4:] + compact[:4]).upper():
        v = ord(c) - 48 if c.isdigit() else ord(c) - 55
        remainder = (remainder * (100 if v >= 10 else 10) + v) % 97
    return remainder == 1


def ipv4_factor(value: str) -> float | None:
    """1.0 for a public address, 0.6 for reserved ranges, ``None`` if invalid."""

    parts = value.split(".")
    if len(parts) != 4:
        return None
    octets: list[int] = []
    for part in parts:
        if not part or len(part) > 3 or not all("0" <= c <= "9" for c in part):
            return None
        if len(part) > 1 and part[0] == "0":
            return None
        v = int(part)
        if v > 255:
            return None
        octets.append(v)
    a, b, c, _ = octets
    if octets in ([0, 0, 0, 0], [255, 255, 255, 255]):
        return None
    reserved = (
        a == 10
        or a == 127
        or (a == 172 and 16 <= b <= 31)
        or (a == 192 and b == 168)
        or (a == 169 and b == 254)
        or (a == 100 and 64 <= b <= 127)
        or (a == 192 and b == 0 and c == 2)
        or (a == 198 and b == 51 and c == 100)
        or (a == 203 and b == 0 and c == 113)
        or a >= 224
        or a == 0
    )
    return 0.6 if reserved else 1.0


def ipv6_factor(value: str) -> float | None:
    """1.0 for global, 0.6 for loopback, unique-local and link-local, ``None`` if invalid."""

    if value.count(":") < 2 or not all(c in "0123456789abcdefABCDEF:" for c in value):
        return None
    try:
        addr = ipaddress.IPv6Address(value)
    except ValueError:
        return None
    if addr.is_unspecified:
        return None
    first = int(addr) >> 112
    local = addr.is_loopback or (first & 0xFE00) == 0xFC00 or (first & 0xFFC0) == 0xFE80
    return 0.6 if local else 1.0


def _b64url_decode(part: str) -> bytes | None:
    out = bytearray()
    buf = 0
    bits = 0
    for ch in part:
        o = ord(ch)
        if 65 <= o <= 90:
            v = o - 65
        elif 97 <= o <= 122:
            v = o - 97 + 26
        elif 48 <= o <= 57:
            v = o - 48 + 52
        elif ch == "-":
            v = 62
        elif ch == "_":
            v = 63
        else:
            return None
        buf = (buf << 6) | v
        bits += 6
        if bits >= 8:
            bits -= 8
            out.append(buf >> bits)
            buf &= (1 << bits) - 1
    if bits >= 6:
        return None
    return bytes(out)


def jwt_valid(value: str) -> bool:
    """Three base64url parts; header and payload decode to JSON objects; header has ``alg``."""

    parts = value.split(".")
    if len(parts) != 3 or not parts[0] or not parts[1]:
        return False
    header = _b64url_decode(parts[0])
    payload = _b64url_decode(parts[1])
    if header is None or payload is None:
        return False
    if parts[2] and _b64url_decode(parts[2]) is None:
        return False

    def is_object(raw: bytes) -> bool:
        text = raw.decode("utf-8", errors="replace").strip()
        return text.startswith("{") and text.endswith("}")

    return (
        is_object(header)
        and is_object(payload)
        and '"alg"' in header.decode("utf-8", errors="replace")
    )


def base62(n: int, width: int) -> str:
    out: list[str] = []
    while n > 0:
        n, r = divmod(n, 62)
        out.append(BASE62[r])
    return "".join(reversed(out)).rjust(width, "0")


def github_checksum(random_part: str) -> str:
    """The 6-character base62 CRC-32 suffix of a GitHub classic token."""

    return base62(zlib.crc32(random_part.encode()), 6)


def github_checksum_valid(value: str) -> bool:
    body = value[4:]
    if len(body) != 36 or not body.isascii():
        return False
    return github_checksum(body[:30]) == body[30:]


def ssn_valid(value: str) -> bool:
    digits = "".join(c for c in value if "0" <= c <= "9")
    if len(digits) != 9:
        return False
    area = digits[:3]
    return (
        area not in ("000", "666")
        and area[0] != "9"
        and digits[3:5] != "00"
        and digits[5:] != "0000"
    )


def phone_factor(value: str) -> float | None:
    n = sum(1 for c in value if "0" <= c <= "9")
    return 1.0 if 7 <= n <= 15 else None


def shannon_entropy(value: str) -> float:
    """Bits per character, summed in code point order (as the Rust core does)."""

    if not value:
        return 0.0
    n = len(value)
    h = 0.0
    for _, count in sorted(Counter(value).items()):
        p = count / n
        h -= p * math.log2(p)
    return h


def _identifier(part: str) -> bool:
    return (
        bool(part)
        and (part[0].isascii() and (part[0].isalpha() or part[0] == "_"))
        and all(c.isascii() and (c.isalnum() or c == "_") for c in part)
    )


def _dotted_identifier(value: str) -> bool:
    """``config.password`` or ``process.env.TOKEN``: code, not a secret."""

    parts = value.split(".")
    return len(parts) >= 2 and all(_identifier(p) for p in parts)


def secret_value_factor(value: str) -> float | None:
    """Reject placeholders and templating, then score by entropy."""

    if not value:
        return None
    lower = value.lower()
    if (
        value[0] in "${<%"
        or any(c in "()[]{}" for c in value)
        or _dotted_identifier(value)
        or "xxxx" in lower
        or "****" in lower
        or lower in _PLACEHOLDERS
    ):
        return None
    h = shannon_entropy(value)
    if h >= 3.5:
        return 1.0
    if h >= 2.5:
        return 0.7
    return None


def _flag(ok: bool) -> float | None:
    return 1.0 if ok else None


VALIDATORS: dict[str, Callable[[str], float | None]] = {
    "luhn": lambda v: _flag(luhn_valid(v)),
    "iban": lambda v: _flag(iban_valid(v)),
    "ipv4": ipv4_factor,
    "ipv6": ipv6_factor,
    "jwt": lambda v: _flag(jwt_valid(v)),
    "github_token": lambda v: 1.0 if github_checksum_valid(v) else 0.8,
    "ssn": lambda v: _flag(ssn_valid(v)),
    "phone": phone_factor,
    "secret_value": secret_value_factor,
    "high_entropy": lambda v: _flag(shannon_entropy(v) >= 3.0),
}

# Checksum and structure validators; their matches win overlaps with tagger spans.
STRUCTURED = frozenset({"luhn", "iban", "ipv4", "ipv6"})


def check(name: str, value: str) -> float | None:
    """Run validator ``name``; unknown names reject."""

    fn = VALIDATORS.get(name)
    return None if fn is None else fn(value)
