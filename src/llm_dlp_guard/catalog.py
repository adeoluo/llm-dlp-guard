"""The pattern catalog: one regular expression per format, with anchors and validators.

Every expression is written in the subset that Rust's `regex` crate and Python's `re` agree on
(no lookaround, no backreferences, explicit ASCII classes), so the Rust engine and the Python
reference return the same spans. ``anchors`` are literals (matched ASCII case-insensitively) of
which at least one occurs in every match; the Aho-Corasick prefilter only runs a pattern when one
of its anchors is present. ``opener`` is a literal every match starts with; streaming needs it
for patterns whose matches can be longer than the stream's hold-back window.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

Category = Literal["pii", "secret"]

# A terminator for tokens whose last character may be '-' or '/' (where \b fails).
_END = r"(?:[^A-Za-z0-9_\-]|$)"
_END64 = r"(?:[^A-Za-z0-9/+=]|$)"
# First four digits of the major card networks: Visa, Mastercard (51-55, 2221-2720), Amex, Diners
# and JCB (30, 36, 38, 35), Discover (6011, 644-649, 65) and UnionPay (62).
_IIN = r"(?:4[0-9]{3}|5[1-5][0-9]{2}|2[2-7][0-9]{2}|3[0-9]{3}|6(?:011|2[0-9]{2}|4[4-9][0-9]|5[0-9]{2}))"


@dataclass(frozen=True)
class Pattern:
    """One detector in the catalog."""

    name: str
    label: str
    category: Category
    regex: str
    score: float
    anchors: tuple[str, ...] = ()
    validator: str | None = None
    group: str | None = None
    opener: str | None = None
    description: str = ""

    def max_length(self) -> int | None:
        """Longest possible match in characters, or ``None`` if unbounded."""

        width = re._parser.parse(self.regex).getwidth()  # type: ignore[attr-defined]
        upper = int(width[1])
        return None if upper >= int(re._constants.MAXREPEAT) else upper  # type: ignore[attr-defined]


@dataclass
class Catalog:
    """An ordered list of patterns (order breaks ties in overlap resolution)."""

    patterns: list[Pattern] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "patterns": [
                    {
                        k: v
                        for k, v in asdict(p).items()
                        if k in {"name", "label", "regex", "group", "anchors", "validator", "score"}
                    }
                    for p in self.patterns
                ]
            }
        )

    def by_name(self, name: str) -> Pattern:
        for p in self.patterns:
            if p.name == name:
                return p
        raise KeyError(name)

    def labels(self) -> list[str]:
        return sorted({p.label for p in self.patterns})

    def categories(self) -> dict[str, Category]:
        return {p.label: p.category for p in self.patterns}

    def extended(self, extra: list[Pattern]) -> Catalog:
        names = {p.name for p in self.patterns}
        for p in extra:
            if p.name in names:
                raise ValueError(f"duplicate pattern name {p.name!r}")
        return Catalog([*self.patterns, *extra])


_PII = [
    Pattern(
        "email",
        "EMAIL",
        "pii",
        r"\b[A-Za-z0-9][A-Za-z0-9._%+\-]{0,63}@(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,30}[A-Za-z0-9])?\.){1,4}[A-Za-z]{2,24}\b",
        0.95,
        anchors=("@",),
        description="RFC 5322 style address with a dotted domain",
    ),
    Pattern(
        "credit_card",
        "CREDIT_CARD",
        "pii",
        r"\b(?:"
        + _IIN
        + r"[ \-][0-9]{4}[ \-][0-9]{4}[ \-][0-9]{1,7}|3[47][0-9]{2}[ \-][0-9]{6}[ \-][0-9]{5}|"
        + _IIN
        + r"[0-9]{8,15})\b",
        0.9,
        validator="luhn",
        description="13 to 19 digit numbers with a card network prefix that pass the Luhn check",
    ),
    Pattern(
        "iban",
        "IBAN",
        "pii",
        r"\b[A-Z]{2}[0-9]{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b",
        0.9,
        validator="iban",
        description="IBAN with registry length and mod-97 check digits",
    ),
    Pattern(
        "us_ssn",
        "US_SSN",
        "pii",
        r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b",
        0.85,
        anchors=("-",),
        validator="ssn",
        description="US social security number with the SSA area, group and serial rules",
    ),
    Pattern(
        "phone",
        "PHONE",
        "pii",
        r"(?:\+[0-9]{1,3}[ .\-]?(?:\([0-9]{1,4}\)[ .\-]?)?[0-9]{1,4}(?:[ .\-]?[0-9]{1,4}){1,4}"
        r"|\(?\b[0-9]{3}\)?[ .\-][0-9]{3}[ .\-][0-9]{4}\b)",
        0.7,
        validator="phone",
        description="International (+ prefix) or North American numbers, 7 to 15 digits",
    ),
    Pattern(
        "ipv4",
        "IP_ADDRESS",
        "pii",
        r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
        0.8,
        validator="ipv4",
        description="Dotted quad; private and reserved ranges score lower",
    ),
    Pattern(
        "ipv6",
        "IP_ADDRESS",
        "pii",
        r"[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7}",
        0.8,
        anchors=(":",),
        validator="ipv6",
        description="IPv6 in full or compressed form; local ranges score lower",
    ),
]

_SECRETS = [
    Pattern(
        "private_key",
        "PRIVATE_KEY",
        "secret",
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |PGP )?PRIVATE KEY(?: BLOCK)?-----"
        r"[\s\S]{0,8000}?"
        r"-----END (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |PGP )?PRIVATE KEY(?: BLOCK)?-----",
        0.99,
        anchors=("-----BEGIN",),
        opener="-----BEGIN",
        description="PEM or OpenSSH private key block",
    ),
    Pattern(
        "aws_access_key_id",
        "AWS_ACCESS_KEY",
        "secret",
        r"\b(?:AKIA|ASIA|ABIA|ACCA)[A-Z2-7]{16}\b",
        0.95,
        anchors=("AKIA", "ASIA", "ABIA", "ACCA"),
        description="AWS access key id (long-term or temporary)",
    ),
    Pattern(
        "aws_secret_access_key",
        "AWS_SECRET_KEY",
        "secret",
        r"(?i:aws)[A-Za-z0-9_ .\-]{0,20}(?i:secret|private)[A-Za-z0-9_ .\-]{0,20}[\"']?[ \t]{0,3}(?:[:=]|=>)[ \t]{0,3}[\"']?(?P<s>[A-Za-z0-9/+]{40})"
        + _END64,
        0.9,
        anchors=("aws",),
        validator="high_entropy",
        group="s",
        description="40-character AWS secret assigned to an aws..secret key",
    ),
    Pattern(
        "github_token",
        "GITHUB_TOKEN",
        "secret",
        r"\bgh[pousr]_[A-Za-z0-9]{36}\b",
        0.99,
        anchors=("ghp_", "gho_", "ghu_", "ghs_", "ghr_"),
        validator="github_token",
        description="GitHub classic token; a valid CRC-32 checksum keeps the full score",
    ),
    Pattern(
        "github_fine_grained_pat",
        "GITHUB_TOKEN",
        "secret",
        r"\bgithub_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{59}\b",
        0.99,
        anchors=("github_pat_",),
        description="GitHub fine-grained personal access token",
    ),
    Pattern(
        "gitlab_pat",
        "GITLAB_TOKEN",
        "secret",
        r"\b(?P<s>glpat-[A-Za-z0-9_\-]{20})" + _END,
        0.95,
        anchors=("glpat-",),
        group="s",
        description="GitLab personal access token",
    ),
    Pattern(
        "slack_token",
        "SLACK_TOKEN",
        "secret",
        r"\bxox[baprs]-[0-9]{10,13}-[0-9]{10,13}(?:-[0-9]{10,13})?-[A-Za-z0-9]{24,34}\b",
        0.95,
        anchors=("xoxb-", "xoxa-", "xoxp-", "xoxr-", "xoxs-"),
        description="Slack bot, user or app token",
    ),
    Pattern(
        "slack_webhook",
        "SLACK_WEBHOOK",
        "secret",
        r"https://hooks\.slack\.com/services/T[A-Z0-9]{8,12}/B[A-Z0-9]{8,12}/[A-Za-z0-9]{24}\b",
        0.95,
        anchors=("hooks.slack.com",),
        description="Slack incoming webhook URL",
    ),
    Pattern(
        "stripe_key",
        "STRIPE_KEY",
        "secret",
        r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{24,99}\b",
        0.95,
        anchors=("sk_live_", "sk_test_", "rk_live_", "rk_test_"),
        description="Stripe secret or restricted key",
    ),
    Pattern(
        "google_api_key",
        "GOOGLE_API_KEY",
        "secret",
        r"\b(?P<s>AIza[0-9A-Za-z_\-]{35})" + _END,
        0.9,
        anchors=("AIza",),
        group="s",
        description="Google API key",
    ),
    Pattern(
        "openai_api_key",
        "OPENAI_API_KEY",
        "secret",
        r"\b(?P<s>sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{16,160}T3BlbkFJ[A-Za-z0-9_\-]{16,160})"
        + _END,
        0.99,
        anchors=("T3BlbkFJ",),
        group="s",
        description="OpenAI API key (legacy, project and service account)",
    ),
    Pattern(
        "anthropic_api_key",
        "ANTHROPIC_API_KEY",
        "secret",
        r"\b(?P<s>sk-ant-(?:api|admin)[0-9]{2}-[A-Za-z0-9_\-]{80,120})" + _END,
        0.99,
        anchors=("sk-ant-",),
        group="s",
        description="Anthropic API key",
    ),
    Pattern(
        "huggingface_token",
        "HUGGINGFACE_TOKEN",
        "secret",
        r"\bhf_[A-Za-z]{34}\b",
        0.9,
        anchors=("hf_",),
        description="Hugging Face access token",
    ),
    Pattern(
        "npm_token",
        "NPM_TOKEN",
        "secret",
        r"\bnpm_[A-Za-z0-9]{36}\b",
        0.95,
        anchors=("npm_",),
        description="npm access token",
    ),
    Pattern(
        "sendgrid_api_key",
        "SENDGRID_API_KEY",
        "secret",
        r"\b(?P<s>SG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43})" + _END,
        0.95,
        anchors=("SG.",),
        group="s",
        description="SendGrid API key",
    ),
    Pattern(
        "twilio_api_key",
        "TWILIO_API_KEY",
        "secret",
        r"\bSK[0-9a-f]{32}\b",
        0.8,
        anchors=("SK",),
        description="Twilio API key sid",
    ),
    Pattern(
        "azure_storage_key",
        "AZURE_STORAGE_KEY",
        "secret",
        r"AccountKey=(?P<s>[A-Za-z0-9+/]{86}==)",
        0.95,
        anchors=("AccountKey=",),
        group="s",
        description="Azure storage account key in a connection string",
    ),
    Pattern(
        "gcp_private_key_id",
        "GCP_KEY_ID",
        "secret",
        r"\"private_key_id\"[ \t]{0,3}:[ \t]{0,3}\"(?P<s>[0-9a-f]{40})\"",
        0.95,
        anchors=("private_key_id",),
        group="s",
        description="Google Cloud service account private key id",
    ),
    Pattern(
        "jwt",
        "JWT",
        "secret",
        r"\beyJ[A-Za-z0-9_\-]{5,1500}\.eyJ[A-Za-z0-9_\-]{5,3000}\.[A-Za-z0-9_\-]{0,1500}",
        0.9,
        anchors=("eyJ",),
        validator="jwt",
        opener="eyJ",
        description="JSON Web Token whose header and payload decode to JSON objects",
    ),
    Pattern(
        "bearer_token",
        "BEARER_TOKEN",
        "secret",
        r"(?i:authorization)[ \t]{0,3}:[ \t]{0,3}(?i:bearer)[ \t]{1,3}(?P<s>[A-Za-z0-9._~+/\-]{16,256}={0,2})",
        0.85,
        anchors=("bearer",),
        validator="high_entropy",
        group="s",
        description="Bearer credential in an Authorization header",
    ),
    Pattern(
        "url_credentials",
        "PASSWORD",
        "secret",
        r"\b[a-z][a-z0-9+.\-]{1,20}://[^ \t\r\n\f\v:/@]{1,64}:(?P<s>[^ \t\r\n\f\v:/@]{3,128})@[A-Za-z0-9.\-]{1,100}",
        0.96,
        anchors=("://",),
        group="s",
        description="Password embedded in a URL or connection string",
    ),
    Pattern(
        "keyword_secret",
        "GENERIC_SECRET",
        "secret",
        r"(?i:[A-Za-z0-9_.\-]{0,30}(?:password|passwd|pwd|passphrase|passwort|kennwort|wachtwoord|"
        r"contrase[nñ]a|mot de passe|secret|token|api[_\-]?key|apikey|access[_\-]?key)"
        r"[A-Za-z0-9_.\-]{0,30}(?:[ \t]{1,3}is)?)[\"']?[ \t]{0,3}(?:[:=]|=>|:=)[ \t]{0,3}[\"']?"
        r"(?P<s>[^ \t\r\n\f\v\"',;]{6,128})",
        0.8,
        anchors=(
            "pass",
            "pwd",
            "wort",
            "wachtwoord",
            "contrase",
            "secret",
            "token",
            "api",
            "key",
        ),
        validator="secret_value",
        group="s",
        description="A high-entropy value assigned to a password, secret, token or key",
    ),
]


def default_catalog() -> Catalog:
    """The built-in catalog: PII formats first, then secrets."""

    return Catalog([*_PII, *_SECRETS])


VALIDATORS = (
    "luhn",
    "iban",
    "ipv4",
    "ipv6",
    "jwt",
    "github_token",
    "ssn",
    "phone",
    "secret_value",
    "high_entropy",
)
