"""A labelled corpus of credentials in real formats with random (fake) values.

No public labelled secrets corpus is freely downloadable (SecretBench needs an access request),
so this generator builds one: every secret has the exact published shape of its provider
(prefixes, lengths, alphabets, the GitHub CRC-32 checksum, decodable JWT parts, PEM framing)
with values drawn from a seeded random generator, so none is a live credential. Secrets are
embedded in the places they leak from (``.env`` and YAML files, JSON, source code, shell
history, logs, chat messages). Hard negatives use the same contexts with look-alikes: hashes,
UUIDs, base64 blobs, placeholders, templated values and prose about passwords.
"""

from __future__ import annotations

import base64
import json
import random
import string
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from llm_dlp_guard.validators import github_checksum

Span = tuple[int, int, str]

ALNUM = string.ascii_letters + string.digits
B64 = ALNUM + "+/"
B64URL = ALNUM + "-_"
BASE32 = string.ascii_uppercase + "234567"
HEX = "0123456789abcdef"


@dataclass(frozen=True)
class Sample:
    text: str
    spans: list[Span]
    kind: str

    def to_json(self) -> str:
        return json.dumps({"text": self.text, "spans": self.spans, "kind": self.kind})


def _r(rng: random.Random, alphabet: str, n: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


# -- secret generators: return (value, label) ------------------------------------------------


def aws_access_key(rng: random.Random) -> str:
    return rng.choice(["AKIA", "ASIA"]) + _r(rng, BASE32, 16)


def aws_secret_key(rng: random.Random) -> str:
    return _r(rng, B64, 40)


def github_token(rng: random.Random) -> str:
    random_part = _r(rng, ALNUM, 30)
    return (
        rng.choice(["ghp_", "gho_", "ghu_", "ghs_", "ghr_"])
        + random_part
        + github_checksum(random_part)
    )


def github_pat(rng: random.Random) -> str:
    return f"github_pat_{_r(rng, ALNUM, 22)}_{_r(rng, ALNUM, 59)}"


def gitlab_token(rng: random.Random) -> str:
    return "glpat-" + _r(rng, ALNUM + "_-", 20)


def slack_token(rng: random.Random) -> str:
    kind = rng.choice("bpa")
    return (
        f"xox{kind}-{_r(rng, string.digits, 12)}-{_r(rng, string.digits, 13)}-{_r(rng, ALNUM, 24)}"
    )


def slack_webhook(rng: random.Random) -> str:
    upper = string.ascii_uppercase + string.digits
    return f"https://hooks.slack.com/services/T{_r(rng, upper, 10)}/B{_r(rng, upper, 10)}/{_r(rng, ALNUM, 24)}"


def stripe_key(rng: random.Random) -> str:
    return f"{rng.choice(['sk', 'rk'])}_{rng.choice(['live', 'test'])}_{_r(rng, ALNUM, rng.randint(24, 99))}"


def google_api_key(rng: random.Random) -> str:
    return "AIza" + _r(rng, ALNUM + "_-", 35)


def openai_key(rng: random.Random) -> str:
    prefix = rng.choice(["sk-", "sk-proj-", "sk-svcacct-"])
    n = 20 if prefix == "sk-" else 74
    return f"{prefix}{_r(rng, ALNUM, n)}T3BlbkFJ{_r(rng, ALNUM, n)}"


def anthropic_key(rng: random.Random) -> str:
    return f"sk-ant-api03-{_r(rng, ALNUM + '_-', 93)}AA"


def huggingface_token(rng: random.Random) -> str:
    return "hf_" + _r(rng, string.ascii_letters, 34)


def npm_token(rng: random.Random) -> str:
    return "npm_" + _r(rng, ALNUM, 36)


def sendgrid_key(rng: random.Random) -> str:
    return f"SG.{_r(rng, ALNUM + '_-', 22)}.{_r(rng, ALNUM + '_-', 43)}"


def twilio_key(rng: random.Random) -> str:
    return "SK" + _r(rng, HEX, 32)


def azure_key(rng: random.Random) -> str:
    return _r(rng, B64, 86) + "=="


def gcp_key_id(rng: random.Random) -> str:
    return _r(rng, HEX, 40)


def jwt(rng: random.Random) -> str:
    header = {"alg": rng.choice(["HS256", "RS256", "ES256"]), "typ": "JWT"}
    payload = {
        "sub": str(rng.randint(10**6, 10**9)),
        "name": rng.choice(["Ana Lima", "Wei Chen", "Sam Okoro", "Ida Berg"]),
        "iat": rng.randint(1_600_000_000, 1_800_000_000),
        "scope": rng.choice(["read", "read write", "admin"]),
    }
    sig = bytes(rng.getrandbits(8) for _ in range(rng.choice([32, 64])))
    return ".".join(
        [_b64url(json.dumps(header).encode()), _b64url(json.dumps(payload).encode()), _b64url(sig)]
    )


def private_key(rng: random.Random) -> str:
    kind = rng.choice(["RSA ", "EC ", "OPENSSH ", ""])
    body = _r(rng, B64, rng.choice([64 * 12, 64 * 25]))
    lines = "\n".join(body[i : i + 64] for i in range(0, len(body), 64))
    return f"-----BEGIN {kind}PRIVATE KEY-----\n{lines}\n-----END {kind}PRIVATE KEY-----"


def password(rng: random.Random) -> str:
    return _r(rng, ALNUM + "!#%&*@^", rng.randint(12, 28))


def url_password(rng: random.Random) -> str:
    return _r(rng, ALNUM + "!%&*^-_", rng.randint(12, 28))


def api_key_generic(rng: random.Random) -> str:
    return _r(rng, ALNUM, rng.choice([32, 40, 48]))


def bearer(rng: random.Random) -> str:
    return _r(rng, B64URL, rng.randint(32, 64))


# -- contexts: (template with {v}, label) -----------------------------------------------------

SecretGen = Callable[[random.Random], str]

_KINDS: list[tuple[str, SecretGen, list[str]]] = [
    ("AWS_ACCESS_KEY", aws_access_key, [
        "AWS_ACCESS_KEY_ID={v}",
        "aws_access_key_id = {v}",
        "export AWS_ACCESS_KEY_ID=\"{v}\"",
        "Can you check why boto3 fails with key {v}?",
        "  accessKeyId: '{v}'",
    ]),
    ("AWS_SECRET_KEY", aws_secret_key, [
        "aws_secret_access_key = {v}",
        "AWS_SECRET_ACCESS_KEY=\"{v}\"",
        "aws_secret_key: {v}",
    ]),
    ("GITHUB_TOKEN", github_token, [
        "GITHUB_TOKEN={v}",
        "git clone https://{v}@github.com/acme/infra.git",
        "token: {v}",
        "Here is my token {v}, why does the push fail?",
    ]),
    ("GITHUB_TOKEN", github_pat, ["GH_PAT={v}", "Authorization: token {v}"]),
    ("GITLAB_TOKEN", gitlab_token, ["GITLAB_TOKEN={v}", "private_token: {v}"]),
    ("SLACK_TOKEN", slack_token, ["SLACK_BOT_TOKEN={v}", "client = WebClient(token=\"{v}\")"]),
    ("SLACK_WEBHOOK", slack_webhook, ["webhook_url: {v}", "curl -X POST -d @msg.json {v}"]),
    ("STRIPE_KEY", stripe_key, ["STRIPE_SECRET_KEY={v}", "stripe.api_key = \"{v}\""]),
    ("GOOGLE_API_KEY", google_api_key, ["GOOGLE_API_KEY={v}", "https://maps.googleapis.com/maps/api/geocode/json?key={v}"]),
    ("OPENAI_API_KEY", openai_key, ["OPENAI_API_KEY={v}", "client = OpenAI(api_key=\"{v}\")"]),
    ("ANTHROPIC_API_KEY", anthropic_key, ["ANTHROPIC_API_KEY={v}", "x-api-key: {v}"]),
    ("HUGGINGFACE_TOKEN", huggingface_token, ["HF_TOKEN={v}", "huggingface-cli login --token {v}"]),
    ("NPM_TOKEN", npm_token, ["//registry.npmjs.org/:_authToken={v}", "NPM_TOKEN={v}"]),
    ("SENDGRID_API_KEY", sendgrid_key, ["SENDGRID_API_KEY={v}", "sg = SendGridAPIClient(\"{v}\")"]),
    ("TWILIO_API_KEY", twilio_key, ["TWILIO_API_KEY={v}", "twilio api:core:keys:fetch --sid {v}"]),
    ("AZURE_STORAGE_KEY", azure_key, [
        "DefaultEndpointsProtocol=https;AccountName=acmedata;AccountKey={v};EndpointSuffix=core.windows.net",
    ]),
    ("GCP_KEY_ID", gcp_key_id, ["{{\"type\": \"service_account\", \"private_key_id\": \"{v}\"}}"]),
    ("JWT", jwt, ["Authorization: Bearer {v}", "id_token={v}", "Why is this JWT rejected? {v}"]),
    ("PRIVATE_KEY", private_key, ["{v}", "Here is the deploy key:\n{v}\nplease add it to the server."]),
    ("PASSWORD", url_password, [
        "DATABASE_URL=postgres://app:{v}@db.internal:5432/orders",
        "mongodb://admin:{v}@10.0.4.12:27017/admin",
        "redis://default:{v}@cache.local:6379/0",
    ]),
    ("GENERIC_SECRET", password, [
        "DB_PASSWORD={v}",
        "password: {v}",
        "\"password\": \"{v}\"",
        "My password is: {v}",
    ]),
    ("GENERIC_SECRET", api_key_generic, [
        "api_key = \"{v}\"",
        "API_KEY={v}",
        "client_secret: {v}",
        "secret_token = '{v}'",
    ]),
    ("BEARER_TOKEN", bearer, ["Authorization: Bearer {v}", "curl -H 'Authorization: Bearer {v}' https://api.example.com/v1/items"]),
]  # fmt: skip

# -- hard negatives --------------------------------------------------------------------------


def _uuid(rng: random.Random) -> str:
    h = _r(rng, HEX, 32)
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


_NEGATIVES: list[Callable[[random.Random], str]] = [
    lambda r: f"commit {_r(r, HEX, 40)}",
    lambda r: f"sha256: {_r(r, HEX, 64)}",
    lambda r: f"md5sum {_r(r, HEX, 32)}  release.tar.gz",
    lambda r: f"request_id={_uuid(r)}",
    lambda r: f'"trace_id": "{_uuid(r)}"',
    lambda r: f"data:image/png;base64,{_r(r, B64, 120)}",
    lambda r: "api_key = YOUR_API_KEY",
    lambda r: "password: ${DB_PASSWORD}",
    lambda r: 'password = os.environ["DB_PASSWORD"]',
    lambda r: "token = tokenizer.encode(text)",
    lambda r: "secret: {{ .Values.secret }}",
    lambda r: "password: ********",
    lambda r: "AWS_ACCESS_KEY_ID=<your-access-key-id>",
    lambda r: "The password must be at least 12 characters long.",
    lambda r: "Reset your password from the account settings page.",
    lambda r: f"version {r.randint(1, 9)}.{r.randint(0, 20)}.{r.randint(0, 99)}",
    lambda r: f"build {r.randint(1000, 9999)}-{_r(r, HEX, 7)}",
    lambda r: (
        f"https://example.com/docs/{_r(r, string.ascii_lowercase, 8)}?page={r.randint(1, 50)}"
    ),
    lambda r: f"git checkout {_r(r, HEX, 7)}",
    lambda r: f'etag: "{_r(r, HEX, 32)}"',
    lambda r: f"session cookie name: sid, max-age {r.randint(60, 86400)}",
    lambda r: f"order #{r.randint(10**7, 10**8)} shipped",
    lambda r: f"container {_r(r, HEX, 12)} restarted",
    lambda r: f'key_name = "{_r(r, string.ascii_lowercase, 6)}_{_r(r, string.ascii_lowercase, 5)}"',
    lambda r: f"docker pull ghcr.io/acme/app@sha256:{_r(r, HEX, 64)}",
    lambda r: f'const id = "{_r(r, ALNUM, 20)}";',
    lambda r: f"nonce={_r(r, ALNUM, 16)}",
]


def _wrap(rng: random.Random, line: str) -> tuple[str, int]:
    """Embed ``line`` in a little surrounding text; return the text and the line's offset."""

    before = rng.choice(
        [
            "",
            "# config\n",
            "Hi, quick question about our deploy.\n",
            "[2025-03-02 10:14:07] INFO loading settings\n",
            "```\n",
        ]
    )
    after = rng.choice(["", "\n", "\nthanks!", "\n```", "\nDEBUG=false"])
    return before + line + after, len(before)


def generate(n_positive: int, n_negative: int, seed: int = 0) -> Iterator[Sample]:
    """``n_positive`` samples with one secret each, then ``n_negative`` hard negatives."""

    rng = random.Random(seed)
    for i in range(n_positive):
        label, gen, templates = _KINDS[i % len(_KINDS)]
        value = gen(rng)
        template = rng.choice(templates)
        before, after = template.split("{v}")
        before = before.replace("{{", "{").replace("}}", "}")
        after = after.replace("{{", "{").replace("}}", "}")
        line = before + value + after
        text, offset = _wrap(rng, line)
        start = offset + len(before)
        yield Sample(text, [(start, start + len(value), label)], label)
    for _ in range(n_negative):
        line = rng.choice(_NEGATIVES)(rng)
        text, _ = _wrap(rng, line)
        yield Sample(text, [], "negative")


def kinds() -> list[str]:
    return sorted({k for k, _, _ in _KINDS})
