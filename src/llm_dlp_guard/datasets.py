"""Loaders for the public PII datasets and the label maps used in evaluation.

* ai4privacy/pii-masking-400k: 406,896 synthetic texts in six languages with 17 PII labels
  (train 325,517, validation 81,379). Academic, non-commercial licence: download it yourself
  with ``scripts/download_data.sh``; it is never redistributed here.
* nvidia/Nemotron-PII: synthetic English documents with 55 labels (CC BY 4.0); the test split
  (100,000 documents: 50,000 personas in a US and an international version) is used for
  cross-dataset evaluation only.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

Span = tuple[int, int, str]

AI4PRIVACY_LANGUAGES = ("en", "de", "es", "fr", "it", "nl")
_AI4PRIVACY_FILES = {"en": "1en.jsonl"}

# ai4privacy label -> label used by the tagger.
AI4PRIVACY_LABELS: dict[str, str] = {
    "GIVENNAME": "GIVEN_NAME",
    "SURNAME": "SURNAME",
    "USERNAME": "USERNAME",
    "CITY": "CITY",
    "EMAIL": "EMAIL",
    "TELEPHONENUM": "PHONE",
    "IDCARDNUM": "ID_CARD",
    "ACCOUNTNUM": "ACCOUNT_NUMBER",
    "DATEOFBIRTH": "DATE_OF_BIRTH",
    "STREET": "STREET",
    "ZIPCODE": "ZIP_CODE",
    "BUILDINGNUM": "BUILDING_NUMBER",
    "SOCIALNUM": "SOCIAL_NUMBER",
    "PASSWORD": "PASSWORD",
    "TAXNUM": "TAX_NUMBER",
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "DRIVERLICENSENUM": "DRIVER_LICENSE",
}
TAGGER_LABELS: list[str] = list(AI4PRIVACY_LABELS.values())

# Pattern labels mapped into the ai4privacy label space for the fine-grained evaluation.
PATTERN_TO_AI4PRIVACY: dict[str, str] = {
    "EMAIL": "EMAIL",
    "CREDIT_CARD": "CREDIT_CARD",
    "PHONE": "PHONE",
    "IBAN": "ACCOUNT_NUMBER",
    "US_SSN": "SOCIAL_NUMBER",
    "GENERIC_SECRET": "PASSWORD",
    "PASSWORD": "PASSWORD",
}

COARSE_TYPES = (
    "PERSON",
    "EMAIL",
    "PHONE",
    "CREDIT_CARD",
    "ADDRESS",
    "DATE_OF_BIRTH",
    "SSN",
    "ID_NUMBER",
    "ACCOUNT_NUMBER",
    "IP_ADDRESS",
    "USERNAME",
    "PASSWORD",
    "API_KEY",
)

# This package's labels (tagger and patterns) -> coarse types shared with Presidio.
OURS_TO_COARSE: dict[str, str] = {
    "GIVEN_NAME": "PERSON",
    "SURNAME": "PERSON",
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "CREDIT_CARD": "CREDIT_CARD",
    "STREET": "ADDRESS",
    "CITY": "ADDRESS",
    "ZIP_CODE": "ADDRESS",
    "BUILDING_NUMBER": "ADDRESS",
    "DATE_OF_BIRTH": "DATE_OF_BIRTH",
    "SOCIAL_NUMBER": "SSN",
    "US_SSN": "SSN",
    "ID_CARD": "ID_NUMBER",
    "DRIVER_LICENSE": "ID_NUMBER",
    "TAX_NUMBER": "ID_NUMBER",
    "ACCOUNT_NUMBER": "ACCOUNT_NUMBER",
    "IBAN": "ACCOUNT_NUMBER",
    "IP_ADDRESS": "IP_ADDRESS",
    "USERNAME": "USERNAME",
    "PASSWORD": "PASSWORD",
    "GENERIC_SECRET": "PASSWORD",
    "AWS_ACCESS_KEY": "API_KEY",
    "AWS_SECRET_KEY": "API_KEY",
    "GITHUB_TOKEN": "API_KEY",
    "GITLAB_TOKEN": "API_KEY",
    "SLACK_TOKEN": "API_KEY",
    "SLACK_WEBHOOK": "API_KEY",
    "STRIPE_KEY": "API_KEY",
    "GOOGLE_API_KEY": "API_KEY",
    "OPENAI_API_KEY": "API_KEY",
    "ANTHROPIC_API_KEY": "API_KEY",
    "HUGGINGFACE_TOKEN": "API_KEY",
    "NPM_TOKEN": "API_KEY",
    "SENDGRID_API_KEY": "API_KEY",
    "TWILIO_API_KEY": "API_KEY",
    "AZURE_STORAGE_KEY": "API_KEY",
    "GCP_KEY_ID": "API_KEY",
    "JWT": "API_KEY",
    "BEARER_TOKEN": "API_KEY",
    "PRIVATE_KEY": "API_KEY",
}

# Presidio's default recognizers -> coarse types (others, such as URL or NRP, are dropped).
PRESIDIO_TO_COARSE: dict[str, str] = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "CREDIT_CARD": "CREDIT_CARD",
    "LOCATION": "ADDRESS",
    "DATE_TIME": "DATE_OF_BIRTH",
    "US_SSN": "SSN",
    "US_DRIVER_LICENSE": "ID_NUMBER",
    "US_PASSPORT": "ID_NUMBER",
    "US_ITIN": "ID_NUMBER",
    "UK_NHS": "ID_NUMBER",
    "IBAN_CODE": "ACCOUNT_NUMBER",
    "US_BANK_NUMBER": "ACCOUNT_NUMBER",
    "IP_ADDRESS": "IP_ADDRESS",
}

# Nemotron-PII labels -> coarse types. Labels not listed are ignore regions in evaluation:
# predictions overlapping them are neither credited nor penalised.
NEMOTRON_TO_COARSE: dict[str, str] = {
    "first_name": "PERSON",
    "last_name": "PERSON",
    "email": "EMAIL",
    "phone_number": "PHONE",
    "fax_number": "PHONE",
    "credit_debit_card": "CREDIT_CARD",
    "street_address": "ADDRESS",
    "city": "ADDRESS",
    "postcode": "ADDRESS",
    "date_of_birth": "DATE_OF_BIRTH",
    "ssn": "SSN",
    "national_id": "ID_NUMBER",
    "tax_id": "ID_NUMBER",
    "account_number": "ACCOUNT_NUMBER",
    "ipv4": "IP_ADDRESS",
    "ipv6": "IP_ADDRESS",
    "user_name": "USERNAME",
    "password": "PASSWORD",
    "api_key": "API_KEY",
}

AI4PRIVACY_TO_COARSE: dict[str, str] = {
    k: OURS_TO_COARSE[v] for k, v in AI4PRIVACY_LABELS.items() if v in OURS_TO_COARSE
}


@dataclass
class Record:
    text: str
    spans: list[Span]
    language: str = "en"
    uid: str = ""
    ignore: list[tuple[int, int]] = field(default_factory=list)


def ai4privacy_files(
    root: str | Path, split: str, languages: tuple[str, ...] | None = None
) -> list[Path]:
    base = Path(root) / split
    langs = languages or AI4PRIVACY_LANGUAGES
    return [base / _AI4PRIVACY_FILES.get(lang, f"{lang}.jsonl") for lang in langs]


def load_ai4privacy(
    root: str | Path,
    split: str = "validation",
    languages: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> Iterator[Record]:
    """Records with spans in the tagger's label names (``AI4PRIVACY_LABELS``)."""

    n = 0
    for path in ai4privacy_files(root, split, languages):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if limit is not None and n >= limit:
                    return
                row = json.loads(line)
                spans = [
                    (int(m["start"]), int(m["end"]), AI4PRIVACY_LABELS[m["label"]])
                    for m in row["privacy_mask"]
                    if m["label"] in AI4PRIVACY_LABELS
                ]
                yield Record(
                    row["source_text"], spans, row.get("language", ""), str(row.get("uid", ""))
                )
                n += 1


def load_nemotron(path: str | Path, limit: int | None = None) -> Iterator[Record]:
    """Documents of a Nemotron-PII split (exact duplicates skipped), spans in coarse types."""

    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["uid", "text", "spans", "locale"])
    seen: set[tuple[str, str]] = set()
    n = 0
    for uid, text, spans, locale in zip(
        table.column("uid").to_pylist(),
        table.column("text").to_pylist(),
        table.column("spans").to_pylist(),
        table.column("locale").to_pylist(),
        strict=True,
    ):
        if (uid, text) in seen:
            continue
        seen.add((uid, text))
        if limit is not None and n >= limit:
            return
        raw = ast.literal_eval(spans) if isinstance(spans, str) else spans
        mapped: list[Span] = []
        ignore: list[tuple[int, int]] = []
        for s in raw:
            coarse = NEMOTRON_TO_COARSE.get(s["label"])
            if coarse is None:
                ignore.append((int(s["start"]), int(s["end"])))
            else:
                mapped.append((int(s["start"]), int(s["end"]), coarse))
        yield Record(text, mapped, "en", f"{uid}:{locale}", ignore)
        n += 1
