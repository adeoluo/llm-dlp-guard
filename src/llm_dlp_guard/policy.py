"""The policy engine: what to do with each finding, per direction.

A policy maps findings to actions with ordered rules (first match wins), an allowlist, a score
floor and optional custom patterns. It loads from YAML or JSON::

    min_score: 0.5
    default_action: tokenize
    rules:
      - labels: [PRIVATE_KEY]
        action: block
      - categories: [secret]
        action: redact
      - categories: [pii]
        directions: [response]
        action: allow
    allow_values: [support@example.com]
    custom_patterns:
      - {name: employee_id, label: EMPLOYEE_ID, regex: "EMP-[0-9]{6}", anchors: ["EMP-"]}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm_dlp_guard.catalog import Catalog, Category, Pattern, default_catalog
from llm_dlp_guard.findings import Finding

Action = Literal["allow", "mask", "redact", "hash", "tokenize", "block"]
Direction = Literal["prompt", "response"]
DIRECTIONS: tuple[Direction, ...] = ("prompt", "response")

# Tagger labels are personal data; anything unknown is treated as PII too.
NER_CATEGORY: Category = "pii"


def _check_regex(value: str) -> None:
    try:
        re.compile(value)
    except re.error as exc:
        raise ValueError(f"invalid regular expression {value!r}: {exc}") from exc


class Rule(BaseModel):
    """Match findings by label or category (empty lists match everything)."""

    model_config = ConfigDict(extra="forbid")

    labels: list[str] = Field(default_factory=list)
    categories: list[Category] = Field(default_factory=list)
    directions: list[Direction] = Field(default_factory=lambda: list(DIRECTIONS))
    action: Action

    def matches(self, label: str, category: Category, direction: Direction) -> bool:
        return (
            direction in self.directions
            and (not self.labels or label in self.labels)
            and (not self.categories or category in self.categories)
        )


class CustomPattern(BaseModel):
    """A user pattern, added after the built-in catalog."""

    model_config = ConfigDict(extra="forbid")

    name: str
    label: str
    regex: str
    category: Category = "pii"
    score: float = Field(0.9, ge=0.0, le=1.0)
    anchors: list[str] = Field(default_factory=list)
    validator: str | None = None
    group: str | None = None
    opener: str | None = None

    @field_validator("regex")
    @classmethod
    def _compiles(cls, value: str) -> str:
        _check_regex(value)
        return value

    def to_pattern(self) -> Pattern:
        return Pattern(
            self.name,
            self.label,
            self.category,
            self.regex,
            self.score,
            anchors=tuple(self.anchors),
            validator=self.validator,
            group=self.group,
            opener=self.opener,
        )


def _default_rules() -> list[Rule]:
    return [
        Rule(labels=["PRIVATE_KEY"], action="block"),
        Rule(categories=["secret"], action="redact"),
        Rule(categories=["pii"], directions=["response"], action="allow"),
    ]


class Policy(BaseModel):
    """Actions for findings; see the module docstring for the file format."""

    model_config = ConfigDict(extra="forbid")

    min_score: float = Field(0.5, ge=0.0, le=1.0)
    ner_min_score: float = Field(0.5, ge=0.0, le=1.0)
    default_action: Action = "tokenize"
    rules: list[Rule] = Field(default_factory=_default_rules)
    allow_values: list[str] = Field(default_factory=list)
    allow_patterns: list[str] = Field(default_factory=list)
    disabled_patterns: list[str] = Field(default_factory=list)
    custom_patterns: list[CustomPattern] = Field(default_factory=list)

    @field_validator("allow_patterns")
    @classmethod
    def _allow_compiles(cls, values: list[str]) -> list[str]:
        for v in values:
            _check_regex(v)
        return values

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        """Read a policy from a ``.yaml``/``.yml`` or ``.json`` file."""

        raw = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw) if str(path).endswith(".json") else yaml.safe_load(raw)
        return cls.model_validate(data or {})

    def catalog(self, base: Catalog | None = None) -> Catalog:
        """The built-in catalog without disabled patterns, plus custom patterns."""

        base = base if base is not None else default_catalog()
        known = {p.name for p in base.patterns}
        unknown = set(self.disabled_patterns) - known
        if unknown:
            raise ValueError(f"unknown patterns in disabled_patterns: {sorted(unknown)}")
        kept = Catalog([p for p in base.patterns if p.name not in self.disabled_patterns])
        return kept.extended([c.to_pattern() for c in self.custom_patterns])

    def category_of(self, label: str, catalog: Catalog) -> Category:
        return catalog.categories().get(label, NER_CATEGORY)

    def is_allowed(self, value: str) -> bool:
        if value in self.allow_values:
            return True
        return any(re.fullmatch(p, value) for p in self.allow_patterns)

    def action_for(self, finding: Finding, category: Category, direction: Direction) -> Action:
        for rule in self.rules:
            if rule.matches(finding.label, category, direction):
                return rule.action
        return self.default_action

    def keep(self, finding: Finding, text: str) -> bool:
        """Score floor and allowlist."""

        return finding.score >= self.min_score and not self.is_allowed(finding.text(text))
