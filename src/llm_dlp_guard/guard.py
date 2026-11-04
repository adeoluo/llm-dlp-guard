"""The guard: detection, policy decisions and redaction for one direction of LLM traffic."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from llm_dlp_guard.catalog import Catalog
from llm_dlp_guard.detector import Detector, Tagger
from llm_dlp_guard.findings import Finding
from llm_dlp_guard.policy import Action, Direction, Policy
from llm_dlp_guard.vault import Vault

Outcome = Literal["allow", "redact", "block"]


@dataclass(frozen=True)
class Applied:
    """A finding with the action taken and the text that replaced it."""

    finding: Finding
    category: str
    action: Action
    replacement: str

    def to_dict(self) -> dict[str, object]:
        return {
            **self.finding.to_dict(),
            "category": self.category,
            "action": self.action,
        }


@dataclass
class Decision:
    """The result of processing one text."""

    outcome: Outcome
    text: str
    applied: list[Applied] = field(default_factory=list)

    @property
    def blocked(self) -> list[Applied]:
        return [a for a in self.applied if a.action == "block"]

    def to_dict(self) -> dict[str, object]:
        return {
            "outcome": self.outcome,
            "text": self.text if self.outcome != "block" else None,
            "findings": [a.to_dict() for a in self.applied],
        }


def replacement(action: Action, value: str, label: str, vault: Vault, key: bytes) -> str:
    """The text that replaces ``value`` under ``action`` (``allow`` and ``block`` keep it)."""

    if action == "mask":
        return "".join("*" if not c.isspace() else c for c in value)
    if action == "redact":
        return f"[{label}]"
    if action == "hash":
        digest = hmac.new(key, f"{label}\x00{value}".encode(), hashlib.sha256).hexdigest()
        return f"[{label}:{digest[:12]}]"
    if action == "tokenize":
        return vault.tokenize(value, label)
    return value


def splice(text: str, edits: Sequence[tuple[int, int, str]]) -> str:
    """Apply non-overlapping ``(start, end, replacement)`` edits."""

    out: list[str] = []
    pos = 0
    for s, e, rep in sorted(edits):
        out.append(text[pos:s])
        out.append(rep)
        pos = e
    out.append(text[pos:])
    return "".join(out)


def overlaps_any(f: Finding, spans: Sequence[tuple[int, int]]) -> bool:
    return any(s < f.end and f.start < e for s, e in spans)


class Guard:
    """Detector plus policy. One guard serves many requests; vaults are per conversation."""

    def __init__(
        self,
        policy: Policy | None = None,
        tagger: Tagger | None = None,
        *,
        backend: str = "auto",
        hash_key: bytes | None = None,
    ) -> None:
        self.policy = policy if policy is not None else Policy()
        self.catalog: Catalog = self.policy.catalog()
        self.detector = Detector(
            self.catalog, tagger, ner_min_score=self.policy.ner_min_score, backend=backend
        )
        self._categories = self.catalog.categories()
        self._hash_key = hash_key if hash_key is not None else b"llm-dlp-guard"

    @classmethod
    def from_files(
        cls,
        policy: str | Path | None = None,
        model: str | Path | None = None,
        *,
        backend: str = "auto",
    ) -> Guard:
        pol = Policy.load(policy) if policy else Policy()
        tagger = Tagger.load(model, backend) if model else None
        return cls(pol, tagger, backend=backend)

    @property
    def has_tagger(self) -> bool:
        return self.detector.tagger is not None

    def category(self, label: str) -> str:
        return self._categories.get(label, "pii")

    def scan(self, text: str) -> list[Finding]:
        """Findings that pass the policy's score floor and allowlist."""

        return [f for f in self.detector.scan(text) if self.policy.keep(f, text)]

    def decide(
        self,
        text: str,
        findings: Sequence[Finding],
        direction: Direction,
        vault: Vault,
        restores: Sequence[tuple[int, int]] = (),
    ) -> Decision:
        """Apply the policy to ``findings`` of ``text``; ``restores`` are vault-token spans to
        replace with their original values (they must not overlap the findings)."""

        applied: list[Applied] = []
        edits: list[tuple[int, int, str]] = [
            (s, e, vault.original(text[s:e]) or text[s:e]) for s, e in restores
        ]
        outcome: Outcome = "allow"
        for f in findings:
            category = self.category(f.label)
            action = self.policy.action_for(f, category, direction)  # type: ignore[arg-type]
            value = f.text(text)
            rep = replacement(action, value, f.label, vault, self._hash_key)
            applied.append(Applied(f, category, action, rep))
            if action == "block":
                outcome = "block"
            elif action != "allow":
                edits.append((f.start, f.end, rep))
                if outcome == "allow":
                    outcome = "redact"
        if outcome == "block":
            return Decision("block", text, applied)
        return Decision(outcome, splice(text, edits), applied)

    def process(
        self, text: str, direction: Direction = "prompt", vault: Vault | None = None
    ) -> Decision:
        """Scan and apply the policy. For responses, vault tokens are restored afterwards."""

        vault = vault if vault is not None else Vault()
        if direction == "response" and len(vault):
            return self._process_response(text, vault)
        return self.decide(text, self.scan(text), direction, vault)

    def _process_response(self, text: str, vault: Vault) -> Decision:
        tokens = vault.find(text)
        findings = [f for f in self.scan(text) if not overlaps_any(f, tokens)]
        return self.decide(text, findings, "response", vault, tokens)

    def restore(self, text: str, vault: Vault) -> str:
        return vault.restore(text)
