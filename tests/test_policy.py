from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_dlp_guard.findings import Finding
from llm_dlp_guard.policy import CustomPattern, Policy, Rule


def test_default_rules() -> None:
    p = Policy()
    key = Finding(0, 1, "PRIVATE_KEY", "x", 1.0)
    tok = Finding(0, 1, "GITHUB_TOKEN", "x", 1.0)
    mail = Finding(0, 1, "EMAIL", "x", 1.0)
    assert p.action_for(key, "secret", "prompt") == "block"
    assert p.action_for(tok, "secret", "response") == "redact"
    assert p.action_for(mail, "pii", "prompt") == "tokenize"
    assert p.action_for(mail, "pii", "response") == "allow"


def test_rule_matching() -> None:
    r = Rule(labels=["EMAIL"], directions=["prompt"], action="mask")
    assert r.matches("EMAIL", "pii", "prompt")
    assert not r.matches("EMAIL", "pii", "response")
    assert not r.matches("PHONE", "pii", "prompt")
    assert Rule(action="allow").matches("ANY", "secret", "response")


def test_allowlist_and_floor() -> None:
    p = Policy(allow_values=["help@example.com"], allow_patterns=[r".*@acme\.test"], min_score=0.6)
    text = "help@example.com bob@acme.test eve@example.org"
    assert not p.keep(Finding(0, 16, "EMAIL", "e", 0.9), text)
    assert not p.keep(Finding(17, 30, "EMAIL", "e", 0.9), text)
    assert p.keep(Finding(31, 46, "EMAIL", "e", 0.9), text)
    assert not p.keep(Finding(31, 46, "EMAIL", "e", 0.5), text)


def test_catalog_customisation() -> None:
    p = Policy(
        disabled_patterns=["phone"],
        custom_patterns=[
            CustomPattern(name="emp", label="EMPLOYEE_ID", regex="EMP-[0-9]{6}", anchors=["EMP-"])
        ],
    )
    cat = p.catalog()
    names = [x.name for x in cat.patterns]
    assert "phone" not in names and names[-1] == "emp"
    assert p.category_of("EMPLOYEE_ID", cat) == "pii"
    assert p.category_of("UNKNOWN_NER_LABEL", cat) == "pii"
    with pytest.raises(ValueError, match="unknown patterns"):
        Policy(disabled_patterns=["nope"]).catalog()


def test_validation_errors() -> None:
    with pytest.raises(ValidationError):
        Policy(min_score=2)
    with pytest.raises(ValidationError):
        Policy(rules=[{"action": "explode"}])  # type: ignore[list-item]
    with pytest.raises(ValidationError):
        Policy(allow_patterns=["("])
    with pytest.raises(ValidationError):
        CustomPattern(name="x", label="X", regex="(")
    with pytest.raises(ValidationError):
        Policy(unknown_field=1)  # type: ignore[call-arg]


def test_load_yaml_and_json(tmp_path: Path) -> None:
    y = tmp_path / "p.yaml"
    y.write_text(
        "min_score: 0.7\nrules:\n  - categories: [secret]\n    action: block\n", encoding="utf-8"
    )
    assert Policy.load(y).min_score == 0.7
    assert Policy.load(y).rules[0].action == "block"
    j = tmp_path / "p.json"
    j.write_text(json.dumps({"default_action": "mask"}), encoding="utf-8")
    assert Policy.load(j).default_action == "mask"
    empty = tmp_path / "e.yml"
    empty.write_text("", encoding="utf-8")
    assert Policy.load(empty).default_action == "tokenize"
