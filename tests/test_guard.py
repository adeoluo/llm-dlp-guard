from __future__ import annotations

from pathlib import Path

from llm_dlp_guard.guard import Guard, replacement, splice
from llm_dlp_guard.policy import Policy, Rule
from llm_dlp_guard.vault import Vault

PROMPT = "Mail jane.doe@example.org about card 4111 1111 1111 1111 and key AKIAIOSFODNN7ABCDEFG."


def test_prompt_tokenizes_pii_and_redacts_secrets(guard: Guard) -> None:
    vault = Vault(b"v" * 32)
    d = guard.process(PROMPT, "prompt", vault)
    assert d.outcome == "redact"
    assert "jane.doe@example.org" not in d.text and "4111 1111 1111 1111" not in d.text
    assert "[AWS_ACCESS_KEY]" in d.text
    actions = {a.finding.label: a.action for a in d.applied}
    assert actions == {"EMAIL": "tokenize", "CREDIT_CARD": "tokenize", "AWS_ACCESS_KEY": "redact"}
    assert d.to_dict()["outcome"] == "redact"
    # The model answers with the tokens; the user gets the originals back.
    answer = f"Sure, I will email {vault.tokenize('jane.doe@example.org', 'EMAIL')} now."
    back = guard.process(answer, "response", vault)
    assert back.text == "Sure, I will email jane.doe@example.org now."


def test_tokens_in_responses_are_not_redetected(guard: Guard) -> None:
    vault = Vault(b"w" * 32)
    token = vault.tokenize("4111 1111 1111 1111", "CREDIT_CARD")
    out = guard.process(f"card {token} and new key AKIAIOSFODNN7ABCDEFG", "response", vault)
    assert out.text == "card 4111 1111 1111 1111 and new key [AWS_ACCESS_KEY]"
    assert [a.finding.label for a in out.applied] == ["AWS_ACCESS_KEY"]


def test_block() -> None:
    guard = Guard(Policy(rules=[Rule(labels=["EMAIL"], action="block")]))
    d = guard.process("write to a.b@example.org", "prompt")
    assert d.outcome == "block"
    assert [a.finding.label for a in d.blocked] == ["EMAIL"]
    assert d.to_dict()["text"] is None


def test_allow_everything_and_scan_filters() -> None:
    guard = Guard(Policy(rules=[], default_action="allow", allow_values=["a.b@example.org"]))
    d = guard.process("a.b@example.org c.d@example.org")
    assert d.outcome == "allow" and d.text == "a.b@example.org c.d@example.org"
    assert [
        f.text("a.b@example.org c.d@example.org")
        for f in guard.scan("a.b@example.org c.d@example.org")
    ] == ["c.d@example.org"]


def test_replacements() -> None:
    v = Vault(b"r" * 32)
    assert replacement("mask", "ab cd", "X", v, b"k") == "** **"
    assert replacement("redact", "abc", "EMAIL", v, b"k") == "[EMAIL]"
    h = replacement("hash", "abc", "EMAIL", v, b"k")
    assert h.startswith("[EMAIL:") and h == replacement("hash", "abc", "EMAIL", v, b"k")
    assert h != replacement("hash", "abc", "EMAIL", v, b"other")
    assert replacement("allow", "abc", "X", v, b"k") == "abc"
    assert splice("hello world", [(6, 11, "there"), (0, 5, "HI")]) == "HI there"


def test_from_files(tmp_path: Path, tagger_path: Path) -> None:
    pol = tmp_path / "p.yaml"
    pol.write_text("default_action: mask\nrules: []\n", encoding="utf-8")
    guard = Guard.from_files(pol, tagger_path)
    assert guard.has_tagger
    d = guard.process("Contact Lena about the Porto office.")
    assert d.text == "Contact **** about the ***** office."
    assert Guard.from_files().has_tagger is False
    assert guard.restore("x", Vault()) == "x"
    assert guard.category("JWT") == "secret" and guard.category("CITY") == "pii"
