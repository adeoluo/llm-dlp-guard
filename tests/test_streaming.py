from __future__ import annotations

import json
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llm_dlp_guard import secrets_corpus as sc
from llm_dlp_guard.catalog import Catalog, Pattern
from llm_dlp_guard.guard import Guard
from llm_dlp_guard.policy import CustomPattern, Policy, Rule
from llm_dlp_guard.streaming import SSEStream, StreamRedactor, redact_stream, stream_window
from llm_dlp_guard.vault import Vault

_RNG = random.Random(4)
PIECES = [
    "jane.doe@example.org", "4111 1111 1111 1111", "+44 20 7946 0958", "8.8.8.8", "123-45-6789",
    sc.aws_access_key(_RNG), sc.github_token(_RNG), sc.jwt(_RNG), sc.private_key(_RNG), sc.stripe_key(_RNG),
    "password: q8Zr2LmP0xTv9KcW", "postgres://app:Zq8rT2mPx@db/x", "Authorization: Bearer " + sc.bearer(_RNG),
    " ", " ", "\n", "hello", "x", "-----BEGIN", "eyJ", "@", "AKIA", "token=", "Zoë", "1234",
]  # fmt: skip

TEXTS = st.lists(st.sampled_from(PIECES), max_size=14).map("".join)
CUTS = st.lists(st.integers(1, 60), min_size=1, max_size=40)


def chunked(text: str, sizes: list[int]) -> list[str]:
    out, i, k = [], 0, 0
    while i < len(text):
        n = sizes[k % len(sizes)]
        out.append(text[i : i + n])
        i += n
        k += 1
    return out


GUARD = Guard(Policy(rules=[Rule(categories=["secret"], action="redact")]))


@settings(max_examples=300, deadline=None)
@given(TEXTS, CUTS, st.sampled_from(["prompt", "response"]))
def test_stream_equals_batch(text: str, sizes: list[int], direction: str) -> None:
    key = b"s" * 32
    batch = GUARD.process(text, direction, Vault(key))  # type: ignore[arg-type]
    streamed = "".join(redact_stream(GUARD, chunked(text, sizes), direction, Vault(key)))  # type: ignore[arg-type]
    assert streamed == batch.text


@settings(max_examples=150, deadline=None)
@given(
    st.lists(
        st.sampled_from(["Lena", "Lena Sato", "the card", "4111 1111 1111 1111", " ", "ok."]),
        max_size=12,
    ).map("".join),
    CUTS,
)
def test_stream_restores_tokens_like_batch(text: str, sizes: list[int]) -> None:
    vault = Vault(b"t" * 32)
    for value, label in (
        ("Lena", "GIVEN_NAME"),
        ("Lena Sato", "PERSON"),
        ("4111 1111 1111 1111", "CREDIT_CARD"),
    ):
        text = (
            text.replace(value, vault.tokenize(value, label))
            if value != "4111 1111 1111 1111"
            else text
        )
    batch = GUARD.process(text, "response", vault)
    streamed = "".join(redact_stream(GUARD, chunked(text, sizes), "response", vault))
    assert streamed == batch.text


def test_window_and_long_patterns() -> None:
    window, long = stream_window(GUARD)
    assert 100 < window < 500
    assert {o for o, _ in long} == {"-----BEGIN", "eyJ"}
    unbounded = Guard(Policy(custom_patterns=[CustomPattern(name="u", label="U", regex="Q+")]))
    with pytest.raises(ValueError, match="unbounded"):
        stream_window(unbounded)


def test_text_is_released_before_the_end() -> None:
    redactor = StreamRedactor(GUARD, "response", Vault())
    out = "".join(redactor.feed("word ") for _ in range(400))
    assert len(out) > 1000
    assert redactor.window == stream_window(GUARD)[0]


def test_private_key_split_across_many_chunks_never_leaks() -> None:
    key = sc.private_key(random.Random(9))
    body_line = key.splitlines()[1]
    text = "here:\n" + key + "\nend"
    out = "".join(redact_stream(GUARD, chunked(text, [3, 7, 11]), "response", Vault()))
    assert body_line not in out and "[PRIVATE_KEY]" in out


def test_block_stops_the_stream() -> None:
    guard = Guard()  # PRIVATE_KEY -> block
    redactor = StreamRedactor(guard, "response", Vault())
    key = sc.private_key(random.Random(2))
    emitted = redactor.feed("safe text " * 60)
    emitted += redactor.feed(key) + redactor.flush()
    assert redactor.blocked
    assert "PRIVATE KEY" not in emitted
    assert redactor.feed("more") == "" and redactor.flush() == ""


def _sse(texts: list[str], finish_inline: bool = False) -> list[str]:
    lines = []
    for i, t in enumerate(texts):
        last = finish_inline and i == len(texts) - 1
        event = {
            "id": "c1",
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"content": t}, "finish_reason": "stop" if last else None}
            ],
        }
        lines += [f"data: {json.dumps(event)}", ""]
    return [*lines, "data: [DONE]"]


def _content(lines: list[str]) -> str:
    out = ""
    for line in lines:
        if line.startswith("data: {"):
            event = json.loads(line[6:])
            for c in event.get("choices", []):
                out += (c.get("delta") or {}).get("content") or ""
    return out


@pytest.mark.parametrize("finish_inline", [False, True])
def test_sse_rewrites_content(finish_inline: bool) -> None:
    vault = Vault(b"e" * 32)
    token = vault.tokenize("jane.doe@example.org", "EMAIL")
    secret = sc.github_token(random.Random(3))
    text = f"I will write to {token}. Your token {secret} was leaked."
    stream = SSEStream(GUARD, vault)
    out: list[str] = []
    for line in [
        ": keep-alive",
        "data: not json",
        "data: [1]",
        *_sse(chunked(text, [5]), finish_inline),
    ]:
        out += stream.feed_line(line)
    assert (
        _content(out)
        == "I will write to jane.doe@example.org. Your token [GITHUB_TOKEN] was leaked."
    )
    assert out[-1] == "data: [DONE]" and stream.done
    assert stream.feed_line("data: late") == []
    assert ": keep-alive" in out and "data: not json" in out
    assert [a.finding.label for a in stream.applied()] == ["GITHUB_TOKEN"]


def test_sse_block_sends_error_event() -> None:
    stream = SSEStream(Guard(), Vault())
    key = sc.private_key(random.Random(5))
    out: list[str] = []
    for line in _sse(chunked("x " + key + " y", [40])):
        out += stream.feed_line(line)
    assert stream.blocked and stream.done
    assert any("dlp_policy_violation" in line for line in out)
    assert "PRIVATE KEY-----\n" not in _content(out)


def test_custom_long_pattern_with_opener_streams() -> None:
    guard = Guard(
        Policy(
            custom_patterns=[
                CustomPattern(
                    name="blob",
                    label="BLOB",
                    regex="BLOB<[a-z]{0,2000}>",
                    opener="BLOB<",
                    category="secret",
                )
            ]
        )
    )
    text = "a BLOB<" + "q" * 900 + "> b"
    assert "".join(redact_stream(guard, chunked(text, [13]), "prompt")) == "a [BLOB] b"
    assert Catalog([Pattern("x", "X", "pii", "x", 0.5)]).to_json().startswith('{"patterns"')
