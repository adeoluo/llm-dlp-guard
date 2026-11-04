"""Streaming redaction for token streams and server-sent events.

:class:`StreamRedactor` accepts text in arbitrary chunks and emits text as soon as it is safe.
It holds back a window as long as the longest possible pattern match (plus two characters for
boundary assertions) and never cuts through a candidate match or a vault token, and it resumes
matching at the cut with the earlier text kept as context. For long patterns that have an
``opener`` literal (private key blocks, JWTs) it holds back from the opener instead. With the
pattern catalog alone the emitted text equals :meth:`Guard.process` on the whole text for any
chunking (``tests/test_streaming.py`` checks this property). The tagger only sees the held text,
so its spans may differ slightly at chunk boundaries.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from llm_dlp_guard.findings import Finding, resolve
from llm_dlp_guard.guard import Applied, Guard, overlaps_any
from llm_dlp_guard.policy import Direction
from llm_dlp_guard.vault import Vault

# Characters of already-emitted text kept as context for assertions like \b.
_CONTEXT = 2
# Extra hold-back while the tagger is on (it uses two tokens of context on each side).
_TAGGER_WINDOW = 64


def stream_window(guard: Guard) -> tuple[int, list[tuple[str, int | None]]]:
    """Hold-back window and the ``(opener, max length)`` of patterns longer than it."""

    window = 0
    long: list[tuple[str, int | None]] = []
    for p in guard.catalog.patterns:
        n = p.max_length()
        if p.opener is not None and (n is None or n > 512):
            long.append((p.opener, n))
            continue
        if n is None:
            raise ValueError(
                f"pattern {p.name!r} has unbounded length and no opener; it cannot be streamed"
            )
        window = max(window, n)
    window += 2
    if guard.has_tagger:
        window = max(window, _TAGGER_WINDOW)
    return window, long


@dataclass
class StreamRedactor:
    """Incremental :meth:`Guard.process` for one stream (one direction, one vault)."""

    guard: Guard
    direction: Direction = "response"
    vault: Vault = field(default_factory=Vault)
    blocked: bool = False
    applied: list[Applied] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._window, self._long = stream_window(self.guard)
        self._buffer = ""  # context + pending text
        self._offset = 0  # where pending text starts in _buffer

    @property
    def window(self) -> int:
        return self._window

    def feed(self, chunk: str) -> str:
        """Add ``chunk``; return the text that is now safe to emit."""

        if self.blocked:
            return ""
        self._buffer += chunk
        return self._drain(final=False)

    def flush(self) -> str:
        """Emit everything still held at the end of the stream."""

        if self.blocked:
            return ""
        return self._drain(final=True)

    def _restoring(self) -> bool:
        return self.direction == "response" and len(self.vault) > 0

    def _drain(self, final: bool) -> str:
        buf, off = self._buffer, self._offset
        n = len(buf)
        if final:
            safe = n
        else:
            hold = self._window
            if self._restoring():
                hold = max(hold, self.vault.max_token_length + 2)
            safe = n - hold
            for opener, max_len in self._long:
                lo = off if max_len is None else max(off, n - max_len - 2)
                q = buf.find(opener, lo)
                if q >= 0:
                    safe = min(safe, q)
            # A partial opener at the very end is covered by the window (openers are short).
        if safe <= off:
            return ""
        candidates = self.guard.detector.candidates(buf, off)
        tokens = self.vault.find(buf, off) if self._restoring() else []
        if not final:
            # Never cut through a candidate or a token; a token touching the end may grow.
            spans = [f.extent for f in candidates]
            spans += [(s, e if e < n else n + 1) for s, e in tokens]
            moved = True
            while moved:
                moved = False
                for s, e in spans:
                    if s < safe < e:
                        safe = s
                        moved = True
            if safe <= off:
                return ""
        inside = [f for f in candidates if f.extent[1] <= safe]
        tokens = [(s, e) for s, e in tokens if e <= safe]
        findings = [
            f
            for f in resolve(inside)
            if self.guard.policy.keep(f, buf) and not overlaps_any(f, tokens)
        ]
        decision = self.guard.decide(
            buf[off:safe],
            [_shift(f, -off) for f in findings],
            self.direction,
            self.vault,
            [(s - off, e - off) for s, e in tokens],
        )
        self.applied.extend(decision.applied)
        if decision.outcome == "block":
            self.blocked = True
            return ""
        keep_from = max(0, safe - _CONTEXT)
        self._buffer = buf[keep_from:]
        self._offset = safe - keep_from
        return decision.text


def _shift(f: Finding, delta: int) -> Finding:
    ms, me = f.extent
    return Finding(
        f.start + delta, f.end + delta, f.label, f.source, f.score, f.rank, ms + delta, me + delta
    )


def redact_stream(
    guard: Guard,
    chunks: Iterable[str],
    direction: Direction = "response",
    vault: Vault | None = None,
) -> Iterator[str]:
    """Yield redacted text for an iterable of chunks (empty strings are skipped)."""

    redactor = StreamRedactor(guard, direction, vault if vault is not None else Vault())
    for chunk in chunks:
        out = redactor.feed(chunk)
        if out:
            yield out
    tail = redactor.flush()
    if tail:
        yield tail


@dataclass
class SSEStream:
    """Rewrites an OpenAI-style chat completion SSE stream through one redactor per choice.

    Feed raw upstream lines; get back lines to forward. ``delta.content`` is replaced by the
    redacted text released so far; held text is released before ``data: [DONE]``. If the policy
    blocks, an error event is sent and the stream ends.
    """

    guard: Guard
    vault: Vault
    direction: Direction = "response"
    done: bool = False
    _redactors: dict[int, StreamRedactor] = field(default_factory=dict)
    _template: dict[str, object] | None = None

    def _redactor(self, index: int) -> StreamRedactor:
        if index not in self._redactors:
            self._redactors[index] = StreamRedactor(self.guard, self.direction, self.vault)
        return self._redactors[index]

    @property
    def blocked(self) -> bool:
        return any(r.blocked for r in self._redactors.values())

    def applied(self) -> list[Applied]:
        return [a for r in self._redactors.values() for a in r.applied]

    def _blocked_event(self) -> list[str]:
        self.done = True
        error = {
            "error": {
                "type": "dlp_policy_violation",
                "message": "the response was stopped by the data loss prevention policy",
            }
        }
        return [f"data: {json.dumps(error)}", "", "data: [DONE]", ""]

    def _tail_events(self) -> list[str]:
        out: list[str] = []
        for index, redactor in sorted(self._redactors.items()):
            tail = redactor.flush()
            if redactor.blocked:
                return self._blocked_event()
            if tail:
                event = dict(self._template or {"object": "chat.completion.chunk"})
                event["choices"] = [
                    {"index": index, "delta": {"content": tail}, "finish_reason": None}
                ]
                out += [f"data: {json.dumps(event)}", ""]
        return out

    def feed_line(self, line: str) -> list[str]:
        """Process one SSE line (without the trailing newline)."""

        if self.done:
            return []
        if not line.startswith("data:"):
            return [line]
        payload = line[5:].strip()
        if payload == "[DONE]":
            out = self._tail_events()
            if self.done:
                return out
            self.done = True
            return [*out, "data: [DONE]"]
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return [line]
        if not isinstance(event, dict):
            return [line]
        self._template = {k: v for k, v in event.items() if k not in ("choices", "usage")}
        choices = event.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                index = int(choice.get("index", 0))
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    redactor = self._redactor(index)
                    delta["content"] = redactor.feed(delta["content"])
                    if redactor.blocked:
                        return self._blocked_event()
                if choice.get("finish_reason") is not None and index in self._redactors:
                    tail = self._redactors[index].flush()
                    if self._redactors[index].blocked:
                        return self._blocked_event()
                    if tail:
                        if not isinstance(delta, dict):
                            delta = {}
                            choice["delta"] = delta
                        delta["content"] = (delta.get("content") or "") + tail
        return [f"data: {json.dumps(event)}"]
