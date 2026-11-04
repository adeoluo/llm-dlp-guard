# ADR-0006: Streaming redaction that equals whole-text redaction

- **Status:** Accepted
- **Date:** 2026-11-04

## Context

Streamed answers arrive a few characters at a time. A secret split across chunks (`ghp_ab` +
`cd...`) is invisible to a per-chunk scanner, and holding the whole answer until the end defeats
streaming. The redactor must never release text that whole-text redaction would have changed.

## Decision

`StreamRedactor` keeps a pending buffer and releases a prefix only when it is safe:

* **Window.** Every pattern is bounded, so its longest match is known (`Pattern.max_length`, from
  Python's regex parser). Text more than `window = max length + 2` characters from the end cannot
  be part of a future match. Today's window is 342 characters.
* **Openers.** Two patterns are longer than that (PEM private keys and JWTs). They start with a
  fixed literal (`-----BEGIN`, `eyJ`), so the redactor holds from the first such opener within
  that pattern's maximum length instead of widening the window for all text.
* **No cuts inside candidates.** The cut moves back to the start of any candidate match (the
  whole match, including keywords before a reported group) or vault token that crosses it.
* **Same scan state.** Matching resumes at the cut with the previous two characters as context,
  so `\b` sees the same neighbours as in a whole-text scan (`find_at` in Rust, `finditer(text,
  pos)` in Python).
* The released prefix goes through the same policy and vault as the batch path.

## Consequences

* With the pattern catalog, streamed output equals `Guard.process` on the whole text for any
  chunking; `tests/test_streaming.py` checks this property on random texts and chunk sizes, in
  both directions and with vault restoration.
* The first characters of an answer are delayed by up to the window (342 characters, about 85
  tokens); text after an opener waits until the key or token ends.
* The tagger sees only the held text, so near cut points its spans can differ from a whole-text
  run; that path is not covered by the equality guarantee.
* A `block` action stops the stream and sends an error event; nothing after the cut is released.
