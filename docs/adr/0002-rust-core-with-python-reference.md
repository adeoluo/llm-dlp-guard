# ADR-0002: A Rust core with a Python reference and parity tests

- **Status:** Accepted
- **Date:** 2026-11-04

## Context

A data loss prevention proxy sits on the request path of every LLM call, in both directions, and
for streamed answers it rescans a sliding window after every token. Scanning cost is paid per
request, so it has to be small next to model latency. The detectors are also security controls:
a silent difference between two implementations (for example a boundary assertion that behaves
differently) is a leak, so the behaviour has to be pinned down exactly.

## Decision

Implement the hot paths in Rust (`rust/src`): the Aho-Corasick prefilter, pattern matching,
validators, overlap resolution, the tokenizer, CRF feature hashing, Viterbi, forward-backward and
training. Compile them into the package with maturin as `llm_dlp_guard.llm_dlp_guard_rs` and use
them by default. Keep a line-for-line Python implementation (`reference.py`, `validators.py`) as
the reference, selectable with `backend="python"`.

* The pattern catalog lives in Python and is handed to Rust as JSON, so custom patterns from a
  policy go through the same engine.
* All offsets crossing the boundary are code point offsets (Python `str` indices).
* Regular expressions use the subset that Rust's `regex` and Python's `re` agree on: no
  lookaround, explicit ASCII classes, no `\s` (whose definitions differ at U+001C to U+001F).

## Alternatives considered

* **Python only:** simplest, but the reference engine is about 20 times slower on the same text
  (see the throughput table), and the tagger's forward-backward in numpy is slower still.
* **Rust only:** fastest to maintain one implementation, but loses the independent check that
  catches semantic drift between what the catalog says and what the engine does.

## Consequences

* `tests/test_parity.py` checks identical candidates, resolved findings, tokens, gold tags,
  feature ids, Viterbi paths, span scores (to 1e-5) and log-likelihoods on property-based text
  and on the secrets corpus, in CI with the extension required.
* The engines agree on which characters are word characters except combining marks, other
  numbers such as `¼`, connector punctuation other than `_`, zero-width joiners, circled Latin
  letters and code points newer than Python's Unicode database. Property tests leave these out;
  they only move `\b` boundaries next to such characters.
* Building from source needs a Rust toolchain; release wheels include the compiled core.
