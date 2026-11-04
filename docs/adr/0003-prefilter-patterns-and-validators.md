# ADR-0003: An anchor prefilter, bounded patterns and validators for precision

- **Status:** Accepted
- **Date:** 2026-11-04

## Context

Credentials have documented shapes (an `AKIA` prefix, a `ghp_` prefix and a CRC-32 suffix, PEM
framing), while many PII formats are just digit runs that also look like order numbers, version
strings or hashes. A detector that only matches shapes either floods users with false alarms or
is tuned so loosely that it misses real data. Running 29 regular expressions over every request
is also wasteful when most texts contain none of their literal prefixes.

## Decision

* Every pattern lists **anchors**, literals of which at least one occurs in every match. One
  Aho-Corasick pass over the text (ASCII case-insensitive) selects the patterns to run; patterns
  without a usable literal (card numbers, phone numbers, IPv4) always run.
* Every match goes through a **validator** where the format has one: Luhn for cards (plus card
  network prefixes), IBAN registry length and mod-97, IPv4 and IPv6 parsing with lower scores for
  private, loopback and documentation ranges, JWT parts that decode to JSON objects with an `alg`,
  the GitHub token checksum (a failed checksum lowers the score instead of dropping the match),
  SSA rules for US social security numbers, and Shannon entropy plus placeholder and code
  rejection for values assigned to secret-like keys.
* Every pattern is **bounded** (no unbounded repetition), which streaming needs (ADR-0006).
* Overlapping matches resolve by score, then length, then position, then catalog order.

## Alternatives considered

* **Entropy everywhere** (as detect-secrets' high-entropy plugins do): high recall on unknown
  token formats, but every hash, UUID and base64 blob becomes a finding; the secrets benchmark
  shows the cost.
* **A RegexSet pass instead of Aho-Corasick:** also selects patterns, but builds a much larger
  automaton than the anchor set and still needs a second pass for positions.

## Consequences

* The card pattern misses card-like numbers without a network prefix or a valid Luhn digit
  (most of the synthetic card numbers in ai4privacy), which the tagger covers instead.
* Keyword secrets need an assignment (`key = value`, `password: value`, `password is: value`);
  a password mentioned in free prose without one is left to the tagger.
