# ADR-0005: Reversible redaction with format-preserving tokens in a per-request vault

- **Status:** Accepted
- **Date:** 2026-11-04

## Context

Removing personal data from prompts protects it from the provider, but a mask such as `[EMAIL]`
breaks tasks that need the value's role ("draft a reply to this customer") and leaves the answer
full of placeholders. Users want the provider to see stand-ins and to get their real values back.

## Decision

* The `tokenize` action replaces a value with a token of the same shape: digits stay digits,
  letters keep case, separators stay. The characters come from HMAC-SHA256 under a random 32-byte
  key per vault, keyed by label and value, so a value maps to one token within a conversation.
* Tokens stay valid where validity matters to the model: cards keep a correct Luhn digit, IBANs
  keep their country and get correct check digits, email addresses keep `@` and the top-level
  domain.
* The vault maps tokens back to values. In responses, whole-token occurrences are restored after
  the response has been scanned, and findings that are vault tokens are skipped, so a tokenized
  card is not re-detected as a new card.
* The proxy creates one vault per request; nothing is persisted and the key never leaves memory.
* Secrets default to `redact` (a label), not `tokenize`: they should not be restored into model
  output, and the model does not need their shape.

## Alternatives considered

* **Format-preserving encryption (FF1/FF3):** reversible without a table, but needs careful key
  handling and still needs post-fixes for checksums; a per-request table is simpler and is
  discarded with the request.
* **Placeholders like `<EMAIL_1>`:** simplest and unambiguous to restore, but models sometimes
  rewrite or drop them and they lose the format cues.

## Consequences

* Restoration needs the model to echo a token verbatim; a model that reformats a token (spaces in
  a card number, a capitalised name) leaves it unrestored, which fails safe.
* A shape-preserving token can collide with a real value in the text by chance; collisions with
  other vault entries are rejected, and short tokens only restore on word boundaries.
