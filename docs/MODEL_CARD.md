# Model / System Card - LLM DLP Guard

## Overview

- **What it is:** a data loss prevention layer for LLM traffic. A pattern engine with
  validators finds credentials and formatted PII; a linear-chain CRF tagger finds names,
  addresses, dates of birth and identifiers without a checkable format. A policy decides, per
  label and direction, whether to allow, mask, redact, hash, tokenize (reversibly) or block.
- **Models:** one CRF tagger trained here on ai4privacy/pii-masking-400k (17 labels, six
  languages, 2^21 hashed feature buckets, learning rate 0.1, 6 epochs, seed 0). No pretrained or
  hosted model is involved, and no LLM is used for detection.
- **Version and date:** 0.1.0, November 2026.

## Intended use

- **Primary use cases:** an OpenAI-compatible proxy in front of LLM providers, a library call in
  applications that build prompts, and a CLI for scanning logs and files before they are shared
  with an assistant.
- **Primary users:** platform and security teams running LLM integrations.

## Out-of-scope use

- A compliance guarantee: detection is probabilistic and the tagger misses entities.
- Scanning for secrets in source repositories at scale (dedicated scanners with verification of
  live credentials do this better).
- Making decisions about individuals.

## Performance

See [EVALUATION_REPORT.md](EVALUATION_REPORT.md). In short, on the ai4privacy validation split
(81,379 records) the tagger reaches micro F1 0.882 with exact span matching (precision 0.927,
recall 0.841). On Nemotron-PII documents, which the tagger never saw, recall falls sharply;
pattern-based types (email, card, IP address, API keys) hold up, names and addresses do not.
English and Dutch are the weakest of the six languages.

## Risks and limitations

- Misses lead to disclosure; false alarms lead to broken prompts. The defaults favour
  tokenizing PII (reversible) and redacting secrets.
- The tagger sees two tokens of context on each side and confuses look-alike number types.
- Encoded or obfuscated secrets are not detected.
- **Dual use:** the secrets generator produces strings in real credential formats. They are
  random and inert, but the generator could be used to create decoys; it creates no working
  credentials.

## Safety mitigations

- Per-request vaults with random keys; no request or response text is logged.
- `block` for private keys by default, with an error that names labels, never values.
- Streaming redaction never releases text that whole-text redaction would change (pattern
  catalog, property tested).

## Environmental and cost footprint

Training the final tagger takes about four minutes on an 8-core Apple M1 CPU; the hyperparameter
grid took about 15 minutes. All benchmarks together ran in about an hour on the same laptop. No
GPU was used.
