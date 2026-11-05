# ADR-0007: Evaluation protocol

- **Status:** Accepted
- **Date:** 2026-11-04

## Context

PII detectors are easy to flatter: tune on the test split, count any overlap as a hit, drop the
labels a tool cannot produce, or compare tools on different texts. Secrets detectors have no
freely available labelled corpus.

## Decision

* **ai4privacy:** tune on a dev split carved from the training data (records whose uid hash ends
  in 9), train once on all training records, score the 81,379-record validation split once. The
  headline is exact entity matching (same start, end and label), micro-averaged, per label and per
  language.
* **Cross-tool comparison (Presidio):** both tools run on the same texts; labels map to shared
  coarse types; adjacent same-type spans separated only by whitespace are merged on both sides;
  predictions of types a dataset does not annotate are dropped; exact and overlap scores are both
  reported. Nemotron-PII annotations without a coarse type (company names, generic dates, URLs)
  are ignore regions: predictions that only overlap them count neither way.
* **Nemotron-PII** is never used for training or tuning; it measures transfer to a different
  generator and document style.
* **Secrets:** a generated corpus of real formats with fake values and hard negatives (ADR-0003);
  detection counts on overlap with the secret, precision counts every reported span, and the
  false-alarm rate is the share of negatives with any report. detect-secrets runs through the
  same code path as `detect-secrets scan`, with two file types because its keyword rules depend
  on the file type.
* Every number in the README comes from a script in `benchmarks/` whose raw output is committed in
  `benchmarks/results/`.

## Consequences

* Exact matching penalises boundary disagreements (a phone number with or without its country
  code), so overlap scores are reported next to it.
* The secrets corpus measures format coverage and false alarms on look-alikes, not recall on
  unknown formats; detect-secrets' entropy plugins are designed for the latter.
