# ADR-0004: A linear-chain CRF over hashed features for free-form PII

- **Status:** Accepted
- **Date:** 2026-11-04

## Context

Names, streets, cities, dates of birth and many national identifiers have no checkable format;
they need a learned tagger. The guard runs in a proxy on CPU, in front of every request, and on a
sliding window of streamed answers. The training data (ai4privacy/pii-masking-400k) has 325,517
records in six languages and 17 labels.

## Decision

A linear-chain CRF with BIO tags, trained and run in Rust:

* 19 hashed features per token (the word, prefixes and suffixes, word shape, length, whether
  whitespace precedes and follows, the two words on each side, neighbouring shapes, bigrams),
  FNV-1a hashed into 2^21 buckets, so the model needs no vocabulary and handles all six languages
  in one set of weights.
* Training minimises the negative log-likelihood with AdaGrad on mini-batches whose gradients are
  computed in parallel; forward-backward runs in scaled probability space.
* Decoding is Viterbi; each span's score is the mean marginal probability of its tokens, and the
  detector drops spans below 0.5.
* Hyperparameters (learning rate, epochs) are chosen on a dev split carved from the training
  data, never on the validation split.

## Alternatives considered

* **A small transformer (for example DistilBERT) fine-tuned for token classification:** stronger
  on context-dependent entities, but 60 to 250 times more compute per token, a GPU for training at
  this scale, and a heavier deployment for a proxy.
* **spaCy's general NER, as Presidio uses:** no training needed, but it finds people,
  organisations and places, not the 17 PII types, and the Presidio comparison shows the gap.

## Consequences

* Training on all of ai4privacy takes about four minutes on 8 cores; the model is 244 MB on disk
  and scans tens of MB per second.
* The CRF only sees two tokens of context on each side, so it confuses look-alike numbers
  (account, tax and ID numbers) and drops in recall on unfamiliar document styles, as the
  Nemotron-PII cross-dataset numbers show.
* The model is trained on data under an academic, non-commercial licence; its weights are not
  distributed (see the data card).
