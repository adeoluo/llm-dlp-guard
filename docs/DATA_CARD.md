# Data Card - LLM DLP Guard

## Sources and provenance

| Dataset | Source | Use here | Size used |
|---|---|---|---|
| ai4privacy/pii-masking-400k | [Hugging Face](https://huggingface.co/datasets/ai4privacy/pii-masking-400k), synthetic, 2024 | Tagger training (train), tuning (a dev split of train), evaluation (validation) | 325,517 train and 81,379 validation records; 48.6 M and 12.1 M characters; 355,376 and 88,810 entities |
| nvidia/Nemotron-PII | [Hugging Face](https://huggingface.co/datasets/nvidia/Nemotron-PII), synthetic, 2025 | Cross-dataset evaluation only (test split) | 100,000 documents (50,000 personas, each in a US and an international version), 95.9 M characters |
| Generated secrets corpus | `llm_dlp_guard.secrets_corpus`, seeded | Secrets evaluation | 46,000 positives (21 credential types) and 46,000 hard negatives per run |

`scripts/download_data.sh` fetches both public datasets into `../.bench-data/llm-dlp-guard`.

## Licensing

- **ai4privacy/pii-masking-400k:** academic and non-commercial use only, no redistribution or
  derivative works without a licence from AI4Privacy. Nothing from it is committed here: no
  records, no samples in tests, and no trained weights (a model trained on it is a derivative
  work). Benchmarks report aggregate metrics only. Train your own model with `make train`.
- **Nemotron-PII:** CC BY 4.0 (NVIDIA). Used for evaluation, cited here.
- **Secrets corpus:** produced by `secrets_corpus.py` in this repository (MIT). Values are random; none is a live
  credential, and the corpus is never committed (it is regenerated from its seed).

## Composition

- **ai4privacy:** short texts (mean 149 characters) in English, German, French, Italian,
  Spanish and Dutch: forms, emails, chat lines, HTML fragments. 17 labels: given name, surname,
  username, city, street, building number, zip code, date of birth, email, phone, ID card,
  driver's licence, passport-like account numbers, social number, tax number, credit card and
  password.
- **Nemotron-PII test:** English documents (mean 959 characters) from 50 industries, US and
  international, structured and unstructured, with 55 span labels; 19 of them map to the coarse
  types compared here, the rest are treated as ignore regions.
- **Secrets corpus:** one-line to multi-line snippets (env files, YAML, JSON, code, shell, logs,
  chat) with one secret each, and negatives with hashes, UUIDs, base64, placeholders, templated
  values, code expressions and prose about passwords.

## Preprocessing and filtering

ai4privacy labels are renamed (`GIVENNAME` to `GIVEN_NAME` and so on) and converted to BIO tags
over the tokenizer's tokens; a token belongs to the first span it overlaps. Nemotron-PII rows
are read as they are (every row is a distinct document). No other cleaning.

## Contamination checks

Hyperparameters were chosen on records of the training split (uid hash ending in 9), not on
validation. Nemotron-PII is a different generator and was not used for any decision. The secrets
corpus uses a different seed for the throughput run (1) than for the detection run (0).

## Known biases and gaps

- Both PII datasets are synthetic. ai4privacy values are often unrealistic (card numbers that
  fail Luhn, random-looking usernames used as names), which favours a learned tagger over
  validators and makes pattern precision look worse than on real data.
- Names come from the generators' name lists; coverage of names from other cultures is unknown.
- The secrets corpus contains only the formats it generates; it says nothing about recall on
  credential formats outside the catalog.

## Ethical considerations

No real personal data or live credentials are used. Test fixtures are fictional sentences built
from short name and city lists, and fake credentials are generated at test time, so the
repository contains no credential-shaped strings that secret scanners would flag.
