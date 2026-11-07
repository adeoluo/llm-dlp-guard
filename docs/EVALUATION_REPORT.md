# Evaluation Report: LLM DLP Guard

Every number here comes from a script in `benchmarks/` with its raw output committed in
`benchmarks/results/`; `make evaluation` reruns all of them (data first: `make data`). The
protocol and its reasons are in [ADR-0007](adr/0007-evaluation-protocol.md).

## Setup

- **Hardware:** Apple M1, 8 cores, 8 GB, macOS, Python 3.12, Rust stable (release build). The
  machine was shared with other jobs during every run; the throughput file records the load
  average at write time.
- **Data** (details in [DATA_CARD.md](DATA_CARD.md)):
  - ai4privacy/pii-masking-400k: 325,517 training and 81,379 validation records in six languages
    (88,810 validation entities, 12.1 M characters). The validation split is used once, for the
    final scores.
  - nvidia/Nemotron-PII test split: 100,000 documents, never used for training or tuning.
  - A generated secrets corpus (`llm_dlp_guard.secrets_corpus`): 46,000 secrets in 21 formats
    with random values, embedded in env files, YAML, JSON, code, shell and logs, and 46,000 hard
    negatives; 10.5 M characters per seed; seeds 0, 1 and 2.
- **Metrics:** entity-level exact match (same start, end and label), micro-averaged, plus macro
  F1, per label and per language; overlap match for the cross-tool comparison. Secrets: recall
  over planted secrets (overlap), precision over reported spans, false-alarm rate as the share of
  negative snippets with any report, label accuracy.
- **Intervals:** 95% percentile bootstrap over documents with 1,000 resamples (seed 0) for PII
  scores, paired bootstrap for differences between systems on the same resampled documents,
  Wilson intervals for secrets proportions, three seeds for the secrets corpus and for the
  smallest learning-curve point.
- **References:** Microsoft Presidio 2.2 (default recognizers, spaCy `en_core_web_lg`, default
  thresholds) and Yelp detect-secrets 1.5 (default plugins, through the same code path as
  `detect-secrets scan`), installed from PyPI in the `validation` dependency group.

## Tagger training

Grid on a dev split carved from train (records whose uid hash ends in 9), then one training run
on all 325,517 records with the best setting (`benchmarks/results/pii_ai4privacy.json`):

| Learning rate | Epochs | Dev micro F1 | Train seconds |
|---:|---:|---:|---:|
| 0.05 | 3 | 0.8675 | 129.5 |
| 0.1 | 3 | 0.8801 | 99.0 |
| 0.2 | 3 | 0.8827 | 98.4 |
| 0.05 | 6 | 0.8775 | 189.8 |
| **0.1** | **6** | **0.8842** | 286.4 |

Final model: 2^21 hashed feature buckets, 17 labels, 229 s to train, 243.6 MB on disk, peak
resident memory 646 MB.

Learning curve (tagger only, validation split, `pii_ablation.json`):

| Training records | Micro F1 |
|---:|---:|
| 10,000 (seeds 0, 1, 2) | 0.680, 0.674, 0.676 |
| 40,000 | 0.796 |
| 160,000 | 0.860 |
| 325,517 (all) | 0.882 |

The curve is still rising at the full split, so more data of this kind would help.

## PII on ai4privacy validation

| System | Precision | Recall | Micro F1 (95% CI) | Paired F1 difference to default |
|---|---:|---:|---:|---:|
| Patterns + tagger (default) | 0.9107 | 0.8392 | 0.8735 (0.8716 to 0.8755) | |
| Tagger only | 0.9274 | 0.8409 | 0.8820 (0.8803 to 0.8841) | +0.0080 to +0.0091 |
| Patterns only | 0.8280 | 0.4291 | 0.5652 (0.5589 to 0.5715) | -0.3147 to -0.3024 |
| Default without pattern validators | 0.8761 | 0.8402 | 0.8578 (0.8558 to 0.8600) | -0.0164 to -0.0150 |
| Default without the structured-match rule | 0.9122 | 0.8407 | 0.8750 (0.8732 to 0.8771) | +0.0013 to +0.0018 |

The structured-match rule drops tagger spans that overlap a validated pattern match. On this
split the tagger alone is slightly better, mostly because ai4privacy annotates some values that
fail their checksum (credit card precision for the default is 0.615) and the tagger learned the
dataset's boundaries. The default keeps patterns first because they carry the secret types and
generalise better; on Nemotron-PII the patterns-only path has precision 0.960.

Score floor for tagger spans (`ner_min_score`, default 0.5):

| Floor | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.0 | 0.894 | 0.860 | 0.877 |
| 0.3 | 0.895 | 0.860 | 0.877 |
| 0.5 | 0.911 | 0.839 | 0.874 |
| 0.7 | 0.941 | 0.762 | 0.842 |
| 0.9 | 0.955 | 0.616 | 0.749 |

Per language (default): German 0.880, English 0.835, Spanish 0.897, French 0.897, Italian 0.894,
Dutch 0.840 F1. Per label, the strongest are email (0.984), username (0.964) and phone (0.934);
the weakest are credit card (0.725; precision 0.615), surname (0.742; recall 0.651), account
number (0.758) and building number (0.793).

## Comparison with Presidio

| Data | System | Exact P | Exact R | Exact F1 (95% CI) | Overlap F1 (95% CI) | MB/s |
|---|---|---:|---:|---:|---:|---:|
| ai4privacy English (17,046 records) | patterns + tagger | 0.915 | 0.813 | 0.861 (0.857 to 0.865) | 0.872 (0.868 to 0.876) | 1.42 |
| | patterns only | 0.747 | 0.128 | 0.219 (0.212 to 0.226) | 0.228 (0.221 to 0.235) | 42.6 |
| | Presidio 2.2 | 0.262 | 0.366 | 0.305 (0.299 to 0.311) | 0.348 (0.342 to 0.354) | 0.022 |
| Nemotron-PII (5,000 seeded docs) | patterns + tagger | 0.677 | 0.550 | 0.607 (0.597 to 0.616) | 0.684 (0.676 to 0.693) | 1.76 |
| | patterns only | 0.960 | 0.307 | 0.466 (0.456 to 0.476) | 0.469 (0.459 to 0.479) | 71.5 |
| | Presidio 2.2 | 0.577 | 0.595 | 0.586 (0.577 to 0.596) | 0.669 (0.659 to 0.678) | 0.018 |
| Nemotron-PII (all 100,000 docs) | patterns + tagger | 0.680 | 0.548 | 0.607 | 0.687 | 0.36 |

Paired F1 differences against Presidio on the same resampled documents: ai4privacy +0.550 to
+0.563 (exact); Nemotron-PII +0.010 to +0.033 (exact) and +0.005 to +0.027 (overlap) for patterns
plus tagger, and -0.131 to -0.109 (exact) for patterns only. So: a large win in distribution, a
small but consistent win out of distribution with lower recall than Presidio, and a loss for
patterns alone, which only cover formatted types. The timing of the full 100,000-document run
(264.6 s) is not comparable to the sample's and is not used as a throughput figure.

## Secrets

| Scanner | Recall (95% CI) | Precision (95% CI) | False-alarm rate (95% CI) | Reported spans | Seconds |
|---|---:|---:|---:|---:|---:|
| llm-dlp-guard | 0.9994 (0.9991 to 0.9996) | 1.0000 (0.9999 to 1.0000) | 0.0000 (0.0000 to 0.0001) | 45,971 | 0.44 |
| llm-dlp-guard, validators off | 1.0000 (0.9999 to 1.0000) | 0.8185 (0.8153 to 0.8216) | 0.2218 (0.2180 to 0.2256) | 56,203 | 0.46 |
| detect-secrets 1.5 (`.yaml`) | 0.6568 (0.6524 to 0.6611) | 0.9212 (0.9186 to 0.9237) | 0.0734 (0.0711 to 0.0758) | 42,858 | 30.9 |
| detect-secrets 1.5 (`.txt`) | 0.5339 (0.5293 to 0.5384) | 0.9443 (0.9417 to 0.9468) | 0.0363 (0.0347 to 0.0381) | 30,024 | 33.1 |

Seeds 1 and 2: llm-dlp-guard 0.9994 recall and 1.000 precision on both; detect-secrets (`.txt`)
recall 0.5315 and 0.5330. The only misses are 29 generic `key = value` secrets (0.7% of that
type); every other format has recall 1.0. detect-secrets (`.yaml`) finds none of the Bearer, GCP
key id and Hugging Face tokens, about half of the OpenAI, Anthropic, GitHub, Slack, npm, Google
and SendGrid keys, and all AWS access, Azure, GitLab, JWT, private key, Stripe, Twilio and Slack
webhook secrets; its false alarms come from its keyword (1,705) and hex-entropy (1,672) plugins.

**Nemotron-PII secrets check.** 8,754 documents with an annotated API key or password and 8,754
without (4,667 API keys and 6,985 passwords in context):

| Scanner | Recall (95% CI) | Precision (95% CI) | False-alarm rate |
|---|---:|---:|---:|
| llm-dlp-guard | 0.189 (0.182 to 0.196) | 0.809 (0.793 to 0.823) | 0.012 |
| detect-secrets 1.5 (`.yaml`) | 0.175 (0.169 to 0.182) | 0.821 (0.806 to 0.835) | 0.008 |
| detect-secrets 1.5 (`.txt`) | 0.071 (0.067 to 0.076) | 0.734 (0.710 to 0.757) | 0.008 |

Both tools miss most of these: Nemotron-PII keys are mostly free-form strings without a provider
prefix, and many passwords appear in prose rather than after a keyword.

## Throughput

Text: 101,378 documents, 14.6 MB (the 81,379 ai4privacy validation records interleaved with
20,000 generated secrets documents, seed 1). Every Rust row is the median of three full passes;
the minimum and maximum are kept in `benchmarks/results/throughput.json`. The Python reference
and detect-secrets ran on the first 3,000 documents, Presidio on the 1,500 ASCII documents among
them. The machine was shared and busy (load average 18.0 when the file was written; an earlier
run in the same session at load 24.6 measured a third to a half of these rates), so the spread between
passes is wide and the absolute values are lower bounds for an idle M1.

| Engine | Documents | MB/s median | MB/s min | MB/s max |
|---|---:|---:|---:|---:|
| Patterns, Rust, 1 thread | 101,378 | 11.04 | 7.70 | 11.75 |
| Patterns, Rust, batch on 8 threads | 101,378 | 35.71 | 19.89 | 40.06 |
| Patterns + tagger, Rust, 1 thread | 101,378 | 0.728 | 0.349 | 0.935 |
| Patterns + tagger, Rust, batch on 8 threads | 101,378 | 3.89 | 3.75 | 4.04 |
| Patterns, Python reference | 3,000 | 1.89 | | |
| detect-secrets 1.5 line scan | 3,000 | 0.195 | | |
| Presidio 2.2 analyze (`en_core_web_lg`) | 1,500 | 0.004 | | |

Every configuration of this package is faster than both reference tools on the same text.
Presidio is slower here than in the PII comparison (0.018 to 0.022 MB/s) because this mix
includes long secret strings.

**Thread scaling** (batch API, median of three, MB/s):

| Threads | Patterns | Patterns + tagger |
|---:|---:|---:|
| 1 | 9.96 | 0.76 |
| 2 | 12.11 | 1.15 |
| 4 | 15.53 | 1.89 |
| 8 | 39.27 | 2.35 |

The tagger path scales 3.1x from 1 to 8 threads. The pattern path is too short (under 1.5 s per
pass) to separate scaling from load noise: its 2- and 4-thread passes ranged from 7.0 to 20.8 and
15.5 to 37.8 MB/s.

**Document size** (same text joined into larger documents, 1 thread, MB/s):

| Target size | Documents | Patterns | Patterns + tagger |
|---:|---:|---:|---:|
| 1 KB | 13,297 | 11.95 | 0.70 |
| 16 KB | 911 | 8.35 | 0.68 |
| 256 KB | 58 | 7.23 | 0.31 |
| 4 MB | 4 | 4.85 | 0.39 |

Throughput falls as documents grow. The cause has not been profiled yet; the proxy scans one
message at a time, where documents are small.

**Streaming** (patterns only, 200 answers of 12 records each):

| Chunk (chars) | Chunks | Median ms per chunk | p99 ms | Max ms | Chars/s | Max held chars |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 463,709 | 0.0090 | 0.125 | 9.9 | 52,639 | 384 |
| 4 | 115,998 | 0.0096 | 0.084 | 1.1 | 223,453 | 384 |
| 16 | 29,070 | 0.0116 | 0.153 | 2.1 | 438,635 | 380 |
| 64 | 7,346 | 0.0159 | 0.170 | 1.4 | 745,435 | 372 |

The held-back text never exceeds the 342-character window plus one chunk. With the tagger
(first 50 answers, 4-character chunks) the median is 0.37 ms per chunk, p99 2.25 ms, 8,381
chars/s, which is still well above the rate a model emits tokens. Peak resident memory for the
whole benchmark was 496 MB.

## Threats to validity

- **Training distribution.** The tagger was trained on ai4privacy, so the ai4privacy comparison
  with Presidio favours it; Nemotron-PII is the fairer comparison and the gap there is small.
- **Synthetic data.** Both PII datasets are generated text; real prompts are messier.
- **Generated secrets.** The corpus covers formats the catalog knows. It measures that the
  patterns and validators work and how many look-alikes they reject, not recall on unknown
  formats.
- **Label mapping.** The comparison uses 13 coarse types; types one tool cannot produce are
  dropped, and Nemotron-PII annotations outside those types are ignore regions. Other mappings
  would move the numbers.
- **Presidio configuration.** Default recognizers and thresholds; a tuned Presidio deployment with
  custom recognizers would score higher.
- **Timing noise.** The machine was shared; Rust throughput rows are medians of three runs and
  the raw minimum and maximum are kept in the results file.

## Reproducibility

```bash
make data                # ai4privacy and Nemotron-PII into ../.bench-data/llm-dlp-guard
make train               # grid, final tagger, ai4privacy scores   -> pii_ai4privacy.json
make bench-ablation      # ablations, score floor, learning curve  -> pii_ablation.json
make bench-compare       # Presidio comparison                     -> presidio_compare.json
make bench-secrets       # secrets corpus and Nemotron-PII secrets -> secrets.json
make bench-throughput    # MB/s and scaling                        -> throughput.json
```
