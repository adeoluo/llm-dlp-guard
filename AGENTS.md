# AGENTS.md

Working notes for coding agents and contributors in **Llm Dlp Guard**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

Data loss prevention for LLM traffic: PII, secrets and credentials in prompts and outputs

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docker-compose.yml`: local container setup
- `docs/`: documentation, architecture diagram source and render, design records
- `rust/`: Rust core compiled into a Python extension with PyO3
- `scripts/`: maintenance scripts (release, assets, data)
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `rust/src/crf.rs`: Linear-chain CRF tagger over hashed token features, with BIO tags.
- `rust/src/detector.rs`: Patterns and the tagger combined into one detector, with parallel batch scanning.
- `rust/src/engine.rs`: The pattern engine: an Aho-Corasick pass over every pattern's literal anchors decides which regular expressions run; matches are validated, scored and resolved to non-overlapping spans.
- `rust/src/finding.rs`: Findings, offset conversion and overlap resolution shared by every detector.
- `rust/src/lib.rs`: Rust core of llm-dlp-guard.
- `rust/src/python.rs`: PyO3 bindings, built by maturin into `llm_dlp_guard.llm_dlp_guard_rs`.
- `rust/src/tokenize.rs`: Tokenizer shared by the tagger: runs of alphanumeric characters (and `_`) form one token, every other non-space character is a token of its own, whitespace separates.
- `rust/src/validators.rs`: Validators run on a candidate match.
- `src/llm_dlp_guard/__init__.py`: Data loss prevention for LLM traffic: PII, secrets and credentials in prompts and outputs.
- `src/llm_dlp_guard/backend.py`: The Rust backend, compiled into the package on install and used by default.
- `src/llm_dlp_guard/catalog.py`: The pattern catalog: one regular expression per format, with anchors and validators.
- `src/llm_dlp_guard/cli.py`: Command line: ``python -m llm_dlp_guard <command>``.
- `src/llm_dlp_guard/configuration.py`: Service settings from ``DLP_*`` environment variables (or a ``.env`` file).
- `src/llm_dlp_guard/datasets.py`: Loaders for the public PII datasets and the label maps used in evaluation.
- `src/llm_dlp_guard/detector.py`: The detector: the pattern catalog plus an optional CRF tagger, on the Rust or Python backend.
- `src/llm_dlp_guard/findings.py`: The finding record and overlap resolution.
- `src/llm_dlp_guard/guard.py`: The guard: detection, policy decisions and redaction for one direction of LLM traffic.
- `src/llm_dlp_guard/metrics.py`: Entity-level precision, recall and F1 for character spans.
- `src/llm_dlp_guard/mock_llm.py`: A local OpenAI-compatible echo model for demos and tests (no network, no weights).
- `src/llm_dlp_guard/observability.py`: Thread-safe metrics in the Prometheus text format.
- `src/llm_dlp_guard/policy.py`: The policy engine: what to do with each finding, per direction.
- `src/llm_dlp_guard/reference.py`: Python reference implementation of the Rust core.
- `src/llm_dlp_guard/secrets_corpus.py`: A labelled corpus of credentials in real formats with random (fake) values.
- `src/llm_dlp_guard/service.py`: HTTP service: scanning endpoints and an OpenAI-compatible proxy that guards both directions.
- `src/llm_dlp_guard/streaming.py`: Streaming redaction for token streams and server-sent events.
- `src/llm_dlp_guard/validators.py`: Validators, the Python reference for ``rust/src/validators.rs``.
- `src/llm_dlp_guard/vault.py`: Reversible redaction: a vault of format-preserving tokens.

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies (compiles the Rust core) and pre-commit hooks |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run tests |
| `make coverage` | Tests with coverage gate |
| `make rust-check` | Rust core: fmt, clippy (with and without Python), unit tests |
| `make rust-develop` | Rebuild the Rust core into .venv after editing rust/ (uv sync also builds it) |
| `make parity` | Rust vs Python parity tests |
| `make data` | Download ai4privacy and Nemotron-PII into ../.bench-data/llm-dlp-guard |
| `make train` | Tune on a dev split, train the tagger on all of ai4privacy train, evaluate on validation |
| `make bench-compare` | Presidio comparison on ai4privacy (English) and Nemotron-PII (needs the validation group) |
| `make bench-secrets` | Secrets corpus: this package vs detect-secrets |
| `make bench-throughput` | Throughput in MB/s for every engine |
| `make bench-ablation` | Component ablations, score-floor sweep and learning curve on ai4privacy |
| `make diagram` | Render architecture diagram to SVG |
| `make docker-build` | Build the image |
| `make docker-up` | Run via docker compose |
| `make clean` | Remove caches and artifacts |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut with `make cut-release`.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.

## Rust notes

- `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` must pass.
- No `unwrap()` in library code paths that can fail at runtime; return a typed error.
- Unsafe code is limited to the places that need a syscall and carries a `SAFETY:` comment.
