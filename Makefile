.PHONY: help setup lint format type test coverage rust-check rust-develop parity data train bench-pii bench-compare bench-secrets bench-throughput diagram docker-build docker-up clean all

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies (compiles the Rust core) and pre-commit hooks
	uv sync --all-extras --dev
	uv run pre-commit install

format: ## Format automatically
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Lint (no fixes)
	uv run ruff format --check .
	uv run ruff check .

type: ## Type-check
	uv run mypy src

test: ## Run tests
	uv run pytest

coverage: ## Tests with coverage gate
	uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=80

rust-check: ## Rust core: fmt, clippy (with and without Python), unit tests
	cd rust && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo clippy --all-targets --features python -- -D warnings && cargo test

rust-develop: ## Rebuild the Rust core into .venv after editing rust/ (uv sync also builds it)
	uv sync --group rust
	uv run --no-sync maturin develop --uv --release

parity: ## Rust vs Python parity tests
	DLP_REQUIRE_RUST=1 uv run --no-sync pytest tests/test_parity.py -v

data: ## Download ai4privacy and Nemotron-PII into ../.bench-data/llm-dlp-guard
	scripts/download_data.sh

train: ## Tune on a dev split, train the tagger on all of ai4privacy train, evaluate on validation
	uv run python benchmarks/pii_ai4privacy.py

bench-compare: ## Presidio comparison on ai4privacy (English) and Nemotron-PII (needs the validation group)
	uv sync --group validation && uv run --no-sync python benchmarks/presidio_compare.py

bench-secrets: ## Secrets corpus: this package vs detect-secrets
	uv sync --group validation && uv run --no-sync python benchmarks/secrets_detection.py

bench-throughput: ## Throughput in MB/s for every engine
	uv sync --group validation && uv run --no-sync python benchmarks/throughput.py

diagram: ## Render architecture diagram to SVG
	npx -y @mermaid-js/mermaid-cli -i docs/architecture/architecture.mmd \
	     -o docs/architecture/architecture.svg \
	     -c mermaid.config.json -b transparent

docker-build: ## Build the image
	docker build -t llm-dlp-guard .

docker-up: ## Run via docker compose
	docker compose up --build

clean: ## Remove caches and artifacts
	rm -rf .pytest_cache .pytest-tmp .mypy_cache .ruff_cache dist build *.egg-info coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +

all: lint type coverage ## Full local gate (mirror of continuous integration)
