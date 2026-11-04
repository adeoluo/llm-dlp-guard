#!/usr/bin/env bash
# Download the public PII datasets used by the benchmarks (about 660 MB on disk).
#
#   scripts/download_data.sh [DEST]      (default: ../.bench-data/llm-dlp-guard)
#
# * ai4privacy/pii-masking-400k (Hugging Face): 406,896 synthetic texts in six languages.
#   Licence: academic and non-commercial use only, no redistribution; read
#   https://huggingface.co/datasets/ai4privacy/pii-masking-400k/blob/main/license.md first.
# * nvidia/Nemotron-PII test split (Hugging Face, CC BY 4.0): 50,000 English documents.
# The secrets corpus is generated, not downloaded: python -m llm_dlp_guard gen-secrets.
set -euo pipefail

DEST="${1:-../.bench-data/llm-dlp-guard}"
HF="https://huggingface.co/datasets"
mkdir -p "$DEST/ai4privacy/train" "$DEST/ai4privacy/validation" "$DEST/nemotron"

for split in train validation; do
  for lang in 1en de es fr it nl; do
    out="$DEST/ai4privacy/$split/$lang.jsonl"
    [[ -s "$out" ]] || curl -fL -o "$out" "$HF/ai4privacy/pii-masking-400k/resolve/main/data/$split/$lang.jsonl"
  done
done

out="$DEST/nemotron/test.parquet"
[[ -s "$out" ]] || curl -fL -o "$out" "$HF/nvidia/Nemotron-PII/resolve/main/data/test-00000-of-00001.parquet"

du -sh "$DEST/ai4privacy" "$DEST/nemotron"
