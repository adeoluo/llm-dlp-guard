#!/usr/bin/env bash
# Build the release assets (sdist and a wheel with the compiled Rust core) into dist/release.
# Called by scripts/release.sh after tagging; the wheel matches the build machine's platform.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
rm -rf dist/release
uv build --sdist --wheel --out-dir dist/release
ls -l dist/release
