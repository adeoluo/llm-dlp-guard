"""Command line: ``python -m llm_dlp_guard <command>``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from llm_dlp_guard import __version__
from llm_dlp_guard.backend import BACKENDS
from llm_dlp_guard.catalog import default_catalog
from llm_dlp_guard.configuration import Settings
from llm_dlp_guard.guard import Guard
from llm_dlp_guard.vault import Vault


def _read(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def _preview(value: str, show: bool) -> str:
    if show:
        return value
    if len(value) <= 6:
        return "*" * len(value)
    return value[:4] + "*" * min(len(value) - 4, 12)


def _guard(args: argparse.Namespace) -> Guard:
    return Guard.from_files(args.policy, args.model, backend=args.backend)


def cmd_scan(args: argparse.Namespace) -> int:
    guard = _guard(args)
    text = _read(args.path)
    start = time.perf_counter()
    findings = guard.scan(text)
    seconds = time.perf_counter() - start
    if args.json:
        print(json.dumps([{**f.to_dict(), "category": guard.category(f.label)} for f in findings]))
        return 0
    print(f"{'span':>13}  {'label':<18} {'category':<8} {'score':>5}  {'source':<24} value")
    for f in findings:
        span = f"{f.start}-{f.end}"
        value = _preview(f.text(text), args.show_values).replace("\n", "\\n")
        print(
            f"{span:>13}  {f.label:<18} {guard.category(f.label):<8} {f.score:5.2f}  "
            f"{f.source:<24} {value}"
        )
    print(f"{len(findings)} findings in {len(text):,} characters ({seconds * 1000:.1f} ms)")
    return 0


def cmd_redact(args: argparse.Namespace) -> int:
    guard = _guard(args)
    text = _read(args.path)
    decision = guard.process(text, args.direction, Vault())
    if args.json:
        print(json.dumps(decision.to_dict()))
        return 0 if decision.outcome != "block" else 2
    if decision.outcome == "block":
        labels = sorted({a.finding.label for a in decision.blocked})
        print(f"BLOCKED by policy: {', '.join(labels)}", file=sys.stderr)
        return 2
    sys.stdout.write(decision.text if decision.text.endswith("\n") else decision.text + "\n")
    counts: dict[str, int] = {}
    for a in decision.applied:
        counts[f"{a.finding.label}:{a.action}"] = counts.get(f"{a.finding.label}:{a.action}", 0) + 1
    summary = ", ".join(f"{k} x{v}" for k, v in sorted(counts.items()))
    print(f"[{decision.outcome}] {summary or 'nothing found'}", file=sys.stderr)
    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    from llm_dlp_guard.streaming import StreamRedactor

    guard = _guard(args)
    text = _read(args.path)
    redactor = StreamRedactor(guard, args.direction, Vault())
    for i in range(0, len(text), args.chunk):
        out = redactor.feed(text[i : i + args.chunk])
        if out:
            sys.stdout.write(out)
            sys.stdout.flush()
        if args.delay:
            time.sleep(args.delay)
    sys.stdout.write(redactor.flush())
    sys.stdout.flush()
    if redactor.blocked:
        print("\n[stream stopped by policy]", file=sys.stderr)
        return 2
    return 0


def cmd_serve(args: argparse.Namespace) -> int:  # pragma: no cover - blocking
    from llm_dlp_guard.service import App, serve

    settings = Settings()
    guard = Guard.from_files(
        args.policy or settings.policy_path,
        args.model or settings.model_path,
        backend=args.backend if args.backend != "auto" else settings.backend,
    )
    app = App(guard, args.upstream or settings.upstream_url, timeout=settings.upstream_timeout)
    host = args.host or settings.host
    port = args.port or settings.port
    print(f"llm-dlp-guard {__version__} on http://{host}:{port} (upstream: {app.upstream})")
    serve(app, host, port, settings.max_body_bytes)
    return 0


def cmd_mock_llm(args: argparse.Namespace) -> int:  # pragma: no cover - blocking
    from llm_dlp_guard.mock_llm import serve_echo

    print(f"echo model on http://{args.host}:{args.port}")
    serve_echo(args.host, args.port, args.chunk)
    return 0


def cmd_train_tagger(args: argparse.Namespace) -> int:
    from llm_dlp_guard.datasets import TAGGER_LABELS, load_ai4privacy
    from llm_dlp_guard.detector import Tagger

    langs = tuple(args.languages.split(",")) if args.languages else None
    records = list(load_ai4privacy(args.data, "train", langs, args.limit))
    print(f"training on {len(records):,} records", file=sys.stderr)
    start = time.perf_counter()
    tagger = Tagger.train(
        [r.text for r in records],
        [r.spans for r in records],
        TAGGER_LABELS,
        n_buckets=1 << args.bucket_bits,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        verbose=True,
    )
    tagger.save(args.out)
    print(f"saved {args.out} after {time.perf_counter() - start:.1f} s", file=sys.stderr)
    return 0


def cmd_gen_secrets(args: argparse.Namespace) -> int:
    from llm_dlp_guard.secrets_corpus import generate

    out = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")  # noqa: SIM115
    try:
        for sample in generate(args.positives, args.negatives, args.seed):
            out.write(sample.to_json() + "\n")
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


def cmd_catalog(args: argparse.Namespace) -> int:
    for p in default_catalog().patterns:
        extra = f" [{p.validator}]" if p.validator else ""
        print(f"{p.name:<24} {p.label:<18} {p.category:<7} {p.score:.2f}{extra}  {p.description}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-dlp-guard", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    def guarded(p: argparse.ArgumentParser) -> None:
        p.add_argument("--policy", help="YAML or JSON policy file")
        p.add_argument("--model", help="CRF tagger weights (enables names and addresses)")
        p.add_argument("--backend", choices=BACKENDS, default="auto")

    p = sub.add_parser("scan", help="list findings in a file or stdin")
    p.add_argument("path", nargs="?", default="-")
    p.add_argument("--json", action="store_true")
    p.add_argument("--show-values", action="store_true", help="print matched values in full")
    guarded(p)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("redact", help="apply the policy and print the redacted text")
    p.add_argument("path", nargs="?", default="-")
    p.add_argument("--direction", choices=["prompt", "response"], default="prompt")
    p.add_argument("--json", action="store_true")
    guarded(p)
    p.set_defaults(func=cmd_redact)

    p = sub.add_parser("stream", help="redact a file chunk by chunk, as a token stream")
    p.add_argument("path", nargs="?", default="-")
    p.add_argument("--direction", choices=["prompt", "response"], default="response")
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--delay", type=float, default=0.0, help="seconds between chunks")
    guarded(p)
    p.set_defaults(func=cmd_stream)

    p = sub.add_parser("serve", help="HTTP service and OpenAI-compatible proxy")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--upstream", help="OpenAI-compatible base URL to forward to")
    guarded(p)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("mock-llm", help="local echo model for demos")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--chunk", type=int, default=4)
    p.set_defaults(func=cmd_mock_llm)

    p = sub.add_parser("train-tagger", help="train the CRF tagger on ai4privacy")
    p.add_argument("--data", required=True, help="directory with train/ and validation/")
    p.add_argument("--out", required=True)
    p.add_argument("--languages", help="comma-separated, default all six")
    p.add_argument("--limit", type=int)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bucket-bits", type=int, default=21)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_train_tagger)

    p = sub.add_parser("gen-secrets", help="write the labelled fake-secrets corpus as JSONL")
    p.add_argument("--positives", type=int, default=1000)
    p.add_argument("--negatives", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="-")
    p.set_defaults(func=cmd_gen_secrets)

    p = sub.add_parser("catalog", help="list the built-in patterns")
    p.set_defaults(func=cmd_catalog)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
