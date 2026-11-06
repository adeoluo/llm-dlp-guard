"""HTTP service: scanning endpoints and an OpenAI-compatible proxy that guards both directions.

``POST /v1/chat/completions`` redacts the messages under the policy (tokenizing PII with a
per-request vault), forwards the request to the upstream provider, then scans the answer,
restores vault tokens and redacts any new secrets, for plain and streamed (SSE) responses.
A ``block`` action returns HTTP 400 for prompts and stops the answer for responses. Request
and response text are never logged.

Also: ``GET /health``, ``GET /ready``, ``GET /metrics`` (Prometheus text), ``POST /v1/scan``,
``POST /v1/redact``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from llm_dlp_guard import __version__
from llm_dlp_guard.guard import Applied, Decision, Guard
from llm_dlp_guard.observability import Metrics
from llm_dlp_guard.policy import DIRECTIONS
from llm_dlp_guard.streaming import SSEStream
from llm_dlp_guard.vault import Vault

MAX_BODY = 4 * 1024 * 1024
_FORWARD_HEADERS = ("authorization", "openai-organization", "openai-project", "x-api-key")


class HTTPError(Exception):
    def __init__(self, status: int, message: str, extra: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.extra = extra or {}

    def body(self) -> dict[str, Any]:
        return {"error": {"message": self.message, **self.extra}}


@dataclass
class Response:
    status: int
    content_type: str
    body: bytes = b""
    stream: Iterator[bytes] | None = None


def _json_response(status: int, payload: Any) -> Response:
    return Response(status, "application/json", json.dumps(payload).encode())


def _violation(applied: list[Applied], where: str) -> dict[str, Any]:
    return {
        "type": "dlp_policy_violation",
        "where": where,
        "findings": [{"label": a.finding.label, "action": a.action} for a in applied],
    }


class App:
    """Routes, independent of the socket server (tests call :meth:`handle` directly)."""

    def __init__(
        self,
        guard: Guard,
        upstream: str | None = None,
        metrics: Metrics | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.guard = guard
        self.upstream = upstream.rstrip("/") if upstream else None
        self.metrics = metrics if metrics is not None else Metrics()
        self.timeout = timeout
        self.metrics.set_gauge("tagger_loaded", 1.0 if guard.has_tagger else 0.0)
        self.metrics.set_gauge("patterns", float(len(guard.catalog.patterns)))

    # -- bookkeeping -------------------------------------------------------------------------

    def _count(self, applied: list[Applied], direction: str) -> None:
        for a in applied:
            self.metrics.inc(
                "findings",
                help="Findings by label, direction and action.",
                label=a.finding.label,
                direction=direction,
                action=a.action,
            )

    def _process(self, text: str, direction: str, vault: Vault) -> Decision:
        self.metrics.inc("scanned_chars", float(len(text)), help="Characters scanned.")
        decision = self.guard.process(text, direction, vault)  # type: ignore[arg-type]
        self._count(decision.applied, direction)
        if decision.outcome == "block":
            self.metrics.inc("blocked", help="Blocked texts by direction.", direction=direction)
        return decision

    # -- routing -----------------------------------------------------------------------------

    def handle(self, method: str, path: str, body: bytes, headers: dict[str, str]) -> Response:
        route = path.split("?", 1)[0]
        start = time.perf_counter()
        try:
            response = self._dispatch(method, route, body, headers)
        except HTTPError as err:
            response = _json_response(err.status, err.body())
        self.metrics.observe_request(route, response.status, time.perf_counter() - start)
        return response

    def _json(self, body: bytes) -> dict[str, Any]:
        try:
            payload = json.loads(body or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPError(400, "body must be JSON") from exc
        if not isinstance(payload, dict):
            raise HTTPError(400, "body must be a JSON object")
        return payload

    def _dispatch(self, method: str, route: str, body: bytes, headers: dict[str, str]) -> Response:
        if method == "GET" and route == "/health":
            return _json_response(200, {"status": "ok", "version": __version__})
        if method == "GET" and route == "/ready":
            return _json_response(
                200,
                {
                    "ready": True,
                    "tagger": self.guard.has_tagger,
                    "patterns": len(self.guard.catalog.patterns),
                    "upstream": self.upstream is not None,
                },
            )
        if method == "GET" and route == "/metrics":
            return Response(200, "text/plain; version=0.0.4", self.metrics.render().encode())
        if method == "POST" and route == "/v1/scan":
            return self._scan(self._json(body))
        if method == "POST" and route == "/v1/redact":
            return self._redact(self._json(body))
        if method == "POST" and route == "/v1/chat/completions":
            return self._chat(self._json(body), headers)
        raise HTTPError(404, f"no route {method} {route}")

    def _texts(self, payload: dict[str, Any]) -> list[str]:
        if isinstance(payload.get("text"), str):
            return [payload["text"]]
        texts = payload.get("texts")
        if isinstance(texts, list) and all(isinstance(t, str) for t in texts):
            return texts
        raise HTTPError(400, "expected 'text' (string) or 'texts' (list of strings)")

    def _scan(self, payload: dict[str, Any]) -> Response:
        texts = self._texts(payload)
        self.metrics.inc("scanned_chars", float(sum(map(len, texts))), help="Characters scanned.")
        results = []
        for text in texts:
            results.append(
                [
                    {**f.to_dict(), "category": self.guard.category(f.label)}
                    for f in self.guard.scan(text)
                ]
            )
        if "text" in payload:
            return _json_response(200, {"findings": results[0]})
        return _json_response(200, {"results": results})

    def _redact(self, payload: dict[str, Any]) -> Response:
        direction = payload.get("direction", "prompt")
        if direction not in DIRECTIONS:
            raise HTTPError(400, "direction must be 'prompt' or 'response'")
        texts = self._texts(payload)
        vault = Vault()
        decisions = [self._process(t, direction, vault).to_dict() for t in texts]
        if "text" in payload:
            return _json_response(200, decisions[0])
        return _json_response(200, {"results": decisions})

    # -- proxy -------------------------------------------------------------------------------

    def _redact_messages(self, payload: dict[str, Any], vault: Vault) -> None:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise HTTPError(400, "'messages' must be a list")
        blocked: list[Applied] = []
        for message in messages:
            if not isinstance(message, dict):
                raise HTTPError(400, "each message must be an object")
            content = message.get("content")
            if isinstance(content, str):
                decision = self._process(content, "prompt", vault)
                blocked += decision.blocked
                message["content"] = decision.text
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        decision = self._process(part["text"], "prompt", vault)
                        blocked += decision.blocked
                        part["text"] = decision.text
        if blocked:
            raise HTTPError(
                400,
                "the request was blocked by the data loss prevention policy",
                _violation(blocked, "prompt"),
            )

    def _upstream(self, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        assert self.upstream is not None
        forward = {k: v for k, v in headers.items() if k.lower() in _FORWARD_HEADERS}
        forward["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.upstream}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers=forward,
            method="POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as err:
            self.metrics.inc("upstream_errors", help="Failed upstream calls.")
            raise HTTPError(err.code, "upstream returned an error") from err
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            self.metrics.inc("upstream_errors", help="Failed upstream calls.")
            raise HTTPError(502, "upstream unreachable") from err

    def _chat(self, payload: dict[str, Any], headers: dict[str, str]) -> Response:
        if self.upstream is None:
            raise HTTPError(503, "no upstream configured (DLP_UPSTREAM_URL or --upstream)")
        vault = Vault()
        self._redact_messages(payload, vault)
        self.metrics.inc("vault_tokens", float(len(vault)), help="Values tokenized in prompts.")
        upstream = self._upstream(payload, headers)
        if payload.get("stream"):
            return Response(200, "text/event-stream", stream=self._stream(upstream, vault))
        with upstream:
            try:
                answer = json.loads(upstream.read())
            except json.JSONDecodeError as exc:
                raise HTTPError(502, "upstream returned invalid JSON") from exc
        blocked: list[Applied] = []
        for choice in answer.get("choices", []) if isinstance(answer, dict) else []:
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                decision = self._process(message["content"], "response", vault)
                blocked += decision.blocked
                message["content"] = decision.text
        if blocked:
            raise HTTPError(
                502,
                "the response was stopped by the data loss prevention policy",
                _violation(blocked, "response"),
            )
        return _json_response(200, answer)

    def _stream(self, upstream: Any, vault: Vault) -> Iterator[bytes]:
        sse = SSEStream(self.guard, vault)
        with upstream:
            for raw in upstream:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                for out in sse.feed_line(line):
                    yield (out + "\n").encode()
                if sse.done:
                    break
            if not sse.done:
                for out in sse.feed_line("data: [DONE]"):
                    yield (out + "\n").encode()
        self._count(sse.applied(), "response")
        if sse.blocked:
            self.metrics.inc("blocked", help="Blocked texts by direction.", direction="response")


def make_handler(app: App, max_body: int = MAX_BODY) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"llm-dlp-guard/{__version__}"

        def _send(self, response: Response) -> None:
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            if response.stream is None:
                self.send_header("Content-Length", str(len(response.body)))
                self.end_headers()
                self.wfile.write(response.body)
                return
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for chunk in response.stream:
                self.wfile.write(chunk)
                self.wfile.flush()

        def _headers(self) -> dict[str, str]:
            return dict(self.headers.items())

        def do_GET(self) -> None:
            self._send(app.handle("GET", self.path, b"", self._headers()))

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length > max_body:
                # Discard a bounded amount of the body first: closing a socket with unread input
                # makes the kernel send a reset, which can reach the client before the 413 does.
                remaining = min(length, 16 * max_body)
                while remaining > 0:
                    got = self.rfile.read(min(remaining, 65_536))
                    if not got:
                        break
                    remaining -= len(got)
                self.close_connection = True
                self._send(_json_response(413, {"error": {"message": "body too large"}}))
                return
            body = self.rfile.read(length)
            self._send(app.handle("POST", self.path, body, self._headers()))

        def log_message(self, format: str, *args: Any) -> None:
            return  # request lines can carry data; stay quiet

    return Handler


def serve(app: App, host: str, port: int, max_body: int = MAX_BODY) -> None:  # pragma: no cover
    server = ThreadingHTTPServer((host, port), make_handler(app, max_body))
    try:
        server.serve_forever()
    finally:
        server.server_close()
