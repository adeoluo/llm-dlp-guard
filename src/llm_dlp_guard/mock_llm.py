"""A local OpenAI-compatible echo model for demos and tests (no network, no weights).

It answers ``POST /v1/chat/completions`` with ``I received: <last user message>``, streamed in
small chunks when ``stream`` is true, and records the last request so tests and demos can show
exactly what a provider would have seen.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class EchoModel:
    def __init__(self, chunk_size: int = 4, extra: str = "") -> None:
        self.chunk_size = chunk_size
        self.extra = extra
        self.last_request: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def reply(self, payload: dict[str, Any]) -> str:
        with self._lock:
            self.last_request = payload
        messages = payload.get("messages") or []
        last = ""
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content")
                last = content if isinstance(content, str) else json.dumps(content)
        return f"I received: {last}{self.extra}"

    def completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": "chatcmpl-echo",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", "echo"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.reply(payload)},
                    "finish_reason": "stop",
                }
            ],
        }

    def events(self, payload: dict[str, Any]) -> list[str]:
        text = self.reply(payload)
        base = {"id": "chatcmpl-echo", "object": "chat.completion.chunk", "model": "echo"}
        out = []
        for i in range(0, len(text), self.chunk_size):
            delta = {"content": text[i : i + self.chunk_size]}
            out.append(
                json.dumps(
                    {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                )
            )
        out.append(
            json.dumps({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        )
        return [f"data: {e}\n\n" for e in out] + ["data: [DONE]\n\n"]


def make_echo_handler(model: EchoModel) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
                return
            if payload.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for event in model.events(payload):
                    self.wfile.write(event.encode())
                    self.wfile.flush()
                return
            body = json.dumps(model.completion(payload)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            body = json.dumps({"last_request": model.last_request}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def serve_echo(host: str, port: int, chunk_size: int = 4) -> None:  # pragma: no cover
    server = ThreadingHTTPServer((host, port), make_echo_handler(EchoModel(chunk_size)))
    try:
        server.serve_forever()
    finally:
        server.server_close()
