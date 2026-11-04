from __future__ import annotations

import json
import random
import threading
import urllib.request
from collections.abc import Iterator
from http.server import ThreadingHTTPServer

import pytest

from llm_dlp_guard import secrets_corpus as sc
from llm_dlp_guard.guard import Guard
from llm_dlp_guard.mock_llm import EchoModel, make_echo_handler
from llm_dlp_guard.observability import Metrics
from llm_dlp_guard.service import App, Response, make_handler


def _serve(handler: type) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture()
def echo() -> Iterator[tuple[EchoModel, str]]:
    model = EchoModel(chunk_size=3)
    server, url = _serve(make_echo_handler(model))
    yield model, url
    server.shutdown()
    server.server_close()


@pytest.fixture()
def proxy(echo: tuple[EchoModel, str]) -> Iterator[tuple[App, str]]:
    app = App(Guard(), upstream=echo[1])
    server, url = _serve(make_handler(app, max_body=200_000))
    yield app, url
    server.shutdown()
    server.server_close()


def _post(url: str, payload: object, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers or {}, method="POST"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def _json(r: Response) -> dict[str, object]:
    data = json.loads(r.body)
    assert isinstance(data, dict)
    return data


def test_health_ready_metrics() -> None:
    app = App(Guard())
    assert _json(app.handle("GET", "/health", b"", {}))["status"] == "ok"
    ready = _json(app.handle("GET", "/ready", b"", {}))
    assert ready["ready"] is True and ready["upstream"] is False and ready["tagger"] is False
    metrics = app.handle("GET", "/metrics", b"", {}).body.decode()
    assert "dlp_requests_total" in metrics and "dlp_patterns" in metrics
    assert app.handle("GET", "/nope", b"", {}).status == 404


def test_scan_and_redact_routes() -> None:
    app = App(Guard())
    r = _json(
        app.handle("POST", "/v1/scan", json.dumps({"text": "mail a.b@example.org"}).encode(), {})
    )
    assert r["findings"] == [
        {
            "start": 5,
            "end": 20,
            "label": "EMAIL",
            "source": "email",
            "score": 0.95,
            "category": "pii",
        }
    ]
    r = _json(app.handle("POST", "/v1/scan", json.dumps({"texts": ["8.8.8.8", "x"]}).encode(), {}))
    assert len(r["results"]) == 2  # type: ignore[arg-type]
    r = _json(
        app.handle(
            "POST", "/v1/redact", json.dumps({"text": "key AKIAIOSFODNN7ABCDEFG"}).encode(), {}
        )
    )
    assert r["text"] == "key [AWS_ACCESS_KEY]" and r["outcome"] == "redact"
    r = _json(
        app.handle(
            "POST", "/v1/redact", json.dumps({"texts": ["a"], "direction": "response"}).encode(), {}
        )
    )
    assert r["results"] == [{"outcome": "allow", "text": "a", "findings": []}]
    for body in (
        b"not json",
        b"[1]",
        json.dumps({"text": 3}).encode(),
        json.dumps({"text": "a", "direction": "up"}).encode(),
    ):
        route = "/v1/redact" if b"direction" in body else "/v1/scan"
        assert app.handle("POST", route, body, {}).status == 400
    assert (
        app.metrics.value("findings", label="AWS_ACCESS_KEY", direction="prompt", action="redact")
        == 1
    )


def test_chat_without_upstream() -> None:
    app = App(Guard())
    r = app.handle("POST", "/v1/chat/completions", json.dumps({"messages": []}).encode(), {})
    assert r.status == 503


def test_proxy_tokenizes_prompt_and_restores_answer(
    echo: tuple[EchoModel, str], proxy: tuple[App, str]
) -> None:
    model, _ = echo
    app, url = proxy
    secret = sc.github_token(random.Random(1))
    prompt = f"Email jane.doe@example.org the token {secret}"
    status, body = _post(
        f"{url}/v1/chat/completions",
        {"model": "m", "messages": [{"role": "user", "content": prompt}]},
        {"Authorization": "Bearer test-key", "Content-Type": "application/json"},
    )
    assert status == 200
    seen = model.last_request["messages"][0]["content"]  # type: ignore[index]
    assert "jane.doe@example.org" not in seen and secret not in seen and "[GITHUB_TOKEN]" in seen
    answer = json.loads(body)["choices"][0]["message"]["content"]
    assert answer == "I received: Email jane.doe@example.org the token [GITHUB_TOKEN]"
    assert app.metrics.value("vault_tokens") == 1


def test_proxy_streaming(echo: tuple[EchoModel, str], proxy: tuple[App, str]) -> None:
    model, _ = echo
    _, url = proxy
    payload = {
        "stream": True,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "card 4111 1111 1111 1111"}]}
        ],
    }
    status, body = _post(f"{url}/v1/chat/completions", payload)
    assert status == 200
    content = ""
    for line in body.decode().splitlines():
        if line.startswith("data: {"):
            for c in json.loads(line[6:])["choices"]:
                content += (c.get("delta") or {}).get("content") or ""
    assert content == 'I received: [{"type": "text", "text": "card 4111 1111 1111 1111"}]'
    seen = json.dumps(model.last_request)
    assert "4111 1111 1111 1111" not in seen
    assert body.decode().rstrip().endswith("data: [DONE]")


def test_proxy_blocks_private_keys(proxy: tuple[App, str]) -> None:
    app, url = proxy
    key = sc.private_key(random.Random(3))
    status, body = _post(
        f"{url}/v1/chat/completions", {"messages": [{"role": "user", "content": key}]}
    )
    assert status == 400
    error = json.loads(body)["error"]
    assert (
        error["type"] == "dlp_policy_violation" and error["findings"][0]["label"] == "PRIVATE_KEY"
    )
    assert "PRIVATE KEY" not in body.decode()
    assert app.metrics.value("blocked", direction="prompt") == 1


def test_proxy_blocks_leaky_responses(echo: tuple[EchoModel, str]) -> None:
    model, upstream = echo
    model.extra = "\n" + sc.private_key(random.Random(4))
    app = App(Guard(), upstream=upstream)
    r = app.handle(
        "POST",
        "/v1/chat/completions",
        json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
        {},
    )
    assert r.status == 502 and b"dlp_policy_violation" in r.body
    r = app.handle(
        "POST",
        "/v1/chat/completions",
        json.dumps({"stream": True, "messages": [{"role": "user", "content": "hi"}]}).encode(),
        {},
    )
    assert r.stream is not None
    out = b"".join(r.stream).decode()
    assert "dlp_policy_violation" in out and "PRIVATE KEY" not in out


def test_proxy_errors(proxy: tuple[App, str]) -> None:
    _, url = proxy
    assert _post(f"{url}/v1/chat/completions", {"messages": "x"})[0] == 400
    assert _post(f"{url}/v1/chat/completions", {"messages": ["x"]})[0] == 400
    big = {"messages": [{"role": "user", "content": "a" * 300_000}]}
    assert _post(f"{url}/v1/chat/completions", big)[0] == 413
    with urllib.request.urlopen(f"{url}/health") as resp:
        assert json.loads(resp.read())["status"] == "ok"
    dead = App(Guard(), upstream="http://127.0.0.1:9")
    r = dead.handle("POST", "/v1/chat/completions", json.dumps({"messages": []}).encode(), {})
    assert r.status == 502 and dead.metrics.value("upstream_errors") == 1


def test_upstream_http_error(echo: tuple[EchoModel, str]) -> None:
    _, upstream = echo
    app = App(Guard(), upstream=upstream + "/missing")
    r = app.handle("POST", "/v1/chat/completions", json.dumps({"messages": []}).encode(), {})
    assert r.status == 404 and app.metrics.value("upstream_errors") == 1


def test_metrics_render_labels() -> None:
    m = Metrics()
    m.inc("x", 2, help="X.", a='q"uote')
    m.set_gauge("g", 1.5)
    m.observe_request("/r", 200, 0.1)
    text = m.render()
    assert 'dlp_x_total{a="q\\"uote"} 2' in text and "dlp_g 1.5" in text
    assert m.value("x", a='q"uote') == 2 and m.value("missing") == 0


def test_echo_model_shapes() -> None:
    model = EchoModel(chunk_size=5)
    payload = {
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hello there"}]
    }
    assert (
        model.completion(payload)["choices"][0]["message"]["content"] == "I received: hello there"
    )
    events = model.events(payload)
    assert events[-1] == "data: [DONE]\n\n" and len(events) == 7
