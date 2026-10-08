"""mind/llm_client.py against a fake LM Studio: parsing, streaming,
thinking, tool calls, the no-think switch, status() and every failure
the owner is likely to meet (server off, HTTP errors, drops, timeouts).

FakeLLM (tests/fake_llm.py) covers the ordinary server. The wire shapes
it can't produce - streamed reasoning and tool-call deltas, HTTP errors,
a connection cut mid-answer - come from _Scripted, a tiny server whose
handler is written by each test.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mind.harness_types import LLMReply, ToolCall
from mind.llm_client import (DEFAULT_URL, ChatReply, LLMClient,
                             LLMUnavailable, assistant_message, main,
                             split_thinking, tool_message)
from tests.fake_llm import FakeLLM

USER = [{"role": "system", "content": "You are Telp's writer."},
        {"role": "user", "content": "What is the capital of Iceland?"}]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("TELP_LLM_URL", "TELP_LLM_MODEL", "TELP_LLM_API_KEY",
                "TELP_LLM_NO_THINK"):
        monkeypatch.delenv(key, raising=False)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _body(content=None, finish="stop", usage=None, tool_calls=None,
          reasoning=None, **extra) -> dict:
    """A full chat-completion response, for shapes FakeLLM won't build."""
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"index": 0, "message": msg,
                         "finish_reason": finish}],
            "usage": usage or {}, **extra}


# ─── a scriptable raw server ────────────────────────────────────────


class _Scripted:
    """An HTTP server whose request handling is a test's own function:
    handle(handler, method, body). Records POST bodies in .requests."""

    def __init__(self, handle):
        outer = self
        self.requests: list[dict] = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                handle(self, "GET", None)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                outer.requests.append(body)
                handle(self, "POST", body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        kwargs={"poll_interval": 0.05},
                                        daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


def _send_json(h, code: int, obj) -> None:
    data = json.dumps(obj).encode()
    h.send_response(code)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(data)))
    h.end_headers()
    h.wfile.write(data)


def _send_sse(h, chunks, done: bool = True) -> None:
    """Chunks are dicts (sent as JSON data events) or raw strings."""
    h.send_response(200)
    h.send_header("Content-Type", "text/event-stream")
    h.end_headers()
    for chunk in chunks:
        raw = chunk if isinstance(chunk, str) else \
            f"data: {json.dumps(chunk)}\n\n"
        h.wfile.write(raw.encode())
        h.wfile.flush()
    if done:
        h.wfile.write(b"data: [DONE]\n\n")


def _models(h, ids=("qwen3.8-27b",)) -> None:
    _send_json(h, 200, {"data": [{"id": m} for m in ids]})


def _delta(**delta) -> dict:
    return {"choices": [{"index": 0, "delta": delta}]}


# ─── settings and model detection ───────────────────────────────────


def test_defaults_and_environment(monkeypatch):
    client = LLMClient(model="m")
    assert client.base_url == DEFAULT_URL
    assert client.timeout == 600 and client.temperature == 0.3
    monkeypatch.setenv("TELP_LLM_URL", "http://192.168.1.20:1234/v1/")
    monkeypatch.setenv("TELP_LLM_MODEL", "qwen-from-env")
    client = LLMClient()
    assert client.base_url == "http://192.168.1.20:1234/v1"
    assert client.model == "qwen-from-env"
    # arguments win; a bare host:port gets http:// and /v1
    client = LLMClient(base_url="localhost:4321", model="x")
    assert client.base_url == "http://localhost:4321/v1"
    assert client.model == "x"


def test_model_auto_detected_from_server():
    with FakeLLM(lambda r: "hi", models=("qwen3.8-27b", "llama-3")) as llm:
        client = LLMClient(base_url=llm.url)
        assert client.model == "qwen3.8-27b"
        client.chat(USER)
        assert llm.requests[-1]["model"] == "qwen3.8-27b"
        assert client.list_models() == ["qwen3.8-27b", "llama-3"]


def test_model_detection_skips_embedding_models():
    # LM Studio lists the embedding model it ships with alongside chat models
    with FakeLLM(lambda r: "hi", models=(
            "text-embedding-nomic-embed-text-v1.5", "qwen3.8-27b")) as llm:
        assert LLMClient(base_url=llm.url).model == "qwen3.8-27b"


def test_model_detection_prefers_what_lm_studio_has_loaded():
    def handle(h, method, body):
        if h.path == "/api/v0/models":
            _send_json(h, 200, {"data": [
                {"id": "big-model", "type": "llm", "state": "not-loaded"},
                {"id": "qwen3.8-27b", "type": "llm", "state": "loaded",
                 "loaded_context_length": 32768}]})
        else:
            _models(h, ("big-model", "qwen3.8-27b"))
    with _Scripted(handle) as srv:
        assert LLMClient(base_url=srv.url).model == "qwen3.8-27b"


def test_no_model_loaded_is_explained():
    with FakeLLM(lambda r: "hi", models=()) as llm:
        client = LLMClient(base_url=llm.url)
        with pytest.raises(LLMUnavailable, match="no model"):
            client.chat(USER)
        info = client.status()
        assert info["reachable"] and info["models"] == []
        assert any("no model" in w for w in info["warnings"])


# ─── plain replies ──────────────────────────────────────────────────


def test_plain_chat_round_trip():
    with FakeLLM(lambda r: "Reykjavík is the capital [1].") as llm:
        client = LLMClient(base_url=llm.url, model="qwen3.8-27b")
        messages = [dict(m) for m in USER]
        reply = client.chat(messages, max_tokens=200)
        assert isinstance(reply, LLMReply)
        assert reply.text == "Reykjavík is the capital [1]."
        assert reply.thinking == "" and reply.tool_calls == []
        assert reply.finish_reason == "stop" and not reply.truncated
        assert reply.notes == [] and reply.seconds >= 0
        sent = llm.requests[-1]
        assert sent["messages"] == USER and messages == USER
        assert sent["temperature"] == 0.3 and sent["max_tokens"] == 200
        assert sent["stream"] is False and "tools" not in sent


def test_temperature_override_and_non_stream_on_token():
    seen = []
    with FakeLLM(lambda r: "Four.") as llm:
        client = LLMClient(base_url=llm.url, model="m")
        reply = client.chat(USER, temperature=0.0, on_token=seen.append)
        assert llm.requests[-1]["temperature"] == 0.0
    assert seen == ["Four."] and reply.text == "Four."


def test_reasoning_content_is_kept_apart():
    with FakeLLM(lambda r: {"content": "It is 4.",
                            "reasoning_content": "2 plus 2 makes 4."}) as llm:
        reply = LLMClient(base_url=llm.url, model="m").chat(USER)
    assert reply.text == "It is 4."
    assert reply.thinking == "2 plus 2 makes 4."


def test_inline_think_is_split_off():
    content = "<think>\nThe user wants the capital.\n</think>\n\nReykjavík [1]."
    with FakeLLM(lambda r: content) as llm:
        reply = LLMClient(base_url=llm.url, model="m").chat(USER)
    assert reply.text == "Reykjavík [1]."
    assert reply.thinking == "The user wants the capital."


def test_unclosed_think_when_cut_off_says_so():
    body = _body("<think>Let me weigh every source carefully, first",
                 finish="length",
                 usage={"prompt_tokens": 7000, "completion_tokens": 1192,
                        "total_tokens": 8192})
    with FakeLLM(lambda r: body) as llm:
        reply = LLMClient(base_url=llm.url, model="m").chat(USER)
    assert reply.text == ""
    assert reply.thinking.startswith("Let me weigh every source")
    assert reply.truncated and reply.finish_reason == "length"
    assert any("ran out of room while still thinking" in n
               for n in reply.notes)


def test_truncated_answer_note():
    body = _body("Reykjavík is the capital and", finish="length")
    with FakeLLM(lambda r: body) as llm:
        reply = LLMClient(base_url=llm.url, model="m").chat(USER)
    assert reply.text == "Reykjavík is the capital and"
    assert reply.truncated
    assert any("cut off" in n for n in reply.notes)


def test_split_thinking_shapes():
    assert split_thinking("plain answer") == ("plain answer", "", False)
    # empty block: what Qwen writes when told /no_think
    assert split_thinking("<think>\n\n</think>\n\nHi.") == ("Hi.", "", False)
    # template opened <think> in the prompt: only the close shows
    assert split_thinking("weighing it</think>\n\nAnswer.") == \
        ("Answer.", "weighing it", False)
    # several blocks, then an unclosed one
    text, thinking, cut = split_thinking(
        "<think>a</think>One. <think>b</think>Two. <think>c")
    assert text == "One. Two." and thinking == "a\n\nb\n\nc" and cut
    assert split_thinking("") == ("", "", False)


def test_reasoning_field_and_inline_think_combine():
    with FakeLLM(lambda r: {"content": "<think>inline</think>Done.",
                            "reasoning_content": "separate"}) as llm:
        reply = LLMClient(base_url=llm.url, model="m").chat(USER)
    assert reply.text == "Done." and reply.thinking == "separate\n\ninline"


# ─── tool calls ─────────────────────────────────────────────────────

TOOLS = [{"type": "function", "function": {
    "name": "search_memory", "description": "Search Telp's memory.",
    "parameters": {"type": "object",
                   "properties": {"query": {"type": "string"}}}}}]


def test_tool_calls_are_parsed():
    out = {"content": None, "tool_calls": [
        {"id": "c1", "name": "search_memory",
         "arguments": {"query": "Iceland capital"}},
        {"id": "c2", "name": "today"}]}
    with FakeLLM(lambda r: out) as llm:
        reply = LLMClient(base_url=llm.url, model="m").chat(USER, tools=TOOLS)
        assert llm.requests[-1]["tools"] == TOOLS
    assert reply.tool_calls == [
        ToolCall("c1", "search_memory", {"query": "Iceland capital"}),
        ToolCall("c2", "today", {})]
    assert reply.finish_reason == "tool_calls" and reply.text == ""
    assert reply.notes == []


def test_bad_tool_arguments_become_empty_with_a_note():
    calls = [{"id": "a", "type": "function", "function": {
                 "name": "search_memory", "arguments": "{query: Iceland"}},
             {"id": "b", "type": "function", "function": {
                 "name": "calculate", "arguments": "[1, 2]"}},
             {"id": "c", "type": "function", "function": {
                 "name": "today", "arguments": ""}},
             {"id": "d", "type": "function", "function": {
                 "name": "calculate",
                 "arguments": json.dumps(json.dumps({"expr": "2+2"}))}}]
    with FakeLLM(lambda r: _body(None, "tool_calls",
                                 tool_calls=calls)) as llm:
        reply = LLMClient(base_url=llm.url, model="m").chat(USER, tools=TOOLS)
    args = {c.id: c.arguments for c in reply.tool_calls}
    assert args == {"a": {}, "b": {}, "c": {}, "d": {"expr": "2+2"}}
    assert len(reply.notes) == 2
    assert "search_memory" in reply.notes[0]
    assert "valid JSON" in reply.notes[0]
    assert "calculate" in reply.notes[1]


def test_tool_call_written_as_text_is_read_back():
    content = ('<tool_call>\n{"name": "search_memory", "arguments": '
               '{"query": "Iceland"}}\n</tool_call>')
    with FakeLLM(lambda r: content) as llm:
        client = LLMClient(base_url=llm.url, model="m")
        reply = client.chat(USER, tools=TOOLS)
        assert reply.tool_calls == [ToolCall("call_text_0", "search_memory",
                                             {"query": "Iceland"})]
        assert reply.text == ""
        # without tools on offer the text is left alone
        assert "<tool_call>" in client.chat(USER).text


def test_assistant_and_tool_messages_for_the_next_round():
    reply = ChatReply(text="", thinking="secret reasoning",
                      tool_calls=[ToolCall("c1", "calculate",
                                           {"expr": "2+2"})])
    msg = assistant_message(reply)
    assert msg["role"] == "assistant" and "secret" not in json.dumps(msg)
    assert msg["tool_calls"][0]["function"] == {
        "name": "calculate", "arguments": '{"expr": "2+2"}'}
    assert tool_message(reply.tool_calls[0], "4") == {
        "role": "tool", "tool_call_id": "c1", "content": "4"}


# ─── usage ──────────────────────────────────────────────────────────


def test_usage_including_cached_tokens():
    usage = {"prompt_tokens": 900, "completion_tokens": 50,
             "total_tokens": 950,
             "prompt_tokens_details": {"cached_tokens": 800},
             "completion_tokens_details": {"reasoning_tokens": 30}}
    with FakeLLM(lambda r: {"content": "ok", "usage": usage}) as llm:
        reply = LLMClient(base_url=llm.url, model="m").chat(USER)
    assert reply.usage == {"prompt_tokens": 900, "completion_tokens": 50,
                           "total_tokens": 950, "cached_tokens": 800,
                           "reasoning_tokens": 30}


def test_llama_cpp_timings_fill_in():
    body = _body("ok", usage={"prompt_tokens": 10}, timings={
        "cache_n": 6, "prompt_ms": 1250.0, "predicted_per_second": 4.5})
    with FakeLLM(lambda r: body) as llm:
        reply = LLMClient(base_url=llm.url, model="m").chat(USER)
    assert reply.usage["cached_tokens"] == 6
    assert reply.usage["prompt_seconds"] == 1.25
    assert reply.usage["tokens_per_second"] == 4.5


# ─── the no-think switch ────────────────────────────────────────────

CHAT = [{"role": "system", "content": "sys"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"}]


def test_thinking_false_appends_switch_to_last_user_message():
    with FakeLLM(lambda r: "ok") as llm:
        client = LLMClient(base_url=llm.url, model="m")
        original = [dict(m) for m in CHAT]
        client.chat(CHAT, thinking=False)
        sent = llm.requests[-1]["messages"]
        assert sent[-1]["content"] == "second question /no_think"
        assert sent[:3] == CHAT[:3]          # the cached prefix is untouched
        assert CHAT == original              # the caller's list isn't changed
        client.chat(CHAT, thinking=None)
        assert llm.requests[-1]["messages"] == CHAT
        client.chat(CHAT, thinking=True)
        assert llm.requests[-1]["messages"] == CHAT
        # already there: not added twice
        client.chat(llm.requests[0]["messages"], thinking=False)
        assert llm.requests[-1]["messages"][-1]["content"] == \
            "second question /no_think"


def test_no_think_switch_is_configurable(monkeypatch):
    with FakeLLM(lambda r: "ok") as llm:
        monkeypatch.setenv("TELP_LLM_NO_THINK", "/nothink")
        LLMClient(base_url=llm.url, model="m").chat(CHAT, thinking=False)
        assert llm.requests[-1]["messages"][-1]["content"].endswith(
            "/nothink")
        monkeypatch.setenv("TELP_LLM_NO_THINK", "")     # a model with none
        LLMClient(base_url=llm.url, model="m").chat(CHAT, thinking=False)
        assert llm.requests[-1]["messages"] == CHAT


def test_no_think_switch_with_content_parts():
    parts = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    with FakeLLM(lambda r: "ok") as llm:
        LLMClient(base_url=llm.url, model="m").chat(parts, thinking=False)
        sent = llm.requests[-1]["messages"][-1]["content"]
    assert sent[-1] == {"type": "text", "text": "/no_think"}
    assert parts[0]["content"] == [{"type": "text", "text": "hi"}]


# ─── streaming ──────────────────────────────────────────────────────


def test_streaming_with_inline_think_split_across_chunks():
    # FakeLLM streams 8 characters at a time, so the tags arrive in pieces
    content = ("<think>The user asks for a capital; source [1] has it."
               "</think>\n\nThe capital is Reykjavík [1].")
    usage = {"prompt_tokens": 120, "completion_tokens": 30,
             "total_tokens": 150, "prompt_tokens_details":
             {"cached_tokens": 100}}
    tokens, thoughts = [], []
    with FakeLLM(lambda r: {"content": content, "usage": usage}) as llm:
        reply = LLMClient(base_url=llm.url, model="m").chat(
            USER, stream=True, on_token=tokens.append,
            on_thinking=thoughts.append)
        sent = llm.requests[-1]
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
    assert "".join(tokens) == "The capital is Reykjavík [1]."
    assert len(tokens) > 1                     # it really streamed
    assert "".join(thoughts) == \
        "The user asks for a capital; source [1] has it."
    assert reply.text == "The capital is Reykjavík [1]."
    assert reply.thinking == "The user asks for a capital; source [1] has it."
    assert reply.finish_reason == "stop"
    assert reply.usage["prompt_tokens"] == 120
    assert reply.usage["cached_tokens"] == 100
    assert reply.usage["first_token_seconds"] >= 0
    assert reply.usage["first_answer_seconds"] >= \
        reply.usage["first_token_seconds"]


def test_streaming_unclosed_think():
    body = _body("<think>still weighing the sources", finish="length")
    tokens = []
    with FakeLLM(lambda r: body) as llm:
        reply = LLMClient(base_url=llm.url, model="m").chat(
            USER, stream=True, on_token=tokens.append)
    assert tokens == [] and reply.text == ""
    assert reply.thinking == "still weighing the sources"
    assert reply.truncated and reply.notes


def test_streaming_tool_call_deltas_and_reasoning_field():
    chunks = [
        _delta(role="assistant"),
        _delta(reasoning_content="I should search "),
        _delta(reasoning_content="and calculate."),
        _delta(tool_calls=[{"index": 0, "id": "call_a", "type": "function",
                            "function": {"name": "search_memory",
                                         "arguments": '{"que'}}]),
        _delta(tool_calls=[{"index": 0,
                            "function": {"arguments": 'ry": "Ice'}}]),
        _delta(tool_calls=[{"index": 1, "id": "call_b", "type": "function",
                            "function": {"name": "calculate",
                                         "arguments": ""}}]),
        _delta(tool_calls=[{"index": 0, "function": {"arguments": 'land"}'}}]),
        _delta(tool_calls=[{"index": 1,
                            "function": {"arguments": '{"expr": "2+2"}'}}]),
        {"choices": [{"index": 0, "delta": {},
                      "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 50,
                                  "completion_tokens": 20,
                                  "total_tokens": 70}},
    ]

    def handle(h, method, body):
        _send_sse(h, chunks) if method == "POST" else _models(h)
    tokens, thoughts = [], []
    with _Scripted(handle) as srv:
        reply = LLMClient(base_url=srv.url, model="m").chat(
            USER, tools=TOOLS, stream=True, on_token=tokens.append,
            on_thinking=thoughts.append)
    assert reply.tool_calls == [
        ToolCall("call_a", "search_memory", {"query": "Iceland"}),
        ToolCall("call_b", "calculate", {"expr": "2+2"})]
    assert tokens == []                        # thinking never reaches it
    assert "".join(thoughts) == "I should search and calculate."
    assert reply.thinking == "I should search and calculate."
    assert reply.finish_reason == "tool_calls"
    assert reply.usage["total_tokens"] == 70
    assert reply.notes == []


def test_streaming_bad_tool_arguments_note():
    chunks = [_delta(tool_calls=[{"index": 0, "id": "x", "function": {
                  "name": "search_memory", "arguments": '{"query": '}}]),
              {"choices": [{"index": 0, "delta": {},
                            "finish_reason": "tool_calls"}]}]

    def handle(h, method, body):
        _send_sse(h, chunks)
    with _Scripted(handle) as srv:
        reply = LLMClient(base_url=srv.url, model="m").chat(
            USER, tools=TOOLS, stream=True)
    assert reply.tool_calls == [ToolCall("x", "search_memory", {})]
    assert any("valid JSON" in n for n in reply.notes)


def test_sse_quirks_comments_no_space_multiline_crlf():
    raw = [": keep-alive\n\n",
           "event: message\r\ndata:" + json.dumps(_delta(content="Hel"))
           + "\r\n\r\n",
           # one event split over two data lines (joined with a newline)
           'data: {"choices": [{"index": 0,\ndata:  "delta": '
           '{"content": "lo"}}]}\n\n',
           "data: not json at all\n\n",
           "data: " + json.dumps({"choices": [{"index": 0, "delta": {},
                                               "finish_reason": "stop"}]})
           + "\n\n"]

    def handle(h, method, body):
        _send_sse(h, raw)
    tokens = []
    with _Scripted(handle) as srv:
        reply = LLMClient(base_url=srv.url, model="m").chat(
            USER, stream=True, on_token=tokens.append)
    assert reply.text == "Hello" and "".join(tokens) == "Hello"
    assert any("weren't valid JSON" in n for n in reply.notes)


def test_stream_error_event_is_reported():
    def handle(h, method, body):
        _send_sse(h, [_delta(content="Rey"),
                      {"error": {"message": "Model unloaded."}}])
    with _Scripted(handle) as srv:
        with pytest.raises(LLMUnavailable, match="Model unloaded") as err:
            LLMClient(base_url=srv.url, model="m").chat(USER, stream=True)
    assert err.value.partial is not None
    assert err.value.partial.text == "Rey"


def test_stream_error_about_context_gets_the_hint():
    def handle(h, method, body):
        _send_sse(h, [{"error": "Context length exceeded: 9000 > 8192"}])
    with _Scripted(handle) as srv:
        with pytest.raises(LLMUnavailable) as err:
            LLMClient(base_url=srv.url, model="m").chat(USER, stream=True)
    assert "9000 > 8192" in str(err.value)
    assert "raise Context Length" in str(err.value)
    assert err.value.partial is None


def test_a_failing_on_token_is_not_blamed_on_the_server():
    def broken_pipe(text):
        raise BrokenPipeError("stdout closed")
    with FakeLLM(lambda r: "A long enough answer to stream.") as llm:
        with pytest.raises(BrokenPipeError, match="stdout closed"):
            LLMClient(base_url=llm.url, model="m").chat(
                USER, stream=True, on_token=broken_pipe)
        assert len(llm.requests) == 1


def test_server_that_ignores_stream_still_works():
    def handle(h, method, body):
        _send_json(h, 200, _body("Plain reply."))
    tokens = []
    with _Scripted(handle) as srv:
        reply = LLMClient(base_url=srv.url, model="m").chat(
            USER, stream=True, on_token=tokens.append)
    assert reply.text == "Plain reply." and tokens == ["Plain reply."]


# ─── failures ───────────────────────────────────────────────────────


def test_server_down_is_explained_plainly():
    url = f"http://127.0.0.1:{_free_port()}/v1"
    client = LLMClient(base_url=url, model="m")
    with pytest.raises(LLMUnavailable) as err:
        client.chat(USER)
    msg = str(err.value)
    assert f"isn't running at {url}" in msg and "Developer tab" in msg
    with pytest.raises(LLMUnavailable, match="isn't running"):
        LLMClient(base_url=url).model          # auto-detection too
    info = client.status()
    assert info["reachable"] is False and "isn't running" in info["error"]
    assert info["warnings"]


def test_http_error_carries_a_short_snippet_and_a_hint():
    long_tail = "x" * 2000

    def handle(h, method, body):
        if method == "GET":
            return _models(h)
        _send_json(h, 400, {"error": {"message": (
            "Trying to keep the first 9000 tokens when context the "
            "overflows. However, the model is loaded with context length "
            "of only 8192 tokens " + long_tail)}})
    with _Scripted(handle) as srv:
        with pytest.raises(LLMUnavailable) as err:
            LLMClient(base_url=srv.url, model="m").chat(USER)
        assert len(srv.requests) == 1          # HTTP errors aren't retried
    msg = str(err.value)
    assert err.value.status == 400
    assert "HTTP 400" in msg and "context length of only 8192" in msg
    assert "raise Context Length" in msg
    assert long_tail not in msg and len(msg) < 700


def test_http_error_plain_text_body():
    def handle(h, method, body):
        data = b"upstream exploded"
        h.send_response(500)
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)
    with _Scripted(handle) as srv:
        with pytest.raises(LLMUnavailable,
                           match="HTTP 500.*upstream exploded") as err:
            LLMClient(base_url=srv.url, model="m").chat(USER)
    assert err.value.status == 500


def test_one_retry_on_a_dropped_connection():
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("simulated drop")   # server hangs up
        return "Back again."
    with FakeLLM(flaky) as llm:
        reply = LLMClient(base_url=llm.url, model="m").chat(USER)
        assert len(llm.requests) == 2
    assert reply.text == "Back again."


def test_two_drops_in_a_row_give_up_politely():
    def always(req):
        raise ConnectionError("simulated drop")
    with FakeLLM(always) as llm:
        with pytest.raises(LLMUnavailable, match="dropped") as err:
            LLMClient(base_url=llm.url, model="m").chat(USER)
        assert len(llm.requests) == 2
    assert "retry" in str(err.value)


def test_stream_drop_before_any_token_is_retried():
    calls = {"n": 0}

    def handle(h, method, body):
        calls["n"] += 1
        if calls["n"] == 1:
            _send_sse(h, [_delta(role="assistant")], done=False)  # cut short
            return
        _send_sse(h, [_delta(content="Fine."),
                      {"choices": [{"index": 0, "delta": {},
                                    "finish_reason": "stop"}]}])
    tokens = []
    with _Scripted(handle) as srv:
        reply = LLMClient(base_url=srv.url, model="m").chat(
            USER, stream=True, on_token=tokens.append)
        assert len(srv.requests) == 2
    assert reply.text == "Fine." and tokens == ["Fine."]


def test_stream_drop_mid_answer_keeps_what_arrived():
    def handle(h, method, body):
        _send_sse(h, [_delta(content="Reykjavík is "),
                      _delta(content="the cap")], done=False)
    tokens = []
    with _Scripted(handle) as srv:
        with pytest.raises(LLMUnavailable, match="part-way") as err:
            LLMClient(base_url=srv.url, model="m").chat(
                USER, stream=True, on_token=tokens.append)
        assert len(srv.requests) == 1          # no retry: it would repeat
    assert "".join(tokens) == "Reykjavík is the cap"
    assert err.value.partial.text == "Reykjavík is the cap"


def test_timeout_is_honoured_and_not_retried():
    def slow(req):
        time.sleep(1.0)
        return "too late"
    with FakeLLM(slow) as llm:
        client = LLMClient(base_url=llm.url, model="m", timeout=0.3)
        started = time.monotonic()
        with pytest.raises(LLMUnavailable, match="within 0.3 seconds"):
            client.chat(USER)
        assert time.monotonic() - started < 0.9
        assert len(llm.requests) == 1


def test_non_json_reply_is_explained():
    def handle(h, method, body):
        data = b"<html>proxy login</html>"
        h.send_response(200)
        h.send_header("Content-Type", "text/html")
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)
    with _Scripted(handle) as srv:
        with pytest.raises(LLMUnavailable, match="isn't a JSON reply"):
            LLMClient(base_url=srv.url, model="m").chat(USER)


def test_local_server_bypasses_proxies(monkeypatch):
    dead = "http://127.0.0.1:9"
    for key in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(key, dead)
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    with FakeLLM(lambda r: "direct") as llm:
        assert LLMClient(base_url=llm.url, model="m").chat(USER).text == \
            "direct"


def test_localhost_is_dialled_as_ipv4():
    with FakeLLM(lambda r: "ok") as llm:             # bound to 127.0.0.1
        url = f"http://localhost:{llm.port}/v1"
        client = LLMClient(base_url=url, model="m")
        assert client.base_url == url                # shown as given
        assert client.chat(USER).text == "ok"


def test_api_key_is_sent_as_bearer_token(monkeypatch):
    seen = {}

    def handle(h, method, body):
        seen["auth"] = h.headers.get("Authorization")
        _send_json(h, 200, _body("ok"))
    monkeypatch.setenv("TELP_LLM_API_KEY", "sk-local")
    with _Scripted(handle) as srv:
        LLMClient(base_url=srv.url, model="m").chat(USER)
    assert seen["auth"] == "Bearer sk-local"


# ─── status ─────────────────────────────────────────────────────────


def test_status_warns_about_a_small_context():
    with FakeLLM(lambda r: "ok", context_length=8192) as llm:
        client = LLMClient(base_url=llm.url)
        info = client.status()
        assert info["reachable"] and info["base_url"] == llm.url
        assert info["models"] == ["qwen3.8-27b"]
        assert info["model"] == "qwen3.8-27b"
        assert info["context_length"] == 8192
        assert any("raise Context Length" in w and "8,192" in w
                   for w in info["warnings"])
        assert client.context_length() == 8192


def test_status_is_quiet_when_all_is_well():
    with FakeLLM(lambda r: "ok", context_length=32768) as llm:
        info = LLMClient(base_url=llm.url).status()
    assert info["context_length"] == 32768 and info["warnings"] == []


def test_status_reads_lm_studio_rest_api():
    def handle(h, method, body):
        if h.path == "/api/v0/models":
            _send_json(h, 200, {"data": [
                {"id": "qwen3.8-27b", "type": "llm", "state": "loaded",
                 "loaded_context_length": 4096,
                 "max_context_length": 40960}]})
        else:
            _models(h)                           # no context fields here
    with _Scripted(handle) as srv:
        info = LLMClient(base_url=srv.url).status()
    assert info["server"] == "LM Studio"
    assert info["context_length"] == 4096
    assert info["max_context_length"] == 40960
    assert any("40,960" in w for w in info["warnings"])


def test_status_warns_about_unloaded_and_unknown_models():
    def handle(h, method, body):
        if h.path == "/api/v0/models":
            _send_json(h, 200, {"data": [
                {"id": "qwen3.8-27b", "type": "llm", "state": "not-loaded",
                 "max_context_length": 40960}]})
        else:
            _models(h)
    with _Scripted(handle) as srv:
        info = LLMClient(base_url=srv.url).status()
        assert any("isn't loaded yet" in w for w in info["warnings"])
        info = LLMClient(base_url=srv.url, model="mistral").status()
        assert any("'mistral' isn't in the server's list" in w
                   for w in info["warnings"])


def test_status_on_a_plain_openai_server_without_context_info():
    def handle(h, method, body):
        if h.path == "/api/v0/models":
            _send_json(h, 404, {"error": "not found"})
        else:
            _models(h)
    with _Scripted(handle) as srv:
        info = LLMClient(base_url=srv.url).status()
    assert info["reachable"] and info["server"] == "OpenAI-compatible"
    assert info["context_length"] is None
    assert any("didn't say how much context" in w for w in info["warnings"])


# ─── the command-line check ─────────────────────────────────────────


def test_main_reports_status_and_streams_an_answer(monkeypatch, capsys):
    with FakeLLM(lambda r: "<think>hm</think>Hello from Qwen.",
                 context_length=8192) as llm:
        monkeypatch.setenv("TELP_LLM_URL", llm.url)
        assert main(["say", "hello"]) == 0
    out = capsys.readouterr().out
    assert "reachable" in out and "qwen3.8-27b" in out
    assert "warning:" in out and "(thinking...)" in out
    assert "Hello from Qwen.\n" in out.splitlines(keepends=True)
    assert "<think>" not in out


def test_main_when_server_is_down(monkeypatch, capsys):
    monkeypatch.setenv("TELP_LLM_URL", f"http://127.0.0.1:{_free_port()}")
    assert main([]) == 1
    assert "isn't running" in capsys.readouterr().out
