"""
A stand-in for LM Studio's local server: an OpenAI-compatible HTTP endpoint
on 127.0.0.1 that answers /v1/models and /v1/chat/completions with replies
your test scripts, and records every request it gets.

    with FakeLLM(lambda req: "Reykjavík is the capital [1].") as llm:
        client = LLMClient(base_url=llm.url, model="qwen3.8-27b")
        ...
        llm.requests      # the JSON bodies the server received

A responder gets the request JSON and returns either a string (the
assistant's text), a dict with "content" / "tool_calls" / "reasoning_content"
/ "usage" keys, or a full chat-completion response dict (has "choices").
Streaming requests ("stream": true) are answered as server-sent events.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _completion(model: str, msg: dict, usage: dict | None,
                finish: str) -> dict:
    return {
        "id": "chatcmpl-fake", "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0,
                           "total_tokens": 0},
    }


class FakeLLM:
    def __init__(self, responder, models=("qwen3.8-27b",),
                 context_length: int = 32768):
        self.responder = responder
        self.models = list(models)
        self.context_length = context_length
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):            # keep test output quiet
                pass

            def _send(self, code: int, body: dict):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.rstrip("/").endswith("/models"):
                    self._send(200, {"object": "list", "data": [
                        {"id": m, "object": "model",
                         "loaded_context_length": outer.context_length}
                        for m in outer.models]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(n) or b"{}")
                outer.requests.append(req)
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    self._send(404, {"error": "not found"})
                    return
                out = outer.responder(req)
                model = req.get("model", outer.models[0])
                if isinstance(out, dict) and "choices" in out:
                    body = out
                else:
                    if isinstance(out, str):
                        out = {"content": out}
                    msg = {"role": "assistant",
                           "content": out.get("content")}
                    if out.get("reasoning_content"):
                        msg["reasoning_content"] = out["reasoning_content"]
                    finish = "stop"
                    if out.get("tool_calls"):
                        msg["tool_calls"] = [
                            {"id": tc.get("id", f"call_{i}"),
                             "type": "function",
                             "function": {"name": tc["name"],
                                          "arguments": json.dumps(
                                              tc.get("arguments", {}))}}
                            for i, tc in enumerate(out["tool_calls"])]
                        finish = "tool_calls"
                    body = _completion(model, msg, out.get("usage"), finish)
                if req.get("stream"):
                    self._stream(body)
                else:
                    self._send(200, body)

            def _stream(self, body: dict):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                msg = body["choices"][0]["message"]
                text = msg.get("content") or ""
                for i in range(0, len(text), 8):
                    chunk = {"choices": [{"index": 0, "delta":
                                          {"content": text[i:i + 8]}}]}
                    self.wfile.write(
                        f"data: {json.dumps(chunk)}\n\n".encode())
                final = {"choices": [{"index": 0, "delta": {},
                                      "finish_reason":
                                      body["choices"][0]["finish_reason"]}],
                         "usage": body.get("usage")}
                self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()

    def last_user_message(self) -> str:
        """The final user message of the most recent request."""
        msgs = self.requests[-1].get("messages", []) if self.requests else []
        users = [m for m in msgs if m.get("role") == "user"]
        return users[-1]["content"] if users else ""
