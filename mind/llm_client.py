"""
mind/llm_client.py - Telp's line to a local language model.

In harness mode Telp does the remembering, searching, calculating and
checking, and a local language model only writes. This module is the wire
between them: it talks to LM Studio's local server (or any server that
speaks the OpenAI chat API) and hands back an LLMReply - the answer with
the model's thinking kept separately, any tool calls it made, and the
server's own token counts.

    client = LLMClient()                  # http://localhost:1234/v1
    reply = client.chat(brief.messages(), thinking=False)
    reply.text, reply.thinking, reply.tool_calls, reply.usage, reply.notes
    client.status()                       # is it up? which model? context?

Settings (arguments win over environment variables):
    TELP_LLM_URL       server address, default http://localhost:1234/v1
    TELP_LLM_MODEL     model id; default the first chat model the server
                       lists (preferring one LM Studio says is loaded)
    TELP_LLM_API_KEY   sent as a Bearer token (LM Studio doesn't need one)
    TELP_LLM_NO_THINK  the "don't think" switch chat(thinking=False) adds
                       to the last user message, default "/no_think"

What this module knows about the owner's setup:
  * Thinking models (Qwen) put their reasoning either in a separate
    reasoning_content field or inline as <think>...</think> - and when
    the output is cut off, the <think> is never closed. Both forms are
    split off, so reply.text is only the answer.
  * The no-think switch is MODEL-SPECIFIC. Qwen3-family models obey
    "/no_think"; other models ignore it or may even repeat it back. Set
    TELP_LLM_NO_THINK to your model's own switch, or to "" to send none.
  * LM Studio loads a model with an 8192-token context unless told
    otherwise, and a thinking model can spend all of it thinking. That
    shows up as finish_reason "length"; the reply says so in its notes,
    and status() warns when the loaded context is small.
  * LM Studio reuses its cached reading of the longest unchanged start
    of the prompt, so this module never rewrites earlier messages: the
    no-think switch goes at the very end, on the last user message.
  * "localhost" is dialled as 127.0.0.1 (on Windows, trying IPv6 first
    costs about two seconds a request), and local addresses never go
    through a proxy.
  * timeout is how long to wait without hearing anything from the
    server. Streaming hears every token, so only the wait for the first
    one (the model reading the prompt) and stalls count; without
    streaming the server says nothing until the whole answer is written.

Every failure is an LLMUnavailable whose message is fit to show the user.
"""
from __future__ import annotations

import contextlib
import http.client
import ipaddress
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mind.harness_types import LLMReply, ToolCall  # noqa: E402

DEFAULT_URL = "http://localhost:1234/v1"
DEFAULT_NO_THINK = "/no_think"      # Qwen3's soft switch - model-specific
GOOD_CONTEXT = 16384                # below this, thinking crowds out answers
_PROBE_TIMEOUT = 15.0               # listing models should never take long
_SNIPPET = 300                      # characters of an error body to show

_OPEN, _CLOSE = "<think>", "</think>"
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.S)
_INLINE_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)


class LLMUnavailable(RuntimeError):
    """The model server can't give an answer right now.

    str(err) is a plain-English message fit to show the user. .status is
    the HTTP status when the server answered with an error, and .partial
    is whatever had already streamed in before things went wrong."""

    def __init__(self, message: str, status: int | None = None,
                 partial: LLMReply | None = None):
        super().__init__(message)
        self.status = status
        self.partial = partial


class _Dropped(Exception):
    """The connection broke before a full reply arrived - worth one retry."""


class _CallerError(Exception):
    """An exception from the caller's own on_token / on_thinking, carried
    past the network-error handling so a broken stdout pipe is never
    reported as a dropped connection to the model server."""

    def __init__(self, original: BaseException):
        super().__init__(str(original))
        self.original = original


def _call_back(callback, text: str) -> None:
    try:
        callback(text)
    except Exception as err:
        raise _CallerError(err) from err


@dataclass
class ChatReply(LLMReply):
    """An LLMReply plus plain-English notes about anything that went
    sideways: a cut-off answer, tool arguments that weren't valid JSON.
    Code written against LLMReply keeps working; .notes is extra."""
    notes: list[str] = field(default_factory=list)

    @property
    def truncated(self) -> bool:
        """True when the model ran out of room (max_tokens or context)."""
        return self.finish_reason == "length"


# ─── thinking ───────────────────────────────────────────────────────


def split_thinking(content: str) -> tuple[str, str, bool]:
    """Separate inline <think>...</think> reasoning from the answer.

    Returns (answer, thinking, cut_off); cut_off is True when a <think>
    was never closed - the output stopped while the model was thinking."""
    text = content or ""
    thoughts: list[str] = []
    # Some chat templates open <think> inside the prompt, so the reply
    # starts mid-thought and only the closing tag shows up.
    close, opened = text.find(_CLOSE), text.find(_OPEN)
    if close >= 0 and (opened < 0 or close < opened):
        thoughts.append(text[:close])
        text = text[close + len(_CLOSE):]

    def keep(match: re.Match) -> str:
        thoughts.append(match.group(1))
        return ""

    text = _THINK_BLOCK.sub(keep, text)
    cut_off = False
    opened = text.find(_OPEN)
    if opened >= 0:                      # never closed: cut off mid-thought
        thoughts.append(text[opened + len(_OPEN):])
        text, cut_off = text[:opened], True
    text = text.replace(_CLOSE, "")
    thinking = "\n\n".join(t.strip() for t in thoughts if t.strip())
    return text.strip(), thinking, cut_off


def _partial_tag(text: str, tag: str) -> int:
    """Length of the longest end of `text` that could be the start of
    `tag` - held back until the next piece shows whether it is one."""
    for k in range(min(len(text), len(tag) - 1), 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


class _ThinkSplitter:
    """Routes streamed text to 'answer' or 'thinking' as it arrives, even
    when a <think> tag is split across two pieces."""

    def __init__(self):
        self.in_think = False
        self._buf = ""

    def feed(self, piece: str) -> tuple[str, str]:
        """Returns the (answer, thinking) text that is now certain."""
        self._buf += piece
        answer: list[str] = []
        thought: list[str] = []
        while self._buf:
            tag = _CLOSE if self.in_think else _OPEN
            out = thought if self.in_think else answer
            at = self._buf.find(tag)
            if at >= 0:
                out.append(self._buf[:at])
                self._buf = self._buf[at + len(tag):]
                self.in_think = not self.in_think
                continue
            hold = _partial_tag(self._buf, tag)
            out.append(self._buf[:len(self._buf) - hold])
            self._buf = self._buf[len(self._buf) - hold:]
            break
        return "".join(answer), "".join(thought)

    def flush(self) -> tuple[str, str]:
        rest, self._buf = self._buf, ""
        return ("", rest) if self.in_think else (rest, "")


# ─── reply parsing ──────────────────────────────────────────────────


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _text_of(content) -> str:
    """Message content as plain text (servers may send a list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text") or "" for p in content
                       if isinstance(p, dict))
    return ""


def _snippet(text: str, limit: int = _SNIPPET) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _arguments(raw, name: str, notes: list[str]) -> dict:
    """A tool call's arguments as a dict. Bad JSON becomes {} plus a note,
    never a crash - the caller decides whether to ask again."""
    if isinstance(raw, dict):
        return raw
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}                        # a tool that takes no arguments
    try:
        value = json.loads(raw)
        if isinstance(value, str):       # some models encode twice
            value = json.loads(value)
    except (TypeError, ValueError):
        notes.append(f"The model's call to '{name}' had arguments that "
                     f"weren't valid JSON ({_snippet(raw, 80)}); "
                     "used no arguments instead.")
        return {}
    if isinstance(value, dict):
        return value
    notes.append(f"The model's call to '{name}' sent {type(value).__name__} "
                 "arguments instead of named ones; used no arguments "
                 "instead.")
    return {}


def _tool_calls(raw_calls, notes: list[str]) -> list[ToolCall]:
    """OpenAI-style tool_calls -> ToolCall list."""
    calls: list[ToolCall] = []
    for i, raw in enumerate(raw_calls or []):
        if not isinstance(raw, dict):
            continue
        fn = _dict(raw.get("function"))
        name = fn.get("name") or raw.get("name") or ""
        args = fn.get("arguments", raw.get("arguments"))
        if not name:
            if args:
                notes.append("The model made a tool call without a tool "
                             "name; it was skipped.")
            continue
        calls.append(ToolCall(id=raw.get("id") or f"call_{i}", name=name,
                              arguments=_arguments(args, name, notes)))
    return calls


def _inline_tool_calls(text: str,
                       notes: list[str]) -> tuple[str, list[ToolCall]]:
    """Qwen sometimes writes its tool call as <tool_call>{json}</tool_call>
    text when the server fails to recognise it. Read those back."""
    calls: list[ToolCall] = []
    for match in _INLINE_CALL.finditer(text):
        try:
            data = json.loads(match.group(1))
        except ValueError:
            data = None
        if not isinstance(data, dict) or not data.get("name"):
            notes.append("The model wrote a tool call Telp couldn't read "
                         f"({_snippet(match.group(1), 80)}); it was "
                         "skipped.")
            continue
        calls.append(ToolCall(
            id=f"call_text_{len(calls)}", name=str(data["name"]),
            arguments=_arguments(data.get("arguments"), data["name"],
                                 notes)))
    if "<tool_call>" in text:
        text = _INLINE_CALL.sub("", text).strip()
    return text, calls


def _usage(body: dict | None) -> dict:
    """The server's token counts, normalised. Only numbers, so a caller
    can add up the usage of several rounds."""
    body = _dict(body)
    raw = _dict(body.get("usage"))
    usage: dict = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if _is_num(raw.get(key)):
            usage[key] = int(raw[key])
    cached = _dict(raw.get("prompt_tokens_details")).get("cached_tokens")
    if _is_num(cached):
        usage["cached_tokens"] = int(cached)
    reasoning = _dict(raw.get("completion_tokens_details")).get(
        "reasoning_tokens")
    if _is_num(reasoning):
        usage["reasoning_tokens"] = int(reasoning)
    timings = _dict(body.get("timings"))        # llama.cpp's own server
    if "cached_tokens" not in usage and _is_num(timings.get("cache_n")):
        usage["cached_tokens"] = int(timings["cache_n"])
    if _is_num(timings.get("prompt_ms")):
        usage["prompt_seconds"] = round(timings["prompt_ms"] / 1000, 3)
    if _is_num(timings.get("predicted_per_second")):
        usage["tokens_per_second"] = round(timings["predicted_per_second"], 2)
    stats = _dict(body.get("stats"))            # LM Studio's REST API
    if _is_num(stats.get("time_to_first_token")):
        usage["first_token_seconds"] = round(stats["time_to_first_token"], 3)
    if _is_num(stats.get("tokens_per_second")):
        usage.setdefault("tokens_per_second",
                         round(stats["tokens_per_second"], 2))
    return usage


def _finish_reply(content: str, reasoning: str, raw_calls, finish: str,
                  usage: dict, tools_offered: bool,
                  notes: list[str] | None = None) -> ChatReply:
    """Assemble the reply: thinking split off, tool calls parsed, and a
    plain-English note for anything the caller must not miss."""
    notes = notes if notes is not None else []
    text, inline_thinking, cut_off = split_thinking(content)
    thinking = "\n\n".join(t for t in (reasoning.strip(), inline_thinking)
                           if t)
    calls = _tool_calls(raw_calls, notes)
    if tools_offered and not calls and "<tool_call>" in text:
        text, calls = _inline_tool_calls(text, notes)
    reply = ChatReply(text=text, thinking=thinking, tool_calls=calls,
                      usage=usage, finish_reason=finish or "", notes=notes)
    if reply.truncated:
        notes.append(_cut_off_note(reply))
    elif cut_off:
        notes.append("The output ended inside <think>: the model stopped "
                     "while it was still thinking, so there may be no "
                     "answer.")
    return reply


def _cut_off_note(reply: LLMReply) -> str:
    if not reply.text and not reply.tool_calls and reply.thinking:
        return ("The model ran out of room while still thinking and never "
                "wrote an answer. Raise Context Length in LM Studio's model "
                "settings, allow more max_tokens, or ask with "
                "thinking=False.")
    return ("The answer was cut off (finish_reason \"length\"): the model "
            "hit its max_tokens limit or the end of its loaded context, so "
            "the last part may be missing.")


def assistant_message(reply: LLMReply) -> dict:
    """The assistant turn to append before sending tool results back.
    Thinking is left out on purpose: Qwen's guidance is not to resend it,
    and it would only make the next prompt longer."""
    msg: dict = {"role": "assistant", "content": reply.text or ""}
    if reply.tool_calls:
        msg["tool_calls"] = [
            {"id": c.id, "type": "function",
             "function": {"name": c.name,
                          "arguments": json.dumps(c.arguments,
                                                  ensure_ascii=False)}}
            for c in reply.tool_calls]
    return msg


def tool_message(call: ToolCall, content: str) -> dict:
    """A tool's result, answering one of the model's tool calls."""
    return {"role": "tool", "tool_call_id": call.id, "content": content}


# ─── streaming ──────────────────────────────────────────────────────


def _sse_events(lines: Iterable[bytes]) -> Iterator[str]:
    """Server-sent events: yields each event's data. Comment lines
    (": keep-alive") and other fields are skipped; several data lines in
    one event are joined with newlines, as the SSE standard says."""
    data: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        if name == "data":
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        yield "\n".join(data)


class _StreamState:
    """Everything a streamed reply has said so far."""

    def __init__(self, on_token, on_thinking):
        self.on_token = on_token
        self.on_thinking = on_thinking
        self.t0 = time.monotonic()
        self.raw: list[str] = []          # content exactly as streamed
        self.reasoning: list[str] = []
        self.calls: list[dict] = []
        self.finish = ""
        self.tail: dict = {}              # usage / timings / stats seen
        self.done = False                 # saw [DONE]
        self.bad_chunks = 0
        self.first_token: float | None = None
        self.first_answer: float | None = None
        self.splitter = _ThinkSplitter()

    @property
    def received(self) -> bool:
        """Has any token arrived? (Then a retry would repeat output.)"""
        return self.first_token is not None

    @property
    def complete(self) -> bool:
        return self.done or bool(self.finish)

    def add(self, chunk: dict) -> None:
        for key in ("usage", "timings", "stats"):
            if chunk.get(key):
                self.tail[key] = chunk[key]
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = _dict(choice.get("delta"))
            reasoning = delta.get("reasoning_content") or delta.get(
                "reasoning")
            if isinstance(reasoning, str) and reasoning:
                self._mark()
                self.reasoning.append(reasoning)
                self._think(reasoning)
            content = _text_of(delta.get("content"))
            if content:
                self._mark()
                self.raw.append(content)
                answer, thought = self.splitter.feed(content)
                self._emit(answer)
                self._think(thought)
            for call in delta.get("tool_calls") or []:
                if isinstance(call, dict):
                    self._mark()
                    self._merge_call(call)
            if choice.get("finish_reason"):
                self.finish = choice["finish_reason"]

    def close(self) -> None:
        answer, thought = self.splitter.flush()
        self._emit(answer)
        self._think(thought)

    def _mark(self) -> None:
        if self.first_token is None:
            self.first_token = time.monotonic() - self.t0

    def _emit(self, answer: str) -> None:
        if self.first_answer is None:     # no blank lines before the answer
            answer = answer.lstrip()
            if not answer:
                return
            self.first_answer = time.monotonic() - self.t0
        if answer and self.on_token:
            _call_back(self.on_token, answer)

    def _think(self, thought: str) -> None:
        if thought and self.on_thinking:
            _call_back(self.on_thinking, thought)

    def _merge_call(self, delta: dict) -> None:
        """Tool calls arrive in pieces: the id and name first, then the
        arguments a few characters at a time, matched up by index."""
        idx = delta.get("index")
        if not isinstance(idx, int):      # servers that omit the index
            ids = [c["id"] for c in self.calls]
            if delta.get("id") in ids:
                idx = ids.index(delta["id"])
            elif delta.get("id") or not self.calls:
                idx = len(self.calls)
            else:
                idx = len(self.calls) - 1
        while len(self.calls) <= idx:
            self.calls.append({"id": "", "name": "", "arguments": ""})
        slot = self.calls[idx]
        if delta.get("id") and not slot["id"]:
            slot["id"] = delta["id"]
        fn = _dict(delta.get("function"))
        name = fn.get("name")
        if name:                          # whole, repeated, or in pieces
            same = not slot["name"] or name == slot["name"]
            slot["name"] = name if same else slot["name"] + name
        args = fn.get("arguments")
        if isinstance(args, dict):
            slot["arguments"] = args
        elif isinstance(args, str) and isinstance(slot["arguments"], str):
            slot["arguments"] += args

    def raw_calls(self) -> list[dict]:
        return [{"id": c["id"], "function": {"name": c["name"],
                                             "arguments": c["arguments"]}}
                for c in self.calls]


# ─── addresses ──────────────────────────────────────────────────────


def _normalise_url(url: str) -> str:
    """'localhost:1234' -> 'http://localhost:1234/v1'."""
    url = url.strip().rstrip("/")
    if "://" not in url:
        url = "http://" + url
    if not urllib.parse.urlsplit(url).path:
        url += "/v1"
    return url


def _dial_url(url: str) -> str:
    """The address actually dialled: 'localhost' as 127.0.0.1."""
    parts = urllib.parse.urlsplit(url)
    if parts.hostname != "localhost":
        return url
    netloc = "127.0.0.1" + (f":{parts.port}" if parts.port else "")
    return urllib.parse.urlunsplit(parts._replace(netloc=netloc))


def _is_local(host: str) -> bool:
    """Loopback, home network or a bare machine name: never via a proxy."""
    if not host or host == "localhost" or "." not in host and ":" not in host:
        return True
    if host.endswith((".local", ".localhost", ".lan", ".home.arpa")):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _error_text(body: str) -> str:
    """The useful part of an error body: the JSON message if there is one."""
    try:
        data = json.loads(body)
    except ValueError:
        return _snippet(body)
    if isinstance(data, dict):
        err = data.get("error", data.get("message"))
        if isinstance(err, dict):
            err = err.get("message") or json.dumps(err)
        if err:
            return _snippet(err)
    return _snippet(body)


def _http_hint(code: int, body: str) -> str:
    low = body.lower()
    if "context" in low and any(w in low for w in (
            "length", "overflow", "exceed", "too long", "n_ctx", "fit")):
        return ("The prompt didn't fit the model's loaded context - raise "
                "Context Length in LM Studio's model settings, or send a "
                "smaller brief.")
    if code in (401, 403):
        return "The server wants an API key - set TELP_LLM_API_KEY."
    if code == 404 and "model" in low:
        return ("That model isn't available - load it in LM Studio, or set "
                "TELP_LLM_MODEL to one the server lists.")
    if code >= 500:
        return "LM Studio's Developer tab log says what went wrong."
    return ""


def _pick_model(ids: list[str], lmstudio: dict[str, dict]) -> str:
    """The model to use when none was named: one LM Studio says is
    loaded, else the first that isn't an embedding model, else the first.
    (LM Studio lists every downloaded model, including the embedding
    model it ships with, when just-in-time loading is on.)"""
    def embedding(mid: str) -> bool:
        return (lmstudio.get(mid, {}).get("type") == "embeddings"
                or "embed" in mid.lower())

    loaded = [m for m in ids if lmstudio.get(m, {}).get("state") == "loaded"
              and not embedding(m)]
    chat = [m for m in ids if not embedding(m)]
    return (loaded or chat or ids)[0]


def _first_int(entry: dict, *keys: str) -> int | None:
    for key in keys:
        value = entry.get(key)
        if _is_num(value) and value > 0:
            return int(value)
    return None


# ─── the client ─────────────────────────────────────────────────────


class LLMClient:
    """Telp's connection to LM Studio (or any OpenAI-compatible server)."""

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: float = 600,
                 temperature: float = 0.3):
        self.base_url = _normalise_url(
            base_url or os.environ.get("TELP_LLM_URL") or DEFAULT_URL)
        self._wire = _dial_url(self.base_url)
        self._model = model or os.environ.get("TELP_LLM_MODEL") or None
        self.api_key = api_key or os.environ.get("TELP_LLM_API_KEY") or None
        self.timeout = float(timeout)
        self.temperature = temperature
        self.no_think = os.environ.get("TELP_LLM_NO_THINK", DEFAULT_NO_THINK)
        self._context: int | None = None
        self._context_checked = False
        host = urllib.parse.urlsplit(self._wire).hostname or ""
        # local servers must never be reached through a proxy
        direct = [urllib.request.ProxyHandler({})] if _is_local(host) else []
        self._opener = urllib.request.build_opener(*direct)

    # ── models ──

    @property
    def model(self) -> str:
        """The model id sent with every request (detected on first use)."""
        if not self._model:
            ids = self.list_models()
            if not ids:
                raise LLMUnavailable(self._no_model_message())
            self._model = ids[0] if len(ids) == 1 else _pick_model(
                ids, self._lmstudio_models())
        return self._model

    @model.setter
    def model(self, value: str | None) -> None:
        self._model = value or None
        self._context_checked = False

    def list_models(self) -> list[str]:
        """Model ids from GET /models."""
        return [e["id"] for e in self._model_entries()]

    def _model_entries(self) -> list[dict]:
        body = self._json_call(self._wire + "/models",
                               timeout=min(self.timeout, _PROBE_TIMEOUT))
        data = body.get("data")
        return [e for e in data if isinstance(e, dict) and e.get("id")] \
            if isinstance(data, list) else []

    def _lmstudio_models(self) -> dict[str, dict]:
        """LM Studio's own model list (/api/v0/models): load state, type
        and context lengths. Empty for servers that aren't LM Studio."""
        parts = urllib.parse.urlsplit(self._wire)
        path = parts.path[:-3] if parts.path.endswith("/v1") else parts.path
        root = urllib.parse.urlunsplit(
            (parts.scheme, parts.netloc, path.rstrip("/"), "", ""))
        try:
            body = self._json_call(root + "/api/v0/models",
                                   timeout=min(self.timeout, _PROBE_TIMEOUT))
        except LLMUnavailable:
            return {}
        data = body.get("data")
        if not isinstance(data, list):
            return {}
        return {e["id"]: e for e in data
                if isinstance(e, dict) and e.get("id")}

    def _no_model_message(self) -> str:
        return (f"The model server at {self.base_url} is running but has no "
                "model to use. Load one (e.g. Qwen) in LM Studio, or set "
                "TELP_LLM_MODEL.")

    # ── status ──

    def status(self) -> dict:
        """Is the server up, which model will answer, how much context is
        loaded - plus plain-English warnings. Never raises."""
        info: dict = {"reachable": False, "base_url": self.base_url,
                      "server": "", "models": [], "model": self._model,
                      "context_length": None, "max_context_length": None,
                      "warnings": [], "error": ""}
        warn = info["warnings"].append
        try:
            entries = self._model_entries()
        except LLMUnavailable as err:
            info["error"] = str(err)
            warn(str(err))
            return info
        info["reachable"] = True
        self._context, self._context_checked = None, True
        ids = [e["id"] for e in entries]
        info["models"] = ids
        lmstudio = self._lmstudio_models()
        info["server"] = "LM Studio" if lmstudio else "OpenAI-compatible"
        if not ids:
            warn(self._no_model_message())
            return info
        model = self._model or _pick_model(ids, lmstudio)
        self._model = info["model"] = model
        if model not in ids:
            warn(f"The model '{model}' isn't in the server's list. Load it "
                 "in LM Studio, or set TELP_LLM_MODEL to one of: "
                 + ", ".join(ids) + ".")
            return info
        entry = next((e for e in entries if e["id"] == model), {})
        lm = lmstudio.get(model, {})
        ctx = (_first_int(entry, "loaded_context_length", "context_length",
                          "n_ctx", "max_model_len")
               or _first_int(lm, "loaded_context_length"))
        info["context_length"] = ctx
        info["max_context_length"] = (_first_int(lm, "max_context_length")
                                      or _first_int(entry,
                                                    "max_context_length"))
        self._context = ctx
        if lm.get("state") and lm["state"] != "loaded":
            warn(f"'{model}' isn't loaded yet - LM Studio will load it on the "
                 "first question, which can take a minute or more. Load it "
                 "in LM Studio first, and set its Context Length there.")
        if ctx is None and lm.get("state") in (None, "loaded"):
            warn("The server didn't say how much context the model has "
                 "loaded. If answers stop part-way, raise Context Length in "
                 "LM Studio's model settings.")
        elif ctx is not None and ctx < GOOD_CONTEXT:
            most = info["max_context_length"]
            room = f"; this model allows up to {most:,}" if most else ""
            warn(f"'{model}' is loaded with a {ctx:,}-token context. Qwen's "
                 "thinking can use thousands of tokens - raise Context "
                 f"Length in LM Studio's model settings (to {GOOD_CONTEXT:,} "
                 f"or more{room}).")
        return info

    def context_length(self, refresh: bool = False) -> int | None:
        """The loaded context length, if the server reports it (cached)."""
        if refresh or not self._context_checked:
            self.status()
        return self._context

    # ── chat ──

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int | None = None,
             temperature: float | None = None, stream: bool = False,
             on_token: Callable[[str], None] | None = None,
             thinking: bool | None = None,
             on_thinking: Callable[[str], None] | None = None) -> ChatReply:
        """Send one chat request and return the parsed reply.

        tools       OpenAI-style function tools the model may call
        stream      read the answer as it is written (server-sent events);
                    on_token(text) gets the answer as it arrives (never
                    the thinking - that goes to on_thinking, if given).
                    Without streaming, on_token gets the whole answer once.
        thinking    False adds the no-think switch (TELP_LLM_NO_THINK,
                    model-specific) to the last user message; None and
                    True leave the prompt exactly as given.
        """
        payload: dict = {
            "model": self.model,
            "messages": self._prepare(messages, thinking),
            "temperature": (self.temperature if temperature is None
                            else temperature),
            "stream": bool(stream),
        }
        if tools:
            payload["tools"] = list(tools)
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        if stream:
            payload["stream_options"] = {"include_usage": True}
        started = time.monotonic()
        emitted = False
        if stream:
            reply, emitted = self._chat_stream(payload, on_token,
                                               on_thinking, bool(tools))
        else:
            body = self._json_call(self._wire + "/chat/completions", payload)
            reply = self._reply_from_body(body, bool(tools))
        if on_token and not emitted and reply.text:
            on_token(reply.text)
        reply.seconds = round(time.monotonic() - started, 3)
        return reply

    def _prepare(self, messages: list[dict],
                 thinking: bool | None) -> list[dict]:
        """Copies of the messages (the caller's list is never changed),
        with the no-think switch on the last user message if asked."""
        msgs = [dict(m) for m in messages]
        if thinking is False and self.no_think:
            for msg in reversed(msgs):
                if msg.get("role") == "user":
                    msg["content"] = _with_switch(msg.get("content"),
                                                  self.no_think)
                    break
        return msgs

    def _reply_from_body(self, body: dict, tools_offered: bool) -> ChatReply:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices \
                or not isinstance(choices[0], dict):
            raise LLMUnavailable(
                f"The model server at {self.base_url} sent a reply with no "
                f"answer in it: {_snippet(json.dumps(body))}")
        choice = choices[0]
        msg = _dict(choice.get("message"))
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        return _finish_reply(_text_of(msg.get("content")),
                             reasoning if isinstance(reasoning, str) else "",
                             msg.get("tool_calls"),
                             choice.get("finish_reason") or "",
                             _usage(body), tools_offered)

    def _chat_stream(self, payload: dict, on_token, on_thinking,
                     tools_offered: bool) -> tuple[ChatReply, bool]:
        """Stream one reply; (reply, whether on_token already saw it).
        A connection that drops before any token arrives is retried once;
        after that a retry would repeat output, so it is reported instead,
        with what had arrived as err.partial."""
        data = json.dumps(payload).encode("utf-8")
        where = "/chat/completions"
        for attempt in (1, 2):
            state = _StreamState(on_token, on_thinking)
            req = urllib.request.Request(
                self._wire + where, data=data, method="POST",
                headers=self._headers(json_body=True, stream=True))
            try:
                with self._errors(where, self.timeout):
                    with self._opener.open(req, timeout=self.timeout) as resp:
                        if resp.headers.get_content_type() != \
                                "text/event-stream":
                            # the server ignored "stream": a plain reply
                            body = self._decode(resp.read(), where)
                            return (self._reply_from_body(body,
                                                          tools_offered),
                                    False)
                        for event in _sse_events(resp):
                            if self._take_event(state, event):
                                break
                if not state.complete:
                    raise _Dropped("the stream ended without finishing")
            except _Dropped as drop:
                if attempt == 1 and not state.received:
                    continue
                partial = (self._stream_reply(state, tools_offered)
                           if state.received else None)
                raise self._dropped_error(drop, attempt, partial) from None
            except LLMUnavailable as err:
                if state.received and err.partial is None:
                    err.partial = self._stream_reply(state, tools_offered)
                raise
            except _CallerError as err:
                raise err.original from None
            return self._stream_reply(state, tools_offered), True
        raise AssertionError("unreachable")     # the loop always returns

    def _take_event(self, state: _StreamState, event: str) -> bool:
        """Feed one SSE event to the stream state; True at [DONE]."""
        if event.strip() == "[DONE]":
            state.done = True
            return True
        try:
            chunk = json.loads(event)
        except ValueError:
            state.bad_chunks += 1
            return False
        if not isinstance(chunk, dict):
            return False
        if chunk.get("error"):
            hint = _http_hint(200, event)
            raise LLMUnavailable(
                f"The model server at {self.base_url} stopped with an error: "
                f"{_error_text(event)}" + (f" {hint}" if hint else ""))
        state.add(chunk)
        return False

    def _stream_reply(self, state: _StreamState,
                      tools_offered: bool) -> ChatReply:
        state.close()
        usage = _usage(state.tail)
        if state.first_token is not None:
            usage["first_token_seconds"] = round(state.first_token, 3)
        if state.first_answer is not None:
            usage["first_answer_seconds"] = round(state.first_answer, 3)
        notes: list[str] = []
        if state.bad_chunks:
            notes.append(f"{state.bad_chunks} streamed piece(s) weren't "
                         "valid JSON and were skipped, so the answer may "
                         "have gaps.")
        return _finish_reply("".join(state.raw), "".join(state.reasoning),
                             state.raw_calls(), state.finish, usage,
                             tools_offered, notes)

    # ── HTTP ──

    def _headers(self, json_body: bool, stream: bool = False) -> dict:
        headers = {"Accept": "text/event-stream" if stream
                   else "application/json", "User-Agent": "telp-harness"}
        if json_body:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _json_call(self, url: str, payload: dict | None = None,
                   timeout: float | None = None) -> dict:
        """GET (or POST, with a payload) and parse the JSON reply. A
        dropped connection is retried once."""
        timeout = timeout or self.timeout
        data = None if payload is None else json.dumps(payload).encode()
        where = urllib.parse.urlsplit(url).path
        for attempt in (1, 2):
            req = urllib.request.Request(
                url, data=data, method="GET" if data is None else "POST",
                headers=self._headers(json_body=data is not None))
            try:
                with self._errors(where, timeout):
                    with self._opener.open(req, timeout=timeout) as resp:
                        raw = resp.read()
            except _Dropped as drop:
                if attempt == 1:
                    continue
                raise self._dropped_error(drop, attempt) from None
            return self._decode(raw, where)
        raise AssertionError("unreachable")     # the loop always returns

    def _decode(self, raw: bytes, where: str) -> dict:
        try:
            body = json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise LLMUnavailable(
                f"The model server at {self.base_url} sent something that "
                f"isn't a JSON reply to {where}: "
                f"{_snippet(raw.decode('utf-8', 'replace'))}")
        return body

    @contextlib.contextmanager
    def _errors(self, where: str, timeout: float):
        """Turn network trouble into LLMUnavailable (or _Dropped)."""
        try:
            yield
        except urllib.error.HTTPError as err:
            raise self._http_error(err, where) from None
        except urllib.error.URLError as err:
            raise self._network_error(err.reason, timeout) from None
        except (OSError, http.client.HTTPException) as err:
            raise self._network_error(err, timeout) from None

    def _http_error(self, err: urllib.error.HTTPError,
                    where: str) -> LLMUnavailable:
        try:
            body = err.read().decode("utf-8", "replace")
        except Exception:                 # the body is a nice-to-have
            body = ""
        message = (f"The model server at {self.base_url} answered HTTP "
                   f"{err.code} to {where}")
        detail = _error_text(body) if body else ""
        message += f": {detail}" if detail else "."
        hint = _http_hint(err.code, body)
        if hint:
            message += f" {hint}"
        return LLMUnavailable(message, status=err.code)

    def _network_error(self, reason, timeout: float) -> Exception:
        if isinstance(reason, TimeoutError) or (
                isinstance(reason, str) and "timed out" in reason):
            return LLMUnavailable(
                f"The model server at {self.base_url} didn't answer within "
                f"{timeout:g} seconds. A big model that runs partly on the "
                "CPU reads long prompts slowly - try again, send less, or "
                "allow a longer timeout.")
        if isinstance(reason, ConnectionRefusedError):
            return LLMUnavailable(
                f"LM Studio's local server isn't running at {self.base_url}. "
                "In LM Studio open the Developer tab and start the server.")
        if isinstance(reason, socket.gaierror):
            return LLMUnavailable(
                f"Can't find the host in {self.base_url} - check "
                "TELP_LLM_URL.")
        if isinstance(reason, (ConnectionResetError, ConnectionAbortedError,
                               BrokenPipeError, http.client.HTTPException)):
            return _Dropped(str(reason) or type(reason).__name__)
        return LLMUnavailable(
            f"Couldn't reach the model server at {self.base_url}: {reason}")

    def _dropped_error(self, drop: _Dropped, attempt: int,
                       partial: LLMReply | None = None) -> LLMUnavailable:
        again = " (and again on a retry)" if attempt > 1 else ""
        when = "part-way through the answer" if partial else \
            "before it answered"
        return LLMUnavailable(
            f"The connection to the model server at {self.base_url} dropped "
            f"{when}{again}: {drop}. Check that LM Studio is still running "
            "and the model is still loaded.", partial=partial)


def _with_switch(content, switch: str):
    """Message content with the no-think switch added at the very end."""
    if isinstance(content, list):
        texts = [p.get("text") or "" for p in content
                 if isinstance(p, dict) and p.get("type") == "text"]
        if texts and texts[-1].rstrip().endswith(switch):
            return content
        return list(content) + [{"type": "text", "text": switch}]
    text = (content or "").rstrip()
    if text.endswith(switch):
        return text
    return f"{text} {switch}" if text else switch


# ─── a quick check from the command line ─────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """python -m mind.llm_client [question]

    Shows whether the model server is reachable, which model answers and
    how much context it has loaded; given a question, streams an answer."""
    args = sys.argv[1:] if argv is None else argv
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    client = LLMClient()
    info = client.status()
    print(f"server:  {info['base_url']} "
          f"({'reachable' if info['reachable'] else 'not reachable'})")
    if info["models"]:
        print("models:  " + ", ".join(info["models"]))
    if info["model"] and info["reachable"]:
        print(f"model:   {info['model']}")
    if info["context_length"]:
        print(f"context: {info['context_length']:,} tokens loaded")
    for warning in info["warnings"]:
        print("warning: " + warning)
    if not info["reachable"] or not args:
        return 0 if info["reachable"] else 1
    shown = {"thinking": False}

    def thinking_dot(_text: str) -> None:
        if not shown["thinking"]:
            print("(thinking...)", flush=True)
            shown["thinking"] = True

    try:
        reply = client.chat([{"role": "user", "content": " ".join(args)}],
                            stream=True, on_thinking=thinking_dot,
                            on_token=lambda t: print(t, end="", flush=True))
    except LLMUnavailable as err:
        print(f"\n{err}")
        return 1
    print()
    used = reply.usage
    print(f"({used.get('prompt_tokens', '?')} prompt tokens, "
          f"{used.get('cached_tokens', 0)} cached, "
          f"{used.get('completion_tokens', '?')} written; first token after "
          f"{used.get('first_token_seconds', '?')}s, {reply.seconds}s in all)")
    for note in reply.notes:
        print("note: " + note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
