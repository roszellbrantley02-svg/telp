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
    TELP_LLM_MODEL     model id. Without one the client picks a chat model
                       itself (one LM Studio says is loaded, else the first
                       it lists, never an embedding model) and checks that
                       choice again as the session goes on, so loading a
                       different model in LM Studio is followed.
    TELP_LLM_API_KEY   sent as a Bearer token (LM Studio doesn't need one)
    TELP_LLM_NO_THINK  the "don't think" switch chat(thinking=False) adds
                       to the last user message, default "/no_think"

What this module knows about the owner's setup:
  * Thinking models (Qwen) put their reasoning either in a separate
    reasoning_content field or inline as <think>...</think> - and when
    the output is cut off, the <think> is never closed. Some chat
    templates write the <think> into the prompt themselves, so the reply
    starts mid-thought and only </think> shows up. Once a reply shows
    that, the client remembers it and treats the next replies as thinking
    until their </think>. reply.text is only ever the answer - and a
    <think> merely mentioned in an answer stays in it.
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
  * Timeouts. Connecting has a few seconds of its own (connect_timeout),
    so a PC that is off or asleep is reported at once. After that,
    timeout is how long to wait without hearing from the server.
    Streaming hears every token. Without streaming the server says
    nothing until the whole answer is written - many minutes, for a 27B
    model that thinks on a small GPU - so after each `timeout` seconds
    of silence the client checks that the server still answers (a quick
    GET /models) and keeps waiting while it does.

Every failure is an LLMUnavailable whose message is fit to show the user.
"""
from __future__ import annotations

import contextlib
import errno
import functools
import http.client
import ipaddress
import json
import math
import os
import re
import selectors
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
_CONNECT_TIMEOUT = 5.0              # a PC that is on accepts in milliseconds
_PROBE_TIMEOUT = 15.0               # listing models should never take long
_TAIL_TIMEOUT = 2.0                 # wait for usage / [DONE] after the finish
_RECHECK_SECONDS = 10.0             # how old an auto-detected model may get
_SNIPPET = 300                      # characters of an error body to show

_OPEN, _CLOSE = "<think>", "</think>"
_INLINE_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)

# signs, in an HTTP error, that the model asked for isn't there
_MISSING_MODEL = ("not found", "not loaded", "isn't loaded", "no model",
                  "does not exist", "doesn't exist", "unknown model",
                  "model_not_found")


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


class _ConnectTimeout(OSError):
    """Connecting took longer than connect_timeout: nothing answered."""

    def __init__(self, seconds: float):
        super().__init__(f"connecting timed out after {seconds:g} seconds")
        self.seconds = seconds


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


def _partial_tag(text: str, tag: str) -> int:
    """Length of the longest end of `text` that could be the start of
    `tag` - held back until the next piece shows whether it is one."""
    for k in range(min(len(text), len(tag) - 1), 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


class _ThinkSplitter:
    """Sorts a reply's text into answer and thinking as it arrives, piece
    by piece, holding back only the few characters that might be the
    start of a tag. Whole replies go through it too (split_thinking), so
    streamed and plain replies follow the same rules:

      * <think> opens thinking only at the start of a line. Qwen writes
        it first thing, and models that think again between steps start
        a new line for it; "a <think> tag" mid-sentence is answer text.
      * </think> closes it.
      * A </think> before any <think>, with a line break right before or
        after it, means the chat template wrote <think> into the prompt:
        all the text before it was thinking (found_template).
      * opens_think=True (that was seen in an earlier reply) starts the
        reply inside thinking. If the reply then ends without a </think>
        and wasn't cut off, the template didn't open one this time: the
        text was the answer, and is handed out as answer at the end.
      * separate=True (the server sends the reasoning in its own field):
        the text is the answer; only a <think> block at the very start
        is taken out.
    """

    def __init__(self, opens_think: bool = False, separate: bool = False):
        self.separate = separate
        self._think = opens_think and not separate     # inside thinking
        self._assumed = self._think      # ...because the template opened it
        self._stray_close_ok = True      # a lone </think> may end thinking
        self._started = False            # any non-blank text yet
        self._line_blank = True          # only blanks since the line began
        self._buf = ""                   # held back: maybe part of a tag
        self.answer: list[str] = []
        self.blocks: list[list[str]] = [[]] if self._think else []
        self.found_template = False      # the reply started mid-thought
        self.opened_itself = False       # ...or, expected to, wrote <think>
        self.cut_off = False             # ended while still thinking

    def mark_separate(self) -> None:
        """The server turned out to send the reasoning in its own field
        (only taken into account before any text has arrived)."""
        if self._started or self.separate:
            return
        self.separate = True
        if self._assumed:
            self._think = self._assumed = False
            self.blocks = []

    def feed(self, piece: str) -> tuple[str, str]:
        """Returns the (answer, thinking) text that is now certain."""
        self._buf += piece
        return self._run(eof=False)

    def flush(self, cut: bool) -> tuple[str, str]:
        """The reply has ended - cut=True when it was cut off. Returns the
        last (answer, thinking) text to hand out."""
        said, thought = self._run(eof=True)
        if self._think:
            block = "".join(self.blocks[-1])
            if self._assumed and not cut:
                # no </think> came: the template didn't open one this time
                self.blocks.pop()
                self._think = self._assumed = False
                said += self._keep_answer(block, [])
            elif self.separate and not cut:
                # the reasoning came separately, so this text is the answer
                self.blocks.pop()
                self._think = False
                said += self._keep_answer(_OPEN + block, [])
            else:
                self.cut_off = True
        return said, thought

    def result(self) -> tuple[str, str, bool]:
        """(answer, thinking, cut_off) of everything fed so far."""
        thoughts = ("".join(b).strip() for b in self.blocks)
        return ("".join(self.answer).strip(),
                "\n\n".join(t for t in thoughts if t), self.cut_off)

    # ── the steps ──

    def _run(self, eof: bool) -> tuple[str, str]:
        said: list[str] = []
        thought: list[str] = []
        while self._buf:
            step = self._think_step if self._think else self._answer_step
            if not step(eof, said, thought):
                break
        return "".join(said), "".join(thought)

    def _think_step(self, eof: bool, said: list, thought: list) -> bool:
        """Consume thinking text; False when more text is needed."""
        buf = self._buf
        if self._assumed and not self._started:
            # the reply may still write its own <think> first thing
            head = buf.lstrip()
            if not eof and (not head or (len(head) < len(_OPEN)
                                         and _OPEN.startswith(head))):
                return False
            if head.startswith(_OPEN):
                self._assumed, self.opened_itself = False, True
                self._started = True
                self._buf = head[len(_OPEN):]
                return True
        at = buf.find(_CLOSE)
        if at >= 0:
            self._keep_thought(buf[:at], thought)
            self._buf = buf[at + len(_CLOSE):]
            if self._assumed:
                self.found_template = True      # confirmed once more
            self._think = self._assumed = False
            self._stray_close_ok = False
            self._line_blank = True
            return True
        hold = 0 if eof else _partial_tag(buf, _CLOSE)
        self._keep_thought(buf[:len(buf) - hold], thought)
        self._buf = buf[len(buf) - hold:]
        return False

    def _answer_step(self, eof: bool, said: list, thought: list) -> bool:
        """Consume answer text; False when more text is needed."""
        buf = self._buf
        at = buf.find("<")
        if at < 0:
            self._keep_answer(buf, said)
            self._buf = ""
            return False
        if at > 0:
            self._keep_answer(buf[:at], said)
            self._buf = buf[at:]
            return True
        if buf.startswith(_OPEN):
            if self._line_blank and not (self.separate and self._started):
                self._think = True
                self.blocks.append([])
                self._stray_close_ok = False
                self._started = True
                self._buf = buf[len(_OPEN):]
                return True
        elif buf.startswith(_CLOSE):
            verdict = self._stray_close(buf[len(_CLOSE):], eof)
            if verdict is None:
                return False                    # wait for the next piece
            if verdict:
                self._end_template_thinking()
                self._buf = buf[len(_CLOSE):]
                return True
        elif not eof and (_OPEN.startswith(buf) or _CLOSE.startswith(buf)):
            return False                        # might become a tag
        self._keep_answer("<", said)            # just a "<" in the answer
        self._buf = buf[1:]
        return True

    def _stray_close(self, after: str, eof: bool) -> bool | None:
        """Does a </think> with no <think> before it end thinking that
        the template opened? None: can't tell until more text arrives."""
        if not self._stray_close_ok:
            return False
        if self.separate:                       # only a leftover tag
            return not self._started
        if self._line_blank:
            return True
        rest = after.lstrip(" \t")
        if not rest:
            return True if eof else None
        return rest[0] in "\r\n"

    def _end_template_thinking(self) -> None:
        """Everything so far was thinking the chat template opened."""
        self._stray_close_ok = False
        self._line_blank = True
        if self.separate:
            return                              # nothing but blanks before
        self.blocks.insert(0, self.answer)
        self.answer = []
        self.found_template = True

    def _keep_answer(self, text: str, said: list) -> str:
        if not text:
            return ""
        self.answer.append(text)
        said.append(text)
        if text.strip():
            self._started = True
        head, newline, tail = text.rpartition("\n")
        if newline:
            self._line_blank = not tail.strip()
        elif text.strip():
            self._line_blank = False
        return text

    def _keep_thought(self, text: str, thought: list) -> None:
        if not text:
            return
        self.blocks[-1].append(text)
        thought.append(text)
        if text.strip():
            self._started = True


def split_thinking(content: str, opens_think: bool = False,
                   separate: bool = False,
                   cut: bool = False) -> tuple[str, str, bool]:
    """Separate inline <think>...</think> reasoning from the answer.

    Returns (answer, thinking, cut_off); cut_off is True when the output
    stopped while the model was still thinking. opens_think: the chat
    template writes <think> into the prompt, so the text starts inside
    thinking. separate: the reasoning came in its own field, so this is
    the answer. cut: the output was cut off (finish_reason "length")."""
    splitter = _ThinkSplitter(opens_think, separate)
    splitter.feed(content or "")
    splitter.flush(cut)
    return splitter.result()


def _thoughts(reasoning: str, inline: str) -> str:
    return "\n\n".join(t for t in ((reasoning or "").strip(), inline) if t)


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


def _snippet(text, limit: int = _SNIPPET) -> str:
    """A short, printable piece of text for a message (control bytes from
    a server that doesn't speak HTTP become '?')."""
    text = "".join(ch if ch.isprintable() or ch.isspace() else "?"
                   for ch in str(text))
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _json_kind(value) -> str:
    """What a JSON value is, in words (for notes the user may read)."""
    if isinstance(value, list):
        return "a list"
    if isinstance(value, str):
        return "a piece of text"
    if isinstance(value, bool):
        return "true/false"
    if isinstance(value, (int, float)):
        return "a number"
    return "something else"


def _arguments(raw, name: str, notes: list[str]) -> dict:
    """A tool call's arguments as a dict. Bad JSON becomes {} plus a note,
    never a crash - the caller decides whether to ask again."""
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}                        # a tool that takes no arguments
    if not isinstance(raw, str):
        value = raw
    elif not raw.strip():
        return {}
    else:
        try:
            value = json.loads(raw)
        except ValueError:
            notes.append(f"The model's call to '{name}' had arguments that "
                         f"weren't valid JSON ({_snippet(raw, 80)}); "
                         "used no arguments instead.")
            return {}
        if isinstance(value, str):       # some models encode twice
            with contextlib.suppress(ValueError):
                value = json.loads(value)
    if value is None:                    # "null": no arguments
        return {}
    if isinstance(value, dict):
        return value
    notes.append(f"The model's call to '{name}' sent {_json_kind(value)} "
                 "instead of named arguments; used no arguments instead.")
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
    text when the server fails to recognise it. Read those back (the
    arguments may be called "arguments" or "parameters")."""
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
        name = str(data["name"])
        args = data["arguments"] if "arguments" in data \
            else data.get("parameters")
        calls.append(ToolCall(id=f"call_text_{len(calls)}", name=name,
                              arguments=_arguments(args, name, notes)))
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


def _finish_reply(text: str, thinking: str, cut_off: bool, raw_calls,
                  finish: str, usage: dict, tools_offered: bool,
                  notes: list[str] | None = None) -> ChatReply:
    """Assemble the reply from text already split from its thinking: tool
    calls parsed, and a plain-English note for anything the caller must
    not miss."""
    notes = notes if notes is not None else []
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


def _sse_events(lines: Iterable[bytes]) -> Iterator[tuple[str, str]]:
    """Server-sent events as (kind, data). kind is "error" for an
    "event: error" event and for llama.cpp's own "error:" field (which
    its server sends, then still sends [DONE]); otherwise the event's
    name, "message" by default. Comment lines (": keep-alive") are
    skipped; several data lines in one event are joined with newlines,
    as the SSE standard says."""
    data: list[str] = []
    errors: list[str] = []
    kind = ""
    for raw in lines:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if errors:
                yield "error", "\n".join(errors)
            elif kind == "error" or data:
                yield kind or "message", "\n".join(data)
            data, errors, kind = [], [], ""
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if name == "data":
            data.append(value)
        elif name == "event":
            kind = value.strip()
        elif name == "error":
            errors.append(value)
    if errors:
        yield "error", "\n".join(errors)
    elif kind == "error" or data:
        yield kind or "message", "\n".join(data)


def _complete_json(args) -> bool:
    """Are a streamed call's arguments already a whole JSON value?"""
    if isinstance(args, dict):
        return True
    if not isinstance(args, str) or not args.strip():
        return False
    try:
        json.loads(args)
    except ValueError:
        return False
    return True


class _StreamState:
    """Everything a streamed reply has said so far."""

    def __init__(self, on_token, on_thinking, opens_think: bool = False):
        self.on_token = on_token
        self.on_thinking = on_thinking
        self.t0 = time.monotonic()
        self.raw: list[str] = []          # content exactly as streamed
        self.reasoning: list[str] = []
        self.calls: list[dict] = []
        self._slot_of: dict[int, int] = {}   # streamed index -> position
        self.finish = ""
        self.tail: dict = {}              # usage / timings / stats seen
        self.done = False                 # saw [DONE]
        self.winding_down = False         # finish seen: only the tail left
        self.bad_chunks = 0
        self.first_token: float | None = None
        self.first_answer: float | None = None
        self.splitter = _ThinkSplitter(opens_think=opens_think)

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
                self.splitter.mark_separate()
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

    def close(self, cut: bool, emit: bool = True) -> None:
        """The stream is over: hand out what the splitter still holds
        (not for a partial reply kept after an error)."""
        answer, thought = self.splitter.flush(cut)
        if emit:
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
        cid = delta.get("id") or ""
        fn = _dict(delta.get("function"))
        name = fn.get("name") or ""
        slot = self._slot_for(delta.get("index"), cid, name)
        if cid and not slot["id"]:
            slot["id"] = cid
        if name:                          # whole, repeated, or in pieces
            same = not slot["name"] or name == slot["name"]
            slot["name"] = name if same else slot["name"] + name
        args = fn.get("arguments")
        if isinstance(args, dict):
            slot["arguments"] = args
        elif isinstance(args, str) and isinstance(slot["arguments"], str):
            slot["arguments"] += args

    def _slot_for(self, idx, cid: str, name: str) -> dict:
        """The call this delta belongs to. Servers differ: some reuse
        index 0 for every call, some send no index at all, so a new id -
        or a new name once the arguments are under way - starts a new
        call."""
        if isinstance(idx, int) and not isinstance(idx, bool):
            pos = self._slot_of.get(idx)
            if pos is not None and not _starts_new(self.calls[pos], cid,
                                                   name):
                return self.calls[pos]
            self._slot_of[idx] = len(self.calls)
            return self._new_slot()
        if cid:                           # no index: match by id
            for slot in self.calls:
                if slot["id"] == cid:
                    return slot
            return self._new_slot()
        if self.calls and not _starts_new(self.calls[-1], cid, name):
            return self.calls[-1]
        return self._new_slot()

    def _new_slot(self) -> dict:
        self.calls.append({"id": "", "name": "", "arguments": ""})
        return self.calls[-1]

    def raw_calls(self) -> list[dict]:
        return [{"id": c["id"], "function": {"name": c["name"],
                                             "arguments": c["arguments"]}}
                for c in self.calls]


def _starts_new(slot: dict, cid: str, name: str) -> bool:
    """Does a tool-call delta begin another call rather than continue
    `slot`? A different id does; so does a name once the slot's
    arguments are under way - unless it is the same name repeated with
    every piece, before the arguments are complete."""
    if cid and slot["id"] and cid != slot["id"]:
        return True
    if name and slot["name"] and slot["arguments"]:
        return name != slot["name"] or _complete_json(slot["arguments"])
    return False


# ─── connections ────────────────────────────────────────────────────


class _Dialling:
    """Mixed into http.client's connections. Connecting gets its own
    short timeout, so a PC that is off or a firewall that drops packets
    is noticed in seconds, not after the whole answer timeout. A request
    may bring a waiter, called before the reply is read (see
    LLMClient._await_reply); the reply then carries its socket, so a
    stream can shorten its wait once the answer is finished."""

    def __init__(self, *args, connect_timeout: float = _CONNECT_TIMEOUT,
                 waiter: Callable | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._connect_timeout = connect_timeout
        self._waiter = waiter

    def connect(self):
        wait = self.timeout if _is_num(self.timeout) else None
        self.timeout = (min(self._connect_timeout, wait) if wait
                        else self._connect_timeout)
        try:
            super().connect()
        except TimeoutError:
            raise _ConnectTimeout(self.timeout) from None
        finally:
            self.timeout = wait
        self.sock.settimeout(wait)

    def getresponse(self):
        sock = self.sock            # (http.client lets go of it on reading
        if self._waiter is not None and sock is not None:   # a closing reply)
            self._waiter(sock)
        response = super().getresponse()
        response.telp_sock = sock
        return response


class _HTTPConnection(_Dialling, http.client.HTTPConnection):
    pass


def _dial_args(req: urllib.request.Request) -> dict:
    return {"connect_timeout": getattr(req, "telp_connect", _CONNECT_TIMEOUT),
            "waiter": getattr(req, "telp_waiter", None)}


class _HTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(functools.partial(_HTTPConnection,
                                              **_dial_args(req)), req)


def _handlers() -> list:
    handlers: list = [_HTTPHandler()]
    if hasattr(http.client, "HTTPSConnection"):     # Python built with ssl
        class _HTTPSConnection(_Dialling, http.client.HTTPSConnection):
            pass

        class _HTTPSHandler(urllib.request.HTTPSHandler):
            def https_open(self, req):
                return self.do_open(functools.partial(
                    _HTTPSConnection, **_dial_args(req)), req,
                    context=self._context)

        handlers.append(_HTTPSHandler())
    return handlers


# ─── addresses ──────────────────────────────────────────────────────


def _normalise_url(url: str) -> str:
    """'localhost:1234' -> 'http://localhost:1234/v1'; a pasted endpoint
    ('.../v1/chat/completions') becomes its base address."""
    url = url.strip().rstrip("/")
    if "://" not in url:
        url = "http://" + url
    try:
        path = urllib.parse.urlsplit(url).path
    except ValueError:
        return url                      # reported by _url_problem
    for tail in ("/chat/completions", "/completions", "/models"):
        if path.endswith(tail) and url.endswith(tail):
            url, path = url[:-len(tail)], path[:-len(tail)]
            break
    if not path:
        url += "/v1"
    return url


def _url_problem(url: str) -> str:
    """Why `url` can't be used as a server address ('' when it can)."""
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError as err:
        if "port" in str(err).lower():
            return "the port must be a number from 1 to 65535"
        return "it can't be read as a web address"
    if parts.scheme not in ("http", "https"):
        return "it should start with http:// or https://"
    if not parts.hostname:
        return "it has no host name"
    if port == 0:
        return "the port must be a number from 1 to 65535"
    return ""


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


def _seconds(value) -> float | None:
    """A timeout in seconds; None (or infinity) means no limit."""
    if value is None:
        return None
    try:
        secs = float(value)
    except (TypeError, ValueError):
        raise ValueError("timeout must be a number of seconds, or None for "
                         f"no limit (got {value!r})") from None
    if math.isinf(secs) and secs > 0:
        return None
    if not secs > 0:
        raise ValueError("timeout must be more than 0 seconds, or None for "
                         f"no limit (got {value!r})")
    return secs


def _secs(value: float | None) -> str:
    return "unlimited" if value is None else f"{value:g}"


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
    if code == 404:
        if "model" in low:
            return ("That model isn't available - load it in LM Studio, or "
                    "set TELP_LLM_MODEL to one the server lists.")
        return ("Check TELP_LLM_URL: it should be the server's base "
                f"address, like {DEFAULT_URL}.")
    if code >= 500:
        return "LM Studio's Developer tab log says what went wrong."
    return ""


def _model_missing(err: LLMUnavailable) -> bool:
    """Is this HTTP error the server saying the model isn't there?"""
    low = str(err).lower()
    return (err.status in (400, 404) and "model" in low
            and any(sign in low for sign in _MISSING_MODEL))


def _is_embedding(mid: str, lmstudio: dict[str, dict]) -> bool:
    return (lmstudio.get(mid, {}).get("type") == "embeddings"
            or "embed" in mid.lower())


def _pick_model(ids: list[str], lmstudio: dict[str, dict],
                current: str | None = None) -> str | None:
    """The model to use when the owner named none, or None when the
    server has no chat model. A chat model LM Studio says is loaded comes
    first (the current one, if it still is); else the current choice, if
    still listed - when LM Studio unloads an idle model, asking for the
    same one reloads it rather than some other download - else the first
    chat model listed. Embedding models never: LM Studio lists the one it
    ships with, and it can't write answers."""
    chat = [m for m in ids if not _is_embedding(m, lmstudio)]
    loaded = [m for m in chat if lmstudio.get(m, {}).get("state") == "loaded"]
    if current in loaded:
        return current
    if loaded:
        return loaded[0]
    if current in chat:
        return current
    return chat[0] if chat else None


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
                 api_key: str | None = None, timeout: float | None = 600,
                 temperature: float = 0.3):
        self.base_url = _normalise_url(
            base_url or os.environ.get("TELP_LLM_URL") or DEFAULT_URL)
        # a bad address is reported by every request (and by status()),
        # not raised here, so a typo never crashes Telp on start-up
        self._url_problem = _url_problem(self.base_url)
        self._wire = self.base_url if self._url_problem else \
            _dial_url(self.base_url)
        self._model = model or os.environ.get("TELP_LLM_MODEL") or None
        self._model_auto = self._model is None   # picked here, not by owner
        self._model_checked = 0.0
        self.api_key = api_key or os.environ.get("TELP_LLM_API_KEY") or None
        self.timeout = _seconds(timeout)
        self.connect_timeout = _CONNECT_TIMEOUT
        self.temperature = temperature
        self.no_think = os.environ.get("TELP_LLM_NO_THINK", DEFAULT_NO_THINK)
        # learnt from the replies: the chat template writes <think> into
        # the prompt, so a reply starts inside thinking
        self.template_opens_think = False
        self._context: int | None = None
        self._context_checked = False
        host = "" if self._url_problem else \
            urllib.parse.urlsplit(self._wire).hostname or ""
        # local servers must never be reached through a proxy
        direct = [urllib.request.ProxyHandler({})] if _is_local(host) else []
        self._opener = urllib.request.build_opener(*direct, *_handlers())

    # ── models ──

    @property
    def model(self) -> str:
        """The model id sent with every request (detected on first use)."""
        if not self._model:
            self._detect_model()
        return self._model

    @model.setter
    def model(self, value: str | None) -> None:
        self._model = value or None
        self._model_auto = self._model is None
        self._forget_model()

    def list_models(self) -> list[str]:
        """Model ids from GET /models."""
        return [e["id"] for e in self._model_entries()]

    def _model_entries(self) -> list[dict]:
        body = self._json_call(self._wire + "/models")
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
            body = self._json_call(root + "/api/v0/models")
        except LLMUnavailable:
            return {}
        data = body.get("data")
        if not isinstance(data, list):
            return {}
        return {e["id"]: e for e in data
                if isinstance(e, dict) and e.get("id")}

    def _detect_model(self) -> str:
        """Pick the model from the server's list (see _pick_model)."""
        ids = self.list_models()
        if not ids:
            raise LLMUnavailable(self._no_model_message())
        choice = _pick_model(ids, self._lmstudio_models(), self._model)
        if not choice:
            raise LLMUnavailable(self._only_embeddings_message(ids))
        self._use_detected(choice)
        return choice

    def _use_detected(self, choice: str) -> None:
        if choice != self._model:
            self._forget_model()
        self._model = choice
        self._model_checked = time.monotonic()

    def _forget_model(self) -> None:
        """What was learnt about one model doesn't carry over to another."""
        self._context, self._context_checked = None, False
        self.template_opens_think = False

    def _model_for_request(self) -> str:
        """The model to send. One the client picked itself is checked
        again when it is more than a few seconds old, so a model the
        owner loads in LM Studio mid-session is followed (with
        just-in-time loading on, asking for the old one would make LM
        Studio load it back - minutes on a small GPU). A check that fails
        keeps the old choice: the request itself will report real trouble."""
        if self._model_auto and self._model and \
                time.monotonic() - self._model_checked > _RECHECK_SECONDS:
            with contextlib.suppress(LLMUnavailable):
                self._detect_model()
        return self.model

    def _no_model_message(self) -> str:
        return (f"The model server at {self.base_url} is running but has no "
                "model to use. Load one (e.g. Qwen) in LM Studio, or set "
                "TELP_LLM_MODEL.")

    def _only_embeddings_message(self, ids: list[str]) -> str:
        return (f"The model server at {self.base_url} only has embedding "
                f"models ({', '.join(ids)}), which can't write answers. Load "
                "a chat model (e.g. Qwen) in LM Studio, or set "
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
            # an HTTP error still means a server answered: it is up, but
            # wants a key, or the address points at the wrong place
            info["reachable"] = err.status is not None
            info["error"] = str(err)
            warn(str(err))
            return info
        info["reachable"] = True
        ids = [e["id"] for e in entries]
        info["models"] = ids
        lmstudio = self._lmstudio_models()
        info["server"] = "LM Studio" if lmstudio else "OpenAI-compatible"
        if self._model_auto:
            # look again rather than trust an old pick: the owner may have
            # loaded another model since
            choice = _pick_model(ids, lmstudio, self._model) if ids else None
            if choice:
                self._use_detected(choice)
            info["model"] = choice
        self._context, self._context_checked = None, True
        if not ids:
            warn(self._no_model_message())
            return info
        if not info["model"]:
            warn(self._only_embeddings_message(ids))
            return info
        model = info["model"] = self._model
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
            "model": self._model_for_request(),
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
        send = functools.partial(self._send, payload, stream, on_token,
                                 on_thinking, bool(tools))
        try:
            reply, emitted = send()
        except LLMUnavailable as err:
            # the model the client picked itself has gone (the owner
            # loaded another): pick again, and ask once more if it changed
            if not (self._model_auto and _model_missing(err)):
                raise
            if self._detect_model() == payload["model"]:
                raise
            payload["model"] = self._model
            reply, emitted = send()
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

    def _send(self, payload: dict, stream: bool, on_token, on_thinking,
              tools_offered: bool) -> tuple[ChatReply, bool]:
        """One request; (reply, whether on_token already saw it)."""
        if stream:
            return self._chat_stream(payload, on_token, on_thinking,
                                      tools_offered)
        body = self._json_call(self._wire + "/chat/completions", payload)
        return self._reply_from_body(body, tools_offered), False

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
        reasoning = reasoning if isinstance(reasoning, str) else ""
        finish = choice.get("finish_reason") or ""
        splitter = _ThinkSplitter(self.template_opens_think,
                                  separate=bool(reasoning.strip()))
        splitter.feed(_text_of(msg.get("content")))
        splitter.flush(cut=finish == "length")
        self._learn(splitter)
        text, inline, cut_off = splitter.result()
        return _finish_reply(text, _thoughts(reasoning, inline), cut_off,
                             msg.get("tool_calls"), finish, _usage(body),
                             tools_offered)

    def _learn(self, splitter: _ThinkSplitter) -> None:
        """Remember whether this model's replies start inside thinking."""
        if splitter.found_template:
            self.template_opens_think = True
        elif splitter.opened_itself:
            self.template_opens_think = False

    def _chat_stream(self, payload: dict, on_token, on_thinking,
                     tools_offered: bool) -> tuple[ChatReply, bool]:
        """Stream one reply; (reply, whether on_token already saw it).
        A connection that drops before any token arrives is retried once;
        after that a retry would repeat output, so it is reported instead,
        with what had arrived as err.partial. An error the server reports
        is never retried."""
        self._check_url()
        data = json.dumps(payload).encode("utf-8")
        where = "/chat/completions"
        for attempt in (1, 2):
            state = _StreamState(on_token, on_thinking,
                                 self.template_opens_think)
            req = self._request(self._wire + where, data, stream=True,
                                patient=True)
            try:
                with self._errors(where, lambda: self._stall_message(state)):
                    with self._opener.open(req, timeout=self.timeout) as resp:
                        if resp.headers.get_content_type() != \
                                "text/event-stream":
                            # the server ignored "stream": a plain reply
                            body = self._decode(resp.read(), where)
                            return (self._reply_from_body(body,
                                                          tools_offered),
                                    False)
                        self._read_stream(resp, state)
                if not state.complete:
                    raise _Dropped("the stream ended without finishing")
                return self._stream_reply(state, tools_offered), True
            except _Dropped as drop:
                if attempt == 1 and not state.received:
                    continue
                partial = (self._stream_reply(state, tools_offered, True)
                           if state.received else None)
                raise self._dropped_error(drop, attempt, partial) from None
            except LLMUnavailable as err:
                if state.received and err.partial is None:
                    err.partial = self._stream_reply(state, tools_offered,
                                                     True)
                raise
            except _CallerError as err:
                raise err.original from None
        raise AssertionError("unreachable")     # the loop always returns

    def _read_stream(self, resp, state: _StreamState) -> None:
        """Feed the events to `state` until [DONE]. Once the finish_reason
        has arrived the answer is complete: the wait for the usage and
        [DONE] that follow it is short, and a stall or a drop then loses
        nothing (some servers never send [DONE] and keep the line open)."""
        sock = getattr(resp, "telp_sock", None)
        try:
            for kind, data in _sse_events(resp):
                if self._take_event(state, kind, data):
                    return
                if state.finish and not state.winding_down:
                    state.winding_down = True
                    if sock is not None:
                        sock.settimeout(_TAIL_TIMEOUT if self.timeout is None
                                        else min(self.timeout, _TAIL_TIMEOUT))
        except (OSError, http.client.HTTPException):
            if not state.finish:
                raise

    def _take_event(self, state: _StreamState, kind: str, data: str) -> bool:
        """Feed one SSE event to the stream state; True at [DONE]."""
        if kind == "error":
            raise self._stream_error(data)
        if data.strip() == "[DONE]":
            state.done = True
            return True
        try:
            chunk = json.loads(data)
        except ValueError:
            state.bad_chunks += 1
            return False
        if not isinstance(chunk, dict):
            return False
        if chunk.get("error"):
            raise self._stream_error(data)
        state.add(chunk)
        return False

    def _stream_error(self, data: str) -> LLMUnavailable:
        """The server stopped the stream with an error of its own."""
        detail = _error_text(data) if data.strip() else "no details given"
        hint = _http_hint(200, data)
        return LLMUnavailable(
            f"The model server at {self.base_url} stopped with an error: "
            f"{detail}" + (f" {hint}" if hint else ""))

    def _stream_reply(self, state: _StreamState, tools_offered: bool,
                      partial: bool = False) -> ChatReply:
        """The reply a stream added up to. partial: kept after an error,
        so it counts as cut off and nothing more is handed out."""
        state.close(cut=partial or state.finish == "length",
                    emit=not partial)
        if not partial:
            self._learn(state.splitter)
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
        if not partial and state.done and not state.finish:
            notes.append("The server ended the stream without saying the "
                         "answer was finished, so it may be incomplete.")
        text, inline, cut_off = state.splitter.result()
        return _finish_reply(text, _thoughts("".join(state.reasoning), inline),
                             cut_off, state.raw_calls(), state.finish, usage,
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

    def _request(self, url: str, data: bytes | None = None,
                 stream: bool = False,
                 patient: bool = False) -> urllib.request.Request:
        """A request, with the connect timeout and (patient=True, for a
        chat request) the wait for a model that writes before it speaks."""
        req = urllib.request.Request(
            url, data=data, method="GET" if data is None else "POST",
            headers=self._headers(json_body=data is not None, stream=stream))
        req.telp_connect = (self.connect_timeout if self.timeout is None
                            else min(self.timeout, self.connect_timeout))
        req.telp_waiter = self._await_reply if patient else None
        return req

    def _check_url(self) -> None:
        if self._url_problem:
            raise LLMUnavailable(
                f"The model server address {self.base_url} isn't valid: "
                f"{self._url_problem}. Check TELP_LLM_URL (LM Studio's "
                f"usual address is {DEFAULT_URL}).")

    def _probe_timeout(self) -> float:
        return _PROBE_TIMEOUT if self.timeout is None else \
            min(self.timeout, _PROBE_TIMEOUT)

    def _json_call(self, url: str, payload: dict | None = None) -> dict:
        """GET (or POST, with a payload) and parse the JSON reply. A
        dropped connection is retried once. A GET only lists models, so it
        gets a short timeout; a POST is a chat request, which waits for
        the model as long as the server stays alive."""
        self._check_url()
        posting = payload is not None
        timeout = self.timeout if posting else self._probe_timeout()
        data = json.dumps(payload).encode() if posting else None
        where = urllib.parse.urlsplit(url).path
        if posting:
            def on_timeout() -> str:
                return (f"The model server at {self.base_url} stopped "
                        "part-way through sending its answer (nothing for "
                        f"{_secs(timeout)} seconds).")
        else:
            def on_timeout() -> str:
                return (f"The model server at {self.base_url} didn't answer "
                        f"within {_secs(timeout)} seconds.")
        for attempt in (1, 2):
            req = self._request(url, data, patient=posting)
            try:
                with self._errors(where, on_timeout):
                    with self._opener.open(req, timeout=timeout) as resp:
                        raw = resp.read()
            except _Dropped as drop:
                if attempt == 1:
                    continue
                raise self._dropped_error(drop, attempt) from None
            return self._decode(raw, where)
        raise AssertionError("unreachable")     # the loop always returns

    def _await_reply(self, sock) -> None:
        """Wait for the server to start its reply to a chat request.

        A model that writes its whole answer before sending anything keeps
        the line silent for as long as it thinks and writes, so silence
        alone proves nothing. After each `timeout` seconds of it, check
        that the server itself still answers; keep waiting while it does,
        and give up only when it doesn't."""
        if self.timeout is None:
            return                              # no limit: just read
        with selectors.DefaultSelector() as watch:
            watch.register(sock, selectors.EVENT_READ)
            while not watch.select(self.timeout):
                if not self._still_there():
                    raise LLMUnavailable(
                        f"The model server at {self.base_url} has sent "
                        f"nothing for {_secs(self.timeout)} seconds and "
                        "didn't answer a quick check either, so it seems "
                        "stuck or gone. Check LM Studio - its Developer tab "
                        "log shows what it is doing.")

    def _still_there(self) -> bool:
        """Does the server still answer at all (any HTTP reply counts)?"""
        req = self._request(self._wire + "/models")
        try:
            with self._opener.open(req, timeout=self._probe_timeout()) as r:
                r.read()
        except urllib.error.HTTPError:
            return True                         # an answer, if an error one
        except (OSError, http.client.HTTPException, ValueError):
            return False
        return True

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
    def _errors(self, where: str, on_timeout: Callable[[], str]):
        """Turn network trouble into LLMUnavailable (or _Dropped).
        on_timeout() words a read timeout for the request at hand."""
        try:
            yield
        except urllib.error.HTTPError as err:
            raise self._http_error(err, where) from None
        except urllib.error.URLError as err:
            raise self._network_error(err.reason, on_timeout) from None
        except (OSError, http.client.HTTPException) as err:
            raise self._network_error(err, on_timeout) from None

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

    def _network_error(self, reason,
                       on_timeout: Callable[[], str]) -> Exception:
        url = self.base_url
        if isinstance(reason, _ConnectTimeout):
            return LLMUnavailable(
                f"Couldn't connect to the model server at {url} within "
                f"{reason.seconds:g} seconds - is that PC on, and LM "
                "Studio's server started? (Check TELP_LLM_URL too.)")
        if isinstance(reason, TimeoutError) or (
                isinstance(reason, str) and "timed out" in reason):
            return LLMUnavailable(on_timeout())
        if isinstance(reason, ConnectionRefusedError):
            return LLMUnavailable(
                f"LM Studio's local server isn't running at {url}. "
                "In LM Studio open the Developer tab and start the server.")
        if isinstance(reason, socket.gaierror):
            return LLMUnavailable(
                f"Can't find the host in {url} - check TELP_LLM_URL.")
        if isinstance(reason, (ConnectionResetError, ConnectionAbortedError,
                               BrokenPipeError, http.client.IncompleteRead)):
            # (RemoteDisconnected is a ConnectionResetError too)
            return _Dropped(_snippet(str(reason) or type(reason).__name__,
                                     120))
        if isinstance(reason, (http.client.BadStatusLine,
                               http.client.UnknownProtocol,
                               http.client.LineTooLong)):
            return LLMUnavailable(
                f"Something answered at {url}, but not in HTTP "
                f"({_snippet(reason, 60)}) - is that the right port? Check "
                f"TELP_LLM_URL (LM Studio's usual address is {DEFAULT_URL}).")
        if isinstance(reason, http.client.InvalidURL):
            return LLMUnavailable(
                f"The model server address {url} isn't valid "
                f"({_snippet(reason, 80)}). Check TELP_LLM_URL.")
        if isinstance(reason, http.client.HTTPException):
            return _Dropped(_snippet(str(reason) or type(reason).__name__,
                                     120))
        if _is_tls_error(reason):
            return LLMUnavailable(
                f"Couldn't make a secure (https) connection to {url} "
                f"({_snippet(reason, 80)}). LM Studio's server speaks plain "
                "http:// - check TELP_LLM_URL.")
        if isinstance(reason, OSError) and reason.errno in _UNREACHABLE:
            return LLMUnavailable(
                f"Couldn't reach the model server at {url}: no route to "
                "that machine - is it on, and on this network? (Check "
                "TELP_LLM_URL too.)")
        return LLMUnavailable(
            f"Couldn't reach the model server at {url}: "
            f"{_snippet(reason, 120)}")

    def _stall_message(self, state: _StreamState) -> str:
        """A streamed reply went quiet for `timeout` seconds."""
        if state.received:
            return (f"The model server at {self.base_url} went quiet for "
                    f"{_secs(self.timeout)} seconds part-way through the "
                    "answer. Check that LM Studio is still running - its "
                    "Developer tab log shows what it is doing.")
        return (f"The model server at {self.base_url} started its reply but "
                f"sent nothing for {_secs(self.timeout)} seconds: the model "
                "may still be reading the prompt, which a big model that "
                "runs partly on the CPU does slowly. Send a smaller brief, "
                "or allow a longer timeout.")

    def _dropped_error(self, drop: _Dropped, attempt: int,
                       partial: LLMReply | None = None) -> LLMUnavailable:
        again = " (and again on a retry)" if attempt > 1 else ""
        when = "part-way through the answer" if partial else \
            "before it answered"
        return LLMUnavailable(
            f"The connection to the model server at {self.base_url} dropped "
            f"{when}{again}: {drop}. Check that LM Studio is still running "
            "and the model is still loaded.", partial=partial)


# errno values meaning "no route to that machine" (Windows has its own)
_UNREACHABLE = {code for code in (
    getattr(errno, "EHOSTUNREACH", None), getattr(errno, "ENETUNREACH", None),
    getattr(errno, "WSAEHOSTUNREACH", None),
    getattr(errno, "WSAENETUNREACH", None)) if code is not None}


def _is_tls_error(reason) -> bool:
    try:
        import ssl
    except ImportError:                     # Python built without ssl
        return False
    return isinstance(reason, ssl.SSLError)


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
