"""
mind/harness.py - Telp in front of a local language model.

In harness mode Telp is the memory, the filter and the worker, and a local
language model (Qwen in LM Studio, on the owner's PC) only WRITES. On that
PC the model runs partly on the CPU, so READING a long prompt is the slow
part; Telp therefore sends a small, focused, cited brief instead of whole
documents and the whole chat. One question goes like this:

  1. Commands Telp handles himself, with no model call: teaching
     ("remember that ..."), statements about the user ("my name is ..."),
     forgetting ("forget ...") and "how do you know that?". They go
     through FluentTelp's own routes, so they behave as in chat mode.
  2. Worker pre-pass: deterministic tools run first when the question
     plainly needs them (arithmetic, today's date and time), and their
     results become numbered sources.
  3. The brief (mind/brief.py): fixed instructions, what Telp knows about
     the user, a constant-size summary of the conversation (follow-ups
     like "when was he born?" are searched with the names just
     discussed), and numbered sources for this question.
  4. The model writes. It may call Telp's tools to dig deeper
     (search_memory, calculate, today, and get_facts when the fact layer
     works); each result comes back as NEW numbered sources - numbers
     keep increasing - for up to max_tool_rounds rounds.
  5. Telp checks every sentence against the sources he supplied
     (mind/checker.py), marks what isn't backed, and records the turn in
     the conversation memory - never in his memory of facts: the model's
     prose is not something Telp knows.

If the model server can't be reached, the user is told how to start it and
gets Telp's own answer (no model) instead, marked as such.

    harness = Harness(FluentTelp(), LLMClient())
    turn = harness.ask("who was Galileo?")
    print(turn.answer)
    print(meter_line(turn))   # tokens sent vs a send-everything prompt
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mind.brief import BriefBuilder, ConversationState  # noqa: E402
from mind.harness_types import (Evidence, LLMReply, SentenceCheck,  # noqa: E402
                                ToolCall, TurnResult, estimate_tokens)
from mind.llm_client import (LLMUnavailable, assistant_message,  # noqa: E402
                             tool_message)


# ─── limits ─────────────────────────────────────────────────────────
#
# Every tool result is read by the model on the next round, so it is kept
# small: a few new sources, each a sentence or two.

SEARCH_LIMIT = 4          # new sources one search_memory call may add
FACTS_LIMIT = 8           # fact lines one get_facts call may add
TOOL_ITEM_CHARS = 320     # one source in a tool result, at most
PREPASS_CALCS = 3         # inline sums worked out before the model reads
CITE_CHARS = 200          # a source's text when Telp cites it himself

START_HINT = ("To use the model: open LM Studio, load a model (for example "
              "Qwen), and start its local server in the Developer tab (or "
              "run `lms server start`). `python telp.py llm-status` checks "
              "the connection.")
START_TAIL = ("(Or run `lms server start`.) `python telp.py llm-status` "
              "checks the connection.")


# ─── the tools the model may call ───────────────────────────────────
#
# Defined once, in a fixed order: chat templates (Qwen's among them) write
# the tool list into the system prompt, so it has to be byte-identical on
# every turn for LM Studio to reuse its cached reading of that prefix.

def _tool(name: str, description: str, params: dict | None = None,
          required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": params or {},
                       "required": required or []}}}


TOOL_SEARCH = _tool(
    "search_memory",
    "Search Telp's memory for more sources when the numbered sources "
    "don't cover the question. Returns new numbered sources to cite.",
    {"query": {"type": "string",
               "description": "What to look for, in a few words; name the "
                              "subject (not 'he' or 'it')."}},
    ["query"])
TOOL_CALCULATE = _tool(
    "calculate",
    "Exact arithmetic, e.g. '17 * 23', '(3 + 4) ** 2' or 'sqrt(2)'. "
    "Returns the result as a numbered source.",
    {"expression": {"type": "string",
                    "description": "Numbers with + - * / ** ( ) % sqrt()."}},
    ["expression"])
TOOL_TODAY = _tool(
    "today",
    "Today's date and the current local time, as a numbered source.")
TOOL_FACTS = _tool(
    "get_facts",
    "Telp's stored facts about one person, place or thing, as numbered "
    "sources.",
    {"entity": {"type": "string",
                "description": "The name, e.g. 'Galileo Galilei'."}},
    ["entity"])
BASE_TOOLS = (TOOL_SEARCH, TOOL_CALCULATE, TOOL_TODAY)


# ─── recognising commands and obvious work ──────────────────────────

# FluentTelp's own "remember that <world fact>" test; a first-person
# "remember that I ..." is a fact about the user instead. A question
# ("remember when Galileo was born?") is never taught.
_TEACH_RX = re.compile(r"^\s*remember\s+(?:that\s+)?(.+?)[\s.!]*$", re.I)
_FIRST_PERSON_RX = re.compile(r"\b(my|i|me|mine|we|our)\b", re.I)
# "forget ..." as a whole word: "forgetting curve" is a question
_FORGET_RX = re.compile(r"^\s*forget\b", re.I)
_FORGET_CHAT_RX = re.compile(
    r"^\s*forget\s+(?:all\s+of\s+|everything\s+in\s+)?(?:this|our|the)\s+"
    r"(?:whole\s+)?(?:conversation|chat)[\s.!]*$", re.I)
_FORGET_LEAD_RX = re.compile(
    r"^(?:about|that|the\s+fact\s+that|what\s+i\s+(?:said|told\s+you)\s+"
    r"about|everything\s+about|all\s+about|the)\s+", re.I)
_VAGUE = frozenset({"it", "that", "this", "them", "those", "these",
                    "everything", "all", "something", "stuff", "things"})

# questions that need today's date or the time of day
_TIME_RX = re.compile(
    r"\b(?:today|tonight|tomorrow|yesterday|date|what\s+time|time\s+is\s+it|"
    r"current\s+time|what\s+day|day\s+is\s+it|this\s+(?:week|month|year)|"
    r"current\s+(?:year|month|date)|what\s+year|ago|until|"
    r"how\s+many\s+(?:days|weeks|months|years)|how\s+old|"
    r"right\s+now)\b", re.I)

# an inline sum: "17*23", "(3 + 4) ** 2", "1642 - 1564". + * × ^ are
# arithmetic on their own; "/" and "-" also write ratings, ranges, dates and
# names ("24/7", "1564-1642", "COVID-19"), so they count only with spaces
# around them AND when the question asks for a calculation. ("15% of 80"
# and "3x4" are left to try_arithmetic on the whole message.)
_NUM = r"\d+(?:\.\d+)?"
_MATH_SPAN_RX = re.compile(
    rf"(?<![\w.,])\(*\s*{_NUM}\s*\)*"
    rf"(?:\s*(?:\*\*|[-+*/×^])\s*\(*\s*{_NUM}\s*\)*)+(?![\w.,])")
_STRONG_OP_RX = re.compile(r"[+*×^]")
_SPACED_WEAK_RX = re.compile(r"[\d)]\s+[-/]\s+[\d(]")
_MATH_CUE_RX = re.compile(
    r"\b(?:calculate|compute|evaluate|work\s+out|how\s+much|what(?:'s|\s+is)|"
    r"equals?|plus|minus|times|divided|multiplied|sum|product)\b", re.I)

_CITE_RX = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_THINK_RX = re.compile(r"<think>.*?(?:</think>|$)", re.S | re.I)
_TOOL_TEXT_RX = re.compile(r"<tool_call>.*?(?:</tool_call>|$)", re.S)
_SENTENCE_END_RX = re.compile(r"(?<=[.!?])\s+")

_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
         "Sunday")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")


# ─── what a turn returns ────────────────────────────────────────────

@dataclass
class HarnessTurn(TurnResult):
    """A TurnResult plus what the harness noticed along the way. Code
    written against TurnResult keeps working; these fields are extra."""
    notes: list[str] = field(default_factory=list)   # plain-English
    fallback: bool = False      # the model failed; Telp answered alone
    measured: bool = False      # brief_tokens is the server's own count
    raw_answer: str = ""        # the model's text before Telp's marks


@dataclass
class _Exchange:
    """One question's conversation with the model, as it grows."""
    messages: list[dict]
    evidence: list[Evidence]
    tools_used: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    rounds: int = 0
    tools_tokens: int = 0       # tool definitions in the last request
    prompt_estimate: int = 0    # tokens of the last request, estimated
    prompt_measured: int = 0    # ... as the server counted them, if it did


# ─── the harness ────────────────────────────────────────────────────

class Harness:
    """Telp as the memory, filter and worker in front of a model.

        harness = Harness(telp, LLMClient(), budget_tokens=1800)
        turn = harness.ask("what is the capital of Iceland?")

    telp is a FluentTelp; client an LLMClient (anything with the same
    chat() works). thinking=False asks the model not to think (faster on
    a slow machine); None leaves it to the model. fact_mind is the
    optional fact layer: "auto" uses it only when it has already been
    built in this memory (or TELP_HARNESS_FACTS=1), None turns it off, or
    pass a FactMind. checker defaults to mind/checker.py; now() to the
    local clock (tests pass fixed ones)."""

    def __init__(self, telp, client, budget_tokens: int = 1800,
                 max_tool_rounds: int = 3, check: bool = True,
                 session: str = "default", thinking: bool | None = None,
                 *, fact_mind="auto", checker=None,
                 now: Callable[[], datetime] | None = None):
        self.telp = telp
        self.client = client
        self.budget_tokens = int(budget_tokens)
        self.max_tool_rounds = max(0, int(max_tool_rounds))
        self.check = bool(check)
        self.session = session
        self.thinking = thinking
        self._now = now or datetime.now
        agent = telp.agent
        self.fact_mind = _load_fact_mind(agent, fact_mind)
        self.builder = BriefBuilder(agent, user_facts=telp.user_facts,
                                    fact_mind=self.fact_mind,
                                    budget_tokens=self.budget_tokens)
        # the conversation lives in the memory file (harness_turns), so it
        # survives restarts; pass the memory's own connection (see brief.py)
        self.state = ConversationState(agent.lattice._con, session=session,
                                       encoder=agent.encoder)
        self._checker = checker         # None: load mind/checker on first use
        self._tools_ok = True           # False once the server refuses tools
        self.last: HarnessTurn | None = None
        self._last_model_turn: tuple[str, HarnessTurn] | None = None
        self.totals = {"turns": 0, "model_turns": 0, "calls": 0,
                       "sent": 0, "naive": 0}

    # ── one question ───────────────────────────────────────────────
    def ask(self, question: str, stream: bool = False, on_token=None,
            on_thinking=None) -> HarnessTurn:
        """Answer one message. stream=True passes the model's answer to
        on_token as it is written (on_thinking gets its thinking), and an
        answer Telp gives himself in one piece; without streaming,
        on_token, if given, gets the finished answer once."""
        started = time.monotonic()
        question = (question or "").strip()
        streamed: list[str] = []

        def relay(piece: str) -> None:
            streamed.append(piece)
            on_token(piece)

        if not question:
            turn = HarnessTurn(answer="Ask me something.", handled_by="telp")
        else:
            turn = (self._command(question)
                    or self._model_turn(question, stream,
                                        relay if on_token else None,
                                        on_thinking))
        turn.seconds = round(time.monotonic() - started, 3)
        if on_token is not None and (not stream or turn.handled_by != "llm"):
            # a fallback after a partial stream starts on a fresh line
            on_token(("\n" if streamed else "") + turn.answer)
        self._count(turn)
        self.last = turn
        return turn

    def tools(self) -> list[dict] | None:
        """The tools offered to the model, or None when the server
        refused them. Always the same list while nothing changes."""
        if not self._tools_ok:
            return None
        return list(BASE_TOOLS) + ([TOOL_FACTS] if self.fact_mind else [])

    # ── 1. commands Telp handles himself ───────────────────────────
    def _command(self, question: str) -> HarnessTurn | None:
        """Teach, facts about the user, forget, provenance - in
        FluentTelp's order. None when the message is none of these."""
        for route in (self._teach, self._user_facts, self._forget,
                      self._provenance):
            turn = route(question)
            if turn is not None:
                return turn
        return None

    def _teach(self, question: str) -> HarnessTurn | None:
        m = _TEACH_RX.match(question)
        if (not m or question.rstrip().endswith("?")
                or _FIRST_PERSON_RX.search(m.group(1))):
            return None
        fact = m.group(1).strip()
        # FluentTelp's teach route comes first in respond(): it files the
        # sentence in the one memory, dated and sourced "user_taught"
        reply = self.telp.respond(question)
        evidence = []
        row = self._memory_row(fact, "user_taught")
        if row is not None:
            evidence = [Evidence(n=1, text=fact, source="user_taught",
                                 created_at=row[1], kind="memory",
                                 memory_id=row[0])]
        self._record(question, reply, evidence, {e.n for e in evidence})
        return HarnessTurn(answer=reply, evidence=evidence, handled_by="telp")

    def _user_facts(self, question: str) -> HarnessTurn | None:
        """Statements about the user, exactly as FluentTelp handles them:
        facts are captured from every message, and a message that only
        states facts is acknowledged. One that also asks something goes on
        to the model, which then sees the new fact in the standing block."""
        uf = self.telp.user_facts
        if uf is None:
            return None
        try:
            added = uf.capture(question)
        except Exception:
            return None
        if not added or question.rstrip().endswith("?") \
                or getattr(uf, "last_had_request", False):
            return None
        ack = "Got it - I'll remember that."
        olds = getattr(uf, "last_superseded", [])
        if olds:
            prev = "; ".join(o.rstrip(".") for o in olds[:2])
            ack = f"Got it - updated what I believe. (Previously: {prev}.)"
        self.telp.agent.turns.append({
            "user": question, "agent": ack, "retrieved_memories": [],
            "similarity": 1.0, "domain": "user_facts:capture"})
        texts = dict(zip(getattr(uf, "_ids", []), getattr(uf, "_texts", [])))
        evidence = [Evidence(n=i, text=texts[fid], source="user_facts",
                             created_at=self._now().isoformat(), kind="fact")
                    for i, fid in enumerate(
                        [f for f in added if f in texts], 1)]
        self._record(question, ack, evidence, {e.n for e in evidence})
        return HarnessTurn(answer=ack, evidence=evidence, handled_by="telp")

    def _forget(self, question: str) -> HarnessTurn | None:
        if not _FORGET_RX.match(question):
            return None
        if _FORGET_CHAT_RX.match(question):
            n = self.state.clear()
            self._last_model_turn = None
            return HarnessTurn(
                answer=f"Done - I've forgotten this conversation "
                       f"({n} turn{'s' if n != 1 else ''}).",
                handled_by="telp")
        try:
            reply = self.telp._forget_route(question, None)
        except Exception:
            reply = None
        # the conversation must not keep alive what was just forgotten
        phrase = _forget_phrase(question)
        dropped = 0
        if phrase:
            try:
                dropped = self.state.forget(phrase)
            except Exception:
                dropped = 0
        if reply is None:
            reply = ("Tell me what to forget - for example 'forget the "
                     "Zorblax river'.") if not dropped else "Done."
        if dropped:
            reply += (f" I've also dropped {dropped} turn"
                      f"{'s' if dropped != 1 else ''} of our conversation "
                      f"that mentioned it.")
        # deliberately not recorded: a turn about it would bring it back
        return HarnessTurn(answer=reply, handled_by="telp")

    def _provenance(self, question: str) -> HarnessTurn | None:
        try:
            asked = self.telp._is_provenance_question(question)
        except Exception:
            asked = False
        if not asked:
            return None
        answer, shown = self._provenance_answer()
        # not recorded either: "how do you know?" asked twice must cite
        # the same answer, not itself
        return HarnessTurn(answer=answer, evidence=shown, handled_by="telp")

    def _provenance_answer(self) -> tuple[str, list[Evidence]]:
        """Cite the previous turn's sources, with their dates - read from
        the conversation memory, so it works after a restart too."""
        rows = self.state.turns(last=1)
        if not rows:
            return ("You haven't asked me anything yet - ask me a question "
                    "first."), []
        refs = [r for r in rows[0]["sources"] if isinstance(r, dict)]
        items = [self._ref_evidence(r) for r in refs]
        if not items:
            return ("Nothing I have stored backs that last answer - I had no "
                    "sources for it, so treat it as unverified."), []
        cited = [e for e, r in zip(items, refs) if r.get("cited")]
        if cited:
            head, shown = "I can show you exactly - that answer came from:", \
                cited
        else:
            head = ("That last answer didn't rest on any of my sources (it "
                    "cited none), so nothing I know backs it. What I had "
                    "given the model was:")
            shown = items[:5]
        lines = [head] + ["  " + _cite_line(e) for e in shown]
        warning = self._unsupported_warning(rows[0]["question"])
        if warning:
            lines.append(warning)
        return "\n".join(lines), shown

    def _ref_evidence(self, ref: dict) -> Evidence:
        """A stored source reference back as Evidence; memory rows are
        read again (a forgotten one comes back with no text)."""
        mid = ref.get("memory_id")
        text = str(ref.get("text") or "")
        if mid is not None:
            try:
                row = self.telp.agent.lattice.get(int(mid))
            except Exception:
                row = None
            text = row["text"] if row else ""
        return Evidence(n=int(ref.get("n") or 0), text=text,
                        source=str(ref.get("source") or ""),
                        created_at=ref.get("created_at"),
                        kind=str(ref.get("kind") or "memory"),
                        memory_id=mid)

    def _unsupported_warning(self, question: str) -> str:
        last = self._last_model_turn
        if not last or last[0] != question:
            return ""
        bad = sum(1 for c in last[1].checks if c.status == "unsupported")
        if not bad:
            return ""
        return (f"Careful: {bad} sentence{'s' if bad != 1 else ''} of that "
                f"answer {'were' if bad != 1 else 'was'} not backed by these "
                f"sources - marked in the answer, and not kept in my notes "
                f"on our conversation.")

    # ── 2. the worker pre-pass ─────────────────────────────────────
    def _pre_pass(self, question: str) -> list[Evidence]:
        """Deterministic work done before the model reads anything:
        arithmetic in the question, and today's date when it matters."""
        now = self._now()
        stamp = now.isoformat(timespec="seconds")
        found = [Evidence(n=0, text=r, source="tool:calculate",
                          created_at=stamp, kind="tool")
                 for r in _arithmetic(question)]
        if _TIME_RX.search(question):
            found.append(Evidence(n=0, text=_now_line(now),
                                  source="tool:today", created_at=stamp,
                                  kind="tool"))
        return found

    # ── 3-6. a turn written by the model ───────────────────────────
    def _model_turn(self, question: str, stream: bool, on_token,
                    on_thinking) -> HarnessTurn:
        # the builder resolves "he"/"it" for its search with the names the
        # conversation was just about (or the agent's own pronoun stack)
        brief = self.builder.build(question, self.state,
                                   extra_evidence=self._pre_pass(question),
                                   today=self._now())
        ex = _Exchange(messages=brief.messages(),
                       evidence=list(brief.evidence))
        brief_estimate = _prompt_tokens(ex.messages, None)
        try:
            reply = self._converse(ex, stream, on_token, on_thinking)
        except LLMUnavailable as err:
            return self._fallback(question, str(err), ex)
        text = _clean_text(reply.text)
        if not text:
            why = " ".join(getattr(reply, "notes", []) or []) or \
                "The model returned an empty answer."
            return self._fallback(question, why, ex, reached=True)

        checks, shown = self._check(text, ex.evidence, ex.notes)
        cited = _cited(text, checks)
        self._record(question, _kept_answer(text, checks), ex.evidence,
                     cited)
        try:                          # feed chat mode's pronoun stack too
            self.telp._track_entities(question, text)
        except Exception:
            pass
        measured = ex.prompt_measured > 0
        # what tool rounds added, and the tool definitions: a send-
        # everything prompt with the same tools carries them too
        growth = max(0, _prompt_tokens(ex.messages, None) - brief_estimate)
        growth += ex.tools_tokens
        turn = HarnessTurn(
            answer=shown, evidence=ex.evidence, checks=checks,
            brief_tokens=ex.prompt_measured if measured
            else ex.prompt_estimate,
            naive_tokens=max(brief.naive_tokens + growth,
                             ex.prompt_estimate),
            rounds=ex.rounds, tools_used=ex.tools_used, usage=ex.usage,
            handled_by="llm", notes=ex.notes, measured=measured,
            raw_answer=text)
        self._last_model_turn = (question, turn)
        return turn

    def _converse(self, ex: _Exchange, stream: bool, on_token,
                  on_thinking) -> LLMReply:
        """Call the model; run the tools it asks for and call again, up to
        max_tool_rounds times. The last call offers no tools, so the model
        has to answer. (On Qwen that changes the system prompt for that one
        call, a cache miss accepted only when every tool round was used.)"""
        tool_rounds = 0
        while True:
            offer = self.tools() if tool_rounds < self.max_tool_rounds \
                else None
            reply = self._call(ex, offer, stream, on_token, on_thinking,
                               self.thinking)
            if offer is None or not reply.tool_calls \
                    or not self._tools_ok:
                break
            tool_rounds += 1
            ex.messages.append(assistant_message(reply))
            for call in reply.tool_calls:
                ex.tools_used.append(call.name)
                ex.messages.append(tool_message(
                    call, self._run_tool(call, ex.evidence)))
        if (not _clean_text(reply.text) and not reply.tool_calls
                and reply.finish_reason == "length"
                and self.thinking is None):
            # LM Studio's default 8192-token context can be spent entirely
            # on thinking: ask once more with thinking switched off
            ex.notes.append("The model ran out of room while thinking, so "
                            "Telp asked again with thinking switched off.")
            reply = self._call(ex, None, stream, on_token, on_thinking,
                               False)
        return reply

    def _call(self, ex: _Exchange, tools: list[dict] | None, stream: bool,
              on_token, on_thinking, thinking: bool | None) -> LLMReply:
        """One request. A server that refuses tools is asked again once
        without them, and isn't offered them again this session."""
        extra = {"on_thinking": on_thinking} if on_thinking else {}
        try:
            reply = self.client.chat(ex.messages, tools=tools, stream=stream,
                                     on_token=on_token if stream else None,
                                     thinking=thinking, **extra)
        except LLMUnavailable as err:
            if not tools or not _refuses_tools(err):
                raise
            self._tools_ok = False
            ex.notes.append("The model server doesn't accept tools, so the "
                            "model answered from Telp's brief alone (it "
                            "can't ask Telp to dig deeper).")
            tools = None
            reply = self.client.chat(ex.messages, tools=None, stream=stream,
                                     on_token=on_token if stream else None,
                                     thinking=thinking, **extra)
        ex.rounds += 1
        ex.tools_tokens = _tools_tokens(tools)
        ex.prompt_estimate = _prompt_tokens(ex.messages, tools)
        ex.prompt_measured = int(reply.usage.get("prompt_tokens") or 0)
        _add_usage(ex.usage, reply.usage)
        ex.notes.extend(n for n in getattr(reply, "notes", []) or []
                        if n not in ex.notes)
        return reply

    # ── the tools ──────────────────────────────────────────────────
    def _run_tool(self, call: ToolCall, evidence: list[Evidence]) -> str:
        """Run one tool call; the result is a short text for the model.
        Anything it finds is appended to the evidence, numbered on from
        the last source, so the model can cite it and Telp can check it."""
        args = call.arguments if isinstance(call.arguments, dict) else {}
        runners = {"search_memory": self._tool_search,
                   "calculate": self._tool_calculate,
                   "today": self._tool_today}
        if self.fact_mind:
            runners["get_facts"] = self._tool_facts
        run = runners.get(call.name)
        if run is None:
            return (f"Telp has no tool called '{call.name}'. Tools: "
                    + ", ".join(runners) + ".")
        try:
            return run(args, evidence)
        except Exception as err:
            return f"Telp's {call.name} tool failed ({err}); answer without it."

    def _tool_search(self, args: dict, evidence: list[Evidence]) -> str:
        query = " ".join(str(args.get("query") or args.get("q") or "").split())
        if not query:
            return "search_memory needs a query."
        found = self.builder.search(query, limit=SEARCH_LIMIT,
                                    exclude=evidence)
        new = _append(evidence, found)
        if not new:
            return f"Nothing more in Telp's memory for '{query}'."
        return "New sources:\n" + "\n".join(e.line() for e in new)

    def _tool_calculate(self, args: dict, evidence: list[Evidence]) -> str:
        from mind.code_synthesis import try_arithmetic
        expr = str(args.get("expression") or args.get("expr") or "").strip()
        result = try_arithmetic(expr) if expr else None
        if not result:
            return (f"Telp couldn't calculate '{expr}'. Use numbers with "
                    "+ - * / ** ( ) % and sqrt().")
        stamp = self._now().isoformat(timespec="seconds")
        ev = Evidence(n=0, text=result, source="tool:calculate",
                      created_at=stamp, kind="tool")
        return _append(evidence, [ev], keep_existing=True)[0].line()

    def _tool_today(self, args: dict, evidence: list[Evidence]) -> str:
        now = self._now()
        ev = Evidence(n=0, text=_now_line(now), source="tool:today",
                      created_at=now.isoformat(timespec="seconds"),
                      kind="tool")
        return _append(evidence, [ev], keep_existing=True)[0].line()

    def _tool_facts(self, args: dict, evidence: list[Evidence]) -> str:
        entity = " ".join(str(args.get("entity") or args.get("name")
                              or "").split())
        if not entity:
            return "get_facts needs an entity."
        fm = self.fact_mind
        fm.sync()
        name = fm.memory.resolve(entity)
        facts = fm.memory.facts_about(name) if name else []
        items = []
        for f in facts:
            line = _fact_line(f)
            if line:
                items.append(Evidence(n=0, text=line,
                                      source=f.source or "facts",
                                      created_at=f.created_at, kind="fact",
                                      memory_id=f.memory_id))
        new = _append(evidence, items[:FACTS_LIMIT])
        if not new:
            return f"Telp has no more stored facts about '{entity}'."
        return "New sources:\n" + "\n".join(e.line() for e in new)

    # ── 5. checking and remembering the conversation ──────────────
    def _check(self, text: str, evidence: list[Evidence],
               notes: list[str]) -> tuple[list[SentenceCheck], str]:
        """Every sentence checked against the sources (mind/checker.py);
        returns the checks and the answer as the user sees it."""
        if not self.check:
            return [], text
        checker = self._load_checker()
        if checker is None:
            notes.append("Telp's checker (mind/checker.py) isn't available, "
                         "so this answer is unchecked.")
            return [], text
        try:
            checks = list(checker.check_answer(
                text, evidence, encoder=self.telp.agent.encoder))
            shown = checker.annotate(text, checks, evidence)
        except Exception as err:
            notes.append(f"Telp couldn't check this answer ({err}), so it "
                         "is unchecked.")
            return [], text
        return checks, shown or text

    def _load_checker(self):
        if self._checker is None:
            try:
                import mind.checker as checker
                self._checker = checker
            except Exception:
                self._checker = False
        return self._checker or None

    def _record(self, question: str, answer: str, evidence: list[Evidence],
                cited: set[int]) -> None:
        """Keep the turn in the conversation memory (harness_turns), with
        its sources as references and which of them the answer cited -
        what "how do you know that?" reads next turn."""
        refs = [{"n": e.n, "kind": e.kind, "source": e.source,
                 "created_at": e.created_at, "memory_id": e.memory_id,
                 "score": round(float(e.score), 4), "text": e.text,
                 "cited": e.n in cited} for e in evidence]
        try:
            self.state.add_turn(question, answer, refs)
        except Exception:
            pass

    # ── when the model can't answer ────────────────────────────────
    def _fallback(self, question: str, problem: str, ex: _Exchange,
                  reached: bool = False) -> HarnessTurn:
        """Say what went wrong (and how to fix it), then give Telp's own
        answer from chat mode - marked, so nobody takes it for the
        model's."""
        turns = getattr(self.telp.agent, "turns", None)
        before = len(turns) if isinstance(turns, list) else 0
        try:
            own = (self.telp.respond(question) or "").strip()
        except Exception as err:
            own = f"(Telp couldn't answer on his own either: {err})"
        head = problem.strip()
        if not reached:
            head += "\n" + (START_TAIL if "start the server" in head
                            else START_HINT)
        answer = (f"{head}\n\nTelp's own answer, without the model:\n"
                  f"{own or '(nothing)'}")
        evidence = self._own_sources(before)
        self._record(question, own, evidence, {e.n for e in evidence})
        return HarnessTurn(answer=answer, evidence=evidence,
                           rounds=ex.rounds, tools_used=ex.tools_used,
                           usage=ex.usage, handled_by="telp",
                           notes=ex.notes, fallback=True)

    def _own_sources(self, before: int) -> list[Evidence]:
        """The memory rows Telp's own answer was built from: only a turn
        record his answer just added (turns[before:]) counts - an older
        one would cite the wrong answer's sources."""
        turns = getattr(self.telp.agent, "turns", None)
        new = turns[before:] if isinstance(turns, list) else []
        mems = (new[-1].get("retrieved_memories") or []) if new else []
        out: list[Evidence] = []
        for m in mems[:5]:
            text = m.get("text") if isinstance(m, dict) else m
            row = self._memory_row(str(text or ""))
            if row is not None:
                out.append(Evidence(n=len(out) + 1, text=str(text),
                                    source=row[2] or "memory",
                                    created_at=row[1], kind="memory",
                                    memory_id=row[0]))
        return out

    # ── helpers ────────────────────────────────────────────────────
    def _memory_row(self, text: str, source: str | None = None
                    ) -> tuple | None:
        """(id, created_at, source) of the newest memory row with exactly
        this text (and source, when given)."""
        if not text:
            return None
        sql = "SELECT id, created_at, source FROM memories WHERE text=?"
        args: tuple = (text,)
        if source is not None:
            sql += " AND source=?"
            args += (source,)
        try:
            return self.telp.agent.lattice._con.execute(
                sql + " ORDER BY id DESC LIMIT 1", args).fetchone()
        except sqlite3.Error:
            return None

    def _count(self, turn: HarnessTurn) -> None:
        t = self.totals
        t["turns"] += 1
        t["calls"] += turn.rounds
        if turn.handled_by == "llm":
            t["model_turns"] += 1
            t["sent"] += turn.brief_tokens
            t["naive"] += turn.naive_tokens


# ─── the optional fact layer ────────────────────────────────────────

def _load_fact_mind(agent, spec):
    """The fact layer is unfinished work: used only when it works, and by
    default only when it has already been built in this memory - building
    it reads every stored sentence, which can take a long time."""
    if spec is None or spec is False:
        return None
    if spec != "auto":
        return spec if _fact_layer_works(spec) else None
    wanted = os.environ.get("TELP_HARNESS_FACTS", "")
    if wanted == "0":
        return None
    if wanted != "1" and not _facts_built(agent):
        return None
    try:
        from mind.fact_mind import FactMind
        fm = FactMind(agent)
    except Exception:
        return None
    return fm if _fact_layer_works(fm) else None


def _facts_built(agent) -> bool:
    try:
        con = agent.lattice._con
        if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                           "AND name='facts'").fetchone():
            return False
        return con.execute("SELECT 1 FROM facts LIMIT 1").fetchone() \
            is not None
    except Exception:
        return False


def _fact_layer_works(fm) -> bool:
    try:
        fm.sync()
        fm.memory.resolve("Telp")
        return True
    except Exception:
        return False


def _fact_line(fact) -> str | None:
    """'Galileo Galilei - place of birth: Pisa' - brief.py's wording when
    it is there, a plain one otherwise."""
    try:
        from mind.brief import _fact_line as brief_line
        return brief_line(fact)
    except Exception:
        pass
    try:
        return f"{fact.subject} - {fact.relation.replace('_', ' ')}: " \
               f"{fact.obj}"
    except Exception:
        return None


# ─── small helpers ──────────────────────────────────────────────────

def _arithmetic(question: str) -> list[str]:
    """Sums in the question, worked out exactly: the whole message when it
    is a calculation ("what's 15% of 80?"), else inline expressions."""
    from mind.code_synthesis import try_arithmetic
    whole = try_arithmetic(question)
    if whole:
        return [whole]
    if re.search(r"\d,\d{3}", question):
        return []                     # "1,000 + 2,000": left to the model
    cue = bool(_MATH_CUE_RX.search(question))
    out: list[str] = []
    for m in _MATH_SPAN_RX.finditer(question):
        span = m.group(0).strip()
        if not _STRONG_OP_RX.search(span) and not (
                cue and _SPACED_WEAK_RX.search(span)):
            continue
        result = try_arithmetic(span)
        if result and result not in out:
            out.append(result)
        if len(out) >= PREPASS_CALCS:
            break
    return out


def _now_line(now: datetime) -> str:
    """'Today is Thursday, 8 October 2026; the local time is 14:05.' -
    spelled out without the locale, the same on every machine."""
    return (f"Today is {_DAYS[now.weekday()]}, {now.day} "
            f"{_MONTHS[now.month - 1]} {now.year}; the local time is "
            f"{now:%H:%M}.")


def _forget_phrase(question: str) -> str:
    """What "forget ..." names, for dropping conversation turns about it;
    "" when it is too vague to match safely ("forget it")."""
    phrase = re.sub(r"^\s*forget\b", "", question, flags=re.I)
    phrase = phrase.strip(" :,.-!?\"'")
    while True:
        shorter = _FORGET_LEAD_RX.sub("", phrase).strip()
        if shorter == phrase:
            break
        phrase = shorter
    words = [w for w in re.findall(r"[a-z0-9']+", phrase.lower())
             if w not in _VAGUE]
    if not any(len(w) >= 4 for w in words):
        return ""
    return phrase


def _clip(text: str, max_chars: int) -> str:
    """Whole sentences up to max_chars; a single long one is cut at a word
    with the cut marked."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    out = ""
    for sentence in _SENTENCE_END_RX.split(text):
        joined = f"{out} {sentence}".strip()
        if len(joined) > max_chars:
            break
        out = joined
    if out:
        return out
    return text[:max_chars - 3].rsplit(" ", 1)[0] + "..."


def _append(evidence: list[Evidence], items: list[Evidence],
            keep_existing: bool = False) -> list[Evidence]:
    """Number new items on from the last source and add them. An item
    whose text is already shown is skipped - or, with keep_existing,
    returned as the source already there."""
    out: list[Evidence] = []
    for item in items:
        text = _clip(item.text, TOOL_ITEM_CHARS)
        if not text:
            continue
        # one sentence gives several facts, so only memory sentences are
        # matched by their row id
        same = next((e for e in evidence if e.text == text
                     or (item.kind == e.kind == "memory"
                         and item.memory_id is not None
                         and e.memory_id == item.memory_id)), None)
        if same is not None:
            if keep_existing:
                out.append(same)
            continue
        new = Evidence(n=max((e.n for e in evidence), default=0) + 1,
                       text=text, source=item.source,
                       created_at=item.created_at, kind=item.kind,
                       score=item.score, memory_id=item.memory_id)
        evidence.append(new)
        out.append(new)
    return out


def _cited(text: str, checks: list[SentenceCheck]) -> set[int]:
    """Sources the answer rests on: those cited by (or found to back)
    sentences the checker accepted; without checks, every [n] cited."""
    if checks:
        nums: set[int] = set()
        for c in checks:
            if c.status == "supported":
                nums.update(int(n) for n in c.cites)
                if c.best_evidence:
                    nums.add(int(c.best_evidence))
        return nums
    return {int(n) for group in _CITE_RX.findall(text)
            for n in re.split(r"\s*[,;]\s*", group)}


def _kept_answer(text: str, checks: list[SentenceCheck]) -> str:
    """The answer as kept in the conversation memory: sentences no source
    backs are left out, so a slip can't come back next turn dressed as
    something already established."""
    for c in checks:
        if c.status == "unsupported" and c.sentence:
            text = text.replace(c.sentence, " ")
    return " ".join(text.split())


def _clean_text(text: str) -> str:
    """The answer without leftover thinking or tool-call markup."""
    text = _THINK_RX.sub("", text or "")
    text = _TOOL_TEXT_RX.sub("", text)
    return text.strip()


def _refuses_tools(err: LLMUnavailable) -> bool:
    """The server answered with an HTTP error that is about tools."""
    return getattr(err, "status", None) is not None \
        and "tool" in str(err).lower()


def _tools_tokens(tools: list[dict] | None) -> int:
    return estimate_tokens(json.dumps(tools)) if tools else 0


def _prompt_tokens(messages: list[dict], tools: list[dict] | None) -> int:
    """Estimated tokens of one request: every message, the tool calls the
    model made, and the tool definitions."""
    total = _tools_tokens(tools)
    for m in messages:
        total += estimate_tokens(str(m.get("content") or ""))
        if m.get("tool_calls"):
            total += estimate_tokens(json.dumps(m["tool_calls"]))
    return total


def _add_usage(total: dict, usage: dict) -> None:
    """Add up the numbers of several calls (timings keep the latest)."""
    for key, value in (usage or {}).items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if key.endswith("_tokens"):
            total[key] = total.get(key, 0) + value
        else:
            total[key] = value


def _cite_line(e: Evidence) -> str:
    """One source as Telp cites it: what it says, where and when from."""
    when = (e.created_at or "")[:10]
    if e.memory_id is not None and not e.text:
        return (f"[{e.n}] a memory from {e.source or 'my memory'} that has "
                f"since been forgotten")
    text = _clip(e.text, CITE_CHARS)
    if e.kind == "tool":
        where = f"worked out by Telp ({e.source}{', ' + when if when else ''})"
    else:
        where = (e.source or "my memory") + (f", saved {when}" if when
                                             else "")
    return f"[{e.n}] \"{text}\" - {where}"


# ─── what the command line prints ───────────────────────────────────

def meter_line(turn: TurnResult) -> str:
    """'Telp sent 812 tokens (a send-everything prompt: ~24,300) · 1 model
    call · 3.1s' - how much the model had to read, against what a
    send-everything chat app would have sent."""
    secs = f"{turn.seconds:.1f}s"
    if getattr(turn, "fallback", False):
        return f"The model didn't answer - Telp answered on his own · {secs}"
    if turn.handled_by != "llm":
        return f"Telp handled this himself - no model call · {secs}"
    sent = f"{turn.brief_tokens:,}" if getattr(turn, "measured", False) \
        else f"~{turn.brief_tokens:,}"
    calls = f"{turn.rounds} model call{'s' if turn.rounds != 1 else ''}"
    if turn.tools_used:
        calls += " (" + ", ".join(turn.tools_used) + ")"
    parts = [f"Telp sent {sent} tokens (a send-everything prompt: "
             f"~{turn.naive_tokens:,})", calls]
    cached = turn.usage.get("cached_tokens")
    if cached:
        parts.append(f"{cached:,} read from the model's cache")
    parts.append(secs)
    return " · ".join(parts)


def session_line(harness: Harness) -> str:
    """Running totals for the REPL's /meter."""
    t = harness.totals
    if not t["model_turns"]:
        return (f"This session: {t['turns']} turn(s), none written by the "
                "model yet.")
    saved = 1 - t["sent"] / t["naive"] if t["naive"] else 0.0
    return (f"This session: {t['model_turns']} model turn(s), "
            f"{t['calls']} model call(s); Telp sent ~{t['sent']:,} tokens in "
            f"all, send-everything prompts would have been "
            f"~{t['naive']:,} ({saved:.0%} less to read).")


def sources_text(turn: TurnResult | None) -> str:
    """The last turn's evidence, for the REPL's /sources."""
    if turn is None:
        return "No turn yet."
    if not turn.evidence:
        return "That turn had no sources."
    cited = _cited(getattr(turn, "raw_answer", "") or turn.answer,
                   turn.checks)
    lines = []
    for e in turn.evidence:
        mark = " (cited)" if e.n in cited else ""
        lines.append(f"{e.line()}  [{e.kind}]{mark}")
    return "\n".join(lines)


def check_lines(turn: TurnResult) -> list[str]:
    """What follows a streamed answer, which went out before Telp checked
    it: one line per sentence he couldn't back, then the sources the
    answer rests on (the non-streamed answer carries both already)."""
    lines = [f"Telp's check: not backed by the sources -> \"{c.sentence}\""
             for c in turn.checks if c.status == "unsupported"]
    cited = _cited(getattr(turn, "raw_answer", "") or turn.answer,
                   turn.checks)
    used = [e for e in turn.evidence if e.n in cited]
    if used:
        lines.append("Sources: " + "; ".join(
            f"[{e.n}] {e.source or e.kind}"
            + (f", {e.created_at[:10]}" if e.created_at else "")
            for e in used))
    return lines


def status_lines(info: dict, budget_tokens: int = 1800) -> list[str]:
    """LLMClient.status() as plain lines, plus what it means for Telp's
    briefs (how much context is left for thinking and the answer)."""
    url = info.get("base_url", "")
    if not info.get("reachable"):
        lines = [f"server:  {url} (not reachable)"]
        lines += [f"problem: {w}" for w in info.get("warnings", [])]
        told = any("start the server" in w for w in info.get("warnings", []))
        return lines + [START_TAIL if told else START_HINT]
    server = info.get("server") or "OpenAI-compatible"
    lines = [f"server:  {url} (reachable, {server})"]
    if info.get("models"):
        lines.append("models:  " + ", ".join(info["models"]))
    if info.get("model"):
        lines.append(f"model:   {info['model']}")
    ctx = info.get("context_length")
    if ctx:
        most = info.get("max_context_length")
        lines.append(f"context: {ctx:,} tokens loaded"
                     + (f" (the model allows up to {most:,})" if most else ""))
        # a brief, the tool definitions, and a few tool rounds of sources
        prompt = (budget_tokens + _tools_tokens(list(BASE_TOOLS))
                  + 3 * SEARCH_LIMIT * TOOL_ITEM_CHARS // 4)
        left = ctx - prompt
        lines.append(f"brief:   about {budget_tokens:,} tokens a turn (up to "
                     f"~{prompt:,} after tool rounds), leaving ~{left:,} for "
                     "the model's thinking and answer")
    else:
        lines.append("context: not reported by the server")
    lines += [f"warning: {w}" for w in info.get("warnings", [])]
    return lines
