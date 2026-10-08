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
     through FluentTelp's own routes, so they behave as in chat mode -
     with extra care where harness mode differs: a pasted document's
     "my name is ..." is somebody else's, and "Forget that, what's ...?"
     is a change of subject, not an order to delete.
  2. Worker pre-pass: deterministic tools run first when the question
     plainly needs them (arithmetic, the date and time now), and their
     results become numbered sources.
  3. The brief (mind/brief.py): fixed instructions, what Telp knows about
     the user, a constant-size summary of the conversation (follow-ups
     like "when was he born?" are searched with the names just
     discussed), and numbered sources for this question.
  4. The model writes. It may call Telp's tools to dig deeper
     (search_memory, calculate, today, and get_facts when the fact layer
     works); each result comes back as NEW numbered sources - numbers
     keep increasing - for up to max_tool_rounds rounds. Tool calls and
     tool results are capped per question, and what the model may write
     (thinking included) is capped to what the loaded context has left,
     so a prompt never crowds out the model's thinking on a small
     context.
  5. Telp checks every sentence against the sources he supplied
     (mind/checker.py), marks what isn't backed, and records the turn in
     the conversation memory - never in his memory of facts: the model's
     prose is not something Telp knows.

If the model server can't be reached, the user is told what went wrong
(and, when nothing answered at all, how to start it) and gets Telp's own
answer (no model) instead, marked as such.

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
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
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
# small: a few new sources, each a sentence or two - and the results of
# one question share a budget of their own.

SEARCH_LIMIT = 4          # new sources one search_memory call may add
FACTS_LIMIT = 8           # fact lines one get_facts call may add
TOOL_ITEM_CHARS = 320     # one source in a tool result, at most
PREPASS_CALCS = 3         # inline sums worked out before the model reads
CITE_CHARS = 200          # a source's text when Telp cites it himself
CALLS_PER_ROUND = 4       # tool calls run from one model reply
TOOL_SHARE = 0.75         # tool results for one question: this share of
                          # the brief's budget, at most
MIN_TOOL_ROOM = 40        # tokens of tool room below which none is given
REPLY_TOKENS = 4096       # what the model may write per call (thinking
                          # included) unless the caller says otherwise
MIN_REPLY_TOKENS = 256    # never ask for less than this
MIN_REPLY_RESERVE = 2048  # context always kept for thinking + the answer
PROMPT_SLACK = 1.25       # chars/4 undercounts real tokens (more so for
                          # languages other than English)
USER_FACT_WORDS = 40      # longer messages aren't read for user facts

START_HINT = ("To use the model: open LM Studio, load a model (for example "
              "Qwen), and start its local server in the Developer tab (or "
              "run `lms server start`). `python telp.py llm-status` checks "
              "the connection.")
START_TAIL = ("(Or run `lms server start`.) `python telp.py llm-status` "
              "checks the connection.")

# the bookkeeping entry stored with a model turn's sources: how many of
# its sentences the checker couldn't back (read by "how do you know?")
_CHECK_KIND = "check"


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

# "forget ..." as a whole word: "forgetting curve" and "forget-me-nots"
# are not the command
_FORGET_RX = re.compile(r"^\s*forget\b(?![-'])", re.I)
# a forget clause ends at a sentence break, a comma or a dash; the rest of
# the message (if it is a clause of its own) is something else entirely
_CLAUSE_BREAK_RX = re.compile(r"[.!?;:](?=\s)|,(?=\s)|\s[-–—]\s")
# phrases about the conversation itself, not about anything Telp knows:
# they clear the conversation, or drop its last exchange
_CHAT_ALL_RX = re.compile(
    r"^forget\s+(?:about\s+)?(?:"
    r"(?:all\s+(?:of\s+)?|everything\s+(?:in|from|about)\s+|"
    r"the\s+whole\s+of\s+)?"
    r"(?:this|our|the|that)\s+(?:whole\s+|entire\s+|last\s+|previous\s+|"
    r"earlier\s+)?(?:conversation|chat|discussion|talk)"
    r"(?:\s+we(?:'ve|\s+have)?\s+(?:just\s+)?had)?"
    r"|(?:everything|all|what|anything)\s+(?:that\s+)?"
    r"(?:we(?:'ve|\s+have)?|you\s+and\s+i(?:'ve|\s+have)?)\s+"
    r"(?:just\s+|already\s+)?(?:talked|spoke|spoken|discussed|chatted|said|"
    r"covered|been\s+(?:talking|discussing))(?:\s+about)?"
    r"(?:\s+(?:so\s+far|today|before|earlier|until\s+now))?"
    r")(?:\s+(?:please|now))?$", re.I)
_CHAT_LAST_RX = re.compile(
    r"^forget\s+(?:about\s+)?(?:"
    r"(?:that|this|your|the|my)\s+(?:last\s+|previous\s+|latest\s+)?"
    r"(?:answer|reply|response|question|message|exchange)"
    r"|(?:everything|what|all)\s+(?:that\s+)?you\s+(?:just\s+)?"
    r"(?:said|told\s+me|wrote|answered)"
    r"|what\s+i\s+just\s+(?:said|asked)"
    r")(?:\s+(?:please|now))?$", re.I)
_CHAT_TOPIC_RX = re.compile(
    r"^forget\s+(?:about\s+)?(?:"
    r"(?:the|our|that|this)\s+(?:conversation|chat|discussion|talk)\s+"
    r"(?:about|on|regarding)"
    r"|(?:what|everything)\s+(?:we|you)\s+(?:said|talked|discussed|"
    r"told\s+me)\s+about"
    r")\s+(.+)$", re.I)
_FORGET_LEAD_RX = re.compile(
    r"^(?:about|that|the\s+fact\s+that|what\s+i\s+(?:said|told\s+you)\s+"
    r"about|everything\s+about|all\s+about|the)\s+", re.I)
_VAGUE = frozenset({
    "it", "that", "this", "them", "those", "these", "everything", "all",
    "something", "anything", "whatever", "stuff", "things", "thing", "about",
    "the", "a", "an", "what", "i", "you", "me", "him", "her", "he", "she",
    "they", "there", "here", "one", "said", "told", "just", "please", "now",
    "so", "ok", "okay", "never", "mind", "thanks", "thank", "pls", "plz"})
_WORD_RX = re.compile(r"[^\W_]+(?:'[^\W_]+)?")

# a message that is (or quotes) somebody else's text: its "my name is
# ..." is not about the user. Asking to work on a text, a letter's
# greeting, or reported words ("she wrote:").
_DOC_CUE_RX = re.compile(
    r"\b(?:summari[sz]e|summary|tl;?dr|translate|translation|proof-?read|"
    r"paraphrase|rewrite|re-?word|reply\s+to|respond\s+to|"
    r"(?:this|that|the\s+following|the\s+attached|the\s+below|"
    r"(?:my\s+)?(?:friend|colleague|boss|client|teacher|mother|father|"
    r"sister|brother|son|daughter|wife|husband|partner)'?s?|his|her|their)"
    r"\s+(?:cover\s+)?(?:letter|e-?mail|message|text|document|essay|"
    r"article|post|note|paragraph|story|resume|cv|bio|profile|transcript|"
    r"review|draft)|"
    r"dear\s+\w+|(?:wrote|writes|says|said|sent\s+me|forwarded)\s*:)",
    re.I)
_QUOTED_RX = re.compile(r"[\"“”«»„]([^\"“”«»„\n]+)[\"“”«»„]")

# questions about the present moment, which need the date or the time as
# a source to cite. Not "the date of the Battle of Hastings", "until
# 1500" or "long ago": the brief states today's date anyway.
_NOW_RX = re.compile(
    r"\b(?:today|tonight|tomorrow|yesterday|right\s+now|nowadays|"
    r"what\s+(?:time|day|date|year|month)(?:\s+of\s+the\s+(?:week|month|"
    r"year))?\s+is\s+it|"
    r"(?:what(?:'s|\s+is)\s+the|the\s+current|current|local)\s+"
    r"(?:time|date)(?!\s+(?:of|for|in|on|when|that|zone|difference|"
    r"between|period))|"
    r"(?:this|current)\s+(?:week|month|year)|"
    r"how\s+old|"
    r"how\s+long\s+(?:ago|since|until|till|before|has\s+it\s+been|"
    r"is\s+it\s+(?:until|till|since))|"
    r"how\s+many\s+(?:hours|days|weeks|months|years)\s+(?:ago|since|until|"
    r"till|left|from\s+now|before|is\s+it)|"
    r"(?:days|weeks|months|years)\s+(?:ago|from\s+now))\b", re.I)

# an inline sum: "17*23", "(3 + 4) ** 2", "1642 - 1564". + * × ^ are
# arithmetic on their own; "/" and "-" also write ratings, ranges, dates and
# names ("24/7", "1564-1642", "COVID-19"), so they count only with spaces
# around them AND when the question asks for a calculation. ("15% of 80"
# and "3x4" are left to try_arithmetic on the whole message.) A span with
# anything number-like next to it is never worked out: "1 500 + 2 500"
# (digit groups), "10:30 - 11:45" (times), "John 3:16 - 3:18" (verses).
_NUM = r"\d+(?:\.\d+)?"
_MATH_SPAN_RX = re.compile(
    rf"(?<![\w.,])\(*\s*{_NUM}\s*\)*"
    rf"(?:\s*(?:\*\*|[-+*/×^])\s*\(*\s*{_NUM}\s*\)*)+(?![\w.,])")
_STRONG_OP_RX = re.compile(r"[+*×^]")
_SPACED_WEAK_RX = re.compile(r"[\d)]\s+[-/]\s+[\d(]")
_MATH_CUE_RX = re.compile(
    r"\b(?:calculate|compute|evaluate|work\s+out|how\s+much|what(?:'s|\s+is)|"
    r"equals?|plus|minus|times|divided|multiplied|sum|product)\b", re.I)
_TOUCH_BEFORE = set(":.,-+*/×^=")
_TOUCH_AFTER = set(":-+*/×^=%")

_CITE_RX = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_THINK_RX = re.compile(r"<think>.*?(?:</think>|$)", re.S | re.I)
_TOOL_TEXT_RX = re.compile(r"<tool_call>.*?(?:</tool_call>|$)", re.S)
_INLINE_CALL_RX = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)",
                             re.S)
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
    measured: bool = False      # token counts are the server's own units
    raw_answer: str = ""        # the model's text before Telp's marks


@dataclass
class _Exchange:
    """One question's conversation with the model, as it grows."""
    messages: list[dict]
    evidence: list[Evidence]
    thinking: bool | None = None    # this question's setting (a retry
                                    # may switch thinking off)
    brief_user: str = ""            # the brief's own user message
    brief_count: int = 0            # sources the brief itself gave
    tools_used: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    rounds: int = 0
    tool_room: int = 0              # tokens of tool results allowed
    tool_tokens: int = 0            # ... and given so far
    skipped: int = 0                # tool calls not run (over a limit)
    tools_done: bool = False        # no more tool rounds this question
    problem: str = ""               # why there is no answer, if known
    tools_tokens: int = 0           # tool definitions in the last request
    prompt_estimate: int = 0        # tokens of the last request, estimated
    prompt_measured: int = 0        # ... as the server counted them, if it did


# ─── the harness ────────────────────────────────────────────────────

class Harness:
    """Telp as the memory, filter and worker in front of a model.

        harness = Harness(telp, LLMClient(), budget_tokens=1800)
        turn = harness.ask("what is the capital of Iceland?")

    telp is a FluentTelp; client an LLMClient (anything with the same
    chat() works). thinking=False asks the model not to think (faster on
    a slow machine); None leaves it to the model, and asks again without
    thinking when it runs out of room or time. reply_tokens caps what the
    model may write per call, thinking included (0 or None: only the
    loaded context limits it). fact_mind is the optional fact layer:
    "auto" uses it only when it has already been built in this memory
    (or TELP_HARNESS_FACTS=1), None turns it off, or pass a FactMind.
    checker defaults to mind/checker.py; now() to the local clock (tests
    pass fixed ones)."""

    def __init__(self, telp, client, budget_tokens: int = 1800,
                 max_tool_rounds: int = 3, check: bool = True,
                 session: str = "default", thinking: bool | None = None,
                 *, fact_mind="auto", checker=None,
                 now: Callable[[], datetime] | None = None,
                 reply_tokens: int | None = REPLY_TOKENS):
        self.telp = telp
        self.client = client
        self.budget_tokens = int(budget_tokens)
        self.max_tool_rounds = max(0, int(max_tool_rounds))
        self.check = bool(check)
        self.session = session
        self.thinking = thinking
        self.reply_tokens = int(reply_tokens) if reply_tokens else None
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
        self._probed = False            # asked the server for its context
        self._context: int | None = None    # the model's loaded context
        self._ratio = 0.0               # server tokens per estimated token
        self.last: HarnessTurn | None = None
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
        notes: list[str] = []

        def relay(piece: str) -> None:
            streamed.append(piece)
            on_token(piece)

        if not question:
            turn = HarnessTurn(answer="Ask me something.", handled_by="telp")
        else:
            turn = (self._command(question, notes)
                    or self._model_turn(question, stream,
                                        relay if on_token else None,
                                        on_thinking))
        turn.notes = notes + [n for n in turn.notes if n not in notes]
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
    def _command(self, question: str,
                 notes: list[str]) -> HarnessTurn | None:
        """Teach, facts about the user, forget, provenance - in
        FluentTelp's order. None when the message is none of these (a
        route may still leave a note for the model's turn)."""
        for route in (self._teach, self._user_facts, self._forget,
                      self._provenance):
            turn = route(question, notes)
            if turn is not None:
                return turn
        return None

    def _teach(self, question: str, notes: list[str]) -> HarnessTurn | None:
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

    def _user_facts(self, question: str,
                    notes: list[str]) -> HarnessTurn | None:
        """Statements about the user, as FluentTelp handles them: a
        message that only states facts is acknowledged; one that also
        asks something goes on to the model, which then sees the new fact
        in the standing block - with a note, so a change to what Telp
        believes about the user is never silent. A pasted document or a
        quoted passage is not read at all: its "my name is ..." is
        somebody else's."""
        uf = self.telp.user_facts
        if uf is None or _not_about_the_user(question):
            return None
        try:
            added = uf.capture(question)
        except Exception:
            return None
        if not added:
            return None
        texts = dict(zip(getattr(uf, "_ids", []), getattr(uf, "_texts", [])))
        olds = list(getattr(uf, "last_superseded", []) or [])
        if question.rstrip().endswith("?") \
                or getattr(uf, "last_had_request", False):
            notes.append(_noted_line([texts[f] for f in added if f in texts],
                                     olds))
            return None
        ack = "Got it - I'll remember that."
        if olds:
            prev = "; ".join(o.rstrip(".") for o in olds[:2])
            ack = f"Got it - updated what I believe. (Previously: {prev}.)"
        self.telp.agent.turns.append({
            "user": question, "agent": ack, "retrieved_memories": [],
            "similarity": 1.0, "domain": "user_facts:capture"})
        evidence = [Evidence(n=i, text=texts[fid], source="user_facts",
                             created_at=self._now().isoformat(), kind="fact")
                    for i, fid in enumerate(
                        [f for f in added if f in texts], 1)]
        self._record(question, ack, evidence, {e.n for e in evidence})
        return HarnessTurn(answer=ack, evidence=evidence, handled_by="telp")

    def _forget(self, question: str, notes: list[str]) -> HarnessTurn | None:
        """'forget ...' - deleting is permanent, so only a message that is
        nothing but a forget command deletes anything. "Forget that,
        what's the capital of Iceland?" changes the subject and goes to
        the model; "forget everything we talked about" is about the
        conversation, not about Telp's memory."""
        kind, clause, target = _forget_intent(question)
        if not kind:
            return None
        if kind == "ask":
            if target:
                notes.append(f"Telp didn't forget anything: to make him "
                             f"forget something, send '{clause}' on its "
                             f"own.")
            return None
        if kind == "chat_all":
            n = self.state.clear()
            return HarnessTurn(
                answer=f"Done - I've forgotten this conversation "
                       f"({n} turn{'s' if n != 1 else ''}).",
                handled_by="telp")
        if kind == "chat_last":
            rows = self.state.turns(last=1)
            if not rows:
                answer = "There's nothing in our conversation to forget yet."
            else:
                self._drop_turns([rows[0]["id"]])
                answer = ("Done - I've dropped our last exchange from the "
                          "conversation.")
            return HarnessTurn(answer=answer, handled_by="telp")
        if kind == "chat_topic":
            return HarnessTurn(answer=self._forget_topic(target),
                               handled_by="telp")
        if kind == "vague":
            return HarnessTurn(
                answer=("Nothing forgotten - tell me what to forget, for "
                        "example 'forget the Zorblax river', or say 'forget "
                        "this conversation' to clear our chat."),
                handled_by="telp")
        return self._forget_memory(question)

    def _forget_memory(self, question: str) -> HarnessTurn:
        """FluentTelp's own forget route (it deletes only on a strong
        match), then the conversation turns that rested on exactly what
        it deleted - so the conversation can't bring it back."""
        lat = self.telp.agent.lattice
        uf = self.telp.user_facts
        mems = dict(zip(getattr(lat, "_ids", []), getattr(lat, "_texts", [])))
        facts = dict(zip(getattr(uf, "_ids", []), getattr(uf, "_texts", []))) \
            if uf is not None else {}
        try:
            reply = self.telp._forget_route(question, None)
        except Exception:
            reply = None
        left = set(getattr(lat, "_ids", []))
        gone = {mid: text for mid, text in mems.items() if mid not in left}
        left_facts = set(getattr(uf, "_ids", [])) if uf is not None else set()
        gone_texts = list(gone.values()) + [
            text for fid, text in facts.items() if fid not in left_facts]
        dropped = self._drop_turns_resting_on(set(gone), gone_texts)
        if reply is None:
            reply = ("I couldn't forget that - tell me what to forget, for "
                     "example 'forget the Zorblax river'.")
        if dropped:
            reply += (f" I've also dropped {dropped} turn"
                      f"{'s' if dropped != 1 else ''} of our conversation "
                      f"that rested on it.")
        # deliberately not recorded: a turn about it would bring it back
        return HarnessTurn(answer=reply, handled_by="telp")

    def _forget_topic(self, topic: str) -> str:
        """'forget what we said about Rome': the conversation turns that
        name every word of the topic as whole words (never 'Chromebook'
        for 'Rome'). Telp's memory itself is left alone."""
        words = [w for w in _WORD_RX.findall(topic.lower()) if w not in _VAGUE]
        if not words:
            return "Tell me which part of our conversation to forget."
        pats = [re.compile(rf"(?<![^\W_]){re.escape(w)}(?![^\W_])", re.I)
                for w in words]
        doomed = [r["id"] for r in self.state.turns()
                  if all(p.search(f"{r['question']} {r['answer']}")
                         for p in pats)]
        n = self._drop_turns(doomed)
        if not n:
            return (f"Nothing in our conversation was about '{topic}', so "
                    f"there was nothing to drop.")
        return (f"Done - I've dropped {n} turn{'s' if n != 1 else ''} of our "
                f"conversation about {topic}. My memory itself is unchanged "
                f"('forget {topic}' would forget it there too).")

    def _drop_turns_resting_on(self, ids: set[int],
                               texts: list[str]) -> int:
        """Drop the turns whose answer rested on a deleted memory row (or
        user fact), or that quote a deleted row word for word."""
        if not ids and not texts:
            return 0
        keys = [k for k in (_match_text(t) for t in texts) if len(k) >= 12]
        doomed = []
        for row in self.state.turns():
            refs = _refs(row)
            flagged = any("cited" in r for r in refs)
            rested = [r for r in refs if r.get("cited") or
                      r.get("cited_unbacked")] if flagged else refs
            hit = any(_ref_id(r) in ids for r in rested) or any(
                _match_text(str(r.get("text") or "")) in keys for r in rested)
            if not hit and keys:
                said = _match_text(f"{row['question']} {row['answer']}")
                hit = any(k in said for k in keys)
            if hit:
                doomed.append(row["id"])
        return self._drop_turns(doomed)

    def _drop_turns(self, ids: list[int]) -> int:
        """Delete these conversation turns (by row id)."""
        if not ids:
            return 0
        con = self.state.con
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            con.execute("DELETE FROM harness_turns WHERE id IN "
                        f"({','.join('?' * len(chunk))})", chunk)
        con.commit()
        cache = getattr(self.state, "_hv_cache", None)
        if isinstance(cache, dict):
            cache.clear()
        return len(ids)

    def _provenance(self, question: str,
                    notes: list[str]) -> HarnessTurn | None:
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
        the conversation memory, so it works after a restart too, along
        with how many of that answer's sentences the checker rejected."""
        rows = self.state.turns(last=1)
        if not rows:
            return ("You haven't asked me anything yet - ask me a question "
                    "first."), []
        refs = _refs(rows[0])
        check = next((r for r in rows[0]["sources"] if isinstance(r, dict)
                      and r.get("kind") == _CHECK_KIND), {})
        items = [self._ref_evidence(r) for r in refs]
        if not items:
            return ("Nothing I have stored backs that last answer - I had no "
                    "sources for it, so treat it as unverified."), []
        backed = [e for e, r in zip(items, refs) if r.get("cited")]
        unbacked = [e for e, r in zip(items, refs)
                    if r.get("cited_unbacked") and not r.get("cited")]
        if backed:
            head, shown = ("I can show you exactly - that answer came from:",
                           backed)
        elif unbacked:
            nums = ", ".join(f"[{e.n}]" for e in unbacked)
            one = len(unbacked) == 1
            head = (f"That last answer cited {nums}, but "
                    f"{'that source doesn' if one else 'those sources don'}'t "
                    f"back what it said, so nothing I know backs it. "
                    f"{'It says' if one else 'They say'}:")
            shown = unbacked
        else:
            head = ("That last answer didn't rest on any of my sources (it "
                    "cited none), so nothing I know backs it. What I had "
                    "given the model was:")
            shown = items[:5]
        lines = [head] + ["  " + _cite_line(e) for e in shown]
        bad = int(check.get("unsupported") or 0)
        if bad and backed:
            lines.append(
                f"Careful: {bad} sentence{'s' if bad != 1 else ''} of that "
                f"answer {'were' if bad != 1 else 'was'} not backed by these "
                f"sources - marked in the answer, and not kept in my notes "
                f"on our conversation.")
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

    # ── 2. the worker pre-pass ─────────────────────────────────────
    def _pre_pass(self, question: str) -> list[Evidence]:
        """Deterministic work done before the model reads anything:
        arithmetic in the question, and the date and time now when the
        question is about the present."""
        now = self._now()
        stamp = now.isoformat(timespec="seconds")
        found = [Evidence(n=0, text=r, source="tool:calculate",
                          created_at=stamp, kind="tool")
                 for r in _arithmetic(question)]
        if _NOW_RX.search(question):
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
        messages = brief.messages()
        ex = _Exchange(messages=messages, evidence=list(brief.evidence),
                       thinking=self.thinking,
                       brief_user=messages[-1]["content"],
                       brief_count=len(brief.evidence))
        brief_estimate = _prompt_tokens(ex.messages, None)
        try:
            self._probe()
            ex.tool_room = tool_room(
                self.budget_tokens, self._context,
                brief_estimate + _tools_tokens(self.tools()))
            reply = self._converse(ex, stream, on_token, on_thinking)
        except LLMUnavailable as err:
            self._probed = False          # look at the server afresh
            return self._fallback(question, str(err), ex, failure_kind(err))
        if ex.skipped:
            ex.notes.append(
                f"The model asked for {ex.skipped} tool call"
                f"{'s' if ex.skipped != 1 else ''} beyond Telp's limits "
                f"({CALLS_PER_ROUND} a round, about {ex.tool_room:,} tokens "
                f"of tool results a question); "
                f"{'they were' if ex.skipped != 1 else 'it was'} not run.")
        text = _clean_text(reply.text)
        if not text:
            why = ex.problem or " ".join(getattr(reply, "notes", []) or []) \
                or "The model returned an empty answer."
            return self._fallback(question, why, ex, "")

        checks, shown = self._check(text, ex.evidence, ex.notes)
        cited = _cited(text, checks)
        unbacked = _cited_unbacked(checks) - cited
        bad = sum(1 for c in checks if c.status == "unsupported")
        self._record(question, _kept_answer(text, checks), ex.evidence,
                     cited, unbacked, check={"unsupported": bad,
                                             "checked": bool(checks)})
        try:                          # feed chat mode's pronoun stack too
            self.telp._track_entities(question, text)
        except Exception:
            pass
        sent, naive, measured = self._meter(ex, brief, brief_estimate)
        return HarnessTurn(
            answer=shown, evidence=ex.evidence, checks=checks,
            brief_tokens=sent, naive_tokens=naive, rounds=ex.rounds,
            tools_used=ex.tools_used, usage=ex.usage, handled_by="llm",
            notes=ex.notes, measured=measured, raw_answer=text)

    def _meter(self, ex: _Exchange, brief,
               brief_estimate: int) -> tuple[int, int, bool]:
        """(sent, naive, measured): the final request against a send-
        everything prompt for the same turn, in the same units - the
        server's own count when it gave one (the estimate of a send-
        everything prompt is then scaled the same way), else estimates."""
        # what tool rounds added, and the tool definitions: a send-
        # everything prompt with the same tools carries them too
        growth = max(0, _prompt_tokens(ex.messages, None) - brief_estimate)
        growth += ex.tools_tokens
        naive = max(brief.naive_tokens + growth, ex.prompt_estimate)
        sent = ex.prompt_estimate
        measured = ex.prompt_measured > 0 and ex.prompt_estimate > 0
        if measured:
            naive = round(naive * ex.prompt_measured / ex.prompt_estimate)
            sent = ex.prompt_measured
        return sent, max(naive, sent), measured

    def _probe(self) -> None:
        """Ask the server, once a session (and again after a failure),
        how much context the model has loaded: what the model may write
        and how many tool results fit depend on it. A server that plainly
        isn't there is reported now, without also waiting on a chat
        request; one that is merely slow is left to the chat request."""
        if self._probed:
            return
        status = getattr(self.client, "status", None)
        if not callable(status):
            self._probed = True
            return
        try:
            info = status()
        except Exception:
            return
        if not isinstance(info, dict):
            self._probed = True
            return
        if not info.get("reachable"):
            err = LLMUnavailable(str(info.get("error")
                                     or "The model server can't be reached."))
            if failure_kind(err) in ("down", "address"):
                raise err
            return
        ctx = info.get("context_length")
        self._context = int(ctx) if isinstance(ctx, (int, float)) \
            and not isinstance(ctx, bool) and ctx > 0 else None
        self._probed = True

    def _converse(self, ex: _Exchange, stream: bool, on_token,
                  on_thinking) -> LLMReply:
        """Call the model; run the tools it asks for and call again, up to
        max_tool_rounds times or until the question's tool room is used.
        The last call offers no tools, so the model has to answer. (On
        Qwen that changes the system prompt for that one call, a cache
        miss accepted only when the tool rounds ran out.) A model that
        still asks for a tool is told, once, to answer from what it has."""
        tool_rounds = 0
        told = False
        while True:
            last = (tool_rounds >= self.max_tool_rounds or ex.tools_done
                    or told or ex.tool_room - ex.tool_tokens < MIN_TOOL_ROOM)
            offer = None if last else self.tools()
            reply = self._call(ex, offer, stream, on_token, on_thinking)
            if offer and reply.tool_calls and not ex.tools_done:
                tool_rounds += 1
                self._run_round(ex, reply)
                continue
            if _only_tool_call(reply) and not told:
                told = True
                self._no_more_tools(ex, reply)
                continue
            break
        if _only_tool_call(reply):
            ex.problem = (f"The model kept asking for more tool calls after "
                          f"Telp's limit ({tool_rounds} tool round"
                          f"{'s' if tool_rounds != 1 else ''}) and never "
                          f"wrote an answer.")
        elif (not _clean_text(reply.text) and not reply.tool_calls
                and reply.finish_reason == "length" and ex.thinking is None):
            # LM Studio's default 8192-token context can be spent entirely
            # on thinking: ask once more with thinking switched off
            ex.thinking = False
            reply = self._call(ex, None, stream, on_token, on_thinking)
            ex.notes.append("The model ran out of room while thinking, so "
                            "Telp asked again with thinking switched off.")
        return reply

    def _call(self, ex: _Exchange, tools: list[dict] | None, stream: bool,
              on_token, on_thinking) -> LLMReply:
        """One request, with three recoveries, each tried once:
          * a server that refuses tools on a question's first request is
            asked again without them, and isn't offered them again this
            session;
          * a server that fails once tool results are in the conversation
            (a chat template that can't render them, say) gets them folded
            into the brief as plain numbered sources instead - tools stay
            on for later questions;
          * a model that takes too long, thinking, is asked again with
            thinking switched off."""
        try:
            return self._send(ex, tools, stream, on_token, on_thinking)
        except LLMUnavailable as err:
            kind = failure_kind(err)
            history = _has_tool_history(ex.messages)
            if tools and not history and _refuses_tools(err):
                self._tools_ok = False
                reply = self._send(ex, None, stream, on_token, on_thinking)
                ex.notes.append("The model server doesn't accept tools, so "
                                "the model answered from Telp's brief alone "
                                "(it can't ask Telp to dig deeper).")
                return reply
            if history and kind == "http":
                self._flatten(ex)
                reply = self._send(ex, None, stream, on_token, on_thinking)
                ex.notes.append(
                    f"The model server couldn't take the tool results back "
                    f"(it answered HTTP {err.status}), so Telp gave them to "
                    f"the model as plain numbered sources instead.")
                return reply
            if kind == "timeout" and ex.thinking is None \
                    and not _answered(err):
                ex.thinking = False
                ex.notes.append("The model didn't answer in time, so Telp "
                                "asked again with thinking switched off.")
                return self._send(ex, tools, stream, on_token, on_thinking)
            raise

    def _send(self, ex: _Exchange, tools: list[dict] | None, stream: bool,
              on_token, on_thinking) -> LLMReply:
        """One chat request, its size and the server's counts noted."""
        estimate = _prompt_tokens(ex.messages, tools)
        extra: dict = {"on_thinking": on_thinking} if on_thinking else {}
        cap = self._reply_cap(estimate)
        if cap:
            extra["max_tokens"] = cap
        reply = self.client.chat(ex.messages, tools=tools, stream=stream,
                                 on_token=on_token if stream else None,
                                 thinking=ex.thinking, **extra)
        ex.rounds += 1
        ex.tools_tokens = _tools_tokens(tools)
        ex.prompt_estimate = estimate
        ex.prompt_measured = int(reply.usage.get("prompt_tokens") or 0)
        if ex.prompt_measured and estimate:
            self._ratio = ex.prompt_measured / estimate
        _add_usage(ex.usage, reply.usage)
        ex.notes.extend(n for n in getattr(reply, "notes", []) or []
                        if n not in ex.notes)
        return reply

    def _reply_cap(self, prompt_estimate: int) -> int | None:
        """max_tokens for one call - what the model may write, thinking
        included: at most reply_tokens, and never more than the loaded
        context has left after the prompt. Without it a thinking model on
        LM Studio's default 8,192 tokens can think until the context runs
        out, which on a CPU-offloaded 27B takes many minutes."""
        cap = self.reply_tokens
        if self._context:
            factor = self._ratio * 1.05 if self._ratio else PROMPT_SLACK
            left = self._context - int(prompt_estimate * factor) - 64
            left = max(MIN_REPLY_TOKENS, left)
            cap = min(cap, left) if cap else left
        return cap

    def _flatten(self, ex: _Exchange) -> None:
        """Fold the tool rounds back into the brief: the system message and
        the brief's user message with every source found since added
        before the question - no tool calls or tool messages left for the
        server to render. Tools are not offered again for this question."""
        found = ex.evidence[ex.brief_count:]
        user = ex.brief_user
        if found:
            more = "More sources, found while answering:\n" + "\n".join(
                e.line() for e in found)
            cut = user.rfind("\n\nQuestion: ")
            user = (f"{user[:cut]}\n\n{more}{user[cut:]}" if cut >= 0
                    else f"{user}\n\n{more}")
        ex.messages[:] = [ex.messages[0], {"role": "user", "content": user}]
        ex.tools_done = True

    def _run_round(self, ex: _Exchange, reply: LLMReply) -> None:
        """Run one reply's tool calls and put their results in the
        conversation: at most CALLS_PER_ROUND of them (the rest are left
        out of the history altogether, so they cost the next request
        nothing), and no more results in all than the question's tool
        room - a call past it is answered 'limit reached'."""
        calls = reply.tool_calls[:CALLS_PER_ROUND]
        extra = len(reply.tool_calls) - len(calls)
        ex.skipped += extra
        ex.messages.append(assistant_message(
            LLMReply(text=reply.text or "", tool_calls=calls)))
        for i, call in enumerate(calls):
            room = ex.tool_room - ex.tool_tokens
            if room < MIN_TOOL_ROOM:
                result = _limit_text(ex.evidence)
                ex.skipped += 1
            else:
                result = self._run_tool(call, ex.evidence, room)
                ex.tools_used.append(call.name)
            if extra and i == len(calls) - 1:
                result += (f"\n(Telp ran only the first {len(calls)} of your "
                           f"{len(reply.tool_calls)} tool calls.)")
            ex.tool_tokens += estimate_tokens(result)
            ex.messages.append(tool_message(call, result))

    def _no_more_tools(self, ex: _Exchange, reply: LLMReply) -> None:
        """The model asked for a tool after the last round (Qwen often
        does: its earlier tool calls are still in the conversation). Answer
        that call with 'no more tool calls - answer now' and ask again."""
        last = max((e.n for e in ex.evidence), default=0)
        what = (f"sources [1]-[{last}]" if last > 1 else
                "source [1]" if last == 1 else "what you already have")
        say = (f"No more tool calls for this question - Telp's limit is "
               f"reached. Answer now from {what}, and say plainly what "
               f"they don't cover.")
        calls = _inline_calls(reply.text)
        if calls and self._tools_ok:
            ex.messages.append(assistant_message(
                LLMReply(text="", tool_calls=calls)))
            ex.messages.extend(tool_message(c, say) for c in calls)
        else:
            ex.messages.append({"role": "assistant",
                                "content": reply.text or ""})
            ex.messages.append({"role": "user", "content": "Telp: " + say})
        ex.notes.append("The model asked for another tool after Telp's "
                        "limit; Telp told it to answer from the sources it "
                        "had.")

    # ── the tools ──────────────────────────────────────────────────
    def _run_tool(self, call: ToolCall, evidence: list[Evidence],
                  room_tokens: int) -> str:
        """Run one tool call; the result is a short text for the model,
        within room_tokens. Anything it finds is appended to the evidence,
        numbered on from the last source, so the model can cite it and
        Telp can check it."""
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
            return run(args, evidence, room_tokens * 4)
        except Exception as err:
            return f"Telp's {call.name} tool failed ({err}); answer without it."

    def _tool_search(self, args: dict, evidence: list[Evidence],
                     room_chars: int) -> str:
        query = " ".join(str(args.get("query") or args.get("q") or "").split())
        if not query:
            return "search_memory needs a query."
        found = self.builder.search(query, limit=SEARCH_LIMIT,
                                    exclude=evidence)
        new = _append(evidence, found, room_chars=room_chars - 16)
        if not new:
            return f"Nothing more in Telp's memory for '{query}'."
        return "New sources:\n" + "\n".join(e.line() for e in new)

    def _tool_calculate(self, args: dict, evidence: list[Evidence],
                        room_chars: int) -> str:
        from mind.code_synthesis import try_arithmetic
        expr = str(args.get("expression") or args.get("expr") or "").strip()
        result = try_arithmetic(expr) if expr else None
        if not result:
            return (f"Telp couldn't calculate '{expr}'. Use numbers with "
                    "+ - * / ** ( ) % and sqrt().")
        stamp = self._now().isoformat(timespec="seconds")
        ev = Evidence(n=0, text=result, source="tool:calculate",
                      created_at=stamp, kind="tool")
        got = _append(evidence, [ev], keep_existing=True,
                      room_chars=room_chars)
        return got[0].line() if got else _limit_text(evidence)

    def _tool_today(self, args: dict, evidence: list[Evidence],
                    room_chars: int) -> str:
        now = self._now()
        ev = Evidence(n=0, text=_now_line(now), source="tool:today",
                      created_at=now.isoformat(timespec="seconds"),
                      kind="tool")
        got = _append(evidence, [ev], keep_existing=True,
                      room_chars=room_chars)
        return got[0].line() if got else _limit_text(evidence)

    def _tool_facts(self, args: dict, evidence: list[Evidence],
                    room_chars: int) -> str:
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
        new = _append(evidence, items[:FACTS_LIMIT],
                      room_chars=room_chars - 16)
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
                cited: set[int], unbacked: set[int] = frozenset(),
                check: dict | None = None) -> None:
        """Keep the turn in the conversation memory (harness_turns), with
        its sources as references, which of them the answer rested on
        (cited) or cited without their backing it (cited_unbacked), and
        for a model turn how many sentences the checker rejected - what
        "how do you know that?" reads next turn, in any process."""
        refs = [{"n": e.n, "kind": e.kind, "source": e.source,
                 "created_at": e.created_at, "memory_id": e.memory_id,
                 "score": round(float(e.score), 4), "text": e.text,
                 "cited": e.n in cited,
                 "cited_unbacked": e.n in unbacked and e.n not in cited}
                for e in evidence]
        if check is not None:
            refs.append({"kind": _CHECK_KIND, "n": 0, **check})
        try:
            self.state.add_turn(question, answer, refs)
        except Exception:
            pass

    # ── when the model can't answer ────────────────────────────────
    def _fallback(self, question: str, problem: str, ex: _Exchange,
                  kind: str) -> HarnessTurn:
        """Say what went wrong (and, when nothing answered at all, how to
        start the server), then give Telp's own answer from chat mode -
        marked, so nobody takes it for the model's. FluentTelp is asked
        the message without a leading "remember"/"forget" and with user-
        fact capture off: the harness has already decided this message
        teaches, deletes and captures nothing (or captured it already)."""
        turns = getattr(self.telp.agent, "turns", None)
        before = len(turns) if isinstance(turns, list) else 0
        asked = _fallback_text(question)
        try:
            with _capture_off(self.telp.user_facts):
                own = (self.telp.respond(asked) or "").strip() if asked \
                    else ""
        except Exception as err:
            own = f"(Telp couldn't answer on his own either: {err})"
        head = problem.strip()
        if kind == "down":
            head += "\n" + (START_TAIL if "start the server" in head.lower()
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


# ─── sizes: how much the model reads and may write ──────────────────

def _reply_reserve(context_length: int) -> int:
    """Context kept free for the model's thinking and answer: half of it,
    and never less than MIN_REPLY_RESERVE tokens."""
    return max(MIN_REPLY_RESERVE, context_length // 2)


def tool_room(budget_tokens: int, context_length: int | None = None,
              prompt_tokens: int | None = None) -> int:
    """Tokens of tool results Telp gives the model for one question: a
    share of the brief's budget, and on a small loaded context no more
    than keeps half of it free for the model's thinking and answer.
    prompt_tokens is the request before any tool round (default: the
    budget)."""
    room = int(budget_tokens * TOOL_SHARE)
    if context_length:
        ceiling = context_length - _reply_reserve(context_length)
        before = budget_tokens if prompt_tokens is None else prompt_tokens
        room = min(room, ceiling - before)
    return max(0, room)


def failure_kind(err) -> str:
    """What an LLMUnavailable (or its message) says went wrong:
      "down"     nothing answered at that address - LM Studio's server
                 isn't started, or that PC is off
      "timeout"  the server is there but went quiet (busy, or a model
                 thinking for a long time)
      "http"     the server answered with an error
      "address"  the address itself is wrong
      "other"    anything else (a dropped connection, a reply that isn't
                 JSON, an error inside a stream)
    The client words its messages for people, so they are read here;
    .status marks an HTTP error."""
    if getattr(err, "status", None) is not None:
        return "http"
    low = str(err).lower()
    if any(s in low for s in ("isn't valid", "can't find the host",
                              "not in http", "secure (https)")):
        return "address"
    if any(s in low for s in ("isn't running", "couldn't connect",
                              "no route", "couldn't reach")):
        return "down"
    if "seconds" in low and any(s in low for s in (
            "sent nothing", "went quiet", "didn't answer within",
            "nothing for", "stuck")):
        return "timeout"
    return "other"


# ─── small helpers ──────────────────────────────────────────────────

def _arithmetic(question: str) -> list[str]:
    """Sums in the question, worked out exactly: the whole message when it
    is a calculation ("what's 15% of 80?"), else inline expressions that
    stand alone."""
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
        if not _stands_alone(question, m.start(), m.end()):
            continue
        if not _STRONG_OP_RX.search(span) and not (
                cue and _SPACED_WEAK_RX.search(span)):
            continue
        result = try_arithmetic(span)
        if result and result not in out:
            out.append(result)
        if len(out) >= PREPASS_CALCS:
            break
    return out


def _stands_alone(text: str, start: int, end: int) -> bool:
    """Nothing number-like touches the span, even across a space: a digit
    ('1 500 + 2 500', a digit group), ':' ('10:30 - 11:45', 'John 3:16 -
    3:18'), a sign or operator it can't see ('-3 + 5'), a decimal point.
    Such a span is a fragment of something else; worked out on its own
    it would be a wrong source Telp vouches for."""
    before = text[:start].rstrip()
    after = text[end:].lstrip()
    if before and (before[-1].isdigit() or before[-1] in _TOUCH_BEFORE):
        return False
    if after and (after[0].isdigit() or after[0] in _TOUCH_AFTER):
        return False
    return True


def _now_line(now: datetime) -> str:
    """'Today is Thursday, 8 October 2026; the local time is 14:05.' -
    spelled out without the locale, the same on every machine."""
    return (f"Today is {_DAYS[now.weekday()]}, {now.day} "
            f"{_MONTHS[now.month - 1]} {now.year}; the local time is "
            f"{now:%H:%M}.")


def _not_about_the_user(question: str) -> bool:
    """A pasted document, a quoted passage, or a request to work on a
    text: whatever 'my name is ...' or 'I live in ...' it holds is
    somebody else's (the letter's writer), not the user's. Long messages
    count too: statements about oneself are short."""
    text = question.strip()
    if "\n\n" in text or text.count("\n") >= 2:
        return True
    if len(text.split()) > USER_FACT_WORDS:
        return True
    if any(len(m.group(1).split()) >= 3 for m in _QUOTED_RX.finditer(text)):
        return True
    return bool(_DOC_CUE_RX.search(text))


def _noted_line(new: list[str], olds: list[str]) -> str:
    """The note a model turn carries when its message also said something
    about the user - a change to what Telp believes is never silent."""
    said = "; ".join(t.rstrip(".") for t in new) or "a fact about you"
    if olds:
        prev = "; ".join(o.rstrip(".") for o in olds[:2])
        return (f"Telp updated what he believes about you: {said}. "
                f"(Previously: {prev}.) If that's wrong, tell him again.")
    return f"Telp noted about you: {said}."


def _first_clause(text: str) -> tuple[str, str]:
    """('forget that', "what's the capital of Iceland?") - a message's
    first clause, and the rest when the rest is a clause of its own (two
    words or more, or a question). A short tail ('Washington, D.C.', 'Dr.
    Smith') stays part of the first clause."""
    for m in _CLAUSE_BREAK_RX.finditer(text):
        rest = text[m.end():].strip(" ,;:-–—")
        if not rest:
            break
        if len(rest.split()) >= 2 or rest.endswith("?"):
            return text[:m.start()].strip(), rest
    return text.strip().rstrip(" .!?;:,"), ""


def _forget_target(clause: str) -> str:
    """What a forget clause names ('the Zorblax river'), or "" when it
    names nothing ('forget it', 'forget about that', 'forget everything')."""
    phrase = re.sub(r"^\s*forget\b", "", clause, flags=re.I)
    phrase = phrase.strip(" :,.-!?\"'")
    while True:
        shorter = _FORGET_LEAD_RX.sub("", phrase).strip()
        if shorter == phrase:
            break
        phrase = shorter
    words = [w for w in _WORD_RX.findall(phrase.lower()) if w not in _VAGUE]
    return phrase if words else ""


def _forget_intent(question: str) -> tuple[str, str, str]:
    """What a message starting with 'forget' asks for, as (kind, its
    forget clause, the thing named):
      ""          not a forget command at all
      "ask"       a forget clause inside a question, or followed by more
                  ("Forget that, what's the capital of Iceland?"): the
                  model answers the message and nothing is deleted
      "chat_all"  the whole conversation ("forget everything we talked
                  about", "forget our last conversation")
      "chat_last" its last exchange ("forget your last answer")
      "chat_topic" the conversation about something ("forget what we
                  said about Rome")
      "vague"     names nothing ("forget it", "forget about that")
      "memory"    names something Telp may know ("forget the Zorblax
                  river") - FluentTelp's own forget route"""
    text = " ".join((question or "").split())
    if not _FORGET_RX.match(text):
        return "", "", ""
    clause, rest = _first_clause(text)
    target = _forget_target(clause)
    if text.endswith("?") or rest:
        return "ask", clause, target
    if _CHAT_ALL_RX.match(clause):
        return "chat_all", clause, ""
    if _CHAT_LAST_RX.match(clause):
        return "chat_last", clause, ""
    m = _CHAT_TOPIC_RX.match(clause)
    if m and _forget_target("forget " + m.group(1)):
        return "chat_topic", clause, m.group(1).strip()
    if not target:
        return "vague", clause, ""
    return "memory", clause, target


def _fallback_text(question: str) -> str:
    """The message as Telp's own (no-model) answer is asked it. FluentTelp
    takes a leading 'remember ...' as teaching and 'forget ...' as an
    order to delete; the harness has already decided this message is
    neither, so those words go first ("remember when Galileo was born?"
    -> "when Galileo was born?", "Forget that, what's 2+2?" -> "what's
    2+2?")."""
    text = question.strip()
    for _ in range(3):
        if _FORGET_RX.match(text):
            _, rest = _first_clause(text)
            text = rest or re.sub(r"^\s*forget\b[\s,:.-]*(?:about\s+)?", "",
                                  text, flags=re.I)
        elif _TEACH_RX.match(text):
            text = re.sub(r"^\s*remember\b\s*(?:that\s+)?", "", text,
                          flags=re.I)
        else:
            break
    if _FORGET_RX.match(text) or _TEACH_RX.match(text):
        return ""
    return text.strip()


def _capture_nothing(user_msg: str) -> list[int]:
    return []


@contextmanager
def _capture_off(uf):
    """FluentTelp's respond() reads facts about the user out of every
    message; for a fallback answer the harness has already done that (or
    decided against it, for a pasted document), so capture is switched
    off for the call."""
    patched = False
    if uf is not None:
        try:
            uf.capture = _capture_nothing
            patched = True
        except Exception:
            pass
    try:
        yield
    finally:
        if patched:
            try:
                del uf.capture
            except Exception:
                pass


def _refs(row: dict) -> list[dict]:
    """A stored turn's source references (without the check entry)."""
    return [r for r in row.get("sources") or []
            if isinstance(r, dict) and r.get("kind") != _CHECK_KIND]


def _ref_id(ref: dict) -> int | None:
    try:
        return int(ref["memory_id"]) if ref.get("memory_id") is not None \
            else None
    except (TypeError, ValueError):
        return None


def _match_text(text: str) -> str:
    """Text as forgetting compares it: case folded, spacing collapsed,
    without a final full stop."""
    return " ".join((text or "").split()).casefold().rstrip(" .")


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
            keep_existing: bool = False,
            room_chars: int | None = None) -> list[Evidence]:
    """Number new items on from the last source and add them, within
    room_chars of source lines when given (an item is shortened to fit,
    and items stop when even a short one won't). An item whose text is
    already shown is skipped - or, with keep_existing, returned as the
    source already there."""
    out: list[Evidence] = []
    used = 0
    for item in items:
        limit = TOOL_ITEM_CHARS
        if room_chars is not None:
            overhead = len(replace(item, n=999, text="").line()) + 1
            limit = min(limit, room_chars - used - overhead)
            if limit < 40:
                break
        text = _clip(item.text, limit)
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
        used += len(new.line()) + 1
    return out


def _limit_text(evidence: list[Evidence]) -> str:
    """The tool result once the question's tool room is used up."""
    last = max((e.n for e in evidence), default=0)
    what = (f"sources [1]-[{last}]" if last > 1 else
            "source [1]" if last == 1 else "what you already have")
    return (f"Limit reached: Telp has given all the tool results he can for "
            f"one question. Answer now from {what}, and say plainly what "
            f"they don't cover.")


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


def _cited_unbacked(checks: list[SentenceCheck]) -> set[int]:
    """Sources cited by sentences the checker rejected."""
    return {int(n) for c in checks if c.status == "unsupported"
            for n in c.cites}


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


def _only_tool_call(reply: LLMReply) -> bool:
    """The model wrote a tool call as text, and nothing else - what it
    does when it wants a tool and none is offered."""
    return (not reply.tool_calls and "<tool_call>" in (reply.text or "")
            and not _clean_text(reply.text))


def _inline_calls(text: str) -> list[ToolCall]:
    """Tool calls written as <tool_call>{json}</tool_call> text."""
    calls: list[ToolCall] = []
    for m in _INLINE_CALL_RX.finditer(text or ""):
        try:
            data = json.loads(m.group(1))
        except ValueError:
            continue
        if not isinstance(data, dict) or not data.get("name"):
            continue
        args = data.get("arguments", data.get("parameters"))
        calls.append(ToolCall(id=f"call_late_{len(calls)}",
                              name=str(data["name"]),
                              arguments=args if isinstance(args, dict)
                              else {}))
    return calls


def _has_tool_history(messages: list[dict]) -> bool:
    return any(m.get("role") == "tool" or m.get("tool_calls")
               for m in messages)


def _refuses_tools(err: LLMUnavailable) -> bool:
    """The server rejected the request (HTTP 400 or 422) because of its
    tools - what a server without function calling answers."""
    low = str(err).lower()
    return getattr(err, "status", None) in (400, 422) \
        and ("tool" in low or "function" in low)


def _answered(err: LLMUnavailable) -> bool:
    """Some of the answer already streamed out before the failure."""
    partial = getattr(err, "partial", None)
    return bool(partial is not None and (partial.text or "").strip())


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


def _tool_counts(names: list[str]) -> str:
    """'search_memory ×3, calculate' - each tool once, with a count."""
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(n if c == 1 else f"{n} ×{c}" for n, c in counts.items())


# ─── what the command line prints ───────────────────────────────────

def meter_line(turn: TurnResult) -> str:
    """'Telp sent 812 tokens (a send-everything prompt: ~24,300) · 1 model
    call · 3.1s' - how much the model had to read, against what a
    send-everything chat app would have sent (in the same units)."""
    secs = f"{turn.seconds:.1f}s"
    if getattr(turn, "fallback", False):
        return f"The model didn't answer - Telp answered on his own · {secs}"
    if turn.handled_by != "llm":
        return f"Telp handled this himself - no model call · {secs}"
    sent = f"{turn.brief_tokens:,}" if getattr(turn, "measured", False) \
        else f"~{turn.brief_tokens:,}"
    calls = f"{turn.rounds} model call{'s' if turn.rounds != 1 else ''}"
    if turn.tools_used:
        calls += " (" + _tool_counts(turn.tools_used) + ")"
    naive = max(turn.naive_tokens, turn.brief_tokens)
    parts = [f"Telp sent {sent} tokens (a send-everything prompt: "
             f"~{naive:,})", calls]
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
    head = (f"This session: {t['model_turns']} model turn(s), "
            f"{t['calls']} model call(s); Telp sent ~{t['sent']:,} tokens in "
            f"all, send-everything prompts would have been "
            f"~{max(t['naive'], t['sent']):,}")
    if t["naive"] <= t["sent"]:
        return head + (" (no saving yet: the conversation and the memory "
                       "are still small).")
    return head + f" ({1 - t['sent'] / t['naive']:.0%} less to read)."


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
        if failure_kind(LLMUnavailable(str(info.get("error") or ""))) \
                != "down":
            return lines
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
        # a brief, the tool definitions, and the question's tool results
        tools = _tools_tokens(list(BASE_TOOLS))
        prompt = budget_tokens + tools + tool_room(budget_tokens, ctx,
                                                   budget_tokens + tools)
        lines.append(f"brief:   about {budget_tokens:,} tokens a turn (at "
                     f"most ~{prompt:,} with tool results), leaving "
                     f"~{ctx - prompt:,} for the model's thinking and answer")
    else:
        lines.append("context: not reported by the server")
    lines += [f"warning: {w}" for w in info.get("warnings", [])]
    return lines
