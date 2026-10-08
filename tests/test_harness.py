"""Harness mode: Telp as the memory, filter and worker in front of a local
language model (a FakeLLM here) - commands never reach the model, tools
add numbered sources, answers are checked, the conversation stays small,
and a dead server falls back to Telp's own answer."""
import json
import os
import re
import socket
import subprocess
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from mind.harness import (BASE_TOOLS, Harness, check_lines, meter_line,
                          sources_text, status_lines)
from mind.harness_types import SentenceCheck
from mind.llm_client import LLMClient
from tests.fake_llm import FakeLLM

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 8, 14, 5)
MODEL = "qwen3.8-27b"

WORLD = [
    ("Reykjavik is the capital and largest city of Iceland.",
     "wikipedia:Iceland"),
    ("Iceland is an island country in the North Atlantic Ocean.",
     "wikipedia:Iceland"),
    ("Galileo Galilei was an Italian astronomer and physicist.",
     "wikipedia:Galileo Galilei"),
    ("Galileo Galilei was born in Pisa in 1564.",
     "wikipedia:Galileo Galilei"),
    ("Marie Curie was born in Warsaw in 1867.", "wikipedia:Marie Curie"),
    ("Bananas are rich in potassium and grow in tropical climates.",
     "wikipedia:Banana"),
    ("The Pacific Ocean is the largest ocean on Earth.",
     "wikipedia:Pacific Ocean"),
]


# ─── a stand-in checker (mind/checker.py's API) ─────────────────────
#
# A sentence is supported when it cites [n] and every n exists; an
# uncited "I don't know" makes no claim. The real checker judges meaning
# too; test_the_real_checker_marks_an_unsupported_sentence uses it.

def _fake_check(answer, evidence, encoder=None):
    shown = {e.n for e in evidence}
    checks = []
    for s in re.split(r"(?<=[.!?])\s+", answer.strip()):
        cites = [int(n) for n in re.findall(r"\[(\d+)\]", s)]
        ok = bool(cites) and all(n in shown for n in cites)
        status = "supported" if ok else "unsupported"
        if not cites and re.search(r"\b(?:no|not|don't|couldn't|can't)\b",
                                   s, re.I):
            status = "no_claim"
        checks.append(SentenceCheck(
            sentence=s, status=status, cites=cites,
            best_evidence=cites[0] if ok else None))
    return checks


def _fake_annotate(answer, checks, evidence):
    for c in checks:
        if c.status == "unsupported":
            answer = answer.replace(c.sentence,
                                    c.sentence + " [not in my sources]")
    return answer


FAKE_CHECKER = SimpleNamespace(
    check_answer=_fake_check, annotate=_fake_annotate,
    summarize=lambda checks: {"checked": len(checks)})


# ─── helpers ────────────────────────────────────────────────────────

def _teach_world(telp) -> dict:
    return {text: telp.agent.lattice.add(text, source=src)
            for text, src in WORLD}


def _harness(telp, llm_or_url, **kw) -> Harness:
    url = llm_or_url if isinstance(llm_or_url, str) else llm_or_url.url
    kw.setdefault("fact_mind", None)
    kw.setdefault("checker", FAKE_CHECKER)
    kw.setdefault("now", lambda: NOW)
    return Harness(telp, LLMClient(base_url=url, model=MODEL), **kw)


def _user(req: dict) -> str:
    return [m for m in req["messages"] if m["role"] == "user"][-1]["content"]


def _tool_msgs(req: dict) -> list[dict]:
    return [m for m in req["messages"] if m["role"] == "tool"]


def _num(text: str, starts: str) -> int:
    """The [n] of the source line that starts with `starts`."""
    m = re.search(r"\[(\d+)\] " + re.escape(starts), text)
    assert m, f"no source starting {starts!r} in:\n{text}"
    return int(m.group(1))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _answer_capital(req):
    n = _num(_user(req), "Reykjavik is the capital")
    return f"Reykjavik is the capital of Iceland [{n}]."


# ─── 1. commands Telp handles himself ───────────────────────────────

def test_teaching_never_calls_the_model_and_is_cited_after(telp):
    with FakeLLM(lambda req: "should never be asked") as llm:
        h = _harness(telp, llm)
        turn = h.ask("remember that the Zorblax river flows through Quendia")
        assert turn.handled_by == "telp" and turn.rounds == 0
        assert "remember" in turn.answer.lower()
        row = telp.agent.lattice._con.execute(
            "SELECT source FROM memories WHERE text LIKE '%Zorblax%'"
        ).fetchone()
        assert row == ("user_taught",)

        why = h.ask("how do you know that?")
        assert why.handled_by == "telp"
        assert "user_taught" in why.answer and "2026-" in why.answer
        assert "Zorblax river flows through Quendia" in why.answer
        assert llm.requests == []
        assert "no model call" in meter_line(why)


def test_a_question_starting_with_remember_is_not_taught(telp):
    with FakeLLM(lambda req: "I don't have that in my sources.") as llm:
        h = _harness(telp, llm)
        before = telp.agent.lattice.count()
        turn = h.ask("remember when Galileo was born?")
        assert turn.handled_by == "llm" and len(llm.requests) == 1
        assert telp.agent.lattice.count() == before


def test_statements_about_the_user_are_kept_not_sent(telp):
    with FakeLLM(lambda req: "I can't suggest a dish from my sources.") as llm:
        h = _harness(telp, llm)
        turn = h.ask("my name is Roszell")
        assert turn.handled_by == "telp" and llm.requests == []
        assert "remember" in turn.answer.lower()
        # the next question reaches the model with the fact standing
        h.ask("what should I cook tonight?")
        assert len(llm.requests) == 1
        assert "Roszell" in _user(llm.requests[0])


def test_a_fact_with_a_request_is_kept_and_the_request_answered(telp):
    with FakeLLM(lambda req: "Here is a short plan.") as llm:
        h = _harness(telp, llm)
        turn = h.ask("My name is Eric. Can you help me plan my week?")
        assert turn.handled_by == "llm" and len(llm.requests) == 1
        assert "Eric" in _user(llm.requests[0])
        assert any("Eric" in t for t in telp.user_facts.all_facts())


def test_forget_is_handled_by_telp_and_drops_the_conversation_about_it(telp):
    lat = telp.agent.lattice
    lat.add("The Zorblax river flows through Quendia.", source="user_taught")
    with FakeLLM(_answer_zorblax) as llm:
        h = _harness(telp, llm)
        h.ask("where does the Zorblax river flow?")
        assert h.state.count() == 1
        turn = h.ask("forget the Zorblax river")
        assert turn.handled_by == "telp" and len(llm.requests) == 1
        assert "forgot" in turn.answer.lower() or "dropped" in turn.answer
        assert not any("Zorblax" in t for t in lat._texts)
        assert h.state.count() == 0          # the turn about it is gone too
        # "forgetting" is a word, not the command
        h.ask("what is the forgetting curve?")
        assert len(llm.requests) == 2


def test_forget_this_conversation_clears_it(telp):
    with FakeLLM(lambda req: "Noted.") as llm:
        h = _harness(telp, llm)
        h.ask("what is the capital of Iceland?")
        h.ask("and of Norway?")
        turn = h.ask("forget this conversation")
        assert turn.handled_by == "telp" and "2 turns" in turn.answer
        assert h.state.count() == 0


def test_provenance_cites_the_previous_answer_even_after_a_restart(telp):
    _teach_world(telp)
    with FakeLLM(_answer_capital) as llm:
        h = _harness(telp, llm)
        h.ask("what is the capital of Iceland?")
        why = h.ask("how do you know that?")
        assert why.handled_by == "telp" and len(llm.requests) == 1
        assert "wikipedia:Iceland" in why.answer
        assert "saved 2026-" in why.answer
        assert "Reykjavik is the capital" in why.answer
        assert "Bananas" not in why.answer      # only what was cited
        assert why.evidence and why.evidence[0].source == "wikipedia:Iceland"
        # asked again, it still cites the answer - not itself
        assert h.ask("how do you know that?").answer == why.answer
        # a new process (new Harness, same memory file) can still say
        again = _harness(telp, llm).ask("Really? How do you know that?")
        assert "wikipedia:Iceland" in again.answer
        assert len(llm.requests) == 1


def test_provenance_before_any_question_and_after_an_unsourced_answer(telp):
    with FakeLLM(lambda req: "I don't know that.") as llm:
        h = _harness(telp, llm)
        assert "haven't asked" in h.ask("how do you know that?").answer
        h.ask("what is the airspeed of an unladen swallow?")
        why = h.ask("how do you know that?")
        assert why.handled_by == "telp"
        assert "no sources" in why.answer or "nothing" in why.answer.lower()


# ─── 2. the worker pre-pass ─────────────────────────────────────────

def test_arithmetic_is_worked_out_before_the_model_reads(telp):
    with FakeLLM(lambda req: "It is 391 [1].") as llm:
        h = _harness(telp, llm)
        turn = h.ask("what is 17*23?")
        user = _user(llm.requests[0])
        assert "[1] 17*23 = 391 (tool:calculate" in user
        assert turn.evidence[0].kind == "tool"
        assert turn.checks[0].status == "supported"


def test_inline_sums_are_worked_out_but_ratings_and_ranges_are_not(telp):
    with FakeLLM(lambda req: "Noted.") as llm:
        h = _harness(telp, llm)
        h.ask("Galileo lived 1564-1642; what is 1642 - 1564 in years?")
        assert "1642 - 1564 = 78" in _user(llm.requests[-1])
        for q in ("is the help desk open 24/7?",
                  "what happened in 1564-1642?",
                  "is COVID-19 a virus?"):
            h.ask(q)
            assert "tool:calculate" not in _user(llm.requests[-1]), q


def test_todays_date_is_a_source_when_the_question_needs_it(telp):
    with FakeLLM(lambda req: "It is Thursday [1].") as llm:
        h = _harness(telp, llm)
        h.ask("what day is it today?")
        assert ("[1] Today is Thursday, 8 October 2026; the local time is "
                "14:05. (tool:today") in _user(llm.requests[0])
        h.ask("what is the capital of Iceland?")
        assert "tool:today" not in _user(llm.requests[1])


# ─── 3. the brief: follow-ups and a stable, small prompt ────────────

def test_a_pronoun_follow_up_is_searched_with_the_last_subject(telp):
    _teach_world(telp)

    def answer(req):
        user = _user(req)
        if "where was he born" in user:
            n = _num(user, "Galileo Galilei was born")
            return f"He was born in Pisa [{n}]."
        n = _num(user, "Galileo Galilei was an Italian")
        return f"Galileo Galilei was an Italian astronomer [{n}]."

    with FakeLLM(answer) as llm:
        h = _harness(telp, llm)
        h.ask("who was Galileo Galilei?")
        turn = h.ask("where was he born?")
        assert "born in Pisa" in _user(llm.requests[1])
        assert turn.checks and turn.checks[0].status == "supported"


def test_the_fixed_prefix_is_byte_identical_across_turns(telp):
    _teach_world(telp)
    with FakeLLM(lambda req: "Noted.") as llm:
        h = _harness(telp, llm, fact_mind=None)
        h.ask("what is the capital of Iceland?")
        h.ask("who was Marie Curie?")
        a, b = llm.requests
        assert a["messages"][0] == b["messages"][0]           # system
        assert json.dumps(a["tools"]) == json.dumps(b["tools"])
        assert [t["function"]["name"] for t in a["tools"]] == \
            ["search_memory", "calculate", "today"]


def test_request_size_stays_flat_while_a_naive_prompt_grows(telp):
    _teach_world(telp)
    topics = ["the capital of Iceland", "Galileo Galilei", "Marie Curie",
              "bananas", "the Pacific Ocean", "Iceland's ocean"]
    long_answer = " ".join(
        f"My source number one says something about this, point {k} [1]."
        for k in range(1, 7))
    with FakeLLM(lambda req: long_answer) as llm:
        h = _harness(telp, llm)
        turns = [h.ask(f"tell me about {topics[i % len(topics)]} "
                       f"(question {i + 1})") for i in range(20)]
    sizes = [len(json.dumps(r["messages"])) for r in llm.requests]
    assert len(sizes) == 20
    # turn 20 reads about what turn 2 read, and stays inside the budget...
    assert sizes[19] <= 1.5 * sizes[1], sizes
    assert all(t.brief_tokens <= 1800 + 400 for t in turns)
    # ...while sending everything would have grown with the conversation
    assert turns[19].naive_tokens > turns[1].naive_tokens + 1000
    assert turns[19].naive_tokens > 2 * turns[19].brief_tokens


# ─── 4. the model's tools ───────────────────────────────────────────

def test_search_memory_adds_numbered_sources_that_keep_counting(telp):
    _teach_world(telp)

    def answer(req):
        tools = _tool_msgs(req)
        if not tools:
            return {"tool_calls": [{"id": "c1", "name": "search_memory",
                                    "arguments": {"query": "Marie Curie"}}]}
        n = _num(tools[-1]["content"], "Marie Curie was born")
        return (f"Reykjavik is Iceland's capital [1], and Marie Curie was "
                f"born in Warsaw [{n}].")

    with FakeLLM(answer) as llm:
        h = _harness(telp, llm)
        turn = h.ask("what is the capital of Iceland?")
    brief_n = len(re.findall(r"^\[\d+\] ", _user(llm.requests[0]), re.M))
    tool_text = _tool_msgs(llm.requests[1])[0]["content"]
    assert f"[{brief_n + 1}] Marie Curie was born in Warsaw" in tool_text
    assert [e.n for e in turn.evidence] == list(
        range(1, len(turn.evidence) + 1))
    assert turn.rounds == 2 and turn.tools_used == ["search_memory"]
    assert all(c.status == "supported" for c in turn.checks)
    assert "(search_memory)" in meter_line(turn)
    # the model's tool call went back exactly as it was made
    sent = llm.requests[1]["messages"]
    assert sent[2]["role"] == "assistant" and sent[2]["tool_calls"]
    assert sent[3]["tool_call_id"] == "c1"


def test_calculate_today_and_unknown_tools(telp):
    def answer(req):
        if not _tool_msgs(req):
            return {"tool_calls": [
                {"id": "a", "name": "calculate",
                 "arguments": {"expression": "1642 - 1564"}},
                {"id": "b", "name": "today", "arguments": {}},
                {"id": "c", "name": "calculate",
                 "arguments": {"expression": "import os"}},
                {"id": "d", "name": "browse_web",
                 "arguments": {"url": "x"}}]}
        return "He lived 78 years [1]."

    with FakeLLM(answer) as llm:
        h = _harness(telp, llm)
        turn = h.ask("how long did Galileo live?")
    results = [m["content"] for m in _tool_msgs(llm.requests[1])]
    assert results[0].startswith("[1] 1642 - 1564 = 78 (tool:calculate")
    assert results[1].startswith("[2] Today is Thursday, 8 October 2026")
    assert "couldn't calculate" in results[2]
    assert "no tool called 'browse_web'" in results[3]
    assert [e.text for e in turn.evidence][:2] == [
        "1642 - 1564 = 78",
        "Today is Thursday, 8 October 2026; the local time is 14:05."]


def test_tool_rounds_stop_at_the_limit_and_the_last_call_must_answer(telp):
    def answer(req):
        if req.get("tools"):
            return {"tool_calls": [{"name": "search_memory",
                                    "arguments": {"query": "more"}}]}
        return "I couldn't find more than this."

    with FakeLLM(answer) as llm:
        h = _harness(telp, llm, max_tool_rounds=2)
        turn = h.ask("tell me everything about Quendia")
    assert len(llm.requests) == 3 and turn.rounds == 3
    assert "tools" in llm.requests[0] and "tools" in llm.requests[1]
    assert "tools" not in llm.requests[2]
    assert turn.tools_used == ["search_memory", "search_memory"]
    assert "couldn't find more" in turn.answer


class _NoToolsServer:
    """An OpenAI-compatible server without function calling: a request
    that carries tools gets HTTP 400, as such servers answer."""

    def __init__(self, answer: str):
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(n) or b"{}")
                outer.requests.append(req)
                if req.get("tools"):
                    code, body = 400, {"error": {"message":
                                       "This model does not support tools."}}
                else:
                    code, body = 200, {"choices": [{"index": 0, "message": {
                        "role": "assistant", "content": answer},
                        "finish_reason": "stop"}]}
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        threading.Thread(target=self._server.serve_forever,
                         daemon=True).start()

    def close(self):
        self._server.shutdown()
        self._server.server_close()


def test_a_server_that_refuses_tools_is_asked_again_without_them(telp):
    srv = _NoToolsServer("Reykjavik, as far as my sources go.")
    try:
        h = _harness(telp, srv.url)
        turn = h.ask("what is the capital of Iceland?")
        assert turn.handled_by == "llm" and "Reykjavik" in turn.answer
        assert turn.rounds == 1
        assert any("doesn't accept tools" in n for n in turn.notes)
        assert [bool(r.get("tools")) for r in srv.requests] == [True, False]
        h.ask("and of Norway?")                  # not offered again
        assert not srv.requests[-1].get("tools")
        assert len(srv.requests) == 3
    finally:
        srv.close()


def test_get_facts_is_offered_only_when_the_fact_layer_works(telp):
    _teach_world(telp)
    with FakeLLM(lambda req: "Noted.") as llm:
        broken = SimpleNamespace(sync=lambda: 1 / 0)
        assert _harness(telp, llm, fact_mind=broken).fact_mind is None
        assert _harness(telp, llm, fact_mind=None).tools() == list(BASE_TOOLS)
    try:
        from mind.fact_mind import FactMind
        fm = FactMind(telp.agent)
    except Exception as err:            # the fact layer is optional
        pytest.skip(f"fact layer unavailable: {err}")

    def answer(req):
        if not _tool_msgs(req):
            return {"tool_calls": [{"name": "get_facts", "arguments":
                                    {"entity": "Galileo Galilei"}}]}
        return "Noted."

    with FakeLLM(answer) as llm:
        h = _harness(telp, llm, fact_mind=fm)
        if h.fact_mind is None:
            pytest.skip("fact layer present but not working")
        h.ask("tell me about Galileo")
    names = [t["function"]["name"] for t in llm.requests[0]["tools"]]
    assert names[-1] == "get_facts"
    facts = _tool_msgs(llm.requests[1])[0]["content"]
    assert "Pisa" in facts or "no more stored facts" in facts


# ─── 5. checking, and what is (never) remembered ────────────────────

def test_an_unsupported_sentence_is_marked_and_not_kept(telp):
    _teach_world(telp)

    def answer(req):
        n = _num(_user(req), "Reykjavik is the capital")
        return (f"Reykjavik is the capital of Iceland [{n}]. "
                f"It has nine million residents.")

    with FakeLLM(answer) as llm:
        h = _harness(telp, llm)
        turn = h.ask("what is the capital of Iceland?")
        assert [c.status for c in turn.checks] == ["supported",
                                                   "unsupported"]
        assert "nine million residents. [not in my sources]" in turn.answer
        lines = check_lines(turn)
        assert lines[0] == ("Telp's check: not backed by the sources -> "
                            "\"It has nine million residents.\"")
        assert lines[1].startswith("Sources: [1] wikipedia:Iceland, 2026-")
        # the slip is not carried into the conversation memory
        kept = h.state.turns()[-1]["answer"]
        assert "Reykjavik" in kept and "nine million" not in kept
        why = h.ask("how do you know that?")
        assert "Careful: 1 sentence" in why.answer


def test_the_real_checker_marks_an_unsupported_sentence(telp):
    checker = pytest.importorskip("mind.checker")
    _teach_world(telp)

    def answer(req):
        n = _num(_user(req), "Reykjavik is the capital")
        return (f"Reykjavik is the capital of Iceland [{n}]. "
                f"Iceland has a population of nine million people.")

    with FakeLLM(answer) as llm:
        h = _harness(telp, llm, checker=checker)
        turn = h.ask("what is the capital of Iceland?")
    assert turn.checks and turn.checks[0].status == "supported"
    assert any(c.status == "unsupported" for c in turn.checks)
    assert turn.answer != turn.raw_answer


def test_unchecked_answers_say_so(telp):
    with FakeLLM(lambda req: "Something.") as llm:
        off = _harness(telp, llm, check=False).ask("anything?")
        assert off.checks == [] and off.answer == "Something."
        missing = _harness(telp, llm, checker=False).ask("anything?")
        assert any("unchecked" in n for n in missing.notes)


def test_the_models_prose_never_becomes_a_memory_or_a_fact(telp):
    _teach_world(telp)
    prose = "Zanzibar Quuxworth invented the teleporter in 1802 [1]."
    with FakeLLM(lambda req: prose) as llm:
        h = _harness(telp, llm)
        mems = telp.agent.lattice.count()
        facts = telp.user_facts.count()
        h.ask("who invented the teleporter?")
        h.ask("tell me more about him")
    assert telp.agent.lattice.count() == mems
    assert telp.user_facts.count() == facts
    assert not any("Quuxworth" in t for t in telp.agent.lattice._texts)


# ─── 6. when the model can't answer ─────────────────────────────────

def test_a_dead_server_gets_a_how_to_and_telps_own_answer(telp):
    telp.agent.lattice.add("The Zorblax river flows through Quendia.",
                           source="user_taught")
    h = _harness(telp, f"http://127.0.0.1:{_free_port()}/v1")
    turn = h.ask("where does the Zorblax river flow?")
    assert turn.fallback and turn.handled_by == "telp"
    assert "LM Studio" in turn.answer and "lms server start" in turn.answer
    assert "Telp's own answer, without the model" in turn.answer
    assert "Quendia" in turn.answer
    assert "Telp answered on his own" in meter_line(turn)
    why = h.ask("how do you know that?")
    assert "user_taught" in why.answer


def test_a_model_that_thinks_until_it_runs_out_is_asked_without_thinking(telp):
    def answer(req):
        if "/no_think" not in _user(req):
            return {"choices": [{"index": 0, "finish_reason": "length",
                                 "message": {"role": "assistant",
                                             "content": "<think>hmm " * 40}}]}
        return "I have no sources on that."

    with FakeLLM(answer) as llm:
        h = _harness(telp, llm)
        turn = h.ask("what is the meaning of Quendia?")
    assert turn.rounds == 2 and len(llm.requests) == 2
    assert turn.raw_answer == "I have no sources on that."
    assert any("thinking switched off" in n for n in turn.notes)


def test_no_think_and_streaming(telp):
    _teach_world(telp)
    with FakeLLM(_answer_capital) as llm:
        h = _harness(telp, llm, thinking=False)
        pieces = []
        turn = h.ask("what is the capital of Iceland?", stream=True,
                     on_token=pieces.append)
        assert _user(llm.requests[0]).endswith("/no_think")
        assert llm.requests[0]["stream"] is True
        assert "".join(pieces) == turn.raw_answer
        assert turn.raw_answer.startswith("Reykjavik is the capital")
        # an answer Telp gives himself reaches a streaming caller too
        pieces.clear()
        why = h.ask("how do you know that?", stream=True,
                    on_token=pieces.append)
        assert pieces == [why.answer] and len(llm.requests) == 1


def test_the_meter_uses_the_servers_own_counts_when_it_gives_them(telp):
    usage = {"prompt_tokens": 812, "completion_tokens": 20,
             "total_tokens": 832,
             "prompt_tokens_details": {"cached_tokens": 640}}
    with FakeLLM(lambda req: {"content": "I don't know.",
                              "usage": usage}) as llm:
        turn = _harness(telp, llm).ask("what is Quendia?")
    assert turn.measured and turn.brief_tokens == 812
    line = meter_line(turn)
    assert line.startswith("Telp sent 812 tokens (a send-everything prompt")
    assert "640 read from the model's cache" in line


def test_meter_sources_and_status_text(telp):
    _teach_world(telp)
    with FakeLLM(_answer_capital) as llm:
        h = _harness(telp, llm)
        turn = h.ask("what is the capital of Iceland?")
        line = meter_line(turn)
        assert line.startswith(f"Telp sent ~{turn.brief_tokens:,} tokens "
                               f"(a send-everything prompt: ~")
        assert "1 model call" in line
        listed = sources_text(turn)
        assert "Reykjavik is the capital" in listed and "(cited)" in listed
        info = LLMClient(base_url=llm.url, model=MODEL).status()
    lines = status_lines(info)
    assert any("reachable" in ln for ln in lines)
    assert any("32,768 tokens loaded" in ln for ln in lines)
    assert any(ln.startswith("brief:") for ln in lines)
    down = LLMClient(base_url=f"http://127.0.0.1:{_free_port()}/v1").status()
    assert any("not reachable" in ln for ln in status_lines(down))


def test_empty_message(telp):
    with FakeLLM(lambda req: "x") as llm:
        turn = _harness(telp, llm).ask("   ")
        assert turn.handled_by == "telp" and llm.requests == []


# ─── the command line ───────────────────────────────────────────────

_RUNNER = ("import sys; sys.path.insert(0, sys.argv.pop(1)); "
           "from tests.fake_minilm import install; install(); "
           "import runpy; sys.argv[0] = 'telp.py'; "
           "runpy.run_path(sys.argv[0], run_name='__main__')")


def _telp_cli(state, *args, stdin=None, timeout=120):
    """telp.py in a subprocess with the stand-in sentence model and every
    proxy pointing at a dead port (local model servers bypass proxies)."""
    dead = "http://127.0.0.1:9"
    env = dict(os.environ, TELP_STATE_DIR=str(state),
               TELP_PORT=str(_free_port()), PYTHONDONTWRITEBYTECODE="1")
    for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy",
              "ALL_PROXY", "all_proxy"):
        env[k] = dead
    env["NO_PROXY"] = env["no_proxy"] = ""
    env.pop("TELP_LLM_URL", None)
    env.pop("TELP_LLM_MODEL", None)
    return subprocess.run(
        [sys.executable, "-c", _RUNNER, str(ROOT), *args], cwd=ROOT,
        env=env, input=stdin, capture_output=True, text=True,
        timeout=timeout)


def _answer_zorblax(req):
    user = _user(req)
    m = re.search(r"\[(\d+)\] The Zorblax river", user)
    if not m:
        return "I don't have that in my sources."
    return f"The Zorblax river flows through Quendia [{m.group(1)}]."


def test_cli_llm_answers_with_a_meter(tmp_path):
    r = _telp_cli(tmp_path, "teach", "The Zorblax river flows through Quendia.")
    assert r.returncode == 0, r.stderr
    with FakeLLM(_answer_zorblax) as llm:
        r = _telp_cli(tmp_path, "llm", "where does the Zorblax river flow?",
                      "--url", llm.url, "--no-think")
        assert r.returncode == 0, r.stderr
        assert "flows through Quendia [" in r.stdout
        assert re.search(r"Telp sent ~[\d,]+ tokens \(a send-everything "
                         r"prompt: ~[\d,]+\) · 1 model call · [\d.]+s",
                         r.stdout), r.stdout
        assert len(llm.requests) == 1
        assert "Quendia" in _user(llm.requests[0])
        assert _user(llm.requests[0]).endswith("/no_think")

        # the conversation lives in the memory file: a new process can cite
        r = _telp_cli(tmp_path, "llm", "how do you know that?",
                      "--url", llm.url)
        assert r.returncode == 0, r.stderr
        assert "user_taught" in r.stdout and "no model call" in r.stdout
        assert len(llm.requests) == 1

        r = _telp_cli(tmp_path, "llm", "where does the Zorblax river flow?",
                      "--url", llm.url, "--stream")
        assert r.returncode == 0, r.stderr
        assert "flows through Quendia [" in r.stdout
        assert llm.requests[-1]["stream"] is True


def test_cli_llm_with_the_server_down_falls_back(tmp_path):
    _telp_cli(tmp_path, "teach", "The Zorblax river flows through Quendia.")
    r = _telp_cli(tmp_path, "llm", "where does the Zorblax river flow?",
                  "--url", f"http://127.0.0.1:{_free_port()}/v1")
    assert r.returncode == 0, r.stderr
    assert "LM Studio" in r.stdout and "lms server start" in r.stdout
    assert "Telp's own answer, without the model" in r.stdout
    assert "Quendia" in r.stdout
    assert "Telp answered on his own" in r.stdout


def test_cli_llm_chat_repl(tmp_path):
    _telp_cli(tmp_path, "teach", "The Zorblax river flows through Quendia.")
    script = "\n".join(["where does the Zorblax river flow?", "/sources",
                        "/meter", "how do you know that?", "/new",
                        "how do you know that?", "/quit"]) + "\n"
    with FakeLLM(_answer_zorblax) as llm:
        r = _telp_cli(tmp_path, "llm-chat", "--url", llm.url, stdin=script)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "flows through Quendia [" in out
    assert re.search(r"\[\d+\] The Zorblax river flows through Quendia\. "
                     r"\(user_taught, 2026-\d\d-\d\d\)\s+\[memory\] \(cited\)",
                     out) or "(user_taught" in out, out
    assert "This session: 1 model turn(s)" in out
    assert "I can show you exactly" in out
    assert "fresh conversation (1 earlier turn(s) forgotten)" in out
    assert "haven't asked me anything yet" in out
    assert len(llm.requests) == 1


def test_cli_llm_status(tmp_path):
    with FakeLLM(lambda req: "x", context_length=8192) as llm:
        r = _telp_cli(tmp_path, "llm-status", "--url", llm.url)
    assert r.returncode == 0, r.stderr
    assert "reachable" in r.stdout and MODEL in r.stdout
    assert "8,192 tokens loaded" in r.stdout
    assert "warning:" in r.stdout              # 8192 is small for thinking
    r = _telp_cli(tmp_path, "llm-status", "--url",
                  f"http://127.0.0.1:{_free_port()}/v1")
    assert r.returncode == 1
    assert "not reachable" in r.stdout and "LM Studio" in r.stdout
