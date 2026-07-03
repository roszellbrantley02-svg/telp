"""
lattice/eval_standalone.py - hard evaluation harness for the standalone agent.

Loads the 100-article Wikipedia corpus into a fresh StandaloneAgent,
runs a labeled Q&A set, and scores three axes:

  * topic_match   - did the answer come from the expected article?
  * entity_match  - did the answer mention the expected answer entity?
  * word_overlap  - fraction of the expected answer's key words present
                    in the agent's response

Outputs a per-question table and aggregate accuracy.  No LLM judge —
checks are mechanical so they are reproducible.

Usage:
    python -m lattice.eval_standalone
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
from pathlib import Path

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    # Only wrap if not already utf-8 — avoid orphaning an outer wrapper
    # that a caller (e.g. an ingest CLI) already installed, which would
    # close the underlying buffer when GC'd.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace")
_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.standalone_agent import StandaloneAgent


CORPUS_PATH = _TELP_ROOT / "state" / "wiki_corpus.json"
EVAL_DB     = _TELP_ROOT / "state" / "eval_standalone.db"


# ─── Q&A set ──────────────────────────────────────────────────────
# (question, expected_topic, expected_entity_or_keyword)
# expected_entity_or_keyword is checked case-insensitively in the
# agent's response.  Multiple acceptable words separated by "|".

QA_SET = [
    # Capitals
    ("What is the capital of Germany?",     "Germany",       "Berlin"),
    ("What is the capital of France?",      "France",        "Paris"),
    ("What is the capital of Japan?",       "Japan",         "Tokyo"),
    ("What is the capital of Italy?",       "Italy",         "Rome"),
    ("What is the capital of Spain?",       "Spain",         "Madrid"),
    ("What is the capital of Egypt?",       "Egypt",         "Cairo"),
    ("What is the capital of Australia?",   "Australia",     "Canberra"),
    ("What is the capital of the UK?",      "United_Kingdom","London"),
    # Scientists
    ("Who developed the theory of relativity?", "Albert_Einstein", "Einstein|relativity"),
    ("Who discovered evolution?",           "Charles_Darwin",  "Darwin|evolution"),
    ("Who was Marie Curie?",                "Marie_Curie",     "Curie|physicist|chemist"),
    ("What did Isaac Newton do?",           "Isaac_Newton",    "Newton|mathematician|physicist|gravity"),
    ("Who was Nikola Tesla?",               "Nikola_Tesla",    "Tesla|engineer|inventor"),
    ("Who was Alan Turing?",                "Alan_Turing",     "Turing|computer|mathematician"),
    # Authors
    ("Who wrote Hamlet?",                   "William_Shakespeare", "Shakespeare|playwright"),
    ("Who was Jane Austen?",                "Jane_Austen",         "Austen|novelist|english"),
    ("Who was Leo Tolstoy?",                "Leo_Tolstoy",         "Tolstoy|russian|writer|novelist"),
    ("Who was George Orwell?",              "George_Orwell",       "Orwell|english|writer|novelist"),
    # Composers
    ("Who composed the Brandenburg Concertos?", "Johann_Sebastian_Bach", "Bach|composer|german"),
    ("Who was Mozart?",                     "Wolfgang_Amadeus_Mozart","Mozart|composer|austrian"),
    ("Who was Beethoven?",                  "Ludwig_van_Beethoven",   "Beethoven|composer|german"),
    # Things
    ("What is the Internet?",               "Internet",       "internet|network"),
    ("What is DNA?",                        "DNA",            "dna|molecule|genetic"),
    ("Who invented the telephone?",         "Telephone",      "telephone|bell"),
    # Planets
    ("What is Jupiter?",                    "Jupiter",        "jupiter|planet"),
    ("What is Saturn?",                     "Saturn",         "saturn|planet"),
    ("Tell me about Mars",                  "Mars",           "mars|planet"),
    ("What is a black hole?",               "Black_hole",     "black hole|gravity"),
    # Animals
    ("What is a lion?",                     "Lion",           "lion|cat"),
    ("What is an octopus?",                 "Octopus",        "octopus|mollusc|cephalopod"),
]


# ─── Scoring ──────────────────────────────────────────────────────


_WORD_RE = re.compile(r"[A-Za-z][\w-]*")


def topic_match(agent_response: str, agent_turns: list[dict],
                  expected_topic: str, lattice) -> bool:
    """Did the top retrieved memory come from `wikipedia:<expected_topic>`?"""
    if not agent_turns:
        return False
    last = agent_turns[-1]
    retrieved = last.get("retrieved_memories", [])
    if not retrieved:
        return False
    # Match the first retrieved memory back to its source in the lattice.
    top_text = retrieved[0]
    for text, source in zip(lattice._texts, lattice._sources):
        if text == top_text and f"wikipedia:{expected_topic}" in source:
            return True
    return False


def entity_match(response: str, expected_words: str) -> bool:
    """Did the response contain any of the expected key words?"""
    response_lo = response.lower()
    for opt in expected_words.split("|"):
        opt = opt.strip().lower()
        if opt and opt in response_lo:
            return True
    return False


def word_overlap(response: str, expected_words: str) -> float:
    """Fraction of expected key words that appear in the response."""
    options = [o.strip().lower() for o in expected_words.split("|") if o.strip()]
    if not options:
        return 0.0
    rl = response.lower()
    hits = sum(1 for o in options if o in rl)
    return hits / len(options)


# ─── Ingestion ───────────────────────────────────────────────────


def load_corpus(agent: StandaloneAgent) -> int:
    if not CORPUS_PATH.exists():
        print(f"[eval] no corpus at {CORPUS_PATH}; run lattice.fetch_wiki first.")
        return 0
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    n = 0
    for entry in corpus:
        if "extract" not in entry or not entry["extract"]:
            continue
        topic = entry["topic"]
        text = entry["extract"]
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        for s in sentences:
            agent.encoder.add_sentence(s)
        for s in sentences:
            agent.lattice.add(s, source=f"wikipedia:{topic}",
                                  wiki_title=entry.get("title", topic))
            n += 1
    # Build KG + generator + structured-QA claim store after all ingestion.
    n_pat, n_facts = agent._rebuild_kg_from_corpus()
    n_sent = agent._retrain_predictor()
    n_claims = agent._rebuild_structured_qa()
    print(f"[eval] ingested {n} sentences, {n_pat} patterns, "
          f"{n_facts} KG triples, {n_claims} structured claims, "
          f"predictor on {n_sent} sentences")
    return n


# ─── Main ────────────────────────────────────────────────────────


def main():
    if EVAL_DB.exists():
        EVAL_DB.unlink()

    print(f"[eval] standalone HDC agent — LLM-free Q&A benchmark")
    print(f"[eval] {len(QA_SET)} questions, ground-truth scoring\n")

    agent = StandaloneAgent(lattice_path=EVAL_DB)
    t0 = time.perf_counter()
    n_sent = load_corpus(agent)
    print(f"[eval] corpus load: {time.perf_counter()-t0:.1f}s\n")
    print(f"[eval] {agent.stats()}\n")

    n_topic = n_entity = 0
    total_overlap = 0.0
    rows = []
    t_total = 0.0

    for q, expected_topic, expected_words in QA_SET:
        t0 = time.perf_counter()
        resp = agent.respond(q)
        dt = (time.perf_counter() - t0) * 1000
        t_total += dt

        tm = topic_match(resp, agent.turns, expected_topic, agent.lattice)
        em = entity_match(resp, expected_words)
        wo = word_overlap(resp, expected_words)

        if tm: n_topic += 1
        if em: n_entity += 1
        total_overlap += wo

        marks = ("T" if tm else "-") + ("E" if em else "-")
        rows.append((marks, dt, q, resp[:90]))

    print("─" * 100)
    print("PER-QUESTION RESULTS")
    print("─" * 100)
    for marks, dt, q, resp in rows:
        print(f"  [{marks}] {dt:5.0f}ms  {q[:48]:48s}  -> {resp}")

    n = len(QA_SET)
    print()
    print("=" * 100)
    print("AGGREGATE")
    print("=" * 100)
    print(f"  topic_match (correct article retrieved): {n_topic}/{n} = {n_topic/n*100:.1f}%")
    print(f"  entity_match (key word in response):      {n_entity}/{n} = {n_entity/n*100:.1f}%")
    print(f"  mean word_overlap:                        {total_overlap/n*100:.1f}%")
    print(f"  mean latency:                             {t_total/n:.0f} ms/query")


if __name__ == "__main__":
    main()
