"""
lattice/eval_paraphrase.py - paraphrase robustness eval.

For each fact in the original 30-question set, ask it 3 different
ways.  Tests whether our regex-based question patterns and content-word
heuristics survive when users phrase questions naturally instead of
in the textbook form.

A robust system answers all 3 paraphrases the same way.  A brittle
system answers the canonical form and fails on the rest.

Usage:
    python -m lattice.eval_paraphrase
"""
from __future__ import annotations

import sys
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.eval_standalone import (
    StandaloneAgent, load_corpus, topic_match, entity_match, word_overlap,
)
import time


EVAL_DB = _TELP_ROOT / "state" / "eval_paraphrase.db"


# Each tuple: (paraphrased question, expected_topic, expected_words).
# The 10 underlying facts are the same as the easier half of
# eval_standalone.py — three phrasings each.
QA_SET = [
    # Germany capital
    ("What is the capital of Germany?",     "Germany",        "Berlin"),
    ("What's Germany's capital?",            "Germany",        "Berlin"),
    ("Tell me Germany's capital city",       "Germany",        "Berlin"),
    # Marie Curie
    ("Who was Marie Curie?",                 "Marie_Curie",    "Curie|physicist|chemist"),
    ("Tell me about Marie Curie",            "Marie_Curie",    "Curie|physicist|chemist"),
    ("What did Marie Curie do?",             "Marie_Curie",    "Curie|physicist|chemist"),
    # Einstein
    ("Who developed the theory of relativity?", "Albert_Einstein", "Einstein|relativity"),
    ("Who came up with relativity?",            "Albert_Einstein", "Einstein|relativity"),
    ("Tell me who created the theory of relativity", "Albert_Einstein", "Einstein|relativity"),
    # Tokyo
    ("What is the capital of Japan?",        "Japan",          "Tokyo"),
    ("Japan's capital is?",                  "Japan",          "Tokyo"),
    ("Which city is Japan's capital?",       "Japan",          "Tokyo"),
    # Newton
    ("Who was Isaac Newton?",                "Isaac_Newton",   "Newton|physicist|mathematician"),
    ("Tell me about Isaac Newton",           "Isaac_Newton",   "Newton|physicist|mathematician"),
    ("What did Newton do?",                  "Isaac_Newton",   "Newton|physicist|mathematician"),
    # Internet
    ("What is the Internet?",                "Internet",       "internet|network"),
    ("Tell me what the internet is",         "Internet",       "internet|network"),
    ("Describe the internet",                "Internet",       "internet|network"),
    # Mars
    ("Tell me about Mars",                   "Mars",           "mars|planet"),
    ("What is Mars?",                        "Mars",           "mars|planet"),
    ("Describe Mars",                        "Mars",           "mars|planet"),
    # Octopus
    ("What is an octopus?",                  "Octopus",        "octopus|mollusc|cephalopod"),
    ("Tell me about octopuses",              "Octopus",        "octopus|mollusc|cephalopod"),
    ("Describe an octopus",                  "Octopus",        "octopus|mollusc|cephalopod"),
    # Bach Brandenburg
    ("Who composed the Brandenburg Concertos?","Johann_Sebastian_Bach","Bach|composer"),
    ("Who wrote the Brandenburg Concertos?",  "Johann_Sebastian_Bach","Bach|composer"),
    ("The Brandenburg Concertos were written by whom?", "Johann_Sebastian_Bach","Bach|composer"),
    # Telephone
    ("Who invented the telephone?",          "Telephone",      "telephone|bell|invented"),
    ("Tell me who made the telephone",       "Telephone",      "telephone|bell"),
    ("The telephone was invented by whom?",  "Telephone",      "telephone|bell"),
]


def main():
    if EVAL_DB.exists():
        EVAL_DB.unlink()

    print(f"[paraphrase] {len(QA_SET)} questions, 3 phrasings of 10 facts\n")
    agent = StandaloneAgent(lattice_path=EVAL_DB)
    t0 = time.perf_counter()
    load_corpus(agent)
    print(f"[paraphrase] corpus load: {time.perf_counter()-t0:.1f}s\n")

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
    last_topic = None
    for marks, dt, q, resp in rows:
        # Group by fact (every 3 rows is one fact).
        print(f"  [{marks}] {dt:5.0f}ms  {q[:50]:50s}  -> {resp}")

    n = len(QA_SET)
    print()
    print("=" * 100)
    print("PARAPHRASE AGGREGATE")
    print("=" * 100)
    print(f"  topic_match:    {n_topic}/{n} = {n_topic/n*100:.1f}%")
    print(f"  entity_match:   {n_entity}/{n} = {n_entity/n*100:.1f}%")
    print(f"  word_overlap:   {total_overlap/n*100:.1f}%")
    print(f"  mean latency:   {t_total/n:.0f} ms/query")


if __name__ == "__main__":
    main()
