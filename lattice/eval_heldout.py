"""
lattice/eval_heldout.py - HELD-OUT evaluation, never seen during tuning.

The 30 questions below are about entities NOT used in eval_standalone.py.
Same question shapes (capitals, "Who was X", "What is X"), different
entities.  This measures how much we overfit to the tuning eval.

If the held-out score drops sharply versus the tuning eval, we
over-tuned.  If it holds, the heuristic stack generalises.

Usage:
    python -m lattice.eval_heldout
"""
from __future__ import annotations

import sys
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

# Importing eval_standalone re-wraps sys.stdout for utf-8.
from lattice.eval_standalone import (
    StandaloneAgent, load_corpus, topic_match, entity_match, word_overlap,
)
import time


EVAL_DB = _TELP_ROOT / "state" / "eval_heldout.db"


# Held-out: every entity here is NOT in eval_standalone.py.
QA_SET = [
    # Capitals (different countries)
    ("What is the capital of China?",       "China",        "Beijing"),
    ("What is the capital of India?",       "India",        "Delhi|New Delhi"),
    ("What is the capital of Brazil?",      "Brazil",       "Brasília|Brasilia"),
    ("What is the capital of Canada?",      "Canada",       "Ottawa"),
    ("What is the capital of Russia?",      "Russia",       "Moscow"),
    ("What is the capital of South Africa?","South_Africa", "Pretoria|Cape Town"),
    # Scientists (different ones)
    ("Who was Galileo?",                    "Galileo_Galilei",  "Galileo|astronomer|italian"),
    ("Who was Stephen Hawking?",            "Stephen_Hawking",  "Hawking|physicist|cosmologist"),
    ("Who was Richard Feynman?",            "Richard_Feynman",  "Feynman|physicist|american"),
    ("Who was Carl Sagan?",                 "Carl_Sagan",       "Sagan|astronomer|astrophysicist"),
    ("Who was Rosalind Franklin?",          "Rosalind_Franklin","Franklin|chemist|x-ray"),
    ("Who was Niels Bohr?",                 "Niels_Bohr",       "Bohr|physicist|danish"),
    # Authors (different ones)
    ("Who was Charles Dickens?",            "Charles_Dickens",  "Dickens|english|novelist|writer"),
    ("Who was Virginia Woolf?",             "Virginia_Woolf",   "Woolf|english|writer|modernist"),
    ("Who was Franz Kafka?",                "Franz_Kafka",      "Kafka|writer|czech|prague"),
    ("Who was Mark Twain?",                 "Mark_Twain",       "Twain|american|writer|clemens"),
    ("Who was Ernest Hemingway?",           "Ernest_Hemingway", "Hemingway|american|novelist"),
    # Musicians (different ones)
    ("Who was Frédéric Chopin?",            "Frédéric_Chopin",  "Chopin|polish|composer|pianist"),
    ("Who was Bob Dylan?",                  "Bob_Dylan",        "Dylan|singer|songwriter|american"),
    ("Who was David Bowie?",                "David_Bowie",      "Bowie|english|singer|musician"),
    ("Who was Jimi Hendrix?",               "Jimi_Hendrix",     "Hendrix|guitarist|american|musician"),
    # Things (different)
    ("What is a computer?",                 "Computer",         "computer|machine|electronic|programmable"),
    ("What is the Sun?",                    "Sun",              "sun|star|solar"),
    ("What is the Moon?",                   "Moon",             "moon|earth|satellite|natural"),
    ("What is gold?",                       "Gold",             "gold|metal|chemical|element"),
    ("What is iron?",                       "Iron",             "iron|metal|element"),
    ("What is helium?",                     "Helium",           "helium|gas|element|noble"),
    # Philosophers (different)
    ("Who was Plato?",                      "Plato",            "plato|greek|philosopher"),
    ("Who was Confucius?",                  "Confucius",        "confucius|chinese|philosopher"),
    # Animals (different)
    ("What is a tiger?",                    "Tiger",            "tiger|cat|panthera"),
    ("What is a wolf?",                     "Wolf",             "wolf|canis|canid"),
]


def main():
    if EVAL_DB.exists():
        EVAL_DB.unlink()

    print(f"[heldout] HELD-OUT eval — none of these entities were "
            f"used during tuning")
    print(f"[heldout] {len(QA_SET)} questions\n")

    agent = StandaloneAgent(lattice_path=EVAL_DB)
    t0 = time.perf_counter()
    n_sent = load_corpus(agent)
    print(f"[heldout] corpus load: {time.perf_counter()-t0:.1f}s\n")

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
    print("HELD-OUT AGGREGATE")
    print("=" * 100)
    print(f"  topic_match:    {n_topic}/{n} = {n_topic/n*100:.1f}%")
    print(f"  entity_match:   {n_entity}/{n} = {n_entity/n*100:.1f}%")
    print(f"  word_overlap:   {total_overlap/n*100:.1f}%")
    print(f"  mean latency:   {t_total/n:.0f} ms/query")


if __name__ == "__main__":
    main()
