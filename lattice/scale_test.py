"""
lattice/scale_test.py - measure system behavior at corpus scale.

Loads whatever's in state/wiki_corpus.json (now potentially 1000+
articles), times every ingestion phase, samples query latency, and
reports memory footprint. Useful as a "what breaks at scale" probe.
"""
from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

# Importing load_corpus pulls in eval_standalone which sets up utf-8
# stdout for us — don't double-wrap.

from lattice.standalone_agent import StandaloneAgent
from lattice.eval_standalone import load_corpus

EVAL_DB = _TELP_ROOT / "state" / "scale_test.db"


def main():
    if EVAL_DB.exists():
        EVAL_DB.unlink()

    tracemalloc.start()

    print("=" * 70)
    print("SCALE TEST")
    print("=" * 70)
    # 1. Corpus stats
    corpus = json.loads(
        (_TELP_ROOT / "state" / "wiki_corpus.json").read_text(encoding="utf-8")
    )
    print(f"  corpus topics: {len(corpus)}")
    nonempty = [e for e in corpus if e.get("extract")]
    total_chars = sum(len(e["extract"]) for e in nonempty)
    print(f"  topics with content: {len(nonempty)}")
    print(f"  total text chars: {total_chars:,} (~{total_chars//5:,} words)")
    print()

    # 2. Agent build (encoder + lattice)
    t0 = time.perf_counter()
    agent = StandaloneAgent(lattice_path=EVAL_DB)
    t_init = time.perf_counter() - t0
    print(f"  agent init: {t_init:.2f}s")

    # 3. Corpus ingest (lattice add + structured-QA rebuild + seq predictor)
    t0 = time.perf_counter()
    n = load_corpus(agent)
    t_ingest = time.perf_counter() - t0
    print(f"  corpus ingest: {t_ingest:.2f}s ({n} sentences, "
          f"{n/t_ingest:.0f} sent/s)")
    print()

    # 4. Final-state stats
    cur, peak = tracemalloc.get_traced_memory()
    print(f"  memory (current):    {cur / 1024 / 1024:.1f} MB")
    print(f"  memory (peak):       {peak / 1024 / 1024:.1f} MB")
    print(f"  lattice size:        {agent.lattice.count()} memories")
    print(f"  encoder vocab:       {len(agent.encoder.index_vectors)}")
    print(f"  structured claims:   {len(agent.structured.claim_triple)}")
    print(f"  KG facts:            {len(agent.knowledge.facts)}")
    print()

    # 5. Query latency
    test_qs = [
        "What is the capital of Germany?",
        "Who was Einstein?",
        "Where was Mozart born?",
        "What is the capital of the country where Bach was born?",
        "List all the composers",
        "How many countries do you know?",
        "Who was born in Germany?",
        "When was Einstein born?",
        "Who invented the telephone?",
        "Tell me about Mars",
    ]
    print("  Query latency sample:")
    latencies: list[float] = []
    for q in test_qs:
        t0 = time.perf_counter()
        resp = agent.respond(q)
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)
        print(f"    {dt:5.0f}ms  {q[:50]:50s}  -> {resp[:60]}")
    print()
    print(f"  mean latency:  {sum(latencies)/len(latencies):.0f}ms")
    print(f"  median:        {sorted(latencies)[len(latencies)//2]:.0f}ms")
    print(f"  max:           {max(latencies):.0f}ms")

    tracemalloc.stop()


if __name__ == "__main__":
    main()
