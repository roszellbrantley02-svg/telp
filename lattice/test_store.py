"""
lattice/test_store.py - end-to-end Lattice demonstration.

Five tests, each load-bearing:

  1. ADD + COUNT: add 20 memories, count matches.
  2. QUERY: paraphrased queries find the right memory (top-1).
  3. PERSISTENCE: close and reopen the DB, all memories survive.
  4. COMPOSE: bundling multiple memories produces a hypervector that
     retrieves themes from each constituent.
  5. COUNTERFACTUAL: surgical swap of a concept in a memory shifts the
     retrieval toward the swapped concept.

If all five pass, the Lattice MVP is real: you can talk to it, it
remembers, it can do algebra on its memories.

Usage:
    python -m lattice.test_store
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.store import Lattice
from lattice.text_encoder import TextEncoder


# Use a temp DB so we don't pollute any real one
_TEST_DB = _TELP_ROOT / "state" / "lattice_test.db"


def fresh_lattice(encoder=None) -> Lattice:
    if _TEST_DB.exists():
        _TEST_DB.unlink()
    return Lattice(_TEST_DB, encoder=encoder)


SEED_MEMORIES = [
    ("I had coffee with Sarah on Monday morning.",         "social"),
    ("The dog chased a squirrel through the yard.",        "pets"),
    ("My presentation to the board went well yesterday.",  "work"),
    ("She finally finished her novel after three years.",  "creative"),
    ("The car needs an oil change next week.",             "maintenance"),
    ("We hiked up the mountain at sunrise.",               "outdoors"),
    ("He learned to play guitar during the pandemic.",     "creative"),
    ("The bakery on Main Street has the best bread.",      "food"),
    ("I bought a new pair of running shoes online.",       "shopping"),
    ("The thunderstorm knocked out power for six hours.",  "weather"),
    ("My grandmother's recipe is the best in the family.", "food"),
    ("The new restaurant downtown is way too expensive.",  "food"),
    ("We adopted a kitten from the shelter last spring.",  "pets"),
    ("The job interview lasted longer than expected.",     "work"),
    ("She gave an incredible speech at her wedding.",      "social"),
    ("I lost my keys somewhere in the park.",              "minor"),
    ("The plane was delayed by three hours due to weather.","travel"),
    ("My back has been hurting since I moved that couch.", "health"),
    ("We watched the sunset over the ocean together.",     "romance"),
    ("He proposed during their trip to Paris last June.",  "romance"),
]


# ─── Test 1: add + count ──────────────────────────────────────────


def test_add_count(L: Lattice) -> bool:
    print("\n=== TEST 1: add + count ===")
    for text, src in SEED_MEMORIES:
        L.add(text, source=src)
    n = L.count()
    print(f"  added {len(SEED_MEMORIES)}; count() = {n}")
    return n == len(SEED_MEMORIES)


# ─── Test 2: paraphrased query ────────────────────────────────────


PARAPHRASE_QUERIES = [
    ("Sarah and I got coffee Monday AM.",                 SEED_MEMORIES[0][0]),
    ("A squirrel was pursued by our dog.",                SEED_MEMORIES[1][0]),
    ("Yesterday's board meeting went smoothly.",          SEED_MEMORIES[2][0]),
    ("After three years of work, she completed her book.",SEED_MEMORIES[3][0]),
    ("My car needs maintenance soon.",                    SEED_MEMORIES[4][0]),
    ("We climbed the mountain at dawn.",                  SEED_MEMORIES[5][0]),
    ("During COVID, he picked up guitar.",                SEED_MEMORIES[6][0]),
    ("Best bakery in town is on Main Street.",            SEED_MEMORIES[7][0]),
    ("Six-hour blackout from the storm.",                 SEED_MEMORIES[9][0]),
    ("Got a kitten from the animal shelter in spring.",   SEED_MEMORIES[12][0]),
    ("Her wedding speech was amazing.",                   SEED_MEMORIES[14][0]),
    ("My keys are missing in the park.",                  SEED_MEMORIES[15][0]),
    ("The flight was three hours late.",                  SEED_MEMORIES[16][0]),
    ("He proposed in Paris last summer.",                 SEED_MEMORIES[19][0]),
]


def test_query(L: Lattice) -> bool:
    print("\n=== TEST 2: paraphrased query (top-1 match) ===")
    correct = 0
    for q, target in PARAPHRASE_QUERIES:
        results = L.query(q, k=3)
        if not results:
            print(f"  [MISS] '{q[:40]}' -> no results")
            continue
        top = results[0]
        ok = (top["text"] == target)
        correct += int(ok)
        mark = "OK" if ok else "MISS"
        print(f"  [{mark}] '{q[:40]:40s}' -> top1={top['distance_pct']*100:.0f}% '{top['text'][:40]}'")
    n = len(PARAPHRASE_QUERIES)
    print(f"\n  Top-1 accuracy: {correct}/{n} = {correct/n:.0%}")
    return correct / n >= 0.85


# ─── Test 3: persistence ──────────────────────────────────────────


def test_persistence(encoder: TextEncoder) -> bool:
    """Close the Lattice, reopen from disk, confirm memories survived."""
    print("\n=== TEST 3: persistence across close/reopen ===")
    # Open a new Lattice on the same DB; should re-read all rows
    L2 = Lattice(_TEST_DB, encoder=encoder)
    n2 = L2.count()
    print(f"  reopened DB: count() = {n2} (expected {len(SEED_MEMORIES)})")
    if n2 != len(SEED_MEMORIES):
        return False
    # Re-run a single query to verify the in-memory stack rebuilt correctly
    results = L2.query("Sarah Monday coffee", k=1)
    print(f"  spot-check query '{'Sarah Monday coffee':30s}' -> "
          f"'{results[0]['text'][:50]}'")
    ok = "Sarah" in results[0]["text"]
    L2.close()
    return ok


# ─── Test 4: composition ──────────────────────────────────────────


def test_composition(L: Lattice) -> bool:
    """Bundle two memories; retrieving with the bundled vector should
    surface both source memories near the top of the ranking."""
    print("\n=== TEST 4: composition (bundle 2 memories -> retrieve both) ===")
    a = SEED_MEMORIES[0][0]   # coffee with Sarah
    b = SEED_MEMORIES[10][0]  # grandmother's recipe
    composite = L.compose(a, b)
    results = L.query_vector(composite, k=5)
    texts = [r["text"] for r in results]
    a_rank = next((r["rank"] for r in results if r["text"] == a), 999)
    b_rank = next((r["rank"] for r in results if r["text"] == b), 999)
    print(f"  composed = bundle('{a[:35]}...', '{b[:35]}...')")
    print(f"  top 3 results:")
    for r in results[:3]:
        print(f"    #{r['rank']} d={r['distance_pct']*100:.0f}% '{r['text'][:55]}'")
    print(f"  A rank: {a_rank}   B rank: {b_rank}")
    # Both source memories should be in the top 5
    return a_rank <= 5 and b_rank <= 5


# ─── Test 5: counterfactual swap ──────────────────────────────────


def test_counterfactual(L: Lattice) -> bool:
    """Take a coffee memory, swap 'coffee' for 'tea', verify the
    resulting vector retrieves toward tea/drink-related items more
    than toward the original."""
    print("\n=== TEST 5: counterfactual (swap concept in a memory) ===")
    # Find the coffee memory id
    target_text = SEED_MEMORIES[0][0]  # "I had coffee with Sarah on Monday morning."
    coffee_mem = next(r for r in L._con.execute(
        "SELECT id FROM memories WHERE text=?", (target_text,)
    ).fetchall())
    coffee_id = coffee_mem[0]
    print(f"  source memory #{coffee_id}: '{target_text}'")

    # Apply counterfactual: replace 'coffee' with 'tea'
    cf_hv = L.counterfactual(coffee_id, swap={"coffee": "tea"})
    # Add some tea-related distractor memories to give the swap something
    # to retrieve. Without these, all neighbors are about non-tea things.
    tea_mem_id = L.add("I had tea with Sarah on Tuesday afternoon.",
                         source="social")

    results = L.query_vector(cf_hv, k=5)
    print(f"  top 5 after swap(coffee -> tea):")
    for r in results:
        marker = "(NEW tea memory)" if r["id"] == tea_mem_id else ""
        marker = "(original coffee)" if r["text"] == target_text else marker
        print(f"    #{r['rank']} d={r['distance_pct']*100:.0f}% "
              f"'{r['text'][:50]}' {marker}")
    # Pass if the tea memory ranks above the original coffee memory
    tea_rank    = next((r["rank"] for r in results if r["id"] == tea_mem_id), 999)
    coffee_rank = next((r["rank"] for r in results if r["text"] == target_text), 999)
    print(f"  Tea-memory rank: {tea_rank}   Coffee-memory rank: {coffee_rank}")
    return tea_rank <= coffee_rank


# ─── Main ──────────────────────────────────────────────────────────


def main():
    print("Lattice store - end-to-end MVP demonstration\n")

    # Shared encoder so we only load MiniLM once
    enc = TextEncoder()

    L = fresh_lattice(encoder=enc)
    results = {}
    results["1. add + count"]          = test_add_count(L)
    results["2. paraphrased query"]    = test_query(L)
    L.close()
    results["3. persistence"]          = test_persistence(enc)
    L = Lattice(_TEST_DB, encoder=enc)   # reopen
    results["4. composition"]          = test_composition(L)
    results["5. counterfactual"]       = test_counterfactual(L)
    L.close()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    all_pass = all(results.values())
    print()
    if all_pass:
        print("  ALL TESTS PASS - the Lattice MVP works.")
        print("  You can store memories, retrieve them by meaning, persist")
        print("  across sessions, and manipulate them with HD algebra.")
    else:
        n_pass = sum(results.values())
        print(f"  {n_pass}/{len(results)} pass.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
