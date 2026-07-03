"""
lattice/test_multihop.py - chained inference in pure HDC.

Can RBI derive facts that were never stored, via composition of stored
relations?

Three tests:

  TEST 1: direct query (sanity)
    Germany --IN_CONTINENT--> ?  -> Europe
    (Direct fact present in training. Should be 100%.)

  TEST 2: multi-hop with all facts present
    Germany --BORDERS--> ? --IN_CONTINENT--> ?
    Chain: Germany -> France or Poland -> Europe
    Tests whether the algebra correctly composes two stored relations.

  TEST 3: inferred fact — direct fact REMOVED
    Retrain RBI from the corpus WITHOUT any 'X is in continent' sentences
    for held-out countries. Then query via chain through their borders.
    Tests whether multi-hop derives knowledge the model was never told.

If Test 3 passes, HDC compositional inference is real — it derives
facts from chained algebraic operations on stored facts.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.relation_indexing import RelationBoundEncoder
from lattice.corpus_tagged import build_tagged_corpus
from lattice.corpus import COUNTRIES
from train.v5_hdc_prototype import D


CONTINENTS = ["Europe", "Asia", "Africa", "North America",
                "South America", "Oceania"]
ALL_COUNTRIES = [c[0] for c in COUNTRIES]


# ─── Multi-hop query primitive ────────────────────────────────────


def multihop_query(enc, subject, chain, trace=False):
    """Walk a relation chain. Each hop: cleanup the current entity, then
    query its next relation, recursively.

    chain: list of (relation_name, candidate_list) tuples
    Returns (final_entity, hop_trace)
    """
    current = subject
    trace_log = [{"entity": current, "step": "START"}]
    for relation, candidates in chain:
        results = enc.query(current, relation, candidates, top_k=1)
        if not results:
            trace_log.append({"entity": None, "relation": relation, "step": "DEAD_END"})
            return None, trace_log
        next_entity = results[0][0]
        d = results[0][1]
        trace_log.append({
            "entity": next_entity, "relation": relation,
            "distance": d, "step": "HOP",
        })
        current = next_entity
    return current, trace_log


def print_trace(trace):
    for h in trace:
        if h["step"] == "START":
            print(f"    START: {h['entity']}")
        elif h["step"] == "HOP":
            print(f"    --{h['relation']}--> {h['entity']}  (d={h['distance']})")
        else:
            print(f"    DEAD_END (no result for {h.get('relation')})")


# ─── Build encoders ───────────────────────────────────────────────


def train_full() -> RelationBoundEncoder:
    """Train RBI on the full corpus (all relations present)."""
    enc = RelationBoundEncoder(dim=D, seed=42)
    enc.train(build_tagged_corpus())
    return enc


def train_minus_continent_facts(held_out_countries: list[str]) -> RelationBoundEncoder:
    """Train RBI on the corpus, but FILTER OUT any IN_CONTINENT triples
    where the subject is in held_out_countries. The continent fact for
    those countries is invisible to the encoder.

    Other relations (BORDERS, HAS_CAPITAL, SPEAKS) are kept, so the
    chain Germany -> BORDERS -> France -> IN_CONTINENT -> Europe is
    still possible if the algebra works.
    """
    tagged = build_tagged_corpus()
    filtered = []
    n_removed = 0
    for sent, triples in tagged:
        kept = []
        for t in triples:
            s, r, o = t
            if r == "IN_CONTINENT" and s in held_out_countries:
                n_removed += 1
                continue
            kept.append(t)
        if kept:
            filtered.append((sent, kept))
    enc = RelationBoundEncoder(dim=D, seed=42)
    enc.train(filtered)
    print(f"  Removed {n_removed} IN_CONTINENT triples for "
          f"{held_out_countries} from training.")
    return enc


# ─── Tests ────────────────────────────────────────────────────────


def test_direct(enc):
    print("\n=== TEST 1: direct query (sanity) ===")
    held = [c for c in COUNTRIES[25:]]
    correct = 0
    for country, _cap, cont, *_ in held:
        results = enc.query(country, "IN_CONTINENT", CONTINENTS, top_k=1)
        ok = results and results[0][0] == cont
        correct += int(bool(ok))
        mark = "OK" if ok else "MISS"
        print(f"  [{mark}] {country:14s} -> {results[0][0]} "
              f"(expected {cont})")
    print(f"\n  Direct accuracy: {correct}/{len(held)}")


def test_multihop(enc):
    print("\n=== TEST 2: multi-hop  Germany -- BORDERS --> X -- IN_CONTINENT --> ? ===")
    held = [c for c in COUNTRIES[25:]]
    correct = 0
    for country, _cap, expected_cont, *_ in held:
        chain = [
            ("BORDERS",      ALL_COUNTRIES),
            ("IN_CONTINENT", CONTINENTS),
        ]
        result, trace = multihop_query(enc, country, chain, trace=True)
        ok = (result == expected_cont)
        correct += int(ok)
        mark = "OK" if ok else "MISS"
        print(f"  [{mark}] {country:14s} -> ... -> {result}  (expected {expected_cont})")
        if not ok:
            print_trace(trace)
    print(f"\n  Multi-hop accuracy: {correct}/{len(held)}")


def test_inferred(enc):
    """The held-out countries had their IN_CONTINENT facts REMOVED from training.
    So direct query should fail. But multi-hop via BORDERS should still work."""
    print("\n=== TEST 3: inferred fact (continent fact REMOVED from training) ===")
    held = [c for c in COUNTRIES[25:]]
    print("\n  3a. Direct query (should FAIL because facts removed):")
    direct_correct = 0
    for country, _cap, cont, *_ in held:
        results = enc.query(country, "IN_CONTINENT", CONTINENTS, top_k=1)
        ok = results and results[0][0] == cont
        direct_correct += int(bool(ok))
        mark = "OK" if ok else "MISS"
        print(f"    [{mark}] {country:14s} -> {results[0][0] if results else '(none)'} "
              f"(expected {cont})")
    print(f"\n    Direct accuracy: {direct_correct}/{len(held)}")

    print("\n  3b. Multi-hop via BORDERS (should still work):")
    hop_correct = 0
    for country, _cap, expected_cont, *_ in held:
        chain = [
            ("BORDERS",      ALL_COUNTRIES),
            ("IN_CONTINENT", CONTINENTS),
        ]
        result, trace = multihop_query(enc, country, chain, trace=True)
        ok = (result == expected_cont)
        hop_correct += int(ok)
        mark = "OK" if ok else "MISS"
        print(f"    [{mark}] {country:14s} -> ... -> {result}  (expected {expected_cont})")
        if not ok:
            print_trace(trace)
    print(f"\n    Multi-hop inferred accuracy: {hop_correct}/{len(held)}")

    # Compare
    print(f"\n  Direct (after removal): {direct_correct}/{len(held)}")
    print(f"  Multi-hop chained:      {hop_correct}/{len(held)}")
    if hop_correct > direct_correct:
        delta = hop_correct - direct_correct
        print(f"\n  RESULT: multi-hop recovered {delta} facts that were INVISIBLE")
        print(f"          to the encoder. Compositional inference works.")


def main():
    print("Multi-hop inference in pure HDC\n")

    print("Training RBI on full corpus ...")
    enc_full = train_full()
    stats = enc_full.stats()
    print(f"  vocab={stats['vocab']}, relations={stats['relations']}, "
          f"triples={stats['triples']}")

    test_direct(enc_full)
    test_multihop(enc_full)

    print("\n" + "=" * 60)
    print("RETRAINING WITHOUT held-out continent facts ...")
    print("=" * 60)
    held = [c[0] for c in COUNTRIES[25:]]
    enc_minus = train_minus_continent_facts(held)

    test_inferred(enc_minus)

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    print("  If Test 3b accuracy is high, HDC compositional inference is real.")
    print("  Multi-hop algebra derives facts the encoder never saw.")


if __name__ == "__main__":
    main()
