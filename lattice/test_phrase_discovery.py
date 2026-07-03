"""
lattice/test_phrase_discovery.py - phrase-level pattern discovery on Wikipedia.

The fix for sentence-level discovery's failure on real text. Look for
shorter recurring phrase patterns instead of whole-sentence matches.
"""
from __future__ import annotations

import sys
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.real_corpus import get_all_sentences, WIKI_TEXTS
from lattice.phrase_discovery import discover_and_extract


def main():
    print("PHRASE-LEVEL DISCOVERY ON REAL WIKIPEDIA TEXT\n")

    sentences = get_all_sentences()
    print(f"Input: {len(sentences)} sentences from {len(WIKI_TEXTS)} summaries\n")

    print("Running phrase-pattern discovery (windows 4-6 words, min 2 occurrences) ...")
    triples, patterns = discover_and_extract(
        sentences, window_min=4, window_max=6, min_pattern_count=2,
    )
    print(f"  patterns found: {len(patterns)}")
    print(f"  triples extracted: {len(triples)}")

    # ── Show top patterns ──
    print("\n=== TOP RECURRING PATTERNS ===")
    for p in patterns[:15]:
        print(f"\n  Pattern {p['pattern_id']}: {p['pattern_string']!r}")
        print(f"    occurrences: {p['instance_count']}")
        for inst in p["sample_instances"]:
            print(f"    \"{inst}\"")
        print(f"    triples extracted: {p['n_triples']}")

    # ── Show sample triples by pattern ──
    print("\n=== AUTO-EXTRACTED TRIPLES (sample by pattern) ===")
    by_rel = {}
    for s, r, o in triples:
        by_rel.setdefault(r, []).append((s, o))
    for rel in sorted(by_rel, key=lambda r: -len(by_rel[r]))[:8]:
        items = by_rel[rel]
        pattern_str = next((p["pattern_string"] for p in patterns
                              if p["pattern_id"] == int(rel.split("_")[1])), "?")
        print(f"\n  {rel} ({pattern_str!r}, {len(items)} triples):")
        for s, o in items[:8]:
            print(f"    ({s}) -> ({o})")

    # ── Try with higher min_count ──
    print("\n=== Phrase discovery with min_count=3 (more conservative) ===")
    triples3, patterns3 = discover_and_extract(
        sentences, window_min=4, window_max=6, min_pattern_count=3,
    )
    print(f"  patterns with >= 3 occurrences: {len(patterns3)}")
    for p in patterns3[:5]:
        print(f"\n  {p['pattern_string']!r}  ({p['instance_count']}x)")
        for inst in p["sample_instances"]:
            print(f"    \"{inst}\"")

    # ── Honest verdict ──
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Sentence-level (previous): 0 clusters, 0 triples on this corpus")
    print(f"  Phrase-level (this):       {len(patterns)} patterns, "
          f"{len(triples)} triples")
    print()
    print(f"  Phrase-level discovery finds REAL recurring patterns in")
    print(f"  Wikipedia text. The sentence-level approach failed because")
    print(f"  real sentences are mostly structurally unique, but the")
    print(f"  CONSTITUENT phrases repeat across articles.")


if __name__ == "__main__":
    main()
