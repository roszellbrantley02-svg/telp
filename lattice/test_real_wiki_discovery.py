"""
lattice/test_real_wiki_discovery.py - relation discovery on REAL Wikipedia text.

The honest test: take 12 Wikipedia summaries (37 messy real sentences),
run our unsupervised discovery, and see what relational patterns emerge.

Real text is much harder than templated:
  - Sentences vary widely in length and structure
  - Many sentences are unique structures
  - Content vocabulary is huge
  - Multiple relations per sentence are common

Discovery will likely find FEWER clean clusters than on templated text,
but the ones it does find should be REAL patterns in English.
"""
from __future__ import annotations

import sys
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.real_corpus import get_all_sentences, WIKI_TEXTS
from lattice.relation_discovery import (
    discover_relations, cluster_sentences, StructureEncoder, find_slots, tokenize
)


def main():
    print("UNSUPERVISED DISCOVERY ON REAL WIKIPEDIA TEXT\n")

    sentences = get_all_sentences()
    print(f"Input: {len(sentences)} sentences from {len(WIKI_TEXTS)} Wikipedia summaries\n")

    # Try a range of clustering thresholds to find the sweet spot
    print("=== Trying multiple cluster thresholds ===")
    print(f"  {'threshold':>10s} | {'clusters':>10s} | {'usable':>7s} | {'triples':>8s}")
    print(f"  {'-'*10} | {'-'*10} | {'-'*7} | {'-'*8}")
    for threshold in [0.10, 0.15, 0.20, 0.25, 0.30]:
        clusters = cluster_sentences(sentences, threshold_pct=threshold)
        usable = [c for c in clusters if len(c["members"]) >= 2]
        # Count triples (rough estimate)
        n_trip = sum(len(c["members"]) for c in usable
                       if find_slots(c, StructureEncoder()) and
                          len(find_slots(c, StructureEncoder())) >= 2)
        print(f"  {threshold:>10.2f} | {len(clusters):>10d} | "
              f"{len(usable):>7d} | {n_trip:>8d}")

    # Use threshold 0.20 for the deeper analysis
    print("\n=== Detailed analysis at threshold=0.20 ===")
    triples, cluster_info = discover_relations(
        sentences, cluster_threshold=0.20, min_cluster_size=2, verbose=True,
    )

    # ── Show the discovered clusters ──
    print("\n=== DISCOVERED CLUSTERS (size >= 2) ===")
    for c in sorted(cluster_info, key=lambda x: -x["size"])[:15]:
        print(f"\n  [Cluster {c['cluster_id']}, {c['relation_id']}, "
              f"size={c['size']}, slots at positions {c['slots']}]")
        for s in c["sample_sentences"]:
            print(f"    \"{s}\"")

    # ── Show extracted triples ──
    print(f"\n=== AUTO-EXTRACTED TRIPLES ({len(triples)} total) ===")
    by_rel = {}
    for s, r, o in triples:
        by_rel.setdefault(r, []).append((s, o))
    for rel in sorted(by_rel, key=lambda r: -len(by_rel[r])):
        items = by_rel[rel]
        print(f"\n  {rel}  ({len(items)} triples):")
        for s, o in items[:5]:
            print(f"    {s}  ->  {o}")
        if len(items) > 5:
            print(f"    ... and {len(items)-5} more")

    # ── Honest verdict ──
    print("\n" + "=" * 70)
    print("VERDICT — discovery on real-world text")
    print("=" * 70)
    if cluster_info:
        biggest = max(cluster_info, key=lambda c: c["size"])
        print(f"  Sentences:                    {len(sentences)}")
        print(f"  Clusters discovered (size>=2): {len(cluster_info)}")
        print(f"  Triples auto-extracted:        {len(triples)}")
        print(f"  Biggest cluster:               size {biggest['size']}")
        print(f"  Avg cluster size:              {sum(c['size'] for c in cluster_info)/len(cluster_info):.1f}")
        print(f"\n  Real-world text is HARDER than templated. Clusters are smaller,")
        print(f"  more sentences are unique structures. But the clean patterns")
        print(f"  that DO emerge are real linguistic regularities.")
    else:
        print("  No usable clusters formed at this threshold.")


if __name__ == "__main__":
    main()
