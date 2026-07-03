"""
lattice/test_relation_discovery.py - does HDC find structure without labels?

Take the unified-relation 10-domain corpus (raw sentences only — NO
template tags, no relation labels). Run discovery. Compare to ground
truth.

Tests:
  T1: discovery accuracy — how many of the ground-truth relations does
      the system find?
  T2: triple extraction quality — precision/recall on the discovered
      vs ground-truth triples
  T3: downstream usability — train RBI on auto-discovered triples,
      test if within-domain queries still work
  T4: cross-domain analogy with auto-discovered relations
"""
from __future__ import annotations

import sys
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.corpus_10domain import (
    build_10domain_corpus, PRIMARY_PAIRS_BY_DOMAIN, HELDOUT,
)
from lattice.relation_discovery import discover_relations
from lattice.relation_indexing import RelationBoundEncoder
from train.v5_hdc_prototype import D


def main():
    print("UNSUPERVISED RELATION DISCOVERY @ 10 DOMAINS\n")
    print("Hypothesis: HDC can find relational structure in raw text without")
    print("any labels, parsers, or hand-coded patterns.\n")

    # Build corpus — extract ONLY the sentences (drop the tags)
    tagged = build_10domain_corpus()
    sentences = [s for s, _ in tagged]
    ground_truth = []
    for _, triples in tagged:
        ground_truth.extend(triples)
    print(f"Input: {len(sentences)} raw sentences (no labels)")
    print(f"Ground truth (hidden from system): {len(ground_truth)} triples\n")

    # ── Discovery ──
    print("Running unsupervised discovery ...")
    triples, clusters = discover_relations(sentences,
                                              cluster_threshold=0.15,
                                              min_cluster_size=3,
                                              verbose=True)

    print(f"\n=== TEST 1: cluster inspection ===")
    print(f"  {'cluster':>8s} | {'rel_id':>20s} | {'size':>4s} | "
          f"{'slots':>10s} | sample sentence")
    for c in clusters[:15]:
        print(f"  {c['cluster_id']:>8d} | {c['relation_id']:>20s} | "
              f"{c['size']:>4d} | {str(c['slots']):>10s} | "
              f"{c['sample_sentences'][0][:50]}")

    # ── Compare discovered relations to ground truth ──
    print(f"\n=== TEST 2: how many ground-truth relations did we discover? ===")
    gt_relations = set(r for _, r, _ in ground_truth)
    print(f"  Ground-truth relations: {sorted(gt_relations)}")
    print(f"  Discovered clusters:    {len(clusters)}")
    print(f"  Auto-extracted triples: {len(triples)}")

    # For each ground-truth (subject, relation, object), check if any auto-triple
    # has the same (subject, ?, object) pair (regardless of relation_id name)
    gt_pairs = set((s.lower(), o.lower()) for s, _, o in ground_truth)
    auto_pairs = set((s.lower(), o.lower()) for s, _, o in triples)
    intersection = gt_pairs & auto_pairs
    print(f"\n  Subject-object pairs in ground truth: {len(gt_pairs)}")
    print(f"  Subject-object pairs auto-discovered: {len(auto_pairs)}")
    print(f"  Pairs in both:                        {len(intersection)}")
    if gt_pairs:
        recall = len(intersection) / len(gt_pairs)
        precision = len(intersection) / len(auto_pairs) if auto_pairs else 0
        print(f"  Recall:    {recall*100:.0f}%")
        print(f"  Precision: {precision*100:.0f}%")

    # ── Train RBI on AUTO-DISCOVERED triples ──
    print(f"\n=== TEST 3: train RBI on auto-discovered triples ===")
    enc = RelationBoundEncoder(dim=D, seed=42)
    enc.train([(f"auto_sent_{i}", [t]) for i, t in enumerate(triples)])

    # For querying, we need to match entities. They're now lowercase.
    # Test within-domain HAS_PRIMARY accuracy by checking if the original
    # (entity, primary) pair was rediscovered. For each domain pair,
    # query using a discovered relation and check if the result matches.
    correct = 0
    total = 0
    detail_misses = []
    for domain, pairs in PRIMARY_PAIRS_BY_DOMAIN.items():
        for ent, attr in pairs:
            # Check if (ent, attr) appears in auto-discovered pairs
            ent_l = ent.lower()
            attr_l = attr.lower()
            if (ent_l, attr_l) in auto_pairs:
                # Find which discovered relation it belongs to
                found_rel = None
                for s, r, o in triples:
                    if s == ent_l and o == attr_l:
                        found_rel = r
                        break
                if found_rel:
                    # Try the query
                    candidates = list({a.lower() for _, _, a in triples})
                    results = enc.query(ent_l, found_rel, candidates, top_k=1)
                    if results and results[0][0] == attr_l:
                        correct += 1
                    else:
                        detail_misses.append((ent, attr, results[0][0] if results else "None", found_rel))
            total += 1
    print(f"  RBI on auto-discovered: {correct}/{total} = {correct/total*100:.0f}%")
    if detail_misses[:3]:
        print(f"  Sample misses:")
        for m in detail_misses[:3]:
            print(f"    {m[0]} -> expected {m[1]}, got {m[2]} (rel: {m[3]})")

    # ── Test 4: cross-domain analogy with auto-discovered ──
    print(f"\n=== TEST 4: cross-domain analogies on auto-discovered relations ===")
    # Pick the BIGGEST cluster — likely the dominant "primary feature" relation
    if clusters:
        clusters_sorted = sorted(clusters, key=lambda c: -c["size"])
        biggest = clusters_sorted[0]
        target_rel = biggest["relation_id"]
        print(f"  Using biggest cluster as 'universal relation': {target_rel}")
        print(f"  Cluster size: {biggest['size']}, slots: {biggest['slots']}")
        print(f"  Sample: {biggest['sample_sentences'][0]}")

        # Find anchor pairs from this discovered relation
        cluster_triples = [t for t in triples if t[1] == target_rel]
        print(f"  Cluster contains {len(cluster_triples)} triples\n")

        if len(cluster_triples) >= 2:
            # Get all unique subjects and objects in this cluster
            cluster_subs = list({t[0] for t in cluster_triples})
            cluster_objs = list({t[2] for t in cluster_triples})
            # Try cross-domain one-shot analogies
            import numpy as np
            anchor_s, _, anchor_o = cluster_triples[0]
            a_ctx = enc.get_context(anchor_s)
            a_idx = enc.get_index(anchor_o)
            if a_ctx.sum() and a_idx.sum():
                delta = np.bitwise_xor(a_ctx, a_idx)
                # Apply to other subjects
                correct_xd = 0
                for s, r, expected_o in cluster_triples[1:11]:
                    s_ctx = enc.get_context(s)
                    if s_ctx.sum() == 0: continue
                    pred = np.bitwise_xor(s_ctx, delta)
                    # Find nearest in cluster_objs
                    from train.v5_hdc_prototype import hamming_distance
                    scored = sorted(
                        [(o, hamming_distance(pred, enc.get_index(o)))
                          for o in cluster_objs if enc.get_index(o).sum() > 0],
                        key=lambda x: x[1]
                    )
                    if scored and scored[0][0] == expected_o:
                        correct_xd += 1
                    print(f"  {anchor_s}:{anchor_o} :: {s}:{scored[0][0] if scored else '?':10s} "
                          f"(expected {expected_o}) {'OK' if scored and scored[0][0]==expected_o else 'MISS'}")
                print(f"\n  One-shot accuracy in discovered cluster: {correct_xd}/10")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Clusters discovered:           {len(clusters)}")
    print(f"  Triples auto-extracted:        {len(triples)}")
    print(f"  Recall vs ground truth:        {len(intersection) / max(1, len(gt_pairs))*100:.0f}%")
    print(f"  RBI accuracy on discovered:    {correct/max(1,total)*100:.0f}%")


if __name__ == "__main__":
    main()
