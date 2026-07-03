"""
lattice/test_differentiable_hdc.py - learn entity+relation embeddings via gradient descent.

Test: train differentiable HDC on the country-fact triples. Hold out the
HAS_CAPITAL triples for 7 countries entirely. Test whether the model
can still predict their capitals using only the OTHER relations
(IN_CONTINENT, BORDERS, SPEAKS, FAMOUS_FOR, USES_CURRENCY).

This is the proper "learn semantics from data" test. The held-out
countries appear in the corpus under non-capital relations, so the
model develops embeddings for them. The question: does the learned
HAS_CAPITAL transformation work on those embeddings?

If yes, gradient descent + HDC ops jointly learn structured embeddings
that generalize. This would be the cleanest demonstration of HDC
matching neural-net knowledge graph completion.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

import torch

from lattice.corpus_tagged import build_tagged_corpus
from lattice.corpus import COUNTRIES
from lattice.differentiable_hdc import DifferentiableHDC, train_triples, predict


HELDOUT_COUNTRIES = {c[0] for c in COUNTRIES[25:]}


def main():
    print("Differentiable HDC — learn entity+relation embeddings via backprop\n")

    # ── Build the triple set, hold out HAS_CAPITAL for heldout countries ──
    tagged = build_tagged_corpus()
    all_triples = []
    held_capital_triples = []  # for evaluation only
    for sent, triples in tagged:
        for s, r, o in triples:
            if r == "HAS_CAPITAL" and s in HELDOUT_COUNTRIES:
                held_capital_triples.append((s, r, o))
                continue   # skip from training
            if r == "HAS_CAPITAL_INV" and o in HELDOUT_COUNTRIES:
                continue   # skip the inverse too
            all_triples.append((s, r, o))
    print(f"  training triples: {len(all_triples)}")
    print(f"  held-out HAS_CAPITAL triples: {len(held_capital_triples)}")

    # ── Build vocab: entities + relations ──
    entities = set()
    relations = set()
    for s, r, o in all_triples + held_capital_triples:
        entities.add(s)
        entities.add(o)
        relations.add(r)
    entities = sorted(entities)
    relations = sorted(relations)
    ent_to_id = {e: i for i, e in enumerate(entities)}
    rel_to_id = {r: i for i, r in enumerate(relations)}
    print(f"  vocab: {len(entities)} entities, {len(relations)} relations")

    triple_ids = [(ent_to_id[s], rel_to_id[r], ent_to_id[o])
                    for s, r, o in all_triples]

    # ── Train ──
    dim = 256
    print(f"\nTraining DifferentiableHDC (dim={dim}) ...")
    torch.manual_seed(42)
    model = DifferentiableHDC(
        n_entities=len(entities),
        n_relations=len(relations),
        dim=dim,
    )
    losses = train_triples(model, triple_ids, epochs=200, lr=0.05, verbose=True)

    # ── Evaluate on held-out HAS_CAPITAL ──
    print("\n=== EVALUATION: held-out HAS_CAPITAL ===")
    print("  The model never saw (heldout_country, HAS_CAPITAL, capital).")
    print("  It only saw OTHER relations for these countries.")
    print("  Can it predict the capital from learned embeddings?\n")

    # All capitals as candidate set
    all_capitals = sorted({o for c in COUNTRIES for o in [c[1]]})
    capital_ids = [ent_to_id[c] for c in all_capitals if c in ent_to_id]

    print(f"  {'COUNTRY':14s} | top-1 prediction  | rank of correct")
    print(f"  {'-' * 14} | {'-' * 17} | {'-' * 15}")
    h_top1 = 0
    h_top3 = 0
    n_evaluated = 0
    rel_id_capital = rel_to_id["HAS_CAPITAL"]

    seen_pairs = set()
    for s, r, o in held_capital_triples:
        if (s, o) in seen_pairs:
            continue
        seen_pairs.add((s, o))
        n_evaluated += 1
        results = predict(model, ent_to_id[s], rel_id_capital,
                            capital_ids, top_k=20)
        names = [entities[eid] for eid, _ in results]
        rank = names.index(o) + 1 if o in names else "out"
        top1 = names[0]
        is_top1 = (top1 == o)
        is_top3 = isinstance(rank, int) and rank <= 3
        h_top1 += int(is_top1)
        h_top3 += int(is_top3)
        mark = "T1" if is_top1 else ("T3" if is_top3 else "  ")
        print(f"  {s:14s} | [{mark}] {top1:14s} | {str(rank):>15s}")

    print(f"\n  HELDOUT Top-1: {h_top1}/{n_evaluated} = {h_top1/max(1,n_evaluated):.0%}")
    print(f"  HELDOUT Top-3: {h_top3}/{n_evaluated} = {h_top3/max(1,n_evaluated):.0%}")

    # ── Cross-check on training-set HAS_CAPITAL ──
    print("\n=== sanity: training-set HAS_CAPITAL ===")
    train_triples_cap = [t for t in triple_ids
                          if relations[t[1]] == "HAS_CAPITAL"]
    correct = 0
    for s_id, r_id, o_id in train_triples_cap:
        results = predict(model, s_id, r_id, capital_ids, top_k=1)
        if results[0][0] == o_id:
            correct += 1
    print(f"  Training HAS_CAPITAL accuracy: {correct}/{len(train_triples_cap)} = "
          f"{correct/max(1,len(train_triples_cap)):.0%}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(f"  Final training loss:    {losses[-1]:.4f}")
    print(f"  Heldout HAS_CAPITAL:    {h_top1}/{n_evaluated}")
    print()
    if h_top1 / max(1, n_evaluated) >= 0.50:
        print("  WORKS: differentiable HDC generalizes to held-out facts via")
        print("  learned embeddings. Gradient descent + bipolar bind = true")
        print("  knowledge graph completion in HDC form.")
    elif h_top1 / max(1, n_evaluated) >= 0.20:
        print("  PARTIAL: some generalization but not at hand-designed level.")
    else:
        print("  WEAK: the model can't recover held-out capitals from other")
        print("  relations. Standard knowledge-graph-embedding limitation:")
        print("  embeddings need diverse training signal to develop relational")
        print("  structure.")


if __name__ == "__main__":
    main()
