"""
lattice/test_hierarchical.py - does HDC handle multi-level taxonomies?

Build a 6-level animal/object taxonomy. Test:
  T1: IS-A queries at every level (is labrador a dog? a mammal? an animal?)
  T2: class membership (find all dogs, find all mammals)
  T3: sibling vs cousin similarity (poodle ~ labrador closer than poodle ~ shark)
  T4: bundle capacity (does depth-6 still work, or does noise dominate?)
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.hierarchical import HDCTaxonomy
from train.v5_hdc_prototype import D


# ─── Taxonomy: 6 levels, ~30 entities ──────────────────────────────


TAXONOMY = [
    # Each list: root -> ... -> leaf
    # Mammals
    ["thing", "physical", "animal", "mammal", "dog",     "labrador"],
    ["thing", "physical", "animal", "mammal", "dog",     "poodle"],
    ["thing", "physical", "animal", "mammal", "dog",     "bulldog"],
    ["thing", "physical", "animal", "mammal", "cat",     "persian"],
    ["thing", "physical", "animal", "mammal", "cat",     "siamese"],
    ["thing", "physical", "animal", "mammal", "horse",   "arabian"],
    ["thing", "physical", "animal", "mammal", "horse",   "thoroughbred"],
    ["thing", "physical", "animal", "mammal", "elephant","african_elephant"],
    # Birds
    ["thing", "physical", "animal", "bird",   "raptor",  "eagle"],
    ["thing", "physical", "animal", "bird",   "raptor",  "hawk"],
    ["thing", "physical", "animal", "bird",   "songbird","robin"],
    ["thing", "physical", "animal", "bird",   "songbird","sparrow"],
    # Fish
    ["thing", "physical", "animal", "fish",   "shark",   "great_white"],
    ["thing", "physical", "animal", "fish",   "shark",   "hammerhead"],
    ["thing", "physical", "animal", "fish",   "ray",     "stingray"],
    # Vehicles
    ["thing", "physical", "vehicle","car",    "sedan",   "camry"],
    ["thing", "physical", "vehicle","car",    "sedan",   "accord"],
    ["thing", "physical", "vehicle","car",    "sports",  "porsche"],
    ["thing", "physical", "vehicle","car",    "sports",  "ferrari"],
    ["thing", "physical", "vehicle","plane",  "jet",     "boeing_747"],
    ["thing", "physical", "vehicle","plane",  "jet",     "airbus_a380"],
    ["thing", "physical", "vehicle","boat",   "ship",    "container_ship"],
    ["thing", "physical", "vehicle","boat",   "yacht",   "schooner"],
    # Plants
    ["thing", "physical", "plant",  "tree",   "deciduous","oak"],
    ["thing", "physical", "plant",  "tree",   "deciduous","maple"],
    ["thing", "physical", "plant",  "tree",   "conifer", "pine"],
    ["thing", "physical", "plant",  "flower", "rose",    "red_rose"],
    ["thing", "physical", "plant",  "flower", "rose",    "white_rose"],
    ["thing", "physical", "plant",  "flower", "tulip",   "yellow_tulip"],
    # Abstract concepts (different branch)
    ["thing", "abstract", "emotion","positive","joy",    "happiness"],
    ["thing", "abstract", "emotion","negative","sadness", "grief"],
    ["thing", "abstract", "math",   "number", "integer", "five"],
]


# Convenience: the level of each named concept (where it first appears)
def build_concept_levels():
    levels = {}
    for chain in TAXONOMY:
        for i, c in enumerate(chain):
            if c not in levels:
                levels[c] = i
    return levels


def main():
    print("HDC Hierarchical Encoding — concepts of concepts\n")

    tax = HDCTaxonomy(seed=42)
    print(f"Adding {len(TAXONOMY)} entities to the taxonomy ...")
    for chain in TAXONOMY:
        tax.add_entity(chain)
    print(f"  entities: {len(tax.entities)}")
    print(f"  concepts: {len(tax.concept_ids)}")
    print(f"  levels:   {len(tax.level_roles)}")

    concept_levels = build_concept_levels()

    # ─── TEST 1: IS-A queries at every level ───
    print("\n=== TEST 1: IS-A queries at every level ===")
    print("  For each entity, query each ancestor level and check.\n")

    n_total = 0
    n_correct = 0
    by_level = {i: [0, 0] for i in range(6)}  # [correct, total]
    failures_by_level = {i: [] for i in range(6)}
    for chain in TAXONOMY:
        leaf = chain[-1]
        for level, expected_ancestor in enumerate(chain):
            ans, dist = tax.is_a(leaf, level)
            n_total += 1
            ok = (ans == expected_ancestor)
            n_correct += int(ok)
            by_level[level][1] += 1
            by_level[level][0] += int(ok)
            if not ok:
                failures_by_level[level].append((leaf, expected_ancestor, ans, dist))
    print(f"  Overall: {n_correct}/{n_total} = {n_correct/n_total:.0%}\n")
    for lvl in sorted(by_level):
        c, t = by_level[lvl]
        print(f"  Level {lvl}: {c}/{t} = {c/t*100:.0f}%")
        for leaf, exp, got, d in failures_by_level[lvl][:2]:
            print(f"    miss: {leaf} -> expected {exp}, got {got} (d={d})")

    # ─── TEST 2: class membership ───
    print("\n=== TEST 2: class membership ===")
    queries = [
        ("dog",     4),
        ("mammal",  3),
        ("car",     3),
        ("animal",  2),
        ("vehicle", 2),
        ("plant",   2),
        ("physical", 1),
        ("abstract", 1),
    ]
    for concept, level in queries:
        members = tax.members_of(concept, level)
        # Compute expected from taxonomy
        expected = sorted({chain[-1] for chain in TAXONOMY
                          if len(chain) > level and chain[level] == concept})
        got = sorted(members)
        ok = (got == expected)
        mark = "OK" if ok else "MISS"
        print(f"  [{mark}] members_of('{concept}', lvl={level}): {len(got)} found")
        if not ok:
            extras = set(got) - set(expected)
            missing = set(expected) - set(got)
            if extras: print(f"    extras:  {extras}")
            if missing: print(f"    missing: {missing}")

    # ─── TEST 3: sibling vs cousin similarity ───
    print("\n=== TEST 3: sibling vs cousin similarity ===")
    print("  Same-parent entities should be closer than different-branch entities.\n")
    pairs = [
        # (a, b, c) - test that a~b should be closer than a~c
        ("labrador", "poodle", "ferrari"),       # same dog vs car
        ("ferrari", "porsche", "great_white"),   # same sports vs fish
        ("labrador", "persian", "ferrari"),      # both mammal vs car
        ("eagle",    "hawk",    "stingray"),     # same raptor vs fish
        ("oak",      "maple",   "camry"),        # same deciduous vs car
        ("happiness", "grief",  "labrador"),     # both emotion vs animal
    ]
    n_pairs_correct = 0
    for a, b, c in pairs:
        d_ab = tax.similarity(a, b)
        d_ac = tax.similarity(a, c)
        ok = d_ab < d_ac
        n_pairs_correct += int(ok)
        mark = "OK" if ok else "MISS"
        print(f"  [{mark}] d({a},{b})={d_ab} vs d({a},{c})={d_ac}")
    print(f"\n  Similarity test: {n_pairs_correct}/{len(pairs)}")

    # ─── TEST 4: bundle capacity (deepest entities) ───
    print("\n=== TEST 4: bundle capacity at depth 6 ===")
    print("  Each entity has 6 bound ancestors. Does retrieval degrade?\n")
    print(f"  Mean distance for correct retrieval (lower = clearer signal):")
    for lvl in sorted(by_level):
        # Compute mean distance for correct retrievals at this level
        dists = []
        for chain in TAXONOMY:
            leaf = chain[-1]
            expected = chain[lvl]
            ans, dist = tax.is_a(leaf, lvl)
            if ans == expected:
                dists.append(dist)
        if dists:
            print(f"  Level {lvl} ({len(dists)} hits): mean d={np.mean(dists):.0f}, "
                  f"std={np.std(dists):.0f}")

    # ─── Summary ───
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    overall = n_correct / n_total
    if overall >= 0.95:
        print(f"  IS-A queries:   {overall*100:.0f}% — STRONG")
    elif overall >= 0.80:
        print(f"  IS-A queries:   {overall*100:.0f}% — PARTIAL")
    else:
        print(f"  IS-A queries:   {overall*100:.0f}% — WEAK")
    sim_score = n_pairs_correct / len(pairs)
    print(f"  Similarity:     {sim_score*100:.0f}% sibling < cousin")
    print()
    if overall >= 0.90 and sim_score >= 0.80:
        print("  WORKS: HDC handles 6-level taxonomies with role-bound ancestor")
        print("  bundling. Multi-level concepts coexist in one hypervector.")


if __name__ == "__main__":
    main()
