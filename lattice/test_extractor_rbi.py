"""
lattice/test_extractor_rbi.py - the full pipeline: regex extraction + RBI.

Build the same corpus as before (576 country-fact sentences). But instead
of using template tags directly, run the rule-based extractor over the
raw sentence strings and feed RBI the extracted triples.

If RBI still hits high heldout accuracy on country->capital with parser-
recovered triples, we've shown the architecture works on natural text
(not just on tagged data).

Comparison:
  - RBI with template tags:   100% heldout (the breakthrough we just got)
  - RBI with regex-extracted: ?
  - MiniLM:                   80%
  - Plain RI:                 14%

Usage:
    python -m lattice.test_extractor_rbi
"""
from __future__ import annotations

import sys
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.corpus import build_corpus, COUNTRIES
from lattice.triple_extractor import extract_triples
from lattice.relation_indexing import RelationBoundEncoder
from lattice.corpus_tagged import build_tagged_corpus
from train.v5_hdc_prototype import D


HELDOUT_PAIRS = [(c[0], c[1]) for c in COUNTRIES[25:]]
ALL_CAPITALS = list({c[1] for c in COUNTRIES})


def evaluate_extractor(sentences, tagged):
    """Compare regex-extracted triples to ground-truth template tags."""
    extracted_total = 0
    matched = 0
    truth_total = 0
    extracted_extra = 0
    for sent, gt_triples in tagged:
        gt_set = set(gt_triples)
        truth_total += len(gt_set)
        extracted = set(extract_triples(sent))
        extracted_total += len(extracted)
        # Match by exact tuple equality
        matched += len(gt_set & extracted)
        extracted_extra += len(extracted - gt_set)
    recall = matched / max(1, truth_total)
    precision = matched / max(1, extracted_total)
    return {
        "truth_triples": truth_total,
        "extracted_triples": extracted_total,
        "matched": matched,
        "extra": extracted_extra,
        "precision": precision,
        "recall": recall,
    }


def main():
    print("Full pipeline: regex extraction -> RBI -> country->capital test\n")

    # Build the same corpus (we have both: sentences and ground-truth tags)
    print("Building corpus ...")
    tagged = build_tagged_corpus()
    sentences = [s for s, _ in tagged]
    print(f"  {len(sentences)} sentences")

    # ── Step 1: evaluate the extractor itself ──
    print("\n=== STEP 1: extractor accuracy vs. template-tag ground truth ===")
    metrics = evaluate_extractor(sentences, tagged)
    print(f"  Ground-truth triples (from templates): {metrics['truth_triples']}")
    print(f"  Triples extracted by regex parser:     {metrics['extracted_triples']}")
    print(f"  Matched (correct extraction):          {metrics['matched']}")
    print(f"  Extra (over-extraction):               {metrics['extra']}")
    print(f"  Precision: {metrics['precision']*100:.1f}%   "
          f"Recall: {metrics['recall']*100:.1f}%")

    # ── Step 2: train RBI on extractor-derived triples ──
    print("\n=== STEP 2: train RBI on EXTRACTOR-derived triples ===")
    extracted_corpus = [(s, extract_triples(s)) for s in sentences]
    enc = RelationBoundEncoder(dim=D, seed=42)
    enc.train(extracted_corpus)
    stats = enc.stats()
    print(f"  encoder: vocab={stats['vocab']}, relations={stats['relations']}, "
          f"triples processed={stats['triples']}")

    # ── Step 3: heldout country->capital queries ──
    print("\n=== STEP 3: HELDOUT country->capital recall ===")
    print(f"  {'COUNTRY':14s} | top-1 prediction  | rank of correct")
    print(f"  {'-' * 14} | {'-' * 17} | {'-' * 15}")
    h_top1 = 0
    h_top3 = 0
    for country, expected in HELDOUT_PAIRS:
        results = enc.query(country, "HAS_CAPITAL", ALL_CAPITALS, top_k=20)
        names = [r[0] for r in results]
        rank = names.index(expected) + 1 if expected in names else "out"
        is_top1 = (names[0] == expected)
        is_top3 = isinstance(rank, int) and rank <= 3
        h_top1 += int(is_top1)
        h_top3 += int(is_top3)
        mark = "T1" if is_top1 else ("T3" if is_top3 else "  ")
        print(f"  {country:14s} | [{mark}] {names[0]:14s} | {str(rank):>15s}")

    n = len(HELDOUT_PAIRS)
    print(f"\n  HELDOUT Top-1: {h_top1}/{n} = {h_top1/n:.0%}")
    print(f"  HELDOUT Top-3: {h_top3}/{n} = {h_top3/n:.0%}")

    # ── Test the same on a couple of other relations ──
    print("\n=== STEP 4: cross-relation test (continent / language) ===")
    continents = ["Europe", "Asia", "Africa", "North America",
                    "South America", "Oceania"]
    languages = list({c[3] for c in COUNTRIES})
    print("\n  IN_CONTINENT queries on heldout:")
    cont_ok = 0
    for country, _ in HELDOUT_PAIRS:
        expected_cont = next(c[2] for c in COUNTRIES if c[0] == country)
        results = enc.query(country, "IN_CONTINENT", continents, top_k=1)
        if results:
            ok = (results[0][0] == expected_cont)
            cont_ok += int(ok)
            mark = "OK" if ok else "MISS"
            print(f"    [{mark}] {country:14s} -> {results[0][0]} "
                  f"(expected {expected_cont})")
    print(f"  IN_CONTINENT: {cont_ok}/{len(HELDOUT_PAIRS)}")

    print("\n  SPEAKS queries on heldout:")
    lang_ok = 0
    for country, _ in HELDOUT_PAIRS:
        expected_lang = next(c[3] for c in COUNTRIES if c[0] == country)
        results = enc.query(country, "SPEAKS", languages, top_k=1)
        if results:
            ok = (results[0][0] == expected_lang)
            lang_ok += int(ok)
            mark = "OK" if ok else "MISS"
            print(f"    [{mark}] {country:14s} -> {results[0][0]} "
                  f"(expected {expected_lang})")
    print(f"  SPEAKS: {lang_ok}/{len(HELDOUT_PAIRS)}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("FULL PIPELINE COMPARISON (HELDOUT TOP-1)")
    print("=" * 60)
    print(f"  MiniLM (LLM encoder + Hebbian):              80%")
    print(f"  Random Indexing (bag-of-neighbors):          14%")
    print(f"  Pure HDC (no learning):                      20%")
    print(f"  RBI w/ ground-truth template tags:          100%")
    print(f"  RBI w/ regex-extracted triples (THIS):      {h_top1/n:.0%}")
    print()
    if h_top1 / n >= 0.80:
        print("  BREAKTHROUGH: Full pipeline (parsing + RBI) matches the LLM.")
        print("  Pure HDC + rule-based parsing solves relational fact retrieval")
        print("  from raw English text. No neural net anywhere.")
    elif h_top1 / n >= 0.50:
        print("  REAL PROGRESS: parser noise reduces accuracy but signal survives.")


if __name__ == "__main__":
    main()
