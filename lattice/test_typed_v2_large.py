"""
lattice/test_typed_v2_large.py - TypedHDC v2 on 100-article Wikipedia corpus.

The big test. Richer types + larger corpus.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from collections import Counter

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.typed_discovery_v2 import (
    type_tag_sentence_v2, discover_typed_windows_v2, extract_triples_v2,
    TYPED_TAGS,
)


CORPUS_PATH = _TELP_ROOT / "state" / "wiki_corpus.json"


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def main():
    print("TypedHDC v2 on 100-ARTICLE Wikipedia corpus\n")

    # Load corpus
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    sentences = []
    for entry in corpus:
        if "extract" in entry and entry["extract"]:
            sentences.extend(split_sentences(entry["extract"]))
    print(f"Articles loaded: {len([c for c in corpus if 'extract' in c])}")
    print(f"Sentences extracted: {len(sentences)}")

    # Tag a few for inspection
    print("\n=== Spot-check the richer tagger ===")
    for sent in sentences[:5]:
        tagged = type_tag_sentence_v2(sent)
        typed_words = [(t[0], t[1]) for t in tagged if t[1] in TYPED_TAGS]
        print(f"  '{sent[:80]}'")
        print(f"    typed slots: {typed_words[:6]}")

    # Distribution of types
    print("\n=== Type distribution across corpus ===")
    type_counts = Counter()
    for sent in sentences:
        tagged = type_tag_sentence_v2(sent)
        for _, t in tagged:
            if t in TYPED_TAGS:
                type_counts[t] += 1
    for t, c in type_counts.most_common():
        print(f"  {t:14s}: {c}")

    # Run discovery
    print("\n=== Running discovery (windows 4-7, min_count=3) ===")
    patterns = discover_typed_windows_v2(
        sentences, window_min=4, window_max=7,
        min_pattern_count=3, min_entities_in_window=2,
    )
    print(f"  Recurring patterns found: {len(patterns)}")

    triples, info = extract_triples_v2(patterns)
    print(f"  Triples extracted: {len(triples)}")

    # Top patterns
    print("\n=== TOP 20 PATTERNS ===")
    for p in info[:20]:
        print(f"\n  [{p['label']}]  ({p['instance_count']} instances)")
        print(f"    pattern: {p['pattern_string']}")
        for s in p['sample'][:2]:
            print(f"    \"{s['fragment'][:80]}\"")
            print(f"       subj={s['subj']}  obj={s['obj']}")

    # Sample triples by relation
    print("\n=== TRIPLES GROUPED BY RELATION ===")
    by_rel = {}
    for s, r, o in triples:
        by_rel.setdefault(r, []).append((s, o))
    for rel in sorted(by_rel, key=lambda r: -len(by_rel[r]))[:10]:
        items = by_rel[rel]
        print(f"\n  {rel}  ({len(items)} total, showing unique):")
        seen = set()
        n = 0
        for s, o in items:
            key = (s.lower(), o.lower())
            if key in seen: continue
            seen.add(key)
            print(f"    ({s}) --> ({o})")
            n += 1
            if n >= 6: break

    # Cross-validated entity pairs
    print("\n=== HIGH-CONFIDENCE ENTITY PAIRS (multiple relations agree) ===")
    from lattice.typed_discovery import cross_validate_triples
    high_conf = cross_validate_triples(triples, min_pattern_diversity=2)
    print(f"  Found {len(high_conf)} high-confidence pairs")
    for pair in list(high_conf)[:30]:
        print(f"    ({pair[0]}) <-> ({pair[1]})")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Corpus:                       {len(sentences)} sentences")
    print(f"  Type slots found:             {sum(type_counts.values())}")
    print(f"  Recurring patterns:           {len(patterns)}")
    print(f"  Triples extracted:            {len(triples)}")
    print(f"  High-confidence pairs:        {len(high_conf)}")


if __name__ == "__main__":
    main()
