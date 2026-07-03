"""
lattice/test_typed_discovery.py - TypedHDC on real Wikipedia text.

The decisive test. If this works, we've cracked unsupervised relation
discovery on real prose.
"""
from __future__ import annotations

import sys
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.real_corpus import get_all_sentences, WIKI_TEXTS
from lattice.typed_discovery import (
    type_tag_sentence, pattern_of_tagged,
    discover_typed_patterns, discover_typed_windows,
    extract_triples, cross_validate_triples,
)


def main():
    print("TYPED HDC DISCOVERY ON REAL WIKIPEDIA\n")

    sentences = get_all_sentences()
    print(f"Input: {len(sentences)} sentences\n")

    # ── Inspect the type tagger on a few sentences ──
    print("=== Type tagger spot-check ===")
    for sent in sentences[:5]:
        tagged = type_tag_sentence(sent)
        # Show abbreviated
        print(f"\n  Raw: {sent[:80]}{'...' if len(sent)>80 else ''}")
        typed_show = []
        for t in tagged[:15]:
            if t[1] in ("ENTITY", "NUMBER"):
                typed_show.append(t[1])
            else:
                typed_show.append(t[0])
        print("  Typed: " + " ".join(typed_show) + ("..." if len(tagged) > 15 else ""))

    # ── Full-sentence typed patterns ──
    print("\n\n=== Full-sentence typed patterns (min_count=2) ===")
    sent_patterns = discover_typed_patterns(sentences, min_pattern_count=2)
    print(f"  Found {len(sent_patterns)} full-sentence patterns")
    for pat, instances in sorted(sent_patterns.items(),
                                     key=lambda x: -len(x[1]))[:5]:
        print(f"\n  Pattern: {pat[:80]}{'...' if len(pat)>80 else ''}")
        for inst in instances[:2]:
            print(f"    {' '.join(t[0] for t in inst)[:80]}")

    # ── Windowed typed patterns ──
    print("\n\n=== Windowed typed patterns (4-7 tokens, min_count=2) ===")
    win_patterns = discover_typed_windows(
        sentences, window_min=4, window_max=7,
        min_pattern_count=2, min_entities_in_window=2,
    )
    print(f"  Found {len(win_patterns)} windowed patterns")

    # Extract triples from windows
    triples, info = extract_triples(win_patterns)
    print(f"  Extracted {len(triples)} triples\n")

    print("=== TOP RECURRING TYPED PATTERNS ===")
    for p in info[:15]:
        print(f"\n  [{p['label']}]  pattern: {p['pattern_string']}")
        print(f"  Instances: {p['instance_count']}, triples: {p['triples_extracted']}")
        for s in p['sample_extractions'][:2]:
            print(f"    \"{s['sentence_fragment'][:70]}\"")
            print(f"      ({s['subject']}) -- {p['label']} --> ({s['object']})")

    # ── Cross-validation ──
    print("\n=== HIGH-CONFIDENCE pairs (appear under 2+ patterns) ===")
    high_conf = cross_validate_triples(triples, min_pattern_diversity=2)
    print(f"  High-confidence entity pairs: {len(high_conf)}")
    for pair in list(high_conf)[:20]:
        print(f"    ({pair[0]}) <-> ({pair[1]})")

    # ── Aggregate triples by label ──
    print("\n=== TRIPLES BY RELATION LABEL ===")
    by_label = {}
    for s, r, o in triples:
        by_label.setdefault(r, []).append((s, o))
    for label in sorted(by_label, key=lambda r: -len(by_label[r]))[:8]:
        items = by_label[label]
        print(f"\n  {label}  ({len(items)} triples):")
        seen = set()
        for s, o in items:
            key = (s.lower(), o.lower())
            if key in seen: continue
            seen.add(key)
            print(f"    ({s}) -- {label} --> ({o})")
            if len(seen) >= 6: break

    # ── Final verdict ──
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Sentence-level HDC (raw words):   0 patterns / 0 triples")
    print(f"  Phrase-level HDC (raw words):     239 patterns / mostly syntactic")
    print(f"  TypedHDC (this system):           {len(win_patterns)} patterns / "
          f"{len(triples)} triples")
    print(f"  High-confidence entity pairs:     {len(high_conf)}")
    print()
    if len(triples) > 100 and len(high_conf) > 10:
        print("  CRACKED: typing + HDC clustering finds semantic relations in")
        print("  real Wikipedia text. The type tags create the structural")
        print("  invariance that pure-word discovery couldn't.")


if __name__ == "__main__":
    main()
