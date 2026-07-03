"""
lattice/typed_discovery.py - TypedHDC: relation discovery via typed slots.

The fix for phrase-level discovery's failure on real Wikipedia text.

Insight: real sentences are structurally unique only at the RAW WORD level.
At the TYPE level — "ENTITY is a TYPE in ENTITY" — they cluster cleanly.
The typing is built from heuristics (capitalization, numbers, stop words),
not from neural networks or parsers.

Pipeline:
  1. TYPE TAGGER (rule-based, no learning)
     - Mid-sentence capitalized words -> ENTITY
     - Numbers/years -> NUMBER
     - Function words (stop list) -> stay literal
     - Other content words -> stay literal (predicate vocabulary)

  2. TYPED PATTERN EXTRACTION
     - Each sentence becomes a typed sequence
     - Sentences with the same typed sequence cluster together
     - Each cluster = a relational pattern with entity slots

  3. TRIPLE EXTRACTION
     - For each cluster, find the ENTITY positions
     - Extract triples between them
     - The "relation" is named by the predicate words in the pattern

  4. CROSS-VALIDATION
     - High-confidence triples are those where the same entity pair
       appears under MULTIPLE patterns (consistency check)

No neural network. No dependency parser. Just rule-based typing + HDC algebra.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import D, bundle, hamming_distance


# ─── Type tagger (rule-based, no learning) ────────────────────────


FUNCTION_WORDS = {
    "the","a","an","is","was","were","are","be","been","being","has","have","had",
    "of","in","on","at","by","to","from","with","for","into","onto","through","across",
    "and","or","but","not","nor","yet",
    "this","that","these","those","such",
    "it","its","they","their","them","he","she","his","her","him","i","we","us","our","you","your","my","mine",
    "as","than","then","also","both","either","neither",
    "which","who","whom","whose","what","where","when","why","how",
    "so","very","more","most","much","many","some","any","all","each","every",
    "if","because","while","during","since","before","after","until",
    "now","still","ever","never","always","often","sometimes","quite","just","only","even","too",
    "one","two","three","first","second","last","next","four","five","six","seven","eight","nine","ten",
}


_TOKEN_RE = re.compile(r"\b[\w-]+\b")


def tokenize_with_offsets(sentence: str) -> list[tuple[str, int]]:
    """Return list of (token, original_position) tuples."""
    return [(m.group(), m.start()) for m in _TOKEN_RE.finditer(sentence)]


def type_tag_sentence(sentence: str) -> list[tuple[str, str]]:
    """Return list of (token, type_tag) for each token.

    Type tags:
      ENTITY   - mid-sentence capitalized word (proper noun)
      NUMBER   - all digits or year-like
      <word>   - function word kept literal
      <word>   - content word kept literal (lowercase form)
    """
    tokens = tokenize_with_offsets(sentence)
    tagged = []
    for i, (tok, off) in enumerate(tokens):
        # Position 0 capitalization is ambiguous (sentence start)
        is_first = (i == 0)
        is_cap = tok[0].isupper() if tok else False
        is_num = tok.isdigit() or (tok[:-1].isdigit() and tok.endswith("s"))   # e.g. "1900s"
        if is_num:
            tagged.append((tok, "NUMBER"))
        elif is_cap and not is_first:
            tagged.append((tok, "ENTITY"))
        elif is_cap and is_first:
            # Sentence-start capitalized: could be entity or just first word
            # Heuristic: if it's NOT in function_words, treat as ENTITY
            if tok.lower() not in FUNCTION_WORDS:
                tagged.append((tok, "ENTITY"))
            else:
                tagged.append((tok.lower(), tok.lower()))
        elif tok.lower() in FUNCTION_WORDS:
            tagged.append((tok.lower(), tok.lower()))
        else:
            tagged.append((tok.lower(), tok.lower()))
    return tagged


def pattern_of_tagged(tagged: list[tuple[str, str]]) -> str:
    """Return the pattern string — type tags only, joined."""
    return " ".join(t[1] for t in tagged)


def entity_positions(tagged: list[tuple[str, str]]) -> list[int]:
    return [i for i, (_, t) in enumerate(tagged) if t == "ENTITY"]


# ─── Pattern discovery ────────────────────────────────────────────


def discover_typed_patterns(sentences: list[str],
                              min_pattern_count: int = 2,
                              min_entities_per_pattern: int = 2,
                              ) -> dict[str, list[list[tuple[str, str]]]]:
    """Group sentences by their TYPED pattern.

    Returns dict: pattern_string -> [list of tagged sentences matching]
    """
    patterns = defaultdict(list)
    for sent in sentences:
        tagged = type_tag_sentence(sent)
        if sum(1 for _, t in tagged if t == "ENTITY") < min_entities_per_pattern:
            continue
        pat = pattern_of_tagged(tagged)
        patterns[pat].append(tagged)
    return {p: ts for p, ts in patterns.items() if len(ts) >= min_pattern_count}


# ─── Windowed (phrase-level) typed discovery ──────────────────────


def discover_typed_windows(sentences: list[str],
                             window_min: int = 4,
                             window_max: int = 7,
                             min_pattern_count: int = 2,
                             min_entities_in_window: int = 2,
                             ) -> dict[str, list[list[tuple[str, str]]]]:
    """Find recurring TYPED phrase windows across the corpus.

    More flexible than full-sentence matching — picks up patterns even
    when surrounded by other text.
    """
    patterns = defaultdict(list)
    for sent in sentences:
        tagged = type_tag_sentence(sent)
        for w in range(window_min, window_max + 1):
            for i in range(len(tagged) - w + 1):
                window = tagged[i:i + w]
                n_entities = sum(1 for _, t in window if t == "ENTITY")
                if n_entities < min_entities_in_window:
                    continue
                pat = pattern_of_tagged(window)
                patterns[pat].append(window)
    return {p: ts for p, ts in patterns.items() if len(ts) >= min_pattern_count}


# ─── Triple extraction with relation labeling ─────────────────────


def extract_triples(pattern_dict: dict, name_prefix: str = "REL") \
        -> tuple[list[tuple[str, str, str]], list[dict]]:
    """For each pattern, extract (subject_entity, label, object_entity) triples.

    The label is derived from the content (non-entity, non-function) words
    in the pattern — a human-readable hint about the relation.
    """
    triples = []
    info = []
    for i, (pat, instances) in enumerate(
        sorted(pattern_dict.items(), key=lambda x: -len(x[1]))
    ):
        # Derive label from non-ENTITY, non-NUMBER content words in pattern
        label_parts = []
        for tag in pat.split():
            if tag not in ("ENTITY", "NUMBER") and tag not in FUNCTION_WORDS:
                label_parts.append(tag)
        if label_parts:
            label = name_prefix + "_" + "_".join(label_parts[:3]).upper()
        else:
            label = f"{name_prefix}_{i}"

        instance_triples = []
        sample_extractions = []
        for instance in instances:
            ent_positions = entity_positions(instance)
            if len(ent_positions) < 2:
                continue
            subj_tok = instance[ent_positions[0]][0]
            obj_tok = instance[ent_positions[-1]][0]
            instance_triples.append((subj_tok, label, obj_tok))
            if len(sample_extractions) < 3:
                sample_extractions.append({
                    "sentence_fragment": " ".join(t[0] for t in instance),
                    "subject": subj_tok,
                    "object": obj_tok,
                })

        triples.extend(instance_triples)
        info.append({
            "pattern_id": i,
            "label": label,
            "pattern_string": pat,
            "instance_count": len(instances),
            "triples_extracted": len(instance_triples),
            "sample_extractions": sample_extractions,
        })
    return triples, info


def cross_validate_triples(triples: list[tuple[str, str, str]],
                              min_pattern_diversity: int = 2,
                              ) -> set[tuple[str, str]]:
    """High-confidence entity pairs: pairs that co-occur under MULTIPLE
    different relation labels. These are likely real semantic relations.
    """
    pair_relations: dict[tuple[str, str], set[str]] = defaultdict(set)
    for s, r, o in triples:
        pair_relations[(s.lower(), o.lower())].add(r)
        pair_relations[(o.lower(), s.lower())].add(r)   # also reverse
    return {pair for pair, rels in pair_relations.items()
              if len(rels) >= min_pattern_diversity}
