"""
lattice/phrase_discovery.py - phrase-pattern discovery for real-world text.

Sentence-level clustering fails on real prose because every sentence is
structurally unique. But shorter PHRASES repeat — "X borders Y", "X is a Y",
"X was a Y", "the capital of X", "X was discovered by Y".

Strategy:
  1. Extract all 4-6 word windows from each sentence
  2. For each window, separate content words (capitalized / not function)
     from function words (small grammatical words)
  3. The "pattern" is the function-word skeleton with CONTENT slots
  4. Cluster windows by their pattern (function-word skeleton)
  5. For each pattern that recurs >= K times, extract triples from the
     content slots

Example: "Germany borders Denmark to the north"
         Pattern: <CONTENT> borders <CONTENT> to the <CONTENT>
         Slots:   ('Germany', 'Denmark', 'north')
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


# Function words — kept literal in patterns
FUNCTION_WORDS = {
    "the","a","an","is","was","were","are","be","been","being","has","have","had",
    "of","in","on","at","by","to","from","with","for","into","onto","through","across",
    "and","or","but","not","nor","yet",
    "this","that","these","those","such",
    "it","its","they","their","them","he","she","his","her","him",
    "as","than","then","also","both","either","neither",
    "which","who","whom","whose","what","where","when","why","how",
    "we","us","our","you","your","my","mine",
    "so","very","more","most","much","many","some","any","all","each","every",
    "if","because","while","during","since","before","after","until",
    "now","still","yet","ever","never","always","often","sometimes",
    "very","quite","just","only","even","also","too",
    "one","two","three","first","second","last","next",
    "born","made","known","named","called","said","told",
}


def is_function(w: str) -> bool:
    return w.lower() in FUNCTION_WORDS


_TOKEN_RE = re.compile(r"\b[\w-]+\b")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def pattern_of_window(words: list[str]) -> str:
    """Return a pattern string where content words become <C> and function
    words stay literal. Used as a cluster key."""
    return " ".join("<C>" if not is_function(w) else w.lower() for w in words)


def content_positions(words: list[str]) -> list[int]:
    return [i for i, w in enumerate(words) if not is_function(w)]


def discover_phrase_patterns(sentences: list[str],
                              window_min: int = 4,
                              window_max: int = 6,
                              min_pattern_count: int = 3,
                              min_content_slots: int = 2,
                              ) -> dict[str, list[list[str]]]:
    """Find recurring phrase patterns across the corpus.

    Returns dict: {pattern_string: [list of word-lists matching the pattern]}
    """
    pattern_counts: dict[str, list[list[str]]] = defaultdict(list)
    for sent in sentences:
        toks = tokenize(sent)
        for window_size in range(window_min, window_max + 1):
            for i in range(len(toks) - window_size + 1):
                window = toks[i: i + window_size]
                # Require >= min_content_slots content words
                if sum(1 for w in window if not is_function(w)) < min_content_slots:
                    continue
                # Require at least one function word (not just content)
                if all(not is_function(w) for w in window):
                    continue
                pat = pattern_of_window(window)
                pattern_counts[pat].append(window)
    # Filter to recurring patterns
    return {p: ws for p, ws in pattern_counts.items()
              if len(ws) >= min_pattern_count}


def extract_triples_from_pattern(pattern: str, instances: list[list[str]],
                                    relation_id: str) -> list[tuple[str, str, str]]:
    """Extract (subject, relation, object) triples from instances of a pattern.

    Uses first and last content words in each instance as subject/object.
    """
    triples = []
    for inst in instances:
        positions = content_positions(inst)
        if len(positions) < 2:
            continue
        subj = inst[positions[0]]
        obj = inst[positions[-1]]
        triples.append((subj, relation_id, obj))
    return triples


def discover_and_extract(sentences: list[str],
                          window_min: int = 4,
                          window_max: int = 6,
                          min_pattern_count: int = 3,
                          ) -> tuple[list[tuple[str, str, str]], list[dict]]:
    """End-to-end: sentences -> patterns -> triples."""
    patterns = discover_phrase_patterns(sentences,
                                            window_min=window_min,
                                            window_max=window_max,
                                            min_pattern_count=min_pattern_count)
    triples = []
    pattern_info = []
    for i, (pat, instances) in enumerate(
        sorted(patterns.items(), key=lambda x: -len(x[1]))
    ):
        rel_id = f"PATTERN_{i}"
        tris = extract_triples_from_pattern(pat, instances, rel_id)
        triples.extend(tris)
        pattern_info.append({
            "pattern_id": i,
            "pattern_string": pat,
            "instance_count": len(instances),
            "sample_instances": [" ".join(w) for w in instances[:3]],
            "n_triples": len(tris),
        })
    return triples, pattern_info
