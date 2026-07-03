"""autopilot/code_analogy.py — neurosymbolic code-analogy operators.

Apply HDC analogy primitives to code templates:

    "quicksort for linked lists" ≈ quicksort - array + linked_list

The substrate:
  * Each template / snippet has a description hypervector D.
  * Each axis-of-variation (data_structure, algorithm, mode, ...)
    has a basis vector encoding values within it.
  * The analogy operator finds  D' = D - basis(axis_old) + basis(axis_new)
    and uses D' as a target retrieval query — finding the closest
    actual snippet whose description vector is near the analogical
    target.

For v1, the axes are coarse — data structure (array / linked_list / tree
/ dict), data type (numbers / strings / objects), and modifier (in-place
/ functional / parallel).

Each axis has its own deterministic basis vector, so "swap" semantics
are well-defined.  When no actual snippet matches the analogical query
closely enough, we return a "best similar" with a note that it's an
approximation.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import bind, bundle, hamming_distance, D
from mind.persona import _deterministic_random_vec


# ─── Axis basis vectors ──────────────────────────────────────────


# Each (axis, value) pair maps to a deterministic HDC vector.  Code
# analogy is "swap value A for value B on axis X".
_AXIS_VALUES = {
    "data_structure": {
        "list", "array", "linked_list", "tree", "dict", "set",
        "tuple", "stack", "queue", "graph", "matrix", "iterator",
        "generator", "string",
    },
    "data_type": {
        "numbers", "ints", "floats", "strings", "objects", "bytes",
        "characters",
    },
    "modifier": {
        "in_place", "functional", "recursive", "iterative", "parallel",
        "async", "lazy", "memoized", "vectorized",
    },
    "algorithm": {
        "quicksort", "mergesort", "bubblesort", "heapsort", "insertion_sort",
        "binary_search", "linear_search", "dfs", "bfs", "dijkstra",
        "bellman_ford", "kruskal", "prim", "kmeans", "knn", "regression",
    },
}


_BASIS_CACHE: dict[str, np.ndarray] = {}


def _basis(axis: str, value: str) -> np.ndarray:
    """Get the deterministic basis vector for a (axis, value) pair."""
    key = f"basis::{axis}::{value}"
    if key not in _BASIS_CACHE:
        _BASIS_CACHE[key] = _deterministic_random_vec(key)
    return _BASIS_CACHE[key]


# ─── Surface form detection ──────────────────────────────────────


_PHRASE_TO_AXIS_VALUE = {
    # data structures
    "linked list":   ("data_structure", "linked_list"),
    "binary tree":   ("data_structure", "tree"),
    "tree":          ("data_structure", "tree"),
    "dict":          ("data_structure", "dict"),
    "dictionary":    ("data_structure", "dict"),
    "hash map":      ("data_structure", "dict"),
    "list":          ("data_structure", "list"),
    "array":         ("data_structure", "array"),
    "string":        ("data_structure", "string"),
    "set":           ("data_structure", "set"),
    "queue":         ("data_structure", "queue"),
    "stack":         ("data_structure", "stack"),
    "generator":     ("data_structure", "generator"),
    "iterator":      ("data_structure", "iterator"),
    # data types
    "integers":      ("data_type", "ints"),
    "ints":          ("data_type", "ints"),
    "floats":        ("data_type", "floats"),
    "strings":       ("data_type", "strings"),
    "bytes":         ("data_type", "bytes"),
    # modifiers
    "in-place":      ("modifier", "in_place"),
    "in place":      ("modifier", "in_place"),
    "recursive":     ("modifier", "recursive"),
    "iterative":     ("modifier", "iterative"),
    "parallel":      ("modifier", "parallel"),
    "async":         ("modifier", "async"),
    "lazy":          ("modifier", "lazy"),
    "memoized":      ("modifier", "memoized"),
    # algorithms
    "quicksort":     ("algorithm", "quicksort"),
    "quick sort":    ("algorithm", "quicksort"),
    "merge sort":    ("algorithm", "mergesort"),
    "merge-sort":    ("algorithm", "mergesort"),
    "binary search": ("algorithm", "binary_search"),
    "bfs":           ("algorithm", "bfs"),
    "dfs":           ("algorithm", "dfs"),
}


def _lookup_axis_value(phrase: str) -> Optional[tuple[str, str]]:
    """Look up a phrase in PHRASE_TO_AXIS_VALUE, with plural-stripping
    + last-word fallback."""
    phrase = phrase.strip().lower()
    av = _PHRASE_TO_AXIS_VALUE.get(phrase)
    if av:
        return av
    if phrase.endswith("s"):
        av = _PHRASE_TO_AXIS_VALUE.get(phrase[:-1])
        if av:
            return av
    if " " in phrase:
        last = phrase.rsplit(" ", 1)[-1]
        av = _PHRASE_TO_AXIS_VALUE.get(last)
        if av:
            return av
        if last.endswith("s"):
            av = _PHRASE_TO_AXIS_VALUE.get(last[:-1])
            if av:
                return av
    return None


def detect_axis_swaps(msg: str) -> list[tuple[str, str, str]]:
    """Find (axis, old_value, new_value) swaps implied in the message.

    Heuristic: look for "X for Y" / "X but with Y instead of Z" phrases.
    Returns a list (often 0 or 1 entries for v1).
    """
    swaps: list[tuple[str, str, str]] = []
    low = msg.lower()

    # "X for Y" — most common.  "quicksort for linked lists" → swap
    # data_structure: array → linked_list
    m = re.search(r"\b(\w+(?:\s+\w+){0,2})\s+for\s+(\w+(?:\s+\w+){0,2})\b",
                       low)
    if m:
        old_phrase = m.group(1).strip()
        new_phrase = m.group(2).strip()
        old_av = _lookup_axis_value(old_phrase)
        new_av = _lookup_axis_value(new_phrase)
        # If the old phrase is an algorithm and the new phrase is a
        # data structure, infer the swap: assume the canonical version
        # was for "array" / "list", swap to the new data structure.
        if (old_av and new_av
                and old_av[0] == "algorithm"
                and new_av[0] == "data_structure"):
            swaps.append(("data_structure", "array", new_av[1]))

    # "X but Y" — replace one feature
    m = re.search(r"\bbut\s+(\w+(?:\s+\w+){0,2})\b", low)
    if m:
        phrase = m.group(1).strip()
        av = _lookup_axis_value(phrase)
        if av:
            # Infer the default opposite
            default_opposite = {
                "in_place":   "functional",
                "recursive":  "iterative",
                "iterative":  "recursive",
                "parallel":   None,
                "async":      None,
            }.get(av[1])
            if default_opposite:
                swaps.append((av[0], default_opposite, av[1]))

    # "make it parallel / make it async / make it in-place"
    m = re.search(r"\bmake\s+it\s+(\w+(?:[- ]\w+)?)\b", low)
    if m:
        phrase = m.group(1).strip()
        av = _lookup_axis_value(phrase)
        if av and av[0] == "modifier":
            swaps.append((av[0], "iterative", av[1]))   # rough default

    return swaps


# ─── Analogical query ────────────────────────────────────────────


def analogical_query(base_description_hv: np.ndarray,
                          swaps: list[tuple[str, str, str]]) -> np.ndarray:
    """Apply axis-value swaps to a base description HV.

    Returns base - basis(axis, old) + basis(axis, new), per swap.  HDC
    XOR makes "subtraction" = "addition" (it's the same operation in
    the binary field), so we use bind/bundle to compose.
    """
    q = base_description_hv.astype(np.int8).copy()
    for axis, old_v, new_v in swaps:
        old_hv = _basis(axis, old_v)
        new_hv = _basis(axis, new_v)
        # XOR-out old, XOR-in new — for binary HVs these compose nicely.
        q = bind(q, old_hv)
        q = bind(q, new_hv)
    return q


def try_code_analogy(msg: str, corpus, encoder) -> Optional[dict]:
    """Try to answer a code-request via analogy on a corpus snippet.

    Steps:
      1. Detect swaps from the message.
      2. If at least one swap, find the closest base snippet to the
         remaining (non-swap) message.
      3. Apply the swap to that base's HV → analogical query Q.
      4. Find the closest snippet to Q.  Return it with a note that
         it's an analogical adaptation.

    Returns a dict similar to try_retrieve_code, or None.
    """
    if corpus is None or corpus.count() == 0:
        return None
    swaps = detect_axis_swaps(msg)
    if not swaps:
        return None

    # 1) Encode the message + find closest base
    q_text_hv = encoder.encode(msg).astype(np.int8)

    if corpus._stack is None:
        return None
    xor = np.bitwise_xor(corpus._stack, q_text_hv[None, :])
    dists = xor.sum(axis=1)
    base_idx = int(np.argmin(dists))
    base_hv  = corpus._stack[base_idx]
    base_sim = 1.0 - 2.0 * int(dists[base_idx]) / D

    # 2) Apply analogical swap to the base's stored HV
    analog_q = analogical_query(base_hv, swaps)

    # 3) Find the closest snippet to the analogical query
    xor2 = np.bitwise_xor(corpus._stack, analog_q[None, :])
    dists2 = xor2.sum(axis=1)
    target_idx = int(np.argmin(dists2))
    target_sim = 1.0 - 2.0 * int(dists2[target_idx]) / D

    base_desc = corpus._descriptions[base_idx]
    target_desc = corpus._descriptions[target_idx]
    target_code = corpus._codes[target_idx]

    # Format swap descriptions
    swap_strs = [f"{axis}: {old} → {new}" for axis, old, new in swaps]

    return {
        "code":         target_code,
        "label":        target_desc,
        "template":     f"analogy({base_desc!r} {' + '.join(swap_strs)})",
        "ran":          False,
        "output":       "",
        "customized":   False,
        "analogy": {
            "base":        base_desc,
            "base_sim":    round(base_sim, 4),
            "target":      target_desc,
            "target_sim":  round(target_sim, 4),
            "swaps":       swap_strs,
        },
    }


def _self_test():
    from mind.code_corpus import CodeCorpus, SNIPPETS
    print(f"  detected swaps for various phrasings:")
    examples = [
        "quicksort for linked lists",
        "merge sort for floats",
        "binary search for strings",
        "make it parallel",
        "moving average but recursive",
        "what is einstein known for",   # no swap expected
    ]
    for msg in examples:
        swaps = detect_axis_swaps(msg)
        print(f"  {msg!r:<45} → {swaps}")


if __name__ == "__main__":
    _self_test()
