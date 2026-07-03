"""
lattice/morphology.py - Phase 12.3: decompose words into root + affixes.

WHY
---
running, runs, runner, ran all SHARE THE ROOT `run`.  Without
morpheme decomposition, the lattice treats them as four unrelated
words with arbitrary HV similarity.  WITH morpheme decomposition,
each one's identity HV bundles in the root, so they cluster naturally.

This is the OTHER half of "spelling and the formation of smaller
words" — not just "what letters are in this word" (handled by
grapheme_hdc) but "what MEANING UNITS are in this word."

APPROACH
--------
Hand-coded suffix + prefix table, not a learned stemmer.  English
has ~20 common derivational/inflectional affixes that cover most of
what a toddler corpus needs.  We attempt to strip them iteratively:

  running  -> run    + [-ing]
  bears    -> bear   + [-s]
  walked   -> walk   + [-ed]
  runner   -> run    + [-er]
  unhappy  -> happy  + [un-]
  unhappily-> happy  + [un-, -ly]

For each word we return:
  {"root": str, "prefixes": [str,...], "suffixes": [str,...]}

The morpheme HV for a word can then be:
  bundle( bind(R_ROOT,   spelling_hv(root)),
            bind(R_PREFIX, bundle(spelling_hv(p) for p in prefixes)),
            bind(R_SUFFIX, bundle(spelling_hv(s) for s in suffixes)) )

So `running` and `runs` share their R_ROOT component while differing
on R_SUFFIX -> they cluster in the morpheme subspace.
"""
from __future__ import annotations
from typing import Optional

import numpy as np

from train.v5_hdc_prototype import D, bind, bundle
from lattice.phoneme_hdc import _det_hv, similarity
from lattice.grapheme_hdc import spelling_hv


# --- Affix tables ------------------------------------------------
#
# Order matters: we try longer suffixes first (so we don't strip "-s"
# from "running" before "-ing").

SUFFIXES = [
    # 4+ letter suffixes
    "tion", "sion", "ment", "ness", "ship", "able", "ible", "less",
    "ward", "wise",
    # 3 letter
    "ing", "ies", "est", "ful",
    # 2 letter
    "ed", "er", "ly",
    # 1 letter — try before "es" so "places" -> "place" + "s" not "plac" + "es"
    "s",
    # 2 letter -es (after -s)
    "es",
]

PREFIXES = [
    # 3+ letter prefixes
    "anti", "auto", "over", "pre",
    # 2 letter
    "un", "re", "in", "im", "il", "ir", "de", "ex",
    "be",
]

# Common roots that we should NEVER strip into (i.e., words that
# look like they have a suffix but actually don't)
SUFFIX_EXCEPTIONS = {
    "is", "as", "us", "this", "his", "has", "gas", "bus", "yes",
    "us", "thus", "less", "miss", "kiss", "boss", "loss",
    "spring", "string", "sing", "king", "ring", "thing", "wing",
    "bring", "swing",
    "red", "bed", "fed", "led", "wed", "shed",
    "her", "per", "for", "or", "her", "over", "ever", "never",
    "by", "my", "why", "ply", "fly",
    # Already monosyllabic — don't split
    "the", "and", "but",
}

PREFIX_EXCEPTIONS = {
    # Common words that start with prefix-like letters but aren't prefixed
    "under", "until", "use", "uncle", "into", "inside", "indeed",
    "be", "between", "before", "behind",
    "red", "real", "really", "rest",
    "exit", "extra",
    "imagine", "impossible",
}


def _looks_like_real_root(candidate: str, vocab: Optional[set] = None) -> bool:
    """A stripped candidate is a "real root" if:
      - it's in the supplied vocab, OR
      - it's at least 3 chars (heuristic: shorter is often noise)
    """
    if not candidate or len(candidate) < 2:
        return False
    if vocab is not None and candidate in vocab:
        return True
    return len(candidate) >= 3


def decompose(word: str, vocab: Optional[set] = None) -> dict:
    """Decompose a word into root + prefixes + suffixes.

    `vocab` (optional) is a set of known root words; if provided, we
    only strip an affix when the resulting root is a real known word
    OR is at least 3 chars long.

    Returns: {"root": str, "prefixes": [str,...], "suffixes": [str,...],
              "original": str}
    """
    w = word.lower().strip()
    if w in SUFFIX_EXCEPTIONS or w in PREFIX_EXCEPTIONS:
        return {"root": w, "prefixes": [], "suffixes": [],
                  "original": word}

    prefixes: list[str] = []
    suffixes: list[str] = []
    root = w

    # SUFFIXES FIRST — English morphology is suffix-heavier and
    # prefix stripping must not eat genuine word stems like "bear"
    # via a spurious "be-" prefix.  Strip suffixes iteratively
    changed = True
    while changed:
        changed = False
        if root in SUFFIX_EXCEPTIONS:
            break
        for s in SUFFIXES:
            if root.endswith(s) and len(root) > len(s) + 1:
                candidate = root[:-len(s)]
                # Doubled-consonant handling: "running" -> "run" + "ing"
                #   the second 'n' was added to preserve short vowel sound
                if (len(candidate) >= 2
                        and candidate[-1] == candidate[-2]
                        and candidate[-1] not in "aeiou"
                        and candidate[-1] not in "fls"):
                    candidate2 = candidate[:-1]
                    if _looks_like_real_root(candidate2, vocab):
                        suffixes.append(s)
                        root = candidate2
                        changed = True
                        break
                # E-restoration: "baking" -> "bake" + "ing", "loved" -> "love" + "ed"
                if s in ("ing", "ed", "er", "est"):
                    candidate_e = candidate + "e"
                    if (vocab is not None
                            and candidate_e in vocab
                            and candidate not in vocab):
                        suffixes.append(s)
                        root = candidate_e
                        changed = True
                        break
                # Y-to-i: "happiness" -> "happy" + "ness", "tries" -> "try" + "es"
                if candidate.endswith("i"):
                    candidate_y = candidate[:-1] + "y"
                    if _looks_like_real_root(candidate_y, vocab):
                        suffixes.append(s)
                        root = candidate_y
                        changed = True
                        break
                if _looks_like_real_root(candidate, vocab):
                    suffixes.append(s)
                    root = candidate
                    changed = True
                    break

    # NOW prefix stripping (after suffix-stripping has settled the root)
    changed = True
    while changed:
        changed = False
        if root in PREFIX_EXCEPTIONS:
            break
        # Short prefixes (2-letter: un, re, in, be, de, ex) are too
        # easy to false-positive on. Require resulting root to be in
        # vocab, NOT just "long enough."
        for p in PREFIXES:
            if root.startswith(p) and len(root) > len(p) + 1:
                candidate = root[len(p):]
                # Require vocab confirmation for short prefixes
                if len(p) <= 2 and vocab is not None:
                    if candidate not in vocab:
                        continue
                if _looks_like_real_root(candidate, vocab):
                    prefixes.append(p)
                    root = candidate
                    changed = True
                    break

    return {
        "root":     root,
        "prefixes": prefixes,
        "suffixes": suffixes,
        "original": word,
    }


# --- Morpheme HV composition --------------------------------------


R_ROOT   = _det_hv("morph_role::root")
R_PREFIX = _det_hv("morph_role::prefix")
R_SUFFIX = _det_hv("morph_role::suffix")


def morpheme_hv(word: str, vocab: Optional[set] = None) -> np.ndarray:
    """Bundle a word's morphemes into a single HV.

    Words sharing the same ROOT have similar morpheme HVs even when
    their full spelling differs:
      morpheme_hv("running") shares R_ROOT-bound part with
      morpheme_hv("runs"), morpheme_hv("ran" — if listed), etc.
    """
    decomp = decompose(word, vocab=vocab)
    components = []
    # Root is always present
    components.append(bind(R_ROOT, spelling_hv(decomp["root"])))
    # Prefixes (bundle of all prefix HVs)
    if decomp["prefixes"]:
        pref_hvs = [spelling_hv(p) for p in decomp["prefixes"]]
        components.append(bind(R_PREFIX, bundle(pref_hvs)))
    # Suffixes
    if decomp["suffixes"]:
        suff_hvs = [spelling_hv(s) for s in decomp["suffixes"]]
        components.append(bind(R_SUFFIX, bundle(suff_hvs)))
    return bundle(components)


# --- Smoke test --------------------------------------------------


if __name__ == "__main__":
    # Test on a sample vocab so the stripper has anchors
    vocab = {
        "run", "bear", "walk", "jump", "happy", "place", "sad", "see",
        "look", "watch", "play", "love", "bake", "try", "kind", "love",
        "use", "view", "act", "form", "list", "head", "hand",
    }

    print("=== Morpheme decomposition ===\n")
    test_words = [
        # Inflectional
        "running", "runs", "runner",
        "bears", "walked", "walking", "jumps", "jumped",
        "happier", "happiest",
        # Derivational
        "kindness", "happiness", "useful", "useless",
        "remake", "unhappy", "unhappily",
        "actor", "actors", "action",
        # Tricky exceptions
        "this", "his", "less", "king", "ring", "bring",
        "the", "and", "fox",
        # Doubled consonant
        "running", "running",
        # E-restoration
        "baked", "loving", "places",
        # Y-to-i
        "tries", "happiness",
    ]
    for w in test_words:
        d = decompose(w, vocab=vocab)
        parts = []
        if d["prefixes"]:
            parts.append(f"prefixes={d['prefixes']}")
        parts.append(f"root={d['root']!r}")
        if d["suffixes"]:
            parts.append(f"suffixes={d['suffixes']}")
        print(f"  {w:12} -> {'  '.join(parts)}")

    print(f"\n=== Morpheme HV similarity (root sharing) ===\n")
    pairs = [
        ("running", "runs",    "share 'run' root"),
        ("running", "runner",  "share 'run' root"),
        ("running", "walking", "share '-ing' suffix only"),
        ("bears",   "bear",    "bears = bear + s"),
        ("kindness","unkind",  "share 'kind' root"),
        ("happy",   "happiness","share 'happy' root (with y->i)"),
        ("running", "jumping", "share '-ing' suffix, different roots"),
        ("running", "bear",    "no shared morphemes"),
    ]
    for w1, w2, note in pairs:
        s = similarity(morpheme_hv(w1, vocab),
                              morpheme_hv(w2, vocab))
        bar = "=" * int(round((s - 0.4) * 50)) if s > 0.4 else ""
        print(f"  '{w1:10}' vs '{w2:10}'  morph_sim={s:.3f}  {bar}  {note}")
