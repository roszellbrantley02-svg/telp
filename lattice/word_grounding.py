"""
lattice/word_grounding.py - Phase 12.1: meaning, not just rhyme.

WHY
---
Phase 12 v0.1 taught Telp to predict "star" after "twinkle twinkle
little" by pattern.  But he had no idea what a star IS.

This module gives every word a MEANING substrate, anchored to two
parallel grounding tracks (the user's correction):

  1. DEFINITION — WordNet gloss + semantic relations (is-a, has-part,
     examples). 117K English words, free, comes with nltk.
  2. PICTURE — for v0.1 we don't have a visual referent dataset for
     every noun.  We use the WordNet gloss as a STAND-IN visual
     descriptor (glosses are often visual: "a celestial body that
     produces its own light").  Future Phase 12.2 can swap in
     actual image HVs.

GROUNDING PIPELINE
------------------
For each word in the reading lattice:
  1. Look up WordNet synset (best match for the word's likely sense)
  2. Build a CONCEPT HV by bundling:
       - bind(R_DEFINITION, gloss_hv)    — the textual definition
       - bind(R_IS_A,        hypernym_hv) — "star IS_A celestial_body"
       - bind(R_LEXNAME,     lexname_hv)  — WordNet semantic field
                                             (noun.location, noun.animal, etc.)
       - bind(R_EXAMPLE,     example_hv)  — sample usages
  3. Attach via lattice.bind_semantic(word, "concept", concept_hv)
     and also store the raw gloss text in word_meanings[word]

NOW Telp can:
  - define("star")         -> textual definition
  - concept_hv("star")     -> the meaning HV
  - similar_concepts("star") -> other words with similar CONCEPT HVs
                                  (would find "moon", "sun", etc.
                                   if those are in his vocab)

USAGE
-----
    from lattice.word_grounding import ground_all_lattice_words
    n = ground_all_lattice_words(lattice)
    # Now every word has both phonemic and semantic structure
"""
from __future__ import annotations
import sys
from typing import Optional

import numpy as np

from train.v5_hdc_prototype import D, bind, bundle
from lattice.phoneme_hdc import (
    compose_word, compose_sentence, _det_hv, similarity,
)
from lattice.g2p_and_prosody import word_to_phonemes


# --- Grounding roles ----------------------------------------------


R_DEFINITION = _det_hv("role::definition")
R_IS_A        = _det_hv("role::is_a")
R_LEXNAME     = _det_hv("role::lexname")
R_EXAMPLE     = _det_hv("role::example")
R_SYNSET      = _det_hv("role::synset")


# --- WordNet wrapper ---------------------------------------------


_WN = None


def _wn():
    global _WN
    if _WN is None:
        import nltk
        from nltk.corpus import wordnet
        try:
            wordnet.synsets("test")
            _WN = wordnet
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            _WN = wordnet
    return _WN


def _best_synset(word: str):
    """Pick the most likely sense for a word.  Heuristic:
      - Prefer nouns for content words, verbs for action words
      - Take the first synset (WordNet orders by frequency)
    """
    wn = _wn()
    # Try noun first (most common content sense), then verb, then any
    for pos in (wn.NOUN, wn.VERB, wn.ADJ, wn.ADV):
        syns = wn.synsets(word, pos=pos)
        if syns:
            return syns[0]
    syns = wn.synsets(word)
    return syns[0] if syns else None


# --- HV composition from text ------------------------------------


def text_to_hv(text: str) -> np.ndarray:
    """Encode a short text passage to one HV by composing its words
    through the phoneme path.  This is the same machinery the lattice
    uses for line composition; here we apply it to definition strings.
    """
    if not text:
        return np.zeros(D, dtype=np.int8)
    word_hvs = []
    for w in text.lower().split():
        # Strip punctuation
        clean = "".join(c for c in w if c.isalpha() or c == "'")
        if not clean:
            continue
        phon = [p for p, _ in word_to_phonemes(clean)]
        if phon:
            word_hvs.append(compose_word(phon))
    if not word_hvs:
        return np.zeros(D, dtype=np.int8)
    return compose_sentence(word_hvs)


# --- Build a concept HV for a single word ------------------------


def concept_hv_for(word: str) -> Optional[tuple[np.ndarray, dict]]:
    """Return (concept_hv, meta_dict) for `word`, or None if no WordNet
    entry exists.

    meta_dict contains: synset_name, definition, hypernyms, examples,
    lexname.
    """
    syn = _best_synset(word)
    if syn is None:
        return None

    gloss = syn.definition()
    hypers = [h.lemmas()[0].name() for h in syn.hypernyms()][:5]
    examples = syn.examples()[:3]
    lexname = syn.lexname()  # e.g. "noun.animal", "noun.location"
    synset_name = syn.name()

    components = []

    # 1. Synset name itself (canonical token for this concept)
    components.append(bind(R_SYNSET, _det_hv(f"synset::{synset_name}")))

    # 2. Lexical category (very compact + discriminative)
    components.append(bind(R_LEXNAME, _det_hv(f"lex::{lexname}")))

    # 3. Definition gloss -> HV via the phoneme pipeline
    if gloss:
        components.append(bind(R_DEFINITION, text_to_hv(gloss)))

    # 4. Hypernyms (is-a relations) as a bundle
    if hypers:
        hyper_hvs = [_det_hv(f"synset::{h}") for h in hypers]
        components.append(bind(R_IS_A, bundle(hyper_hvs)))

    # 5. Examples as a bundle
    if examples:
        ex_hvs = [text_to_hv(ex) for ex in examples]
        components.append(bind(R_EXAMPLE, bundle(ex_hvs)))

    concept = bundle(components)
    meta = {
        "synset":      synset_name,
        "definition":  gloss,
        "hypernyms":   hypers,
        "examples":    examples,
        "lexname":     lexname,
    }
    return concept, meta


# --- Ground all words in a reading lattice -----------------------


def ground_all_lattice_words(lattice, verbose: bool = True) -> dict:
    """Attach a concept HV + textual meta to every word in `lattice`.

    Stores under:
      lattice.word_semantics[word]["concept"]  = concept_hv (HV)
      lattice.word_meanings[word]              = meta dict
                                                  (added if missing)
    """
    if not hasattr(lattice, "word_meanings"):
        lattice.word_meanings = {}

    n_grounded = 0
    n_unknown  = 0
    for word in lattice.word_mem.labels():
        if word in lattice.word_meanings:
            continue
        result = concept_hv_for(word)
        if result is None:
            n_unknown += 1
            continue
        concept_hv, meta = result
        lattice.bind_semantic(word, "concept", concept_hv)
        lattice.word_meanings[word] = meta
        n_grounded += 1
        if verbose and n_grounded % 50 == 0:
            print(f"[ground] {n_grounded}/{len(lattice.word_mem)}",
                    file=sys.stderr)

    stats = {
        "n_grounded": n_grounded,
        "n_unknown":  n_unknown,
        "n_total":    len(lattice.word_mem),
    }
    if verbose:
        print(f"[ground] DONE: {stats}", file=sys.stderr)
    return stats


# --- Query APIs ---------------------------------------------------


def define(lattice, word: str) -> Optional[dict]:
    """What does `word` mean?  Returns the meta dict (definition,
    hypernyms, examples, lexname) or None if not grounded.
    """
    meanings = getattr(lattice, "word_meanings", {})
    return meanings.get(word)


def similar_concepts(lattice, word: str,
                              top_k: int = 5) -> list[tuple[str, float]]:
    """Find words whose CONCEPT HV is most similar to `word`'s concept HV.

    This is the semantic neighborhood — finds words that mean SIMILAR
    things, not just words that sound similar.

    Note: "concept similarity" via gloss-bundled HVs is approximate —
    two words with overlapping definition text will land near each other.
    """
    target_concept = lattice.word_semantics.get(word, {}).get("concept")
    if target_concept is None:
        return []
    scored = []
    for w in lattice.word_mem.labels():
        if w == word:
            continue
        ch = lattice.word_semantics.get(w, {}).get("concept")
        if ch is None:
            continue
        sim = similarity(target_concept, ch)
        scored.append((w, sim))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


# --- CLI smoke test ----------------------------------------------


if __name__ == "__main__":
    import argparse
    from lattice.reading_lattice import ReadingLattice
    p = argparse.ArgumentParser(description="Ground lattice words via WordNet")
    p.add_argument("--word", default=None,
                       help="Show meaning + neighbors for a single word")
    args = p.parse_args()

    lattice = ReadingLattice.load()
    print(f"[ground] lattice loaded: {len(lattice.word_mem)} words",
            file=sys.stderr)

    stats = ground_all_lattice_words(lattice)
    lattice.save()
    print(f"[ground] saved lattice with grounded meanings",
            file=sys.stderr)
    print()

    test_words = [args.word] if args.word else [
        "star", "bear", "spider", "lamb", "wonder",
        "twinkle", "diamond", "fleece", "fiddle", "spoon",
    ]
    for w in test_words:
        meaning = define(lattice, w)
        if meaning is None:
            print(f"'{w}': (no meaning grounded)\n")
            continue
        print(f"'{w}'  [{meaning['lexname']}]")
        print(f"  def: {meaning['definition']}")
        if meaning["hypernyms"]:
            print(f"  is-a: {' -> '.join(meaning['hypernyms'][:3])}")
        if meaning["examples"]:
            print(f"  ex:  {meaning['examples'][0]!r}")
        neighbors = similar_concepts(lattice, w, top_k=4)
        if neighbors:
            ns = ", ".join(f"{n}({s:.2f})" for n, s in neighbors)
            print(f"  similar in concept: {ns}")
        print()
