"""
lattice/prediction.py - multi-level prediction from the reading lattice.

Five prediction APIs, one per resolution level:
  next_phoneme(context_phonemes)  -- what sound comes next?
  next_word(context_words)        -- what word comes next?
  next_phrase(context_phrase_hv)  -- what phrase comes next?
  next_line(context_line_hv)      -- what line comes next?
  complete(prompt_text)           -- end-to-end: read prompt, predict next word

Each level uses BOTH:
  1. Statistical co-occurrence (n-gram counts from observations)
  2. HDC similarity + cleanup (compose context HV, snap to nearest)

The HDC path is what makes this developmental rather than n-gram-only:
even on never-seen contexts, the HV similarity finds the closest
remembered pattern.
"""
from __future__ import annotations
import sys
from collections import Counter
from typing import Optional

import numpy as np

from train.v5_hdc_prototype import D, bind, bundle, hamming_distance
from lattice.phoneme_hdc import (
    phoneme_hv, compose_word, compose_sentence, position_hv, similarity,
)
from lattice.g2p_and_prosody import word_to_phonemes, text_to_units
from lattice.reading_lattice import ReadingLattice


# --- Word-level prediction ----------------------------------------


def next_word(lattice: ReadingLattice,
                  context_words: list[str],
                  top_k: int = 5,
                  hdc_blend: float = 0.5) -> list[tuple[str, float]]:
    """Predict the most likely next word given `context_words`.

    Combines two signals:
      1. Statistical: count(context -> next_word) from observed n-grams
      2. HDC: compose context HV from observed words, find words whose
         stored HV is most similar to the "expected next slot"

    Returns top_k (word, score) pairs.  Score is normalized in [0, 1].
    """
    if not context_words:
        return _global_top_words(lattice, top_k)

    # ── 1. Statistical: trigram first, then bigram fallback ─────
    stat_counts: Counter = Counter()
    if len(context_words) >= 2:
        key3 = (context_words[-2], context_words[-1])
        if key3 in lattice.next_word_counts_3:
            stat_counts.update(lattice.next_word_counts_3[key3])
    if not stat_counts and len(context_words) >= 1:
        key2 = context_words[-1]
        if key2 in lattice.next_word_counts:
            stat_counts.update(lattice.next_word_counts[key2])

    stat_total = sum(stat_counts.values()) or 1
    stat_scores = {w: c / stat_total for w, c in stat_counts.items()}

    # ── 2. HDC: compose context HV, find similar lines, extract next word ──
    hdc_scores: dict[str, float] = {}
    # Build context HV from last 2 words (matches line composition)
    available = [w for w in context_words[-3:]
                       if lattice.word_mem.has(w)]
    if available:
        word_hvs = [lattice.word_mem.get_hv(w) for w in available]
        ctx_hv = compose_sentence(word_hvs)
        # Find nearest stored LINES (lines are bundles of word HVs)
        top_lines = lattice.line_mem.snap(ctx_hv, top_k=5)
        # From each matched line, find the word that comes right after
        # the context phrase
        for line_label, line_sim in top_lines:
            line_words = line_label.split()
            # Find context in line_words and grab the next word
            for i in range(len(line_words) - len(available)):
                if line_words[i:i + len(available)] == available:
                    if i + len(available) < len(line_words):
                        nxt = line_words[i + len(available)]
                        hdc_scores[nxt] = max(hdc_scores.get(nxt, 0.0),
                                                          line_sim)
                    break

    # ── Blend ─────────────────────────────────────────────────────
    all_candidates = set(stat_scores.keys()) | set(hdc_scores.keys())
    blended = {
        w: ((1 - hdc_blend) * stat_scores.get(w, 0.0)
              + hdc_blend     * hdc_scores.get(w, 0.0))
        for w in all_candidates
    }
    if not blended:
        return _global_top_words(lattice, top_k)
    ranked = sorted(blended.items(), key=lambda x: -x[1])[:top_k]
    return ranked


def _global_top_words(lattice: ReadingLattice,
                              top_k: int = 5) -> list[tuple[str, float]]:
    """Fallback: most frequent words (when no context match)."""
    total = sum(lattice.word_freq.values()) or 1
    return [(w, c / total) for w, c in lattice.word_freq.most_common(top_k)]


# --- Phoneme-level prediction -------------------------------------


def next_phoneme(lattice: ReadingLattice,
                       context_phonemes: list[str],
                       top_k: int = 5) -> list[tuple[str, float]]:
    """Predict the next phoneme given a partial phoneme sequence.

    Uses HDC similarity: compose the context phoneme HV, then for each
    candidate phoneme in the vocab, check which extends the bundle best.

    Statistical signal comes from EVERY word in the vocabulary that
    starts with the context_phonemes — what phoneme comes next in those?
    """
    if not context_phonemes:
        return []

    # Statistical: scan vocabulary for phoneme prefixes matching context
    stat_counts: Counter = Counter()
    n_ctx = len(context_phonemes)
    for word in lattice.word_mem.labels():
        wp = [p for p, _ in word_to_phonemes(word)]
        if len(wp) > n_ctx and wp[:n_ctx] == context_phonemes:
            # Weight by frequency the word was observed
            stat_counts[wp[n_ctx]] += lattice.word_freq.get(word, 1)

    if not stat_counts:
        # No prefix match — fall back to most-likely-anywhere
        for word in lattice.word_mem.labels():
            wp = [p for p, _ in word_to_phonemes(word)]
            for p in wp:
                stat_counts[p] += lattice.word_freq.get(word, 1)

    total = sum(stat_counts.values()) or 1
    ranked = sorted(stat_counts.items(),
                          key=lambda x: -x[1])[:top_k]
    return [(p, c / total) for p, c in ranked]


# --- Phrase-level prediction --------------------------------------


def next_phrase_after_line(lattice: ReadingLattice,
                                       last_line_text: str,
                                       top_k: int = 3) -> list[tuple[str, float]]:
    """Given a line just observed, predict the most likely next line.

    Uses HDC similarity: compose the line HV, find stored lines that
    immediately followed similar lines.  v0.1 — simpler than it could be,
    just snap-to-nearest-line + return alternatives.
    """
    units = text_to_units(last_line_text)
    line_words = [u for u in units if "word" in u]
    if not line_words:
        return []
    word_hvs = [lattice.word_mem.get_hv(u["word"])
                       for u in line_words
                       if lattice.word_mem.has(u["word"])]
    if not word_hvs:
        return []
    ctx_hv = compose_sentence(word_hvs)
    return lattice.line_mem.snap(ctx_hv, top_k=top_k)


# --- End-to-end "complete the prompt" -----------------------------


def complete(lattice: ReadingLattice,
                  prompt: str,
                  n_words: int = 5,
                  top_k_per_step: int = 5) -> list[str]:
    """Given a text prompt, predict the next n words by greedy decoding."""
    units = text_to_units(prompt)
    ctx_words = [u["word"] for u in units if "word" in u]
    out: list[str] = []
    for _ in range(n_words):
        preds = next_word(lattice, ctx_words, top_k=top_k_per_step)
        if not preds:
            break
        choice = preds[0][0]
        out.append(choice)
        ctx_words.append(choice)
    return out


# --- CLI smoke -----------------------------------------------------


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Multi-level prediction tests")
    p.add_argument("--prompt", default="twinkle twinkle little",
                       help="Word prompt")
    p.add_argument("--n", type=int, default=8,
                       help="Words to generate")
    args = p.parse_args()

    lattice = ReadingLattice.load()
    print(f"[predict] lattice loaded: vocab={len(lattice.word_mem)} "
          f"lines={len(lattice.line_mem)}\n", file=sys.stderr)

    # Test 1: Word-level prediction
    print(f"=== WORD prediction ===")
    contexts = [
        ["twinkle", "twinkle", "little"],
        ["how", "i"],
        ["mary", "had", "a"],
        ["old", "macdonald", "had"],
        ["itsy", "bitsy"],
        ["humpty", "dumpty"],
        ["row", "row"],
    ]
    for ctx in contexts:
        top = next_word(lattice, ctx, top_k=3)
        ctx_str = " ".join(ctx)
        if top:
            preds = ", ".join(f"{w}={s:.2f}" for w, s in top)
            print(f"  '{ctx_str}' -> {preds}")
        else:
            print(f"  '{ctx_str}' -> (no prediction)")

    # Test 2: Phoneme-level prediction
    print(f"\n=== PHONEME prediction ===")
    phoneme_ctxs = [
        ["T", "W"],         # twinkle starts T-W-IH-NG-K-AH-L
        ["B", "EH"],        # bear starts B-EH-R
        ["AY"],             # iy / ay / etc. — vowel-initial words
        ["HH", "AH"],       # how starts HH-AW
    ]
    for pctx in phoneme_ctxs:
        top = next_phoneme(lattice, pctx, top_k=3)
        ctx_str = " ".join(pctx)
        if top:
            preds = ", ".join(f"{p}={s:.2f}" for p, s in top)
            print(f"  [{ctx_str}] -> {preds}")

    # Test 3: Line continuation
    print(f"\n=== LINE prediction (next line after) ===")
    line_prompts = [
        "Twinkle twinkle little star",
        "Mary had a little lamb",
        "Humpty Dumpty sat on a wall",
    ]
    for lp in line_prompts:
        top = next_phrase_after_line(lattice, lp, top_k=3)
        if top:
            preds = "\n    ".join(f"{l!r} ({s:.2f})" for l, s in top)
            print(f"  '{lp}' -> nearest lines:\n    {preds}")

    # Test 4: End-to-end completion (fill-in-the-blank)
    print(f"\n=== COMPLETION (greedy {args.n} words) ===")
    completion = complete(lattice, args.prompt, n_words=args.n)
    print(f"  '{args.prompt}' -> {' '.join(completion)}")
