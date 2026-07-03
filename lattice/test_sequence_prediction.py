"""
lattice/test_sequence_prediction.py - can HDC predict next words?

Three tests:

  T1: training-set recall — give the model a prefix it has seen.
       Should return the correct next word (memorization sanity).

  T2: held-out generation — give the model novel prefixes derived
       from country facts NOT in training. See if it generalizes.

  T3: free generation — give a starter phrase, generate continuations.
       Inspect coherence.

This is the tiny HDC LLM. Results inform whether HDC can plausibly
do language generation at all.

Usage:
    python -m lattice.test_sequence_prediction
"""
from __future__ import annotations

import sys
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.sequence_predictor import HDCSequencePredictor, tokenize
from lattice.corpus import build_corpus, COUNTRIES
from train.v5_hdc_prototype import D


def main():
    print("HDC Sequence Prediction — tiny HDC LLM\n")

    print("Building country-fact corpus ...")
    corpus = build_corpus()
    print(f"  {len(corpus)} sentences")

    # Hold out the last few countries' sentences entirely for generalization test
    held_out_country_names = {c[0] for c in COUNTRIES[27:]}
    train_sents = [s for s in corpus
                     if not any(name in s for name in held_out_country_names)]
    held_out_sents = [s for s in corpus
                        if any(name in s for name in held_out_country_names)]
    print(f"  train: {len(train_sents)}, held-out: {len(held_out_sents)}")

    print("\nTraining HDC sequence predictor (n-gram=3) ...")
    model = HDCSequencePredictor(n_gram=3, dim=D, seed=42)
    model.train(train_sents)
    stats = model.stats()
    print(f"  vocab: {stats['vocab']}, memories: {stats['memories']}")

    # ── TEST 1: training-set recall ──
    print("\n=== TEST 1: training-set recall ===")
    print("  Give the model 100 random training prefixes; check if it recalls")
    print("  the correct next word.\n")
    import random
    rng = random.Random(0)
    sample_sents = rng.sample(train_sents, min(100, len(train_sents)))
    correct = 0
    tested = 0
    for sent in sample_sents:
        tokens = tokenize(sent)
        if len(tokens) < 4:
            continue
        # Pick a random position
        i = rng.randrange(2, len(tokens) - 1)
        prefix = tokens[max(0, i-2):i+1]
        expected = tokens[i+1]
        result = model.predict(prefix)
        ok = (result["next_word"] == expected)
        correct += int(ok)
        tested += 1
    print(f"  Training-set next-word accuracy: {correct}/{tested} = "
          f"{correct/tested:.0%}")

    # ── TEST 2: HELDOUT recall ──
    print("\n=== TEST 2: HELDOUT country sentences ===")
    print("  These sentences are about countries the model never saw.")
    print("  Tests: does it generalize the templated structure?\n")
    correct_h = 0
    tested_h = 0
    examples_shown = 0
    for sent in held_out_sents:
        tokens = tokenize(sent)
        if len(tokens) < 4:
            continue
        for i in range(2, len(tokens) - 1):
            prefix = tokens[max(0, i-2):i+1]
            expected = tokens[i+1]
            result = model.predict(prefix)
            ok = (result["next_word"] == expected)
            correct_h += int(ok)
            tested_h += 1
            if examples_shown < 8 and ok:
                print(f"  [OK] '{' '.join(prefix)}' -> '{result['next_word']}' "
                      f"(d={result['nearest_distance']}, "
                      f"conf={result['confidence']:.2f})")
                examples_shown += 1
            elif examples_shown < 12 and not ok:
                print(f"  [MISS] '{' '.join(prefix)}' -> '{result['next_word']}' "
                      f"(expected '{expected}', d={result['nearest_distance']})")
                examples_shown += 1
    print(f"\n  HELDOUT next-word accuracy: {correct_h}/{tested_h} = "
          f"{correct_h/tested_h:.0%}")

    # ── TEST 3: free generation ──
    print("\n=== TEST 3: free generation (starter prompts) ===")
    starters = [
        "the capital of france",
        "germany is a country",
        "people in italy",
        "tourists visit egypt",
        "the currency used in japan",
    ]
    for prompt in starters:
        gen = model.generate(prompt, n_words=10, temperature=0.0)
        print(f"\n  PROMPT: '{prompt}'")
        print(f"  CONT:   '{gen}'")

    # ── TEST 4: temperature sampling ──
    print("\n=== TEST 4: temperature sampling (variety from same prompt) ===")
    prompt = "the capital of"
    print(f"  PROMPT: '{prompt}'")
    print(f"  Greedy:    '{model.generate(prompt, n_words=8, temperature=0.0)}'")
    for t in [0.5, 1.0, 1.5]:
        # Re-seed the model's RNG for variety
        model.rng = __import__('numpy').random.default_rng(42 + int(t*10))
        gen = model.generate(prompt, n_words=8, temperature=t)
        print(f"  T={t:.1f}:     '{gen}'")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(f"  Training-set recall:  {correct/tested:.0%}")
    print(f"  HELDOUT generalization: {correct_h/tested_h:.0%}")
    if correct_h / tested_h >= 0.50:
        print("\n  WORKS: HDC sequence model generalizes via prefix similarity.")
        print("  Tiny HDC LLM is real — at least for templated text.")
    elif correct_h / tested_h >= 0.25:
        print("\n  PARTIAL: HDC catches some patterns but not all.")
    else:
        print("\n  WEAK: HDC doesn't generalize well; mostly memorization.")


if __name__ == "__main__":
    main()
