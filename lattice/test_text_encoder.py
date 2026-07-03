"""
lattice/test_text_encoder.py - validate the bridge between language and HD.

Four tests, each a yes/no on a load-bearing claim:

  1. Semantic similarity: similar sentences land close in HD space.
  2. Semantic dissimilarity: unrelated sentences land far apart.
  3. Compositional algebra: substituting a word in a sentence via
     XOR/bundle algebra produces a hypervector closer to the target
     sentence than to random sentences.
  4. Retrieval by paraphrase: store 100 sentences, query with a
     paraphrase, the original sentence is the nearest neighbor.

If all four pass, HDC handles language well enough to be the substrate
of a real mind. If any fail, we know what to fix before building more.

Usage:
    python -m lattice.test_text_encoder
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import D, hamming_distance
from lattice.text_encoder import TextEncoder, hv_distance_pct, hv_similarity


# ─── Test 1: semantic similarity ───────────────────────────────────


SIMILAR_PAIRS = [
    ("I love my daughter.", "I adore my child."),
    ("The market crashed today.", "Stocks plummeted this afternoon."),
    ("It is raining outside.", "There's a downpour out there."),
    ("She is running fast.", "She sprinted quickly."),
    ("My coffee is hot.", "The coffee is scalding."),
    ("The cat is sleeping.", "A feline is dozing."),
    ("He laughed loudly.", "He chuckled with great volume."),
    ("The car broke down.", "The vehicle stopped working."),
]


def test_similarity(enc):
    print("\n=== TEST 1: similar sentences should land close ===")
    print(f"  {'sentence A':35s} | {'sentence B':35s} | ham% | sim")
    print(f"  {'-' * 35} | {'-' * 35} | ---- | -----")
    distances = []
    for a, b in SIMILAR_PAIRS:
        ha = enc.encode(a)
        hb = enc.encode(b)
        d = hv_distance_pct(ha, hb)
        s = hv_similarity(ha, hb)
        distances.append(d)
        print(f"  {a[:35]:35s} | {b[:35]:35s} | {d*100:4.1f}% | {s:+.2f}")
    mean_d = np.mean(distances)
    print(f"\n  Mean distance: {mean_d*100:.1f}% of D")
    print(f"  Pass if < 25% (similar should be much closer than random ~50%)")
    return mean_d < 0.25


# ─── Test 2: semantic dissimilarity ────────────────────────────────


DISSIMILAR_PAIRS = [
    ("I love my daughter.", "The S&P 500 closed at 5300 today."),
    ("It is raining outside.", "The cat is sleeping on the couch."),
    ("She is running fast.", "Coffee tastes bitter in the morning."),
    ("The market crashed.", "My favorite book is about dragons."),
    ("He laughed loudly.", "Mars has two small moons."),
    ("The car broke down.", "Python is a programming language."),
    ("My coffee is hot.", "The pyramids are in Egypt."),
    ("She painted a portrait.", "Quantum mechanics is counterintuitive."),
]


def test_dissimilarity(enc):
    print("\n=== TEST 2: dissimilar sentences should land FAR apart ===")
    print(f"  {'sentence A':35s} | {'sentence B':35s} | ham% | sim")
    print(f"  {'-' * 35} | {'-' * 35} | ---- | -----")
    distances = []
    for a, b in DISSIMILAR_PAIRS:
        ha = enc.encode(a)
        hb = enc.encode(b)
        d = hv_distance_pct(ha, hb)
        s = hv_similarity(ha, hb)
        distances.append(d)
        print(f"  {a[:35]:35s} | {b[:35]:35s} | {d*100:4.1f}% | {s:+.2f}")
    mean_d = np.mean(distances)
    print(f"\n  Mean distance: {mean_d*100:.1f}% of D")
    print(f"  Pass if > 40% (unrelated should approach random ~50%)")
    return mean_d > 0.40


# ─── Test 3: compositional algebra ─────────────────────────────────


def test_composition(enc):
    """Can we substitute concepts via XOR algebra?

    Test: encode("the happy cat is sleeping") minus encode("cat")
    plus encode("dog") should be closer to encode("the happy dog is
    sleeping") than to a random sentence.

    Note: this uses subtraction = XOR, addition = XOR (in binary HDC,
    bind and unbind are the SAME operation).
    """
    print("\n=== TEST 3: compositional algebra (substitute a word) ===")

    cases = [
        ("the happy cat is sleeping", "cat", "dog",
         "the happy dog is sleeping",
         "the spaceship landed on mars"),
        ("I love coffee in the morning", "coffee", "tea",
         "I love tea in the morning",
         "the dog barked at the moon"),
        ("she walked to the store", "store", "park",
         "she walked to the park",
         "physics is a difficult subject"),
    ]
    passes = 0
    for src, old_word, new_word, target, distractor in cases:
        h_src      = enc.encode(src)
        h_old      = enc.encode(old_word)
        h_new      = enc.encode(new_word)
        h_target   = enc.encode(target)
        h_distract = enc.encode(distractor)
        # Algebra: source XOR old XOR new = synthesized
        h_synth = np.bitwise_xor(np.bitwise_xor(h_src, h_old), h_new)
        d_target   = hv_distance_pct(h_synth, h_target)
        d_distract = hv_distance_pct(h_synth, h_distract)
        d_source   = hv_distance_pct(h_synth, h_src)
        ok = d_target < d_distract
        passes += int(ok)
        print(f"  SOURCE:    '{src}'")
        print(f"  + remove '{old_word}'  + add '{new_word}'  =>")
        print(f"     dist to TARGET    '{target[:40]}': {d_target*100:.1f}%")
        print(f"     dist to DISTRACT  '{distractor[:40]}': {d_distract*100:.1f}%")
        print(f"     dist to SOURCE    (unchanged):           {d_source*100:.1f}%")
        print(f"     {'PASS' if ok else 'FAIL'}: target closer than distractor? {ok}")
        print()
    print(f"  {passes}/{len(cases)} composition cases pass")
    return passes >= 2   # 2 out of 3 is a reasonable bar


# ─── Test 4: paraphrased retrieval ─────────────────────────────────


CORPUS = [
    "I had coffee with Sarah on Monday morning.",
    "The dog chased a squirrel through the yard.",
    "My presentation to the board went well yesterday.",
    "She finally finished writing her novel after three years.",
    "The car needs an oil change next week.",
    "We hiked up the mountain at sunrise.",
    "He learned to play guitar during the pandemic.",
    "The bakery on Main Street has the best bread.",
    "I bought a new pair of running shoes online.",
    "The thunderstorm knocked out power for six hours.",
    "My grandmother's recipe is the best in the family.",
    "The new restaurant downtown is way too expensive.",
    "We adopted a kitten from the shelter last spring.",
    "The job interview lasted longer than I expected.",
    "She gave an incredible speech at her wedding.",
    "I lost my keys somewhere in the park.",
    "The plane was delayed by three hours due to weather.",
    "My back has been hurting since I moved that couch.",
    "We watched the sunset over the ocean together.",
    "He proposed during their trip to Paris last June.",
    "The construction next door is unbearably loud.",
    "I finally beat my personal best in the marathon.",
    "The kids built an enormous sandcastle at the beach.",
    "We started a vegetable garden in the backyard.",
    "My dentist said I need a root canal next month.",
    "The concert was sold out within an hour.",
    "She graduated top of her class in medical school.",
    "The package arrived three days earlier than promised.",
    "I taught my son how to ride a bicycle today.",
    "The painting sold at auction for over a million dollars.",
]

QUERIES_AND_TARGETS = [
    ("Sarah and I got coffee Monday AM.",                                CORPUS[0]),
    ("A squirrel was pursued by our dog in the garden.",                 CORPUS[1]),
    ("Yesterday's board meeting went smoothly for me.",                  CORPUS[2]),
    ("After three years of work, she completed her book.",               CORPUS[3]),
    ("My car needs maintenance soon.",                                   CORPUS[4]),
    ("We climbed the mountain at dawn.",                                 CORPUS[5]),
    ("During COVID, he picked up guitar.",                               CORPUS[6]),
    ("Best bakery in town is on Main Street.",                           CORPUS[7]),
    ("Bought running shoes from the internet.",                          CORPUS[8]),
    ("Six-hour blackout from the storm.",                                CORPUS[9]),
    ("Grandma makes the best food in our family.",                       CORPUS[10]),
    ("The fancy new restaurant downtown is overpriced.",                 CORPUS[11]),
    ("Got a kitten from the animal shelter in spring.",                  CORPUS[12]),
    ("The interview took ages.",                                         CORPUS[13]),
    ("Her wedding speech was amazing.",                                  CORPUS[14]),
    ("My keys are missing — left them in the park.",                     CORPUS[15]),
    ("The flight was three hours late because of weather.",              CORPUS[16]),
    ("Moving that couch wrecked my back.",                               CORPUS[17]),
    ("We saw the sun set over the sea.",                                 CORPUS[18]),
    ("He asked her to marry him in Paris last summer.",                  CORPUS[19]),
    ("The next-door construction is so noisy.",                          CORPUS[20]),
    ("New personal record in my marathon time.",                         CORPUS[21]),
    ("Huge sandcastle on the beach with the kids.",                      CORPUS[22]),
    ("Planted vegetables in our yard.",                                  CORPUS[23]),
    ("Need a root canal soon, dentist said.",                            CORPUS[24]),
    ("Concert tickets sold out fast.",                                   CORPUS[25]),
    ("She was first in her med school class.",                           CORPUS[26]),
    ("Delivery came early.",                                             CORPUS[27]),
    ("Today my kid learned to ride a bike.",                             CORPUS[28]),
    ("Auction price for the painting hit seven figures.",                CORPUS[29]),
]


def test_retrieval(enc):
    print("\n=== TEST 4: paraphrased retrieval over 30-sentence corpus ===")
    db = enc.encode_many(CORPUS)
    db_stack = np.stack(db) if isinstance(db, list) else db
    correct_top1 = 0
    correct_top3 = 0
    for query, target in QUERIES_AND_TARGETS:
        q = enc.encode(query)
        xor = np.bitwise_xor(db_stack, q[None, :])
        dists = xor.sum(axis=1)
        order = np.argsort(dists)
        top1_idx = order[0]
        top3_idx = order[:3]
        target_idx = CORPUS.index(target)
        if top1_idx == target_idx:
            correct_top1 += 1
        if target_idx in top3_idx:
            correct_top3 += 1
    n = len(QUERIES_AND_TARGETS)
    print(f"  Top-1 accuracy: {correct_top1}/{n} = {correct_top1/n:.0%}")
    print(f"  Top-3 accuracy: {correct_top3}/{n} = {correct_top3/n:.0%}")
    print(f"  Pass if Top-1 >= 70% (paraphrased query finds original)")
    return correct_top1 / n >= 0.70


# ─── Main ──────────────────────────────────────────────────────────


def main():
    print("HDC text encoder validation suite\n")
    print("Loading MiniLM (first run downloads ~80MB) ...")
    enc = TextEncoder()
    print(f"  embedding dim: {enc.embed_dim}")
    print(f"  hypervector dim: {D}")

    results = {}
    results["1. similar sentences close"]      = test_similarity(enc)
    results["2. dissimilar sentences far"]     = test_dissimilarity(enc)
    results["3. compositional algebra works"]  = test_composition(enc)
    results["4. paraphrased retrieval works"]  = test_retrieval(enc)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    all_pass = all(results.values())
    print()
    if all_pass:
        print("  ALL TESTS PASS - text encoder is sound.")
        print("  HDC can hold language. The Lattice substrate is viable.")
    else:
        n_pass = sum(results.values())
        print(f"  {n_pass}/4 tests pass. Investigate failures before building further.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
