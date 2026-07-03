"""
lattice/test_stack.py - can HDC compose transformations across layers?

Build a 2-layer HDC network:
  Layer 1: country -> capital
  Layer 2: capital -> continent

Test compositional generalization by holding out (Germany, Berlin, Europe)
and asking: given just 'Germany', does the chained inference produce
Europe?

If yes: HDC supports multi-layer composition — the building block for
deep HDC networks. If no: cleanup at intermediate layers destroys
signal needed by later layers.

Two modes tested:
  - 'cleanup': commit to a symbol at each layer (decisive)
  - 'raw':    pass noisy XOR results between layers (deferred commitment)

Usage:
    python -m lattice.test_stack
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.feedforward import HDCFeedForward, HDCStack
from lattice.text_encoder import TextEncoder
from train.v5_hdc_prototype import hamming_distance, D


# ─── Training data ─────────────────────────────────────────────────


# (country, capital, continent) triples
KNOWN = [
    ("USA",            "Washington",   "North America"),
    ("France",         "Paris",        "Europe"),
    ("Japan",          "Tokyo",        "Asia"),
    ("United Kingdom", "London",       "Europe"),
    ("Italy",          "Rome",         "Europe"),
    ("Spain",          "Madrid",       "Europe"),
    ("China",          "Beijing",      "Asia"),
    ("Russia",         "Moscow",       "Europe"),
    ("Brazil",         "Brasilia",     "South America"),
    ("Canada",         "Ottawa",       "North America"),
    ("Australia",      "Canberra",     "Oceania"),
    ("Mexico",         "Mexico City",  "North America"),
    ("Argentina",      "Buenos Aires", "South America"),
    ("South Korea",    "Seoul",        "Asia"),
    ("Egypt",          "Cairo",        "Africa"),
    ("Greece",         "Athens",       "Europe"),
    ("Sweden",         "Stockholm",    "Europe"),
    ("Poland",         "Warsaw",       "Europe"),
    ("Turkey",         "Ankara",       "Asia"),
    ("Netherlands",    "Amsterdam",    "Europe"),
    ("South Africa",   "Cape Town",    "Africa"),
    ("Vietnam",        "Hanoi",        "Asia"),
    ("Chile",          "Santiago",     "South America"),
    ("Kenya",          "Nairobi",      "Africa"),
    ("New Zealand",    "Wellington",   "Oceania"),
]

# Held out — countries the layers never see
HELDOUT = [
    ("Germany",   "Berlin",   "Europe"),
    ("Thailand",  "Bangkok",  "Asia"),
    ("Peru",      "Lima",     "South America"),
    ("Nigeria",   "Abuja",    "Africa"),
    ("Norway",    "Oslo",     "Europe"),
]


def main():
    print("HDC stack — compositional generalization (country -> capital -> continent)\n")
    print("Loading encoder ...")
    enc = TextEncoder()

    # ── Encode everything ──
    print("Encoding all entities ...")
    encode_cache: dict[str, np.ndarray] = {}
    def E(name: str) -> np.ndarray:
        if name not in encode_cache:
            encode_cache[name] = enc.encode(name)
        return encode_cache[name]

    for c, cap, cont in KNOWN + HELDOUT:
        E(c); E(cap); E(cont)

    # ── Train Layer 1: country -> capital ──
    print(f"\nTraining Layer 1 (country -> capital) on {len(KNOWN)} pairs ...")
    L1 = HDCFeedForward()
    # Cleanup memory: all capitals (train + heldout) so heldout has a chance
    for _c, cap, _ in KNOWN + HELDOUT:
        L1.add_cleanup_item(cap, E(cap))
    train_pairs_L1 = [(c, E(c), cap, E(cap)) for c, cap, _ in KNOWN]
    L1.train(train_pairs_L1)
    print(f"  cleanup memory: {len(L1.cleanup_items)} capitals")

    # ── Train Layer 2: capital -> continent ──
    print(f"\nTraining Layer 2 (capital -> continent) on {len(KNOWN)} pairs ...")
    L2 = HDCFeedForward()
    # Cleanup memory: all continents (only 6, finite set)
    continents = {"Europe", "Asia", "Africa", "North America",
                    "South America", "Oceania", "Antarctica"}
    for cont in continents:
        L2.add_cleanup_item(cont, E(cont))
    train_pairs_L2 = [(cap, E(cap), cont, E(cont)) for _c, cap, cont in KNOWN]
    L2.train(train_pairs_L2)
    print(f"  cleanup memory: {len(L2.cleanup_items)} continents")

    # ── Build stack ──
    stack = HDCStack([L1, L2])
    print(f"\nStack: {len(stack.layers)} layers")

    # ── Test 1: training-set chain ──
    print("\n=== TEST 1: training-set chain (country -> ... -> continent) ===")
    train_correct = 0
    for c, _cap, cont in KNOWN:
        result = stack.forward(E(c), top_k=1, trace=True)
        trace = result["trace"]
        final = result["final"][0][0]
        ok = (final == cont)
        train_correct += int(ok)
        mark = "OK" if ok else "MISS"
        print(f"  [{mark}] {c:18s} -> {trace[0]['top1_name']:18s} -> {final:18s} "
              f"(expected {cont})")
    print(f"\n  Training chain accuracy: {train_correct}/{len(KNOWN)} = "
          f"{train_correct/len(KNOWN):.0%}")

    # ── Test 2: heldout chain (the key test) ──
    print("\n=== TEST 2: HELDOUT chain (country never seen by either layer) ===")
    print("  (Cleanup mode: commit to a capital at layer 1, pass to layer 2)\n")
    print(f"  {'COUNTRY':14s} | layer-1 -> capital  | layer-2 -> continent  | correct?")
    print(f"  {'-' * 14} | {'-' * 19} | {'-' * 21} | --------")
    held_correct_cleanup = 0
    for c, cap_correct, cont_correct in HELDOUT:
        result = stack.forward(E(c), top_k=1, trace=True)
        trace = result["trace"]
        predicted_cap = trace[0]["top1_name"]
        predicted_cont = result["final"][0][0]
        ok = (predicted_cont == cont_correct)
        held_correct_cleanup += int(ok)
        mark = "OK" if ok else "MISS"
        print(f"  {c:14s} | {predicted_cap:19s} | {predicted_cont:21s} | "
              f"[{mark}] expected {cont_correct}")
    print(f"\n  HELDOUT cleanup-chain accuracy: {held_correct_cleanup}/{len(HELDOUT)} = "
          f"{held_correct_cleanup/len(HELDOUT):.0%}")

    # ── Test 3: raw chain (no cleanup between layers) ──
    print("\n=== TEST 3: HELDOUT chain in RAW mode (no intermediate cleanup) ===")
    print("  Tests whether HDC can compose transformations algebraically")
    print("  without committing to intermediate symbols.\n")
    held_correct_raw = 0
    for c, cap_correct, cont_correct in HELDOUT:
        results = stack.forward_raw(E(c), top_k=1)
        predicted_cont = results[0][0]
        ok = (predicted_cont == cont_correct)
        held_correct_raw += int(ok)
        mark = "OK" if ok else "MISS"
        print(f"  {c:14s} -> {predicted_cont:21s} [{mark}] expected {cont_correct}")
    print(f"\n  HELDOUT raw-chain accuracy: {held_correct_raw}/{len(HELDOUT)} = "
          f"{held_correct_raw/len(HELDOUT):.0%}")

    # ── Test 4: control — does layer 2 alone work given true capital? ──
    print("\n=== TEST 4: control — layer 2 alone on TRUE heldout capitals ===")
    print("  Bypass layer 1, feed the correct capital directly to layer 2.")
    print("  Shows layer 2's capacity in isolation.\n")
    L2_alone_correct = 0
    for c, cap_correct, cont_correct in HELDOUT:
        results = L2.forward(E(cap_correct), top_k=1)
        predicted = results[0][0]
        ok = (predicted == cont_correct)
        L2_alone_correct += int(ok)
        mark = "OK" if ok else "MISS"
        print(f"  {cap_correct:14s} -> {predicted:21s} [{mark}] expected {cont_correct}")
    print(f"\n  Layer 2 alone (true capitals): {L2_alone_correct}/{len(HELDOUT)} = "
          f"{L2_alone_correct/len(HELDOUT):.0%}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Training chain (in-distribution):       "
          f"{train_correct}/{len(KNOWN)}")
    print(f"  Heldout chain (cleanup between layers): "
          f"{held_correct_cleanup}/{len(HELDOUT)}")
    print(f"  Heldout chain (raw, no intermediate cleanup): "
          f"{held_correct_raw}/{len(HELDOUT)}")
    print(f"  Layer 2 alone (true capitals):          "
          f"{L2_alone_correct}/{len(HELDOUT)}")
    print()
    best = max(held_correct_cleanup, held_correct_raw)
    if best >= 4:
        print(f"  STRONG: 2-layer HDC composition WORKS ({best}/5 on novel countries)")
    elif best >= 2:
        print(f"  PARTIAL: composition works sometimes ({best}/5)")
    else:
        print(f"  FAIL: 2-layer composition broke down ({best}/5)")


if __name__ == "__main__":
    main()
