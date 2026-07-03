"""
tools/dream.py — Phase 16 driver: ask Telp to dream up a story.

Usage:
    python -m tools.dream                       # random everything
    python -m tools.dream --seed dragon
    python -m tools.dream --seed owl --arc small_to_brave
    python -m tools.dream --seed robot --n 3    # 3 different stories
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
if str(_TELP_ROOT) not in sys.path:
    sys.path.insert(0, str(_TELP_ROOT))

from lattice.imagination import ImaginationEngine, ARCS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default=None,
                          help="protagonist word; default: random animal")
    ap.add_argument("--arc",  default=None,
                          help=f"arc name; one of {sorted(ARCS.keys())}; "
                                 f"default: random")
    ap.add_argument("--n", type=int, default=1,
                          help="how many stories to generate")
    ap.add_argument("--rng", type=int, default=None,
                          help="RNG seed for reproducibility")
    args = ap.parse_args()

    eng = ImaginationEngine(seed=args.rng)

    # Pick a random protagonist if not specified — sample from the
    # cast pool so it lands on a real noun in Telp's dictionary.
    import random
    proto_rng = random.Random(args.rng)

    for i in range(args.n):
        seed = args.seed or proto_rng.choice(eng.cast_pool())
        frames = eng.imagine_story(seed=seed, arc_name=args.arc)
        arc_name = frames[0].get("_arc", "?")
        story = eng.render(frames)
        if args.n > 1:
            print(f"\n{'=' * 60}")
            print(f"  Story {i+1}/{args.n}  —  seed={seed}  arc={arc_name}")
            print(f"{'=' * 60}\n")
        else:
            print(f"# seed = {seed}    arc = {arc_name}")
            print()
        print(story)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
