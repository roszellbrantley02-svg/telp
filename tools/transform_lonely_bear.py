"""
tools/transform_lonely_bear.py — Phase 14 demo driver.

Takes "The Lonely Bear" (state/books/the_lonely_bear.txt — original
short story for Telp's developmental layer), applies a role-substitution
map to every frame in the story, verifies the HDC algebra is equivalent
to symbolic re-encoding, and renders the resulting frames as a new
original story.

Usage:
    python -m tools.transform_lonely_bear

The substitution map below produces "The Lonely Cat" — same emotional
arc (lonely -> sad -> happy), same structural shape (visits three
candidates, the third becomes a friend), but different surface fillers:

    bear   -> cat
    rabbit -> mouse
    fox    -> sparrow
    frog   -> beetle
    forest -> garden
    hill   -> roof
    cave   -> nest
    pond   -> puddle

No LLM was consulted at any point.  Every output word is traceable
to a specific bind/bundle operation on a specific thematic role of
a specific input frame.
"""
from __future__ import annotations

import sys
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
if str(_TELP_ROOT) not in sys.path:
    sys.path.insert(0, str(_TELP_ROOT))

from lattice.story_transform import (
    FrameSubstitution,
    transform_story_text,
    verify_algebra_equivalence,
)


# ─── The substitution map ─────────────────────────────────────────


SUBSTITUTIONS = [
    # The protagonist
    FrameSubstitution("agent",   "bear",   "cat"),
    FrameSubstitution("patient", "bear",   "cat"),
    FrameSubstitution("goal",    "bear",   "cat"),

    # The three creatures the protagonist meets
    FrameSubstitution("agent",   "rabbit", "mouse"),
    FrameSubstitution("patient", "rabbit", "mouse"),
    FrameSubstitution("goal",    "rabbit", "mouse"),

    FrameSubstitution("agent",   "fox",    "sparrow"),
    FrameSubstitution("patient", "fox",    "sparrow"),
    FrameSubstitution("goal",    "fox",    "sparrow"),

    FrameSubstitution("agent",   "frog",   "beetle"),
    FrameSubstitution("patient", "frog",   "beetle"),
    FrameSubstitution("goal",    "frog",   "beetle"),

    # The four places visited
    FrameSubstitution("goal",     "forest", "garden"),
    FrameSubstitution("location", "forest", "garden"),

    FrameSubstitution("goal",     "hill",   "roof"),
    FrameSubstitution("location", "hill",   "roof"),

    FrameSubstitution("goal",     "cave",   "nest"),
    FrameSubstitution("location", "cave",   "nest"),

    FrameSubstitution("goal",     "pond",   "puddle"),
    FrameSubstitution("location", "pond",   "puddle"),
]


# ─── Run the transform ───────────────────────────────────────────


def main() -> int:
    src_path = _TELP_ROOT / "state" / "books" / "the_lonely_bear.txt"
    if not src_path.exists():
        print(f"source story not found: {src_path}", file=sys.stderr)
        return 1
    text = src_path.read_text(encoding="utf-8")

    print("=" * 70)
    print("  ORIGINAL STORY  (state/books/the_lonely_bear.txt)")
    print("=" * 70)
    for ln in text.splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        print(f"  {ln}")
    print()

    print("=" * 70)
    print("  SUBSTITUTION MAP  (specified by user)")
    print("=" * 70)
    seen = set()
    for s in SUBSTITUTIONS:
        key = (s.from_value, s.to_value)
        if key in seen:
            continue
        seen.add(key)
        print(f"  {s.from_value:<8} ->  {s.to_value}")
    print()

    result = transform_story_text(text, SUBSTITUTIONS, drop_titles=True)

    print("=" * 70)
    print(f"  FRAMES PARSED  ({len(result['original_frames'])} events)")
    print("=" * 70)
    for i, (orig, new) in enumerate(zip(result["original_frames"],
                                                       result["substituted_frames"])):
        clean_orig = {k: v for k, v in orig.items()
                              if not k.startswith("_") and v}
        clean_new  = {k: v for k, v in new.items()
                              if not k.startswith("_") and v}
        marker = "  " if clean_orig == clean_new else " *"
        print(f"  [{i:2d}]{marker} {clean_orig}")
        if clean_orig != clean_new:
            print(f"       ->  {clean_new}")
    print()

    # ─── HDC algebra verification on EVERY frame ──────────────
    print("=" * 70)
    print("  HDC ALGEBRA VERIFICATION  "
              "(per-frame: algebraic = re-encoded)")
    print("=" * 70)
    n_changed = 0
    n_passed = 0
    n_failed = 0
    min_sim  = 1.0
    sum_sim  = 0.0
    for i, orig in enumerate(result["original_frames"]):
        # Only check frames that actually changed under substitution
        applicable = [s for s in SUBSTITUTIONS
                          if (orig.get(s.role) or "").lower()
                              == s.from_value.lower()
                              or (s.from_value.lower() in
                                  (orig.get(s.role) or "").lower()
                                  and " and " in (orig.get(s.role) or ""))]
        if not applicable:
            continue
        n_changed += 1
        check = verify_algebra_equivalence(orig, applicable, tol=0.99)
        sum_sim += check["similarity"]
        min_sim = min(min_sim, check["similarity"])
        if check["passed"]:
            n_passed += 1
        else:
            n_failed += 1
            print(f"  [{i:2d}] FAIL  sim={check['similarity']:.4f}  "
                      f"k={check['k']}  raw={orig.get('_raw')!r}")

    if n_changed > 0:
        print(f"  frames affected by subs: {n_changed}")
        print(f"  algebra-vs-reencode passed: {n_passed}/{n_changed}")
        print(f"  similarity min: {min_sim:.4f}  "
                  f"mean: {sum_sim/n_changed:.4f}")
    else:
        print(f"  no frames matched any substitution — check map")
    print()

    # ─── Render the new story ──────────────────────────────────
    print("=" * 70)
    print("  TELP'S NEW STORY  (rendered from substituted frames)")
    print("=" * 70)
    for line in result["rendered_lines"]:
        print(f"  {line}")
    print()

    # ─── Side-by-side ──────────────────────────────────────────
    print("=" * 70)
    print("  SIDE BY SIDE")
    print("=" * 70)
    orig_lines = [f.get("_raw", "") for f in result["original_frames"]]
    for o, n in zip(orig_lines, result["rendered_lines"]):
        print(f"  {o:<46}  |  {n}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
