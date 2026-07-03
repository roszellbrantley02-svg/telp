"""
lattice/analogy_qa.py - HDC analogy + counterfactual reasoning.

Analogy: "Einstein is to physics as Bach is to ?"  via XOR algebra.
  ?_hv = c_hv XOR a_hv XOR b_hv   (the a->b relation applied to c)

We decode the resulting hypervector back to a known entity by
nearest-neighbour search over the vocabulary and structured-claim
subjects/objects.  The closest concept is the analogy's answer.

Counterfactual: "If Einstein had been Polish, where would Berlin be?"
  Substitute "German" -> "Polish" in a base memory, re-query.
  (Less powerful than analogy here because our extracted facts are
  static — a counterfactual question is really probing what the
  KG would say if a piece of it were different.)

Both use the existing Lattice algebra in store.py.  This module
wraps them with question parsing + nearest-neighbour decoding.
"""
from __future__ import annotations

import re
import numpy as np

from lattice.store import Lattice
from lattice.structured_qa import StructuredQA


# Analogy question shapes.
_ANALOGY_PATTERNS = [
    # "X is to Y as Z is to ?"  / "X:Y::Z:?"
    re.compile(
        r"\b(?P<a>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)"
        r"\s+is\s+to\s+"
        r"(?P<b>[A-Z]?[\w-]+(?:\s+[\w-]+)*?)"
        r"\s+as\s+"
        r"(?P<c>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)"
        r"\s+is\s+to\s+(?:what|whom|\?+)",
        re.I),
    re.compile(
        r"(?P<a>[\w]+)\s*:\s*(?P<b>[\w]+)\s*::\s*"
        r"(?P<c>[\w]+)\s*:\s*\?",
        re.I),
]


# Counterfactual question shapes.
_COUNTERFACTUAL_PATTERNS = [
    # "If X were Y, ..." / "If X had been Y, ..."
    re.compile(
        r"\bif\s+(?P<subj>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)"
        r"\s+(?:were|had\s+been|was)\s+"
        r"(?P<new>[A-Z]?[\w-]+(?:\s+[\w-]+){0,3}?)"
        r"[,?.]",
        re.I),
]


class AnalogyQA:
    """Wraps Lattice.analogy + nearest-neighbour decoding."""

    def __init__(self, lattice: Lattice, structured: StructuredQA):
        self.lattice = lattice
        self.structured = structured

    # ── Public entrypoint ───────────────────────────────────

    def answer(self, query: str) -> dict | None:
        """Try analogy patterns first, then counterfactual."""
        for pat in _ANALOGY_PATTERNS:
            m = pat.search(query)
            if m:
                a = m.group("a").strip()
                b = m.group("b").strip()
                c = m.group("c").strip()
                return self._solve_analogy(a, b, c)
        return None

    # ── Analogy solver ──────────────────────────────────────

    def _solve_analogy(self, a: str, b: str, c: str) -> dict | None:
        """Two strategies:

        1. STRUCTURED: find a relation R where (A, R, B) is a stored
           claim, then look up (C, R, ?).  This is the cleanest path
           when both pairs share a known relation.

        2. XOR-ALGEBRA fallback (Plate 1995 style):
           ?_hv = c_hv XOR a_hv XOR b_hv.  Decode by nearest neighbour.
           Works less well on a corpus-trained RI encoder because the
           encoding doesn't preserve clean linear relations, but is
           kept as a fallback for novel relations not in the claim
           store.
        """
        # Strategy 1: structured-relation lookup.
        # Find any claim (a, R, b), then look up (c, R, ?).
        def _matches(key: str, name: str) -> bool:
            return key.lower() in name.lower() or name.lower() in key.lower()
        rel = None
        for s, v, o in self.structured.claim_triple:
            if _matches(a, s) and _matches(b, o):
                rel = v; break
        if rel is not None:
            # Use the structured QA to resolve (c, R, ?).
            self.structured._ensure_stacks()
            q_subj_hv = self.structured._key_token_hv(c)
            q_verb_hv = self.structured._word_hv(rel)
            sqa = self.structured
            ssim = (1.0 - 2.0
                      * np.bitwise_xor(sqa._subj_stack,
                                          q_subj_hv[None, :]).sum(axis=1)
                      / sqa.dim)
            vsim = (1.0 - 2.0
                      * np.bitwise_xor(sqa._verb_stack,
                                          q_verb_hv[None, :]).sum(axis=1)
                      / sqa.dim)
            scores = (ssim + 2.0 * vsim) / 3.0
            best = int(np.argmax(scores))
            if scores[best] >= 0.5:
                s_, v_, o_ = sqa.claim_triple[best]
                return {
                    "a": a, "b": b, "c": c,
                    "relation": rel,
                    "answer": o_,
                    "similarity": float(scores[best]),
                    "mode": "structured",
                    "top5": [],
                }

        # Strategy 2: XOR-algebra fallback.
        target_hv = self.lattice.analogy(a, b, c)
        candidates = self._candidate_pool(exclude={a, b, c})
        if not candidates:
            return None
        best = None
        best_d = self.lattice.encoder.dim
        scored: list[tuple[str, int]] = []
        for cand, cand_hv in candidates:
            d = int(np.bitwise_xor(target_hv, cand_hv).sum())
            scored.append((cand, d))
            if d < best_d:
                best_d = d
                best = cand
        scored.sort(key=lambda x: x[1])
        sim = 1.0 - 2.0 * best_d / self.lattice.encoder.dim
        return {
            "a": a, "b": b, "c": c,
            "answer": best,
            "similarity": sim,
            "mode": "xor",
            "top5": [(s, 1.0 - 2.0 * d / self.lattice.encoder.dim)
                       for s, d in scored[:5]],
        }

    def _candidate_pool(self, exclude: set[str]
                          ) -> list[tuple[str, np.ndarray]]:
        """Collect (name, hypervector) pairs from the known entities."""
        exclude_lower = {e.lower() for e in exclude}
        seen: set[str] = set()
        out: list[tuple[str, np.ndarray]] = []
        # Structured claim subjects (people, places, things we have facts on).
        for s, _, o in self.structured.claim_triple:
            for name in (s, o):
                key = name.lower()
                if key in exclude_lower or key in seen:
                    continue
                if len(name) < 2:
                    continue
                seen.add(key)
                hv = self.lattice.encoder.encode(name)
                out.append((name, hv))
        return out
