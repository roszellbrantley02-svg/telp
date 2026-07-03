"""
lattice/query_router.py — the recall-vs-creative switch.

The architectural piece that decides per-query whether the HDC system
should answer from FACTS (recall the closest stored knowledge) or
from IMAGINATION (chain forward through transitions to generate
something new).

Three modes:

  recall    — the answer exists in memory; look it up.
              Used for: "what is X", "current price", "show me", definitions,
              factual queries.  Backend: lattice retrieval, direct Markov
              v2 prediction, HDC v4 nearest-neighbour.

  creative  — the answer doesn't exist yet; generate one.
              Used for: "imagine", "what might", "tell me a story", "what
              could happen", multi-step trajectories.  Backend:
              TransitionMemory chain-forward, text continuation generator.

  hybrid    — ground in fact, extend creatively.  This is the typical
              chat default: lookup what's known, then synthesise an answer
              that goes beyond verbatim recall.  Backend: recall to fetch
              relevant memories, then creative to compose them.

Public API:

    router = QueryRouter()
    decision = router.classify("where might MES go next?")
    # -> {"mode": "creative", "matched_cues": ["might", "next"], ...}

    response = router.route(query, agent)
    # executes the appropriate backend, returns the response + the mode
    # used so callers know whether the answer is a fact or a guess
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ── Cue lexicons ───────────────────────────────────────────────────


# Words/phrases that signal the user is asking about something the
# system would have to imagine/extrapolate/synthesise.
CREATIVE_CUES: set[str] = {
    # speculation
    "might", "could", "would", "may", "maybe",
    # future / prediction
    "will", "predict", "prediction", "predicted", "forecast",
    "future", "next", "tomorrow", "later", "upcoming",
    "where will", "what will", "where would", "what would",
    "going to", "expected", "expect", "anticipate", "anticipating",
    "likely", "probably", "possibly", "potential",
    # imagination
    "imagine", "imagining", "envision", "picture", "suppose",
    "what if", "scenario", "hypothetical", "guess",
    # generation
    "create", "generate", "invent", "compose", "make up",
    "write me", "tell me a story", "tell a story", "story",
    "short story", "fairy tale", "fable", "anecdote",
    # multi-step trajectories
    "play out", "unfold", "evolve", "trajectory", "path",
    "simulate", "extrapolate",
}


# Words/phrases that signal the user wants a concrete factual answer
# from stored memory.
RECALL_CUES: set[str] = {
    # factual interrogatives
    "what is", "what's", "what was", "what are", "what were",
    "who is", "who was", "when is", "when was", "when did",
    "where is", "where was",
    # current state
    "current", "currently", "now", "right now", "at this moment",
    "today", "as of",
    # historical
    "last", "previous", "before", "earlier", "yesterday",
    # request for stored info
    "show me", "tell me about", "describe", "describe the",
    "define", "definition", "meaning", "explain",
    "list", "give me", "what does",
    # factual precision
    "fact", "actual", "actually", "exact", "exactly", "literally",
    "real", "really", "true", "truly",
}


# Domain hints — phrases that lock the answer into a specific backend
# regardless of recall/creative classification.
DOMAIN_CUES: dict[str, set[str]] = {
    "trading_state": {
        "price", "close", "open", "high", "low", "bar", "atr",
        "current position", "pnl", "ledger", "trade", "trading",
    },
    "knowledge": {
        "what is", "define", "definition", "meaning of",
        "explain", "describe the", "how does", "tell me about",
    },
    "prediction": {
        "where will", "what will", "predict", "forecast",
        "going to", "next bars", "future return", "imagine the",
    },
}


@dataclass
class RouterDecision:
    """Result of classifying a query.

    `creativity` is the primary output: a continuous dial in [0, 1]
    indicating how much *composition* should happen between the
    grounded HDC substrate and the answer.

      0.0–0.25 : direct        — return nearest single memory
      0.25–0.5 : synthesized   — bundle k nearest, recombine
      0.5–0.75 : extrapolated  — chain forward a few transitions
      0.75–1.0 : imagined      — multi-step composition, weak anchor

    `mode` is kept as a legacy three-class summary derived from the
    creativity float for backward compatibility with existing callers.
    """
    creativity:        float = 0.15
    tag:               str = "direct"   # direct|synthesized|extrapolated|imagined
    mode:              str = "recall"    # legacy: recall|creative|hybrid
    matched_creative:  list[str] = field(default_factory=list)
    matched_recall:    list[str] = field(default_factory=list)
    matched_domains:   list[str] = field(default_factory=list)
    rationale:         str = ""

    def to_dict(self) -> dict:
        return {
            "creativity":       round(self.creativity, 3),
            "tag":              self.tag,
            "mode":             self.mode,
            "matched_creative": self.matched_creative,
            "matched_recall":   self.matched_recall,
            "matched_domains":  self.matched_domains,
            "rationale":        self.rationale,
        }


def creativity_to_tag(c: float) -> str:
    """Map a creativity value in [0, 1] to a four-level tag."""
    if c < 0.25:  return "direct"
    if c < 0.50:  return "synthesized"
    if c < 0.75:  return "extrapolated"
    return "imagined"


def creativity_to_mode(c: float) -> str:
    """Map a creativity value to the legacy three-class mode."""
    if c < 0.35:  return "recall"
    if c > 0.65:  return "creative"
    return "hybrid"


# Cues with extra weight — explicit signals for either direction
STRONG_CREATIVE_CUES: set[str] = {
    "imagine", "imagining", "predict", "prediction", "predicted",
    "tell me a story", "tell a story", "write me", "compose",
    "novel", "poem", "poetry", "story", "short story", "fairy tale",
    "fable", "anecdote",
    "scenario", "what if", "hypothetical", "extrapolate", "simulate",
    "play out", "unfold", "trajectory",
}

STRONG_RECALL_CUES: set[str] = {
    "current", "currently", "what is", "what was", "what's",
    "define", "definition", "exact", "exactly", "literally",
    "actual", "actually", "fact", "show me", "right now",
    "as of",
}


# ── Router ─────────────────────────────────────────────────────────


class QueryRouter:
    """The recall-vs-creative classifier + dispatcher.

    classify(query) inspects the query string and returns a
    RouterDecision.  route(query, handlers) calls the matching handler
    from the supplied dict and returns both the response and the
    decision so the caller knows whether the answer is a fact or a
    guess.
    """

    def __init__(self,
                  creative_cues: set[str] = None,
                  recall_cues:   set[str] = None,
                  domain_cues:   dict[str, set[str]] = None,
                  default_mode:  str = "recall"):
        self.creative_cues = creative_cues or CREATIVE_CUES
        self.recall_cues   = recall_cues   or RECALL_CUES
        self.domain_cues   = domain_cues   or DOMAIN_CUES
        self.default_mode  = default_mode   # safer default: recall
        # Precompile multi-word phrases for fast scanning
        self._creative_phrases = sorted(self.creative_cues,
                                              key=len, reverse=True)
        self._recall_phrases   = sorted(self.recall_cues,
                                              key=len, reverse=True)

    # ─── Classification ───────────────────────────────────────

    def _find_matches(self, q: str, phrases: list[str]) -> list[str]:
        """Return the phrases in `phrases` that appear in lowercase
        query `q`, preferring longer matches (already sorted)."""
        out = []
        for p in phrases:
            # word-bounded match for single tokens; substring for phrases
            if " " in p:
                if p in q:
                    out.append(p)
            else:
                if re.search(rf"\b{re.escape(p)}\b", q):
                    out.append(p)
        return out

    def classify(self, query: str) -> RouterDecision:
        """Classify a query — emit a continuous creativity dial in [0, 1].

        Cues with explicit imagination/prediction wording push higher;
        cues asking for literal facts push lower.  The default for an
        unmarked query is 0.15 (slightly grounded, mostly direct).
        Calls to write fiction / poetry get a floor of 0.85.
        """
        q = (query or "").lower().strip()
        if not q:
            d = RouterDecision(creativity=0.15)
            d.tag  = creativity_to_tag(d.creativity)
            d.mode = creativity_to_mode(d.creativity)
            d.rationale = "empty query"
            return d
        creative_hits = self._find_matches(q, self._creative_phrases)
        recall_hits   = self._find_matches(q, self._recall_phrases)
        domain_hits = [name for name, cues in self.domain_cues.items()
                          if any(c in q for c in cues)]

        # Weight cues — strong ones count more.
        def _signal(hits: list[str], strong_set: set[str]) -> float:
            s = 0.0
            for h in hits:
                s += 1.0 if h in strong_set else 0.5
            return s

        creative_signal = _signal(creative_hits, STRONG_CREATIVE_CUES)
        recall_signal   = _signal(recall_hits,   STRONG_RECALL_CUES)
        total = creative_signal + recall_signal

        if total == 0:
            creativity = 0.15  # safe default: slightly grounded
            rationale = "no cues — default low creativity"
        else:
            # Net creative pull; smoothed via softmax-like ratio
            creativity = creative_signal / total
            rationale = (f"cre={creative_signal:.1f} rec={recall_signal:.1f} "
                            f"-> {creativity:.2f}")

        # Floors / ceilings for explicit phrases
        if any(p in q for p in (
            "write me a poem", "write a poem", "tell me a story",
            "tell a story", "novel about", "compose a", "imagine",
        )):
            creativity = max(creativity, 0.85)
            rationale += "  [explicit fiction -> 0.85+]"
        if any(p in q for p in (
            "exact value", "current price", "what is the price",
            "literally", "exactly what",
        )):
            creativity = min(creativity, 0.15)
            rationale += "  [explicit fact -> 0.15-]"

        creativity = max(0.0, min(1.0, creativity))

        return RouterDecision(
            creativity=creativity,
            tag=creativity_to_tag(creativity),
            mode=creativity_to_mode(creativity),
            matched_creative=creative_hits,
            matched_recall=recall_hits,
            matched_domains=domain_hits,
            rationale=rationale,
        )

    # ─── Dispatch ─────────────────────────────────────────────

    def route(self, query: str,
                handlers: dict[str, Callable[[str], Any]]) -> dict:
        """Classify, dispatch to the right handler, return tagged result.

        `handlers` keys: 'recall', 'creative', 'hybrid' (hybrid is
        optional — defaults to running both and stitching).  Each
        handler takes the query string and returns whatever response
        the caller wants.
        """
        decision = self.classify(query)
        mode = decision.mode
        if mode in handlers:
            response = handlers[mode](query)
        elif mode == "hybrid":
            # Fallback hybrid: run both backends, return both
            response = {
                "recall_part":   (handlers["recall"](query)
                                   if "recall" in handlers else None),
                "creative_part": (handlers["creative"](query)
                                   if "creative" in handlers else None),
            }
        else:
            response = None
        return {
            "mode":     mode,
            "decision": decision.to_dict(),
            "response": response,
        }


# ── Smoke test / demo ──────────────────────────────────────────────


def _demo():
    """Run the router on a battery of test queries to verify routing
    behaves sensibly across factual, predictive, and mixed prompts."""
    router = QueryRouter()
    test_queries = [
        # Pure recall (factual)
        "what is the current price of MES?",
        "show me the last trade",
        "define RSI",
        "tell me about Bollinger bands",
        "what was yesterday's high?",
        "explain support and resistance",
        # Pure creative (predictive / imaginative)
        "where might MES go next?",
        "imagine the chart playing out for 30 minutes",
        "what could happen if VIX spikes?",
        "predict the next 20 bars",
        "tell me a story about a trader",
        "what's likely to happen tomorrow?",
        # Hybrid (factual + creative)
        "what is happening now and what might come next?",
        "show me the current setup and predict where price will go",
        "describe this pattern and tell me what could happen",
        # Ambiguous
        "MES",
        "trading",
    ]

    print("=" * 70)
    print(" Query Router — creativity-dial demo")
    print("=" * 70)
    print()
    print(f"  {'tag':>12s}  {'cre':>5s}  query")
    print("  " + "-" * 70)
    for q in test_queries:
        d = router.classify(q)
        print(f"  {d.tag:>12s}  {d.creativity:>5.2f}  {q}")
        if d.matched_creative or d.matched_recall:
            cre = (','.join(d.matched_creative[:3])
                     if d.matched_creative else '')
            rec = (','.join(d.matched_recall[:3])
                     if d.matched_recall else '')
            ann = []
            if cre: ann.append(f"cre=[{cre}]")
            if rec: ann.append(f"rec=[{rec}]")
            if d.matched_domains:
                ann.append(f"dom=[{','.join(d.matched_domains)}]")
            print(f"  {'':>12s}  {'':>5s}  {'  '.join(ann)}")
    print()
    print("=" * 70)
    print(" Routing with stub handlers")
    print("=" * 70)

    handlers = {
        "recall":   lambda q: f"[FACTS] would lookup: {q!r}",
        "creative": lambda q: f"[IMAGINE] would generate from: {q!r}",
        "hybrid":   lambda q: f"[BOTH] ground in fact then extend: {q!r}",
    }
    samples = [
        "what is the current ATR?",
        "where might price go next?",
        "show me the current setup and predict what comes next",
    ]
    for q in samples:
        r = router.route(q, handlers)
        print(f"\n  query : {q}")
        print(f"  mode  : {r['mode']}")
        print(f"  why   : {r['decision']['rationale']}")
        print(f"  answer: {r['response']}")


if __name__ == "__main__":
    _demo()
