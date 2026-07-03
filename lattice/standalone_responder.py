"""
lattice/standalone_responder.py - LLM-free response generator.

Takes retrieved memories + KG hits and generates a natural-language
response WITHOUT any language model. Three strategies, applied in order:

  1. Question-type detection (what/who/where/when/how many)
     -> pick the KG hit whose relation label matches the question type
     -> render with a template tied to that label

  2. Direct quotation from the most-similar memory
     -> the corpus text is already English; the best-matching sentence
        usually answers the question better than any paraphrase

  3. Fallback: list the top hits

The relation labels were auto-discovered by TypedHDC v2 (REL_HAS_CAPITAL,
REL_BORN_IN, REL_FOUNDED_IN, ...). The responder maps each label to a
small template; unknown labels fall back to a generic "X <relation> Y"
form derived from the label itself.

No LLM. No external API. Deterministic output for a given (query, hits).
"""
from __future__ import annotations

import re


# ─── Question-type detection ───────────────────────────────────────


_QUESTION_PATTERNS = [
    ("what",     re.compile(r"\bwhat\b",     re.I)),
    ("who",      re.compile(r"\bwho\b",      re.I)),
    ("where",    re.compile(r"\bwhere\b",    re.I)),
    ("when",     re.compile(r"\bwhen\b",     re.I)),
    ("how_many", re.compile(r"\bhow\s+many\b", re.I)),
    ("why",      re.compile(r"\bwhy\b",      re.I)),
    ("how",      re.compile(r"\bhow\b",      re.I)),
]


def detect_question_type(text: str) -> str:
    for name, pat in _QUESTION_PATTERNS:
        if pat.search(text):
            return name
    return "statement"


# ─── Relation -> template ──────────────────────────────────────────
#
# Auto-discovered TypedHDC labels look like REL_<TOKEN1>_<TOKEN2>_<TOKEN3>
# from the non-typed content words inside the pattern window.  Common
# examples we've seen on the 77-article Wikipedia corpus:
#   REL_CAPITAL_OF
#   REL_BORN_IN     REL_BORN
#   REL_FOUNDED     REL_FOUNDED_IN
#   REL_KNOWN_FOR
#   REL_LOCATED_IN
#   REL_INVENTED
#   REL_DISCOVERED
#
# Each maps to a 2-position template. {s} is the subject token, {o} the
# object token.  We render in the direction that matches the user's
# question form.

_TEMPLATES = {
    "REL_CAPITAL":     "{s} is the capital of {o}.",
    "REL_CAPITAL_OF":  "{s} is the capital of {o}.",
    "REL_BORN":        "{s} was born in {o}.",
    "REL_BORN_IN":     "{s} was born in {o}.",
    "REL_DIED":        "{s} died in {o}.",
    "REL_DIED_IN":     "{s} died in {o}.",
    "REL_FOUNDED":     "{s} was founded in {o}.",
    "REL_FOUNDED_IN":  "{s} was founded in {o}.",
    "REL_LOCATED":     "{s} is located in {o}.",
    "REL_LOCATED_IN":  "{s} is located in {o}.",
    "REL_KNOWN":       "{s} is known for {o}.",
    "REL_KNOWN_FOR":   "{s} is known for {o}.",
    "REL_INVENTED":    "{s} invented {o}.",
    "REL_DISCOVERED":  "{s} discovered {o}.",
    "REL_WROTE":       "{s} wrote {o}.",
    "REL_COMPOSED":    "{s} composed {o}.",
    "REL_PAINTED":     "{s} painted {o}.",
    "REL_RULED":       "{s} ruled {o}.",
    "REL_LEADER":      "{s} led {o}.",
    "REL_PART_OF":     "{s} is part of {o}.",
    "REL_MEMBER_OF":   "{s} is a member of {o}.",
}


# Which relations are good answers to which question types.  When the
# user asks "where", we prefer hits whose relation is about place.
_Q_TO_REL_HINTS = {
    "where":    {"CAPITAL", "LOCATED", "BORN_IN", "DIED_IN", "FOUNDED_IN"},
    "when":     {"BORN", "DIED", "FOUNDED", "DATE"},
    "who":      {"INVENTED", "DISCOVERED", "WROTE", "COMPOSED",
                 "PAINTED", "FOUNDED", "RULED", "LEADER"},
    "what":     {"CAPITAL", "KNOWN_FOR", "PART_OF", "MEMBER_OF"},
    "how_many": {"NUMBER", "POPULATION", "COUNT"},
}


def _relation_keyword(label: str) -> str:
    """Strip the REL_ prefix and return the inner keyword for matching."""
    return label.replace("REL_", "").strip()


def _render_fact(s: str, label: str, o: str) -> str:
    """Render a (subject, relation, object) triple into English."""
    tpl = _TEMPLATES.get(label)
    if tpl:
        return tpl.format(s=s, o=o)
    # Generic fallback: turn REL_X_Y into "X y"
    kw = _relation_keyword(label).lower().replace("_", " ").strip()
    if not kw:
        return f"{s} is related to {o}."
    return f"{s} {kw} {o}."


def _is_noisy_label(label: str) -> bool:
    """A label like REL_0 or REL_INV (no semantic content words) is noise."""
    kw = _relation_keyword(label)
    if not kw:
        return True
    # Pure digits, or single-letter tokens
    if kw.isdigit():
        return True
    parts = [p for p in kw.split("_") if p]
    if all(len(p) <= 2 for p in parts):
        return True
    return False


def _score_kg_hit_for_question(label: str, qtype: str) -> int:
    """Return a relevance score for this fact given the question type."""
    if _is_noisy_label(label):
        return 0
    if qtype == "statement":
        return 2
    hints = _Q_TO_REL_HINTS.get(qtype, set())
    kw = _relation_keyword(label).upper()
    for h in hints:
        if h in kw:
            return 10
    return 1


# ─── The responder ─────────────────────────────────────────────────


class TemplateResponder:
    """LLM-free response generator.

    .narrate(query, memories, kg_hits) -> str

    Three strategies, applied in order:
      1. KG hit -> template render
      2. Top retrieved memory -> direct quote (optionally with HDC
         generative continuation seeded by query content words)
      3. Pure HDC generation from query content words (no LLM)
    """

    def __init__(self, max_memory_chars: int = 240,
                  sequence_predictor=None,
                  generate_words: int = 14):
        self.max_memory_chars = max_memory_chars
        self.seq = sequence_predictor
        self.generate_words = generate_words

    def narrate(self, query: str,
                  memories: list[dict] | None = None,
                  kg_hits: list[tuple[str, str, str]] | None = None) -> str:
        memories = memories or []
        kg_hits = kg_hits or []
        qtype = detect_question_type(query)

        # 1. If we have KG hits, try to answer with the best one.
        if kg_hits:
            scored = sorted(
                [(s, r, o, _score_kg_hit_for_question(r, qtype))
                 for (s, r, o) in kg_hits],
                key=lambda x: -x[3],
            )
            # Keep only non-zero-scoring hits (drops REL_0, REL_INV junk).
            scored = [x for x in scored if x[3] > 0]
            if scored:
                top_s, top_r, top_o, top_score = scored[0]
                primary = _render_fact(top_s, top_r, top_o)
                extras = [(s, r, o) for (s, r, o, sc) in scored[1:3]
                          if sc >= 5]
                if extras:
                    more = " ".join(_render_fact(s, r, o)
                                      for (s, r, o) in extras)
                    return f"{primary} {more}"
                # If the top score is only 'statement-level' and we ALSO
                # have memories, prefer the memory — it has more context.
                if top_score <= 2 and memories:
                    top_mem = memories[0]
                    text = top_mem["text"]
                    if len(text) > self.max_memory_chars:
                        text = text[: self.max_memory_chars - 3] + "..."
                    return text
                return primary

        # 2. No KG hits — fall back to the top retrieved memory.
        if memories:
            top = memories[0]
            text = top["text"]
            if len(text) > self.max_memory_chars:
                text = text[: self.max_memory_chars - 3] + "..."
            sim = top.get("similarity", 0.0)
            if sim < 0.0:
                return ("I have a vague memory that might be related, but "
                          f"nothing close enough to be reliable: \"{text}\"")
            return text

        # 3. Pure HDC generation seeded by the question's content words.
        generated = self._try_generate(query)
        if generated:
            return generated

        # 4. Nothing.
        return ("I don't have any memories or facts about that yet. "
                  "Teach me by talking, or use /learn <topic> to ingest "
                  "a Wikipedia article.")

    # ─── HDC generative fallback ──────────────────────────────

    _Q_STOPWORDS = {
        "what","who","where","when","why","how","which","is","was","are",
        "were","be","do","does","did","the","a","an","of","tell","me",
        "about","you","your","please","can","you","i",
    }

    def _seed_tokens(self, query: str) -> list[str]:
        toks = [t.lower() for t in re.findall(r"[A-Za-z][\w-]*", query)]
        # Strip question-shaped words; keep the actual subject.
        return [t for t in toks if t not in self._Q_STOPWORDS]

    def _try_generate(self, query: str) -> str | None:
        if self.seq is None:
            return None
        seed = self._seed_tokens(query)
        if not seed:
            return None
        try:
            sentence = self.seq.generate(
                " ".join(seed), n_words=self.generate_words,
            )
        except Exception:
            return None
        if not sentence:
            return None
        # Trim trailing duplicate triples defensively.
        words = sentence.split()
        for i in range(2, len(words)):
            if words[i] == words[i - 1] == words[i - 2]:
                sentence = " ".join(words[:i])
                break
        return sentence.capitalize()


# ─── Quick smoke test ──────────────────────────────────────────────


def main():
    r = TemplateResponder()
    print(r.narrate("What is the capital of Germany?",
                     kg_hits=[("Berlin", "REL_CAPITAL_OF", "Germany")]))
    print(r.narrate("Who invented the telephone?",
                     kg_hits=[("Alexander_Graham_Bell", "REL_INVENTED",
                                "telephone")]))
    print(r.narrate("Where was Einstein born?",
                     kg_hits=[("Einstein", "REL_BORN_IN", "Ulm")]))
    print(r.narrate("Tell me about Mars",
                     memories=[{"text": "Mars is the fourth planet from "
                                          "the Sun.",
                                  "similarity": 0.85}]))
    print(r.narrate("Tell me about quantum gravity"))


if __name__ == "__main__":
    main()
