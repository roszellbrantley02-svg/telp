"""
mind/fact_mind.py - Telp answering from facts, in his own words.

The pipeline for a question:
  1. mind/fact_questions.parse_question reads it into a fact query
     (Lookup, Who, Describe, Analogy, Chain, Check, Compare) - or None,
     in which case Telp's other routes answer as before.
  2. lattice/hdc_facts.FactMemory answers the query with hyperdimensional
     computing: bundled entity records, unbinding, superposed probes,
     clean-up memory - then verifies every proposal against the stored
     facts.
  3. mind/nlg writes the answer as fresh English, and returns the exact
     Facts it states, so "how do you know that?" can cite each one.

Facts come from the memory itself (lattice/fact_store keeps a facts table
in step with the stored sentences) and from structured imports.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lattice.facts import (Analogy, Chain, Check, Compare, Describe, Fact,  # noqa: E402
                           Lookup, Realization, Who)
from lattice.fact_store import FactStore  # noqa: E402

# bump when the extractor's output changes so stored facts are re-read
EXTRACTOR_VERSION = "1"


class FactMind:
    """Fact memory + question reader + sentence writer for one agent."""

    def __init__(self, agent):
        from lattice.hdc_facts import FactMemory
        self.agent = agent
        self.store = FactStore(agent.lattice._con)
        self.memory = FactMemory()
        self._loaded = False
        self._seen = (-1, -1)          # (max memory id, memory count)

    # ── keeping the fact memory current ───────────────────────────────
    def sync(self) -> None:
        """Bring facts in step with the memory: read new sentences, drop
        facts of forgotten ones. Cheap when nothing changed."""
        lat = self.agent.lattice
        seen = (max(lat._ids) if lat._ids else 0, len(lat._ids))
        if self._loaded and seen == self._seen:
            return
        from lattice.fact_extract import extract_from_rows
        added, removed = self.store.sync(extract_from_rows,
                                         EXTRACTOR_VERSION)
        if not self._loaded or removed or added:
            self.memory.clear()
            self.memory.add_many(self.store.all_facts())
            self._loaded = True
        self._seen = seen

    def import_facts(self, facts: list[Fact]) -> int:
        """Add structured facts (no sentence behind them)."""
        n = self.store.add(facts)
        if n:
            self._loaded = False
            self.sync()
        return n

    # ── answering ─────────────────────────────────────────────────────
    def parse(self, question: str):
        from mind.fact_questions import parse_question
        self.sync()
        return parse_question(question, self.memory.resolve)

    def answer(self, question: str) -> Realization | None:
        """Fresh-English answer with its facts, or None (not a fact
        question, or nothing verified to say)."""
        q = self.parse(question)
        if q is None:
            return None
        return self.answer_query(q)

    def answer_query(self, q) -> Realization | None:
        from mind import nlg
        m = self.memory
        if isinstance(q, Lookup):
            answers = m.lookup(q.subject, q.relation)
            return nlg.answer_lookup(q.subject, q.relation, answers) \
                if answers else None
        if isinstance(q, Who):
            matches = m.who(q.constraints, answer_kind=q.answer_kind)
            return nlg.answer_who(q.constraints, matches) if matches else None
        if isinstance(q, Describe):
            facts = m.describe(q.subject)
            stated = [f for f in facts if f.relation != "pronoun"]
            # a description needs substance; thin records fall through to
            # the sentence-retrieval voice
            return nlg.describe(q.subject, facts) if len(stated) >= 3 \
                else None
        if isinstance(q, Analogy):
            answers = m.analogy(q.a, q.b, q.c)
            return nlg.answer_analogy(q.a, q.b, q.c, answers) \
                if answers else None
        if isinstance(q, Chain):
            answers = m.chain(q.subject, q.relations)
            return nlg.answer_chain(q.subject, q.relations, answers) \
                if answers else None
        if isinstance(q, Check):
            verdict, facts = m.check(q.subject, q.relation, q.obj)
            if verdict is None:
                return None            # unknown: let retrieval try
            return nlg.answer_check(q.subject, q.relation, q.obj, verdict,
                                    facts)
        if isinstance(q, Compare):
            shared = m.compare(q.a, q.b)
            return nlg.answer_compare(q.a, q.b, shared) if shared else None
        return None
