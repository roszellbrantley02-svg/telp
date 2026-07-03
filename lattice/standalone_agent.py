"""
lattice/standalone_agent.py - LLM-free conversational HDC agent.

Same shape as HDCAgent (lattice/agent.py) but ZERO neural models:

  TextEncoder (MiniLM)        ->  CorpusRIEncoder (Random Indexing)
  Qwen2.5-0.5B decoder         ->  TemplateResponder (relation templates)
  Triple extraction (rules)    ->  TypedHDC v2 (already LLM-free)
  Knowledge graph              ->  HDCKnowledgeBase (already LLM-free)

The agent loads no transformer weights, no sentence-transformer
checkpoints, no torch.  It depends on numpy and sqlite3 only.

Pipeline per turn is identical to HDCAgent:
  1. Retrieve memories from the Lattice (now with the RI encoder)
  2. Extract triples from the message and add them to the KG
  3. Query KG forward + backward against all known relations
  4. Generate a response via TemplateResponder
  5. Store the (user, agent, conversation) turn back into the Lattice
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.store              import Lattice
from lattice.knowledge_graph    import HDCKnowledgeBase
from lattice.triple_extractor   import extract_triples
from lattice.standalone_encoder import CorpusRIEncoder
from lattice.standalone_responder import TemplateResponder
from lattice.sequence_predictor import HDCSequencePredictor
from lattice.standalone_generator import BackoffBeamGenerator
from lattice.structured_qa import StructuredQA
from lattice.multi_hop_qa import MultiHopQA
from lattice.aggregate_qa import AggregateQA
from lattice.analogy_qa import AnalogyQA
from lattice.compare_qa import CompareQA
from lattice.arithmetic_qa import detect_and_eval as arithmetic_eval
from lattice.code_qa import CodeQA
# Image support is OPTIONAL.  The pure-HDC image encoder uses raw
# pixels + LSH — no neural model, no CLIP.  Imported lazily so PIL
# isn't a hard dependency.
try:
    from lattice.image_encoder import ImageEncoder
    from lattice.image_bank import ImageBank
    _IMAGES_AVAILABLE = True
except ImportError:
    _IMAGES_AVAILABLE = False


DEFAULT_LATTICE_DB = _TELP_ROOT / "state" / "standalone_lattice.db"


class StandaloneAgent:
    """Conversational HDC agent — runs with no LLM at all."""

    def __init__(self, lattice_path: Path = DEFAULT_LATTICE_DB,
                   skip_ri_retrain: bool = False,
                   skip_ngram_retrain: bool = False):
        """
        skip_ri_retrain:    if True, don't retrain the RI encoder on
                            persisted memories.  Useful when the caller
                            will hot-swap a learned encoder (FluentTelp).
        skip_ngram_retrain: if True, don't rebuild the n-gram generator.
                            Useful for pure-ingest scripts that don't
                            need text generation.
        """
        # THE encoder (2026-07-02): semantic by default - meaning-based,
        # deterministic forever. RI remains as fallback (TELP_RI=1 or if
        # sentence-transformers is unavailable).
        import os as _os
        self.encoder = None
        if _os.environ.get("TELP_RI", "0") != "1":
            try:
                from lattice.semantic_encoder import SemanticEncoder
                print("[standalone] loading semantic encoder (MiniLM->HV) ...")
                self.encoder = SemanticEncoder()
            except Exception as e:
                print(f"[standalone] semantic encoder unavailable ({e}); "
                      f"falling back to RI")
        if self.encoder is None:
            print("[standalone] building corpus-trained RI encoder ...")
            self.encoder = CorpusRIEncoder()
        # auto-GPU: mirror the memory stack on cuda once it's big enough
        # to matter (the educated lattice) - one env var, set by default
        try:
            import sqlite3 as _sq
            _n = _sq.connect(str(lattice_path)).execute(
                "SELECT COUNT(*) FROM memories").fetchone()[0]
            if _n > 15000:
                import torch as _torch
                if _torch.cuda.is_available():
                    _os.environ.setdefault("LATTICE_DEVICE", "cuda")
                    print(f"[standalone] {_n} memories -> GPU search mirror on")
        except Exception:
            pass
        print("[standalone] opening lattice memory ...")
        self.lattice = Lattice(lattice_path, encoder=self.encoder)
        print(f"[standalone] lattice: {self.lattice.count()} memories stored")
        self._skip_ngram_retrain = skip_ngram_retrain

        # Feed corpus statistics to the encoder: for RI this trains the word
        # space; for the semantic encoder it only calibrates IDF ranking.
        if self.lattice.count() > 0 and not skip_ri_retrain:
            print("[standalone] calibrating encoder on persisted memories ...")
            for text in self.lattice._texts:
                self.encoder.add_sentence(text)
            print(f"[standalone] vocab: {self.encoder.stats()}")
        elif skip_ri_retrain:
            print("[standalone] skipping retrain (caller will swap encoder)")

        self.knowledge = HDCKnowledgeBase()
        self.turns: list[dict] = []
        # Recently-mentioned entities for pronoun resolution.
        # Each entry is (name, kind) with kind in {"person","thing","any"}.
        # When the user types "he/she", we prefer the most recent
        # "person"; "it/its" prefers "thing"; "they/their" allows
        # either.  Falls back to most-recent if no typed match.
        self._recent_entities: list[tuple[str, str]] = []
        # HDC generator — back-off n-gram with beam search + topic
        # anchoring.  Re-trained whenever new corpus text is ingested.
        self.seq = BackoffBeamGenerator(
            n_gram_sizes=(10, 7, 5, 3),
            encoder=self.encoder,
            beam_width=4,
        )
        # Role-bound HDC Q&A — claims encoded as (S, V, O) triples.
        # Tries first; falls through to bag-of-words retrieval on miss
        # or abstention.  Threshold = 0.45 chosen so exact slot
        # matches dominate but partial-phrase matches still qualify.
        self.structured = StructuredQA(self.encoder, abstain_threshold=0.45)
        self.multi_hop = MultiHopQA(self.structured, abstain_threshold=0.45)
        self.aggregate = AggregateQA(self.structured)
        self.analogy = AnalogyQA(self.lattice, self.structured)
        self.compare = CompareQA(self.structured)
        self.code_qa = CodeQA(self.lattice, self.structured)
        # Image support (optional, pure-HDC, no neural model).
        self._image_bank = None
        self._image_encoder = None
        self.responder = TemplateResponder(sequence_predictor=self.seq)
        # LAZY generator training (2026-07-02): on the educated 22K-row
        # lattice, n-gram training costs ~143s - only creative-extension
        # requests need it, so train on first use (ensure_generator).
        self._generator_trained = False
        if self.lattice.count() > 0:
            if skip_ngram_retrain:
                print("[standalone] skipping n-gram retrain (ingest-only mode)")
            print("[standalone] building role-bound claim store ...")
            n_claims = self._rebuild_structured_qa()
            print(f"[standalone] structured-QA: {n_claims} claims")

    def ensure_generator(self):
        """Train the n-gram generator on demand (first creative request).
        HONEST SCALING NOTE (2026-07-03): trained on the full educated lattice
        (22K rows -> ~450K prefix memories) this generator needs ~18GB RAM,
        ~3min training, and produces incoherent cross-topic word salad; a disk
        cache of its tables hit 37GB. So it now trains on a CAPPED sample of
        the corpus (recent + small) as a light flavor-extender only. Real
        generative composition is the imagination engine's job."""
        if self._generator_trained or self.lattice.count() == 0:
            return
        texts = [t for t, s in zip(self.lattice._texts, self.lattice._sources)
                 if s and s.startswith(self._CORPUS_PREFIXES)]
        cap = 2500
        if len(texts) > cap:
            texts = texts[-cap:]
        print(f"[standalone] training HDC generator on {len(texts)} rows "
              f"(capped; first creative use) ...", flush=True)
        self.seq.train(texts)
        self._generator_trained = True

    # ─── Live learning from Wikipedia ──────────────────────

    # Track corpus-source sentences so discovery sees real text —
    # NOT echoed conversation turns (user_msg/agent_response).
    # We include wikipedia (the main encyclopedic source), user_taught
    # (facts the user types via /teach), and wisdom (curated trading
    # statements) so all three persist across sessions in the claim
    # store on rebuild.
    # sources whose rows feed the claim store + generator at boot.
    # url: added 2026-07-02 (URL-learned facts were invisible to claims);
    # code: added for ingest_self self-code claims (previously vanished
    # every restart despite CodeQA sitting in the live answer chain)
    _CORPUS_PREFIXES = ("wikipedia:", "user_taught", "wisdom:", "legacy:",
                        "url:", "code:")

    def _corpus_sentences(self) -> list[str]:
        return [t for t, s in zip(self.lattice._texts, self.lattice._sources)
                if s.startswith(self._CORPUS_PREFIXES)]

    def _retrain_predictor(self) -> int:
        """Rebuild the HDC generator over the whole corpus."""
        sentences = self._corpus_sentences()
        if not sentences:
            return 0
        self.seq = BackoffBeamGenerator(
            n_gram_sizes=(10, 7, 5, 3),
            encoder=self.encoder,
            beam_width=4,
        )
        self.seq.train(sentences)
        self.responder.seq = self.seq
        return len(sentences)

    def _rebuild_structured_qa(self) -> int:
        """Rebuild role-bound claim store from the whole corpus."""
        sources = []
        sentences = []
        for t, s in zip(self.lattice._texts, self.lattice._sources):
            if s.startswith(self._CORPUS_PREFIXES):
                sentences.append(t)
                sources.append(s)
        self.structured = StructuredQA(self.encoder,
                                            abstain_threshold=0.45)
        n = self.structured.add_corpus(sentences, sources)

        # Layer in Wikidata-sourced birth/death years as claims so
        # date questions work even when the lead text doesn't carry
        # the year explicitly.  See lattice/wikidata_dates.py.
        try:
            import json as _json
            from pathlib import Path as _Path
            dates_path = _TELP_ROOT / "state" / "wikidata_dates.json"
            if dates_path.exists():
                cache = _json.loads(dates_path.read_text(encoding="utf-8"))
                n_date = 0
                for topic, dates in cache.items():
                    name = topic.replace("_", " ")
                    if dates.get("born") is not None:
                        self.structured.claim_text.append(
                            f"{name} was born in {dates['born']}.")
                        self.structured.claim_source.append(
                            f"wikidata:{dates.get('qid','?')}")
                        self.structured.claim_triple.append(
                            (name, "born_year", str(dates["born"])))
                        n_date += 1
                    if dates.get("died") is not None:
                        self.structured.claim_text.append(
                            f"{name} died in {dates['died']}.")
                        self.structured.claim_source.append(
                            f"wikidata:{dates.get('qid','?')}")
                        self.structured.claim_triple.append(
                            (name, "died_year", str(dates["died"])))
                        n_date += 1
                if n_date > 0:
                    self.structured._dirty = True
        except Exception as e:
            print(f"[standalone] could not load wikidata dates: {e}",
                    flush=True)

        self.multi_hop = MultiHopQA(self.structured,
                                        abstain_threshold=0.45)
        self.aggregate = AggregateQA(self.structured)
        self.analogy = AnalogyQA(self.lattice, self.structured)
        self.compare = CompareQA(self.structured)
        return n

    def _rebuild_kg_from_corpus(self) -> tuple[int, int]:
        """Re-run TypedHDC v2 over the entire wikipedia corpus and
        rebuild the KG so relation labels are derived from PATTERNS
        that recur across topics (not within a single summary).

        Returns (n_patterns, n_triples_added).
        """
        from lattice.typed_discovery_v2 import (
            discover_typed_windows_v2, extract_triples_v2,
        )
        sentences = self._corpus_sentences()
        if not sentences:
            return (0, 0)
        # min_pattern_count=2: a pattern must recur somewhere in the
        # whole corpus to count.  This makes labels meaningful.
        patterns = discover_typed_windows_v2(
            sentences, window_min=4, window_max=7,
            min_pattern_count=2, min_entities_in_window=2,
        )
        triples, _ = extract_triples_v2(patterns)
        # Fresh KG.
        self.knowledge = HDCKnowledgeBase()
        for s, r, o in triples:
            self.knowledge.add_fact(s, r, o)
        return (len(patterns), len(triples))

    def learn_wikipedia(self, topic: str) -> dict:
        from lattice.fetch_wiki import fetch_one

        print(f"[standalone] fetching Wikipedia summary for: {topic}")
        result = fetch_one(topic)
        if "error" in result or not result.get("extract"):
            return {"learned": False,
                      "reason": result.get("error", "no extract")}

        text = result["extract"]
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

        # Train the RI encoder on the new sentences BEFORE storing them.
        for s in sentences:
            self.encoder.add_sentence(s)

        n_added = 0
        for s in sentences:
            self.lattice.add(s, source=f"wikipedia:{topic}",
                                wiki_title=result.get("title", topic))
            n_added += 1

        # Re-derive the KG from the WHOLE corpus so labels come from
        # patterns that recur across topics.
        n_patterns, n_facts = self._rebuild_kg_from_corpus()

        # Re-train the sequence predictor on the whole corpus so the
        # generator picks up the latest n-gram statistics.
        self._retrain_predictor()

        # Re-derive structured-QA claims (role-bound triples).
        self._rebuild_structured_qa()

        return {
            "learned":          True,
            "topic":            topic,
            "title":            result.get("title", topic),
            "sentences_added":  n_added,
            "patterns_found":   n_patterns,
            "facts_added":      n_facts,
            "text_chars":       len(text),
            "vocab_size":       self.encoder.stats()["vocab_size"],
        }

    # ─── KG lookup (forward + backward across all relations) ────

    _STOP_ENTITIES = {
        "what", "who", "where", "when", "why", "how", "which", "whom",
        "is", "was", "were", "are", "the", "a", "an", "of", "in", "on",
        "at", "to", "from", "and", "or", "but", "tell", "me", "about",
        "do", "does", "did", "i", "you", "he", "she", "it", "they",
    }

    def _is_noise_token(self, tok: str) -> bool:
        """Skip junk subjects/objects: digits, single chars, stopwords."""
        if not tok or len(tok) < 2:
            return True
        if tok.lower() in self._STOP_ENTITIES:
            return True
        # Pure digit-only or year-only objects without context aren't great
        if tok.isdigit() and len(tok) <= 2:
            return True
        return False

    def _lookup_kg_facts(self, user_msg: str) -> list[tuple[str, str, str]]:
        words = re.findall(r"[A-Za-z][\w-]*", user_msg)
        # Drop question words and other stopwords before matching entities.
        words = [w for w in words if w.lower() not in self._STOP_ENTITIES]
        entity_lookup = {e.lower(): e for e in self.knowledge.entities.keys()}

        matched = set()
        for w in words:
            if w.lower() in entity_lookup:
                matched.add(entity_lookup[w.lower()])
        for i in range(len(words) - 1):
            two = f"{words[i]} {words[i+1]}"
            if two.lower() in entity_lookup:
                matched.add(entity_lookup[two.lower()])

        if not matched:
            return []

        all_entities = list(self.knowledge.entities.keys())
        all_relations = list(self.knowledge.relations.keys())
        hits: list[tuple[str, str, str]] = []
        max_q = 50

        for ent in list(matched)[:5]:
            for rel in all_relations[:max_q]:
                results = self.knowledge.query(
                    ent, rel, top_k=1, restrict_to=all_entities,
                )
                if results and results[0][1] < 4400:
                    obj = results[0][0]
                    if obj == ent or self._is_noise_token(obj):
                        continue
                    hits.append((ent, rel, obj))
            for fact_s, fact_r, fact_o in self.knowledge.facts:
                if fact_o == ent and fact_s != ent \
                        and not self._is_noise_token(fact_s):
                    hits.append((fact_s, fact_r, fact_o))

        # Dedup, preserve order
        seen = set()
        deduped = []
        for s, r, o in hits:
            key = (s, r, o)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((s, r, o))
        return deduped[:20]

    # ─── Per-turn pipeline ─────────────────────────────────

    _STOP = {
        "what","who","where","when","why","how","which","is","was","are","were",
        "be","do","does","did","the","a","an","of","tell","me","about",
        "you","i","in","on","at","to","from","with","for",
    }

    # Pronouns that may refer to a previously-mentioned entity.  When
    # one of these appears in a query, we substitute the most-recent
    # entity from self._recent_entities.  Person vs object distinction
    # is informational — we don't currently have gender/animacy data
    # on the stored entities, so all pronouns resolve to the same
    # latest entity.
    _PRONOUN_PATTERN = re.compile(
        r"\b(he|him|his|she|her|hers|they|them|their|theirs|it|its)\b",
        re.I,
    )

    _POSSESSIVE_PRONOUNS = {"his", "her", "their", "its"}

    # Known place / thing words that strongly suggest the entity is
    # NOT a person.  Used to flip the kind tag when we see them.
    _NON_PERSON_HINTS = {
        # Countries
        "germany","france","japan","italy","spain","china","india","brazil",
        "canada","russia","egypt","australia","poland","greece","austria",
        "england","scotland","ireland","wales","switzerland","belgium",
        "netherlands","portugal","mexico","turkey","iran","iraq","israel",
        "lebanon","syria","morocco","algeria","tunisia","libya","kenya",
        "nigeria","ghana","ethiopia","vietnam","thailand","indonesia",
        "philippines","malaysia","singapore","cuba","colombia","venezuela",
        "peru","chile","united","kingdom","states","emirates",
        # Major cities
        "berlin","paris","tokyo","rome","madrid","cairo","london","beijing",
        "warsaw","moscow","mumbai","delhi","brasilia","ottawa","canberra",
        "vienna","prague","athens","istanbul","new york",
        # Generic
        "city","country","capital","continent","ocean","sea","river",
    }

    def _entity_kind(self, name: str) -> str:
        """Guess whether `name` is a person, a thing/place, or unknown."""
        if not name:
            return "any"
        words = name.lower().split()
        if any(w in self._NON_PERSON_HINTS for w in words):
            return "thing"
        # Multi-word capitalized names are usually people ("Albert
        # Einstein", "Marie Curie", "Bob Dylan").
        if len(words) >= 2 and all(w[:1].isalpha() for w in words):
            return "person"
        return "any"

    def _push_entity(self, entity: str | None, kind: str | None = None) -> None:
        """Push an entity onto the recent-entity stack (FIFO of size 5).
        If `kind` is None, we guess from the name itself.
        """
        if not entity:
            return
        e = entity.strip()
        if not e or len(e) < 2:
            return
        k = kind or self._entity_kind(e)
        # Don't duplicate the most-recent one.
        if self._recent_entities and self._recent_entities[-1][0] == e:
            return
        self._recent_entities.append((e, k))
        if len(self._recent_entities) > 5:
            self._recent_entities.pop(0)

    # Words to NEVER track as entities even if capitalised at the
    # start of a sentence or in a stock phrase.  Avoids the recent-
    # entities stack filling up with "What", "Tell", "Federal", etc.
    _ENTITY_BLACKLIST = {
        "what","who","where","when","why","how","which","whose","whom",
        "tell","please","yes","no",
        "federal","official","officially","republic","republic.",
        "german-born","french-born","british-born","english-born",
        "polish-born","russian-born","austrian-born","italian-born",
        "spanish-born","german","french","british","english","polish",
        "russian","austrian","italian","spanish","american","canadian",
        "australian","japanese","chinese","indian","greek","czech",
        "north","south","east","west","central","western","eastern",
        "northern","southern","new","old",
    }

    @staticmethod
    def _edit_distance(a: str, b: str, cap: int = 2) -> int:
        """Levenshtein distance with early-exit when it exceeds `cap`.
        Returns cap+1 if the true distance is greater than cap.
        Standard DP, but bails out per row if no cell <= cap remains.
        """
        if a == b:
            return 0
        la, lb = len(a), len(b)
        if abs(la - lb) > cap:
            return cap + 1
        if la < lb:
            a, b = b, a
            la, lb = lb, la
        prev = list(range(lb + 1))
        for i in range(1, la + 1):
            cur = [i] + [0] * lb
            row_min = cur[0]
            for j in range(1, lb + 1):
                ins = cur[j - 1] + 1
                dele = prev[j] + 1
                sub = prev[j - 1] + (a[i - 1] != b[j - 1])
                cur[j] = min(ins, dele, sub)
                if cur[j] < row_min:
                    row_min = cur[j]
            if row_min > cap:
                return cap + 1
            prev = cur
        return prev[lb]

    def _typo_correct(self, query: str) -> tuple[str, list[tuple[str, str]]]:
        """For each proper-noun-shaped token in the query, if it's
        not in our vocab/known-entities, look for the closest known
        PROPER-NOUN-LIKE candidate by edit distance.  Replaces in
        place if a unique close match (distance <= 2) exists.

        Candidate pool is restricted to:
        - article topic tokens (capital-of-X kind of names)
        - structured claim subject tokens (entities the system knows)
        We do NOT correct to arbitrary vocabulary words — that would
        rewrite e.g. "Picaso" to "piano".

        Returns (rewritten_query, [(original, corrected), ...]).
        """
        candidates: set[str] = set()
        # Structured-claim subjects: these ARE the entities the system
        # has facts about.
        for s, _, _ in self.structured.claim_triple:
            for tok in re.split(r"[\s_-]+", s.lower()):
                if len(tok) >= 4 and tok.isalpha():
                    candidates.add(tok)
        # Wikipedia article topic tokens: places, people, things we
        # have content for.
        for src in self.lattice._sources:
            if src.startswith("wikipedia:"):
                topic = src.split(":", 1)[1].lower()
                for tok in re.split(r"[\s_-]+", topic):
                    if len(tok) >= 4 and tok.isalpha():
                        candidates.add(tok)

        corrections: list[tuple[str, str]] = []
        out = query

        # Look only at proper-noun-shaped tokens (capitalised, >=4 chars).
        for m in re.finditer(r"\b([A-Z][\w-]{3,})\b", query):
            tok = m.group(1)
            low = tok.lower()
            if low in candidates:
                continue
            if low in self._STOP or low in self._ENTITY_BLACKLIST:
                continue
            # Don't correct common content words that happen to be
            # capitalised (sentence-initial "Tell", question words, etc).
            if low in self.encoder.index_vectors and low not in candidates:
                # In our vocab as a non-entity word - don't replace it.
                continue
            # Find unique closest candidate within edit distance 2.
            best_d = 3
            best = None
            best_count = 0
            for c in candidates:
                if abs(len(c) - len(low)) > 2:
                    continue
                d = self._edit_distance(low, c, cap=2)
                if d < best_d:
                    best_d = d
                    best = c
                    best_count = 1
                elif d == best_d:
                    best_count += 1
            # Only replace if a UNIQUE close match exists.
            if best is not None and best_count == 1 and best_d <= 2:
                # Preserve original case of the first character.
                cased = best.capitalize() if tok[0].isupper() else best
                out = out[:m.start()] + cased + out[m.end():]
                corrections.append((tok, cased))
        return out, corrections

    def _query_entities(self, query: str) -> list[str]:
        """Pull capitalized proper-noun runs out of `query` — these
        are entity candidates we should remember for next turn.
        Filters obvious non-entities like 'What', 'Federal', etc."""
        pat = re.compile(r"(?<!\w)([A-Z][\w-]+(?:\s+[A-Z][\w-]+){0,3})")
        out: list[str] = []
        for m in pat.finditer(query):
            tok = m.group(1)
            if tok.lower() in self._ENTITY_BLACKLIST:
                continue
            # Reject single-token "What"/"Tell" etc. at sentence start
            # whose lowercased form is a question word.
            if len(tok.split()) == 1 and tok.lower() in self._STOP:
                continue
            out.append(tok)
        return out

    # Map pronoun -> preferred entity kind.
    _PRONOUN_KIND_PREFERENCE = {
        "he":     "person", "him": "person", "his": "person",
        "she":    "person", "her": "person", "hers": "person",
        "they":   "any",    "them": "any",   "their": "any", "theirs": "any",
        "it":     "thing",  "its": "thing",
    }

    def _resolve_pronouns(self, query: str) -> tuple[str, bool]:
        """Substitute pronouns in `query` with the most-recent entity
        of the right kind.  "He/she" prefer the most recent person.
        "It/its" prefer the most recent thing/place.  Falls back to
        the most-recent entity of any kind if no typed match exists.
        """
        if not self._recent_entities:
            return query, False
        if not self._PRONOUN_PATTERN.search(query):
            return query, False

        def pick_for(pron: str) -> str | None:
            pref = self._PRONOUN_KIND_PREFERENCE.get(pron, "any")
            # Walk from most-recent backwards looking for the right kind.
            for name, kind in reversed(self._recent_entities):
                if pref == "any" or kind == pref:
                    return name
            # Fallback: most recent of any kind.
            return self._recent_entities[-1][0]

        def _in_sentence_antecedent(before: str, tok: str) -> bool:
            """A sentence that names its own subject keeps its pronouns:
            'how old was Alexander Graham Bell when HE died?' must not
            pull a stale entity off the stack; 'I have 2 apples and eat
            5 of THEM' has its antecedent right there."""
            seg = re.split(r"[.!?]", before)[-1]
            if tok in ("he", "she", "his", "her", "hers", "him",
                       "it", "its"):
                for pm in re.finditer(
                        r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*)", seg):
                    if pm.group(1).lower() not in self._ENTITY_BLACKLIST:
                        return True
            if tok in ("they", "them", "their", "theirs"):
                for nm in re.finditer(r"\b([a-z]{3,}s)\b", seg):
                    w = nm.group(1)
                    if w not in self._STOP and not w.endswith("ss"):
                        return True
            return False

        resolved_kinds: set = set()

        def sub(m: re.Match) -> str:
            tok = m.group(1).lower()
            pref = self._PRONOUN_KIND_PREFERENCE.get(tok, "any")
            # once one pronoun of a kind is resolved, the rest of the
            # query refers to IT in-sentence: "how old was he when he
            # died?" -> "how old was Galileo when he died?" (not
            # "...when Galileo died?", which no route parses)
            if pref in resolved_kinds:
                return m.group(0)
            if _in_sentence_antecedent(query[:m.start()], tok):
                return m.group(0)
            picked = pick_for(tok)
            if picked is None:
                return m.group(0)
            resolved_kinds.add(pref)
            if tok in self._POSSESSIVE_PRONOUNS:
                return f"{picked}'s"
            return picked

        rewritten = self._PRONOUN_PATTERN.sub(sub, query)
        return rewritten, rewritten != query

    # Question patterns: the operative noun/verb whose presence in a
    # candidate sentence indicates that sentence is likely the answer
    # to a question of this shape.  Patterns are forgiving on syntax:
    # they catch "capital of X", "X's capital", "What's the capital
    # of X", etc.
    _QUESTION_TARGETS = [
        # Capital — handles "capital of X", "X's capital",
        # "what is X's capital", etc.
        (re.compile(r"\bcapital\b",          re.I), "capital"),
        # Verb-target patterns.  We map a wider set of verbs to a
        # single canonical target so synonyms ("came up with",
        # "created") all map to the canonical word in the answer
        # sentence.  Includes passive ("was X by ...") and possessive
        # / inverted forms.
        (re.compile(r"\b(?:invented|created|made|built|designed)\b", re.I),
         "invented"),
        (re.compile(r"\b(?:composed|wrote)\b",       re.I), "wrote"),
        (re.compile(r"\b(?:discovered|found)\b",     re.I), "discovered"),
        (re.compile(r"\b(?:founded|established)\b",  re.I), "founded"),
        (re.compile(r"\b(?:developed|came\s+up\s+with|created|formulated)\b",
                       re.I), "developed"),
        (re.compile(r"\bpainted\b",          re.I), "painted"),
        (re.compile(r"\bborn\b",             re.I), "born"),
    ]

    # Common abbreviation -> expanded title tokens.  Used to make queries
    # like "capital of the UK" match the United_Kingdom article.
    _ABBREV_MAP = {
        "uk":  {"united", "kingdom"},
        "us":  {"united", "states"},
        "usa": {"united", "states"},
        "uae": {"united", "arab", "emirates"},
    }

    def _content_words(self, text: str) -> set[str]:
        out = {w.lower() for w in re.findall(r"[A-Za-z][\w-]*", text)
                  if w.lower() not in self._STOP}
        # Expand known abbreviations so title-injection can still match.
        expansions: set[str] = set()
        for w in out:
            if w in self._ABBREV_MAP:
                expansions |= self._ABBREV_MAP[w]
        return out | expansions

    def _stems(self, words: set[str], n: int = 5) -> set[str]:
        """Return a set of length-n prefixes of `words`.  Used as a
        crude lemmatiser so "evolution" matches "evolutionary",
        "developed" matches "developing", etc.
        """
        return {w[:n] for w in words if len(w) >= n}

    def _is_proper_noun_in(self, word: str, original_query: str) -> bool:
        """Was this word capitalised in the user's original query?"""
        for tok in re.findall(r"[A-Za-z][\w-]*", original_query):
            if tok.lower() == word and tok[0].isupper():
                return True
        return False

    def _question_target(self, query: str) -> str | None:
        for pat, target in self._QUESTION_TARGETS:
            if pat.search(query):
                return target
        return None

    # Patterns whose group(1) captures the OBJECT of the question
    # (the thing being asked about).  Order matters — the first match
    # wins.  Includes possessive ("X's capital"), inverted/passive
    # ("written by whom"), short biographical ("what is X").
    _OBJECT_PATTERNS = [
        # Active: "Who Verbed X"
        re.compile(r"\bwho\s+(?:wrote|composed|invented|discovered|founded|"
                     r"developed|painted|made|built|designed|created|came\s+up\s+with|formulated)"
                     r"\s+(?:the\s+)?(.+?)\??$",
                     re.I),
        # Passive: "X was Verbed by whom" / "X was Verbed by ..."
        re.compile(r"\bthe\s+(.+?)\s+(?:was|were)\s+"
                     r"(?:wrote|composed|invented|discovered|founded|"
                     r"developed|painted|made|built|designed|created|written|"
                     r"composed|invented)\s+by\b",
                     re.I),
        # Possessive capital: "X's capital", "What's X's capital"
        re.compile(r"\b([\w-]+)'s\s+capital\b", re.I),
        re.compile(r"\bcapital\s+of\s+(?:the\s+)?(.+?)\??$", re.I),
        # "Which city is X's capital?"
        re.compile(r"\bwhich\s+city\s+is\s+([\w-]+)'s\s+capital\b", re.I),
        # "X's capital is" / "X's capital is?"
        re.compile(r"\b([\w-]+)'s\s+capital\s+is\b", re.I),
        # "X is?" / "X capital is?" — bare-bones
        re.compile(r"\b([\w-]+)'s\s+capital\s+city\b", re.I),
        # Generic biographical
        re.compile(r"\bwhere\s+(?:was|is)\s+(.+?)\s+(?:born|located|from)\??$",
                     re.I),
        re.compile(r"\bwho\s+(?:was|is)\s+(.+?)\??$", re.I),
        re.compile(r"\btell\s+me\s+(?:about|who|what)\s+(.+?)\??$", re.I),
        re.compile(r"\bdescribe\s+(.+?)\??$", re.I),
        re.compile(r"\bwhat\s+(?:is|are)\s+(?:an?\s+|the\s+)?(.+?)\??$",
                     re.I),
    ]

    def _question_object(self, query: str) -> set[str]:
        """Extract the content words of the question's grammatical
        object (the thing being asked about).  Empty set if no
        pattern matches.
        """
        for pat in self._OBJECT_PATTERNS:
            m = pat.search(query)
            if m:
                obj = m.group(1)
                return self._content_words(obj)
        return set()

    def _topic_tokens_for_text(self, text: str) -> set[str]:
        """Find the wikipedia:<topic> source of `text` in the lattice
        and return the tokens of that topic (for title-boost matching).
        """
        for t, src in zip(self.lattice._texts, self.lattice._sources):
            if t == text and src.startswith("wikipedia:"):
                topic = src.split(":", 1)[1].lower()
                return {w for w in re.split(r"[_\s-]+", topic) if w}
        return set()

    def _build_content_index(self) -> dict[str, list[int]]:
        """Inverted index: content-word -> list of lattice indices of
        sentences containing that word.  Used to inject candidates that
        the HDC encoder may have missed.

        Cached and invalidated when the lattice grows.
        """
        if (getattr(self, "_content_idx_cached_size", -1)
                == len(self.lattice._texts)):
            return self._content_idx
        idx: dict[str, list[int]] = {}
        for i, (text, src) in enumerate(zip(self.lattice._texts,
                                              self.lattice._sources)):
            if not src.startswith("wikipedia:"):
                continue
            for w in self._content_words(text):
                idx.setdefault(w, []).append(i)
        self._content_idx = idx
        self._content_idx_cached_size = len(self.lattice._texts)
        return idx

    def _content_injected_candidates(self, q_words: set[str],
                                         max_per_word: int = 8) -> list[dict]:
        """For each rare query word, inject up to N sentences from the
        wikipedia corpus that contain it.  Rare = IDF >= 4.  This
        catches the case where the encoder fails to put the right
        sentence in the HDC top-K but the right sentence does mention
        the query's key word literally.
        """
        idx = self._build_content_index()
        hits: list[dict] = []
        seen: set[int] = set()
        for w in q_words:
            if self.encoder._idf_weight(w) < 4.0:
                continue
            for i in idx.get(w, [])[:max_per_word]:
                if i in seen:
                    continue
                seen.add(i)
                hits.append({
                    "id":           self.lattice._ids[i],
                    "text":         self.lattice._texts[i],
                    "source":       self.lattice._sources[i],
                    "similarity":   0.0,
                    "distance_pct": 0.5,
                    "_injected":    True,
                })
        return hits

    def _build_title_index(self) -> dict[str, list[int]]:
        """Build {topic_token -> [lattice_index, ...]} from all
        wikipedia: sources.  Cached on the instance after first build.
        Invalidated when new memories are added.
        """
        if (getattr(self, "_title_idx_cached_size", -1)
                == len(self.lattice._texts)):
            return self._title_idx
        idx: dict[str, list[int]] = {}
        for i, src in enumerate(self.lattice._sources):
            if not src.startswith("wikipedia:"):
                continue
            topic = src.split(":", 1)[1].lower()
            for tok in re.split(r"[_\s-]+", topic):
                if not tok or len(tok) < 2:
                    continue
                idx.setdefault(tok, []).append(i)
        self._title_idx = idx
        self._title_idx_cached_size = len(self.lattice._texts)
        return idx

    def _title_injected_candidates(self, q_words: set[str]) -> list[dict]:
        """Return all sentences from articles whose topic matches any
        query content word.  Provides recall even when the HDC encoder
        misses the right article entirely.
        """
        idx = self._build_title_index()
        hits: list[dict] = []
        seen: set[int] = set()
        for w in q_words:
            for i in idx.get(w, []):
                if i in seen:
                    continue
                seen.add(i)
                hits.append({
                    "id":           self.lattice._ids[i],
                    "text":         self.lattice._texts[i],
                    "source":       self.lattice._sources[i],
                    "similarity":   0.0,           # unknown until rerank
                    "distance_pct": 0.5,
                    "_injected":    True,
                })
        return hits

    _QUESTION_MARKERS = re.compile(
        r"\?\s*$|^\s*(?:what|who|where|when|why|how|which|whose|whom|"
        r"is|are|was|were|do|does|did|can|could|will|would|should|may|"
        r"might|tell|describe)\b",
        re.I,
    )

    def _looks_like_question(self, text: str) -> bool:
        return bool(self._QUESTION_MARKERS.search(text.strip()))

    # Confidence thresholds for response wrapping.  Values are
    # interpreted in the [-1, 1] balanced-Hamming similarity space.
    # RECALIBRATED 2026-07-02 for the semantic encoder (MiniLM->LSH):
    # measured distribution - verbatim 1.0, true semantic match 0.37-0.55,
    # unrelated <= 0.15.  (Old RI-era bands were 0.80/0.50/0.30.)
    _CONF_HIGH = 0.55    # >= : answer plainly
    _CONF_MED  = 0.38    # >= : prefix with "I think"
    _CONF_LOW  = 0.25    # >= : hedge strongly
    # Below _CONF_LOW we abstain explicitly.

    def _wrap_with_confidence(self, answer: str, sim: float,
                                 prefix_qa: str = "") -> str:
        """Wrap an answer with confidence language calibrated to sim.
        prefix_qa is an optional source tag for the user ("from KG",
        "via 2-hop chain", etc) - currently unused but reserved.
        """
        if sim >= self._CONF_HIGH:
            return answer
        if sim >= self._CONF_MED:
            return f"I think: {answer}"
        if sim >= self._CONF_LOW:
            return (f"I'm not sure, but possibly: {answer}\n"
                      f"(low confidence: {sim:.2f})")
        return (f"I don't have a confident answer. Closest match I found:"
                  f" {answer} (very low confidence: {sim:.2f})")

    def respond(self, user_msg: str) -> str:
        # 0a. Resolve pronouns against recent turns' entities.  "Where
        #     was he born?" becomes "Where was Einstein born?" if the
        #     previous turn established Einstein as the topic.
        original_msg = user_msg
        rewritten, did_sub = self._resolve_pronouns(user_msg)
        if did_sub:
            user_msg = rewritten

        # 0a-typo. Fuzzy-correct proper-noun-shaped tokens that aren't
        #          in our vocabulary.  "Einstien" -> "Einstein",
        #          "Germny" -> "Germany".  Only fires for unique close
        #          matches at edit-distance <= 2, so common words and
        #          ambiguous typos pass through unchanged.
        corrected, corrections = self._typo_correct(user_msg)
        if corrections:
            user_msg = corrected

        # 0a'. If the user is making a DECLARATIVE statement (not a
        #      question), extract claims from it and add them to the
        #      structured QA store immediately.  This is how the
        #      agent "learns" by being told: "Mozart was Austrian."
        #      becomes (Mozart, born, Austria) which the multi-hop
        #      module can chain through next turn.
        if not self._looks_like_question(user_msg):
            n_new = self.structured.add_sentence(user_msg, source="user_taught")
            if n_new > 0:
                # Persist the taught sentence to the lattice for
                # cross-session memory.
                self.lattice.add(user_msg, source="user_taught",
                                    turn=len(self.turns))

        # 0. Snapshot the encoder vocab BEFORE we add the new sentence
        #    — needed by the confidence calibrator to detect query
        #    tokens we haven't seen before.
        self._vocab_before_turn = frozenset(self.encoder.index_vectors.keys())
        # Update the encoder so the new sentence's vocabulary is known.
        self.encoder.add_sentence(user_msg)

        # 0a-pre-arith.  Arithmetic: "What is 247 + 893?".  Safe-eval
        #                a restricted AST — no `eval()`, no name
        #                lookups, just numeric operators.
        ar = arithmetic_eval(user_msg)
        if ar is not None:
            response = ar.explain
            self.turns.append({
                "ts":     datetime.now(timezone.utc).isoformat(),
                "user":   user_msg, "agent": response,
                "retrieved_memories": [], "extracted_triples": [],
                "kg_hits": [],
                "arithmetic": {"expr": ar.expression, "value": ar.value},
            })
            self.lattice.add(user_msg, source="user_msg",
                                turn=len(self.turns) - 1)
            self.lattice.add(response, source="agent_response",
                                turn=len(self.turns) - 1)
            return response

        # 0a-pre-code.  Code questions: "What does function X do?",
        #               "Where is X defined?", "What does X import?",
        #               "What methods does X have?" — route directly
        #               to the code claim store so we don't fall back
        #               to lexical retrieval that returns Wikipedia
        #               sentences on technical questions.
        cq = self.code_qa.answer(user_msg)
        if cq is not None:
            response = cq.answer
            self.turns.append({
                "ts":     datetime.now(timezone.utc).isoformat(),
                "user":   user_msg, "agent": response,
                "retrieved_memories": [], "extracted_triples": [],
                "kg_hits": [],
                "code_qa": {"kind": cq.kind, "target": cq.target},
            })
            self.lattice.add(user_msg, source="user_msg",
                                turn=len(self.turns) - 1)
            self.lattice.add(response, source="agent_response",
                                turn=len(self.turns) - 1)
            return response

        # 0a-pre-1. Comparator: "Was X born before Y?", "Who is older
        #           X or Y?", "Was X born before 1900?".  Fetches the
        #           two values from the claim store and compares.
        cmp_res = self.compare.answer(user_msg)
        if cmp_res is not None and cmp_res.a_value is not None:
            response = cmp_res.explain
            self.turns.append({
                "ts":                 datetime.now(timezone.utc).isoformat(),
                "user":               user_msg,
                "agent":              response,
                "retrieved_memories": [],
                "extracted_triples":  [],
                "kg_hits":            [],
                "compare":            {
                    "kind":   cmp_res.kind,
                    "a":      cmp_res.a, "b": cmp_res.b,
                    "a_val":  cmp_res.a_value, "b_val": cmp_res.b_value,
                    "winner": cmp_res.winner,
                },
            })
            self.lattice.add(user_msg, source="user_msg",
                                turn=len(self.turns) - 1)
            self.lattice.add(response, source="agent_response",
                                turn=len(self.turns) - 1)
            return response

        # 0a-pre0. Analogy: "X is to Y as Z is to ?"  via XOR algebra.
        #          Uses the encoder's word HVs directly — no structured
        #          claim needed.  Decoded by nearest-neighbour over the
        #          known-entity vocabulary.
        ana = self.analogy.answer(user_msg)
        if ana is not None:
            response = (f"{ana['answer']}  "
                          f"(by analogy {ana['a']}:{ana['b']} :: "
                          f"{ana['c']}:?, similarity={ana['similarity']:.2f})")
            self.turns.append({
                "ts":                 datetime.now(timezone.utc).isoformat(),
                "user":               user_msg,
                "agent":              response,
                "retrieved_memories": [],
                "extracted_triples":  [],
                "kg_hits":            [],
                "analogy":            ana,
            })
            self.lattice.add(user_msg, source="user_msg",
                                turn=len(self.turns) - 1)
            self.lattice.add(response, source="agent_response",
                                turn=len(self.turns) - 1)
            return response

        # 0a-pre. Aggregate / enumerative queries return a SET, not
        #         a single answer.  "List all the composers" / "Which
        #         scientists were German?" / "How many countries do
        #         you know?".  Skip the QA chain entirely.
        agg = self.aggregate.answer(user_msg)
        if agg is not None:
            response = agg.format()
            self.turns.append({
                "ts":                 datetime.now(timezone.utc).isoformat(),
                "user":               user_msg,
                "agent":              response,
                "retrieved_memories": [],
                "extracted_triples":  [],
                "kg_hits":            [],
                "aggregate":          {
                    "pattern":  agg.pattern,
                    "filter":   agg.filter_desc,
                    "count":    agg.count,
                },
            })
            self.lattice.add(user_msg, source="user_msg",
                                turn=len(self.turns) - 1)
            self.lattice.add(response, source="agent_response",
                                turn=len(self.turns) - 1)
            return response

        # 0a. Multi-hop pass: chain two structured-QA queries via HDC
        #     substitution.  Catches "the capital of the country where
        #     X was born?" / "where was the composer of X born?" etc.
        mh = self.multi_hop.answer(user_msg)
        if mh is not None:
            # Chain confidence is the WEAKER of the two hop sims —
            # a chain is only as solid as its weakest link.
            chain_sim = min(mh.inner_sim, mh.outer_sim)
            raw_answer = mh.outer_text
            answer = self._wrap_with_confidence(raw_answer, chain_sim)
            self.turns.append({
                "ts":                 datetime.now(timezone.utc).isoformat(),
                "user":               user_msg,
                "agent":              answer,
                "retrieved_memories": [mh.outer_text],
                "extracted_triples":  [],
                "kg_hits":            [],
                "multi_hop": {
                    "final":      mh.final_answer,
                    "inner":      mh.inner_answer,
                    "outer":      mh.outer_answer,
                    "pattern":    mh.pattern_name,
                    "inner_sim":  mh.inner_sim,
                    "outer_sim":  mh.outer_sim,
                },
            })
            self.lattice.add(user_msg, source="user_msg",
                                turn=len(self.turns) - 1)
            self.lattice.add(answer, source="agent_response",
                                turn=len(self.turns) - 1)
            # Push entities for next-turn pronoun resolution: the
            # inner answer (e.g. Einstein) and the final answer
            # (e.g. Berlin) are both candidates for "he/she/it".
            for ent in self._query_entities(user_msg):
                self._push_entity(ent)
            self._push_entity(mh.inner_answer)
            self._push_entity(mh.final_answer)
            return answer

        # 0b. Structured-QA pass.  If the question parses into a known
        #     shape (e.g. "Who Verbed X?", "Capital of X?") and we have
        #     a high-confidence role-bound claim match, return that
        #     sentence directly.  This handles paraphrase and synonym
        #     robustness via the encoder's co-occurrence vectors.
        sqa = self.structured.answer(user_msg)
        if sqa is not None and sqa["similarity"] >= 0.55:
            sentence = self._wrap_with_confidence(
                sqa["sentence"], sqa["similarity"])
            self.turns.append({
                "ts":                 datetime.now(timezone.utc).isoformat(),
                "user":               user_msg,
                "agent":              sentence,
                "retrieved_memories": [sentence],
                "extracted_triples":  [],
                "kg_hits":            [],
                "structured_qa":      {
                    "subj":         sqa["subj"],
                    "verb":         sqa["verb"],
                    "obj":          sqa["obj"],
                    "answer_word":  sqa["answer_word"],
                    "similarity":   sqa["similarity"],
                    "unknown_role": sqa["unknown_role"],
                },
            })
            self.lattice.add(user_msg, source="user_msg",
                                turn=len(self.turns) - 1)
            self.lattice.add(sentence, source="agent_response",
                                turn=len(self.turns) - 1)
            for ent in self._query_entities(user_msg):
                self._push_entity(ent)
            self._push_entity(sqa["subj"])
            self._push_entity(sqa["answer_word"])
            return sentence

        # 1. Memory retrieval — over-fetch, then drop conversation-turn
        #    echoes so the responder doesn't quote its own past output.
        raw = self.lattice.query(user_msg, k=60, threshold=0.49)
        raw = [m for m in raw
                if not str(m.get("source", "")).startswith(
                    ("agent_response", "conversation_turn", "user_msg"))]

        q_words = self._content_words(user_msg)

        # 1b. Title-based candidate injection: if a query word matches a
        #     Wikipedia article topic, pull ALL of that article's
        #     sentences into the candidate pool.  Closes the recall gap
        #     when the encoder fails to retrieve a relevant sentence at
        #     all (e.g. "What is the capital of Australia?").
        if q_words:
            injected = self._title_injected_candidates(q_words)
            seen_ids = {m["id"] for m in raw}
            for m in injected:
                if m["id"] not in seen_ids:
                    raw.append(m)
                    seen_ids.add(m["id"])
            # 1c. Content-word injection: for each rare query word
            #     (high IDF), inject any wikipedia sentence containing
            #     that word.  Catches "Who discovered evolution?" ->
            #     Darwin's "evolutionary biology" sentence even if the
            #     encoder didn't retrieve it.
            cinjected = self._content_injected_candidates(q_words)
            for m in cinjected:
                if m["id"] not in seen_ids:
                    raw.append(m)
                    seen_ids.add(m["id"])

        # 1a. Hybrid rerank with four signals:
        #     - HDC similarity (fuzzy semantic match)
        #     - IDF-weighted lexical overlap (rare proper nouns dominate)
        #     - Title-boost: candidate from an article whose topic name
        #       overlaps the query gets a strong boost
        #     - Question-target boost: for patterns like "capital of X"
        #       or "who invented X", give a fixed bonus to candidates
        #       whose text contains the question's target word
        q_target = self._question_target(user_msg)
        q_stems = self._stems(q_words)
        q_proper = {w for w in q_words
                      if self._is_proper_noun_in(w, user_msg)}
        # Question object: the X in "Who Verbed X?", "Capital of X?",
        # "Tell me about X".  These words identify what the user is
        # asking about and get 3x the lexical weight.
        q_object_words = self._question_object(user_msg)
        if q_words:
            for m in raw:
                m_words = self._content_words(m["text"])
                shared = q_words & m_words
                # Stem-based partial-match catches evolution/evolutionary,
                # developed/developing, composed/composer, etc.
                m_stems = self._stems(m_words)
                stem_overlap = q_stems & m_stems
                # Don't double-count exact matches in the stem bonus.
                exact_stems = {w[:5] for w in shared if len(w) >= 5}
                extra_stems = stem_overlap - exact_stems

                # Object-aware IDF bonus: words that are part of the
                # question's grammatical object count more — BUT only
                # when title_overlap is empty.  Once we know we are
                # already in the right article (title matches the
                # object), we don't need the object multiplier; we
                # want to pick the best sentence inside that article,
                # which is the one matching the question target.
                topic_toks = self._topic_tokens_for_text(m["text"])
                title_overlap = q_words & topic_toks
                has_target = (q_target is not None
                                and q_target in m["text"].lower())
                object_mult = 1.0 if (title_overlap or has_target) else 1.8
                idf_bonus = 0.0
                for w in shared:
                    multiplier = (object_mult if w in q_object_words
                                    else 1.0)
                    idf_bonus += 0.05 * multiplier * self.encoder._idf_weight(w)
                # IDF-weighted stem bonus.  For each stem in
                # extra_stems, look up the query word it came from and
                # weight by that word's IDF (so "evolu" matching from
                # "evolution" carries the IDF of "evolution", not a
                # flat constant).
                stem_bonus = 0.0
                for s in extra_stems:
                    for qw in q_words:
                        if qw.startswith(s) or s.startswith(qw[:5]):
                            multiplier = (object_mult
                                            if qw in q_object_words else 1.0)
                            stem_bonus += (0.04 * multiplier
                                              * self.encoder._idf_weight(qw))
                            break

                title_bonus = sum(
                    0.25 * self.encoder._idf_weight(w)
                    for w in title_overlap
                )

                # Intro bonus: candidates that mention the article's
                # topic name AND have some other link to the query
                # (lexical overlap or title overlap) are likely the
                # canonical intro sentence.  The gate is critical:
                # without it, random biographical intros ("Niels Bohr
                # was a Danish physicist...") would win unrelated
                # queries just because their name has many tokens.
                intro_bonus = 0.0
                has_link = (bool(title_overlap)
                              or idf_bonus > 0
                              or stem_bonus > 0)
                if topic_toks and has_link:
                    matched_toks = [t for t in topic_toks if t in m_words]
                    n_matched = len(matched_toks)
                    if n_matched == 1:
                        intro_bonus = 0.12
                    elif n_matched >= 2:
                        intro_bonus = 0.30 * n_matched

                target_bonus = 0.0
                if q_target and q_target in m["text"].lower():
                    # Strong target bonus only when the candidate is
                    # also in the right topic (title overlap) OR shares
                    # a proper noun from the query.
                    proper_shared = bool(q_proper & m_words)
                    if title_overlap or proper_shared:
                        target_bonus = 0.4
                    else:
                        target_bonus = 0.03    # weak, defensive bonus

                # ── Discriminator-word gate (2026-05-22) ─────────────
                # The previous "any overlap on high-IDF words" check
                # let candidates through when the only shared word was
                # common (e.g. "form").  The right check is whether the
                # candidate contains the *most discriminative* word in
                # the query — the one that actually identifies the
                # topic.  If the query is "explain how trends form" the
                # discriminator is "trends" (or its stem); a candidate
                # that doesn't contain "trends" or "trend" anywhere is
                # off-topic regardless of how it looks in HV space.
                if not hasattr(self, "_disc_cache"):
                    self._disc_cache = {}
                cache_key = id(q_words)
                if cache_key not in self._disc_cache:
                    ranked = sorted(
                        ((self.encoder._idf_weight(w), w) for w in q_words),
                        reverse=True,
                    )
                    # Top 2 most discriminative query words
                    self._disc_cache[cache_key] = [w for _, w in ranked[:2]]
                key_words = self._disc_cache[cache_key]
                m_text_lc = m["text"].lower()
                # Discriminator present if candidate contains the key
                # word OR its 5-char stem (catches trends/trending/trend).
                key_present = any(
                    (kw in m_text_lc)
                    or (len(kw) >= 5 and kw[:5] in m_text_lc)
                    for kw in key_words
                )
                # Title or target evidence also counts as a pass.
                has_strong_evidence = (
                    key_present
                    or bool(title_overlap)
                    or (q_target is not None
                        and q_target in m_text_lc)
                )
                sim_contrib = m.get("similarity", 0.0)
                if not has_strong_evidence:
                    # Candidate doesn't contain the query's discriminator
                    # — likely off-topic.  Eliminate from contention.
                    sim_contrib = -10.0
                m["_topical_key"] = key_words[:2]

                m["_hybrid"] = (sim_contrib
                                  + idf_bonus + stem_bonus
                                  + title_bonus + intro_bonus
                                  + target_bonus)
            raw.sort(key=lambda m: -m["_hybrid"])

            # Intro-swap: when the query has NO title-overlap (we
            # found the right article by CONTENT, not by name), the
            # answer sentence ought to introduce the article's topic
            # by name.  If the top candidate doesn't mention its
            # article's topic, look for a sibling sentence that does
            # and swap it in.  Catches "Who composed the Brandenburg
            # Concertos?" -> the answer should be Bach's intro, not
            # the deeper sentence that just lists works.
            if raw:
                top = raw[0]
                top_topic = self._topic_tokens_for_text(top["text"])
                top_words = self._content_words(top["text"])
                top_has_title = bool(q_words & top_topic)
                top_names_self = bool(top_topic & top_words)
                if top_topic and not top_has_title and not top_names_self:
                    # First look in the candidate pool for an intro
                    # sibling from the same article.
                    swapped = False
                    for m in raw[1:20]:
                        m_topic = self._topic_tokens_for_text(m["text"])
                        if m_topic != top_topic:
                            continue
                        m_words = self._content_words(m["text"])
                        if m_topic & m_words:
                            raw.remove(m)
                            raw.insert(0, m)
                            swapped = True
                            break
                    # Fallback: look DIRECTLY in the lattice for the
                    # article's first sentence containing the topic
                    # name.  Catches the case where the intro sentence
                    # wasn't retrieved by HDC and didn't share rare
                    # words with the query.
                    if not swapped:
                        top_source = top.get("source", "")
                        for i, (text, src) in enumerate(zip(
                            self.lattice._texts, self.lattice._sources
                        )):
                            if src != top_source:
                                continue
                            if top_topic & self._content_words(text):
                                # Wrap in the same shape as raw entries.
                                intro = {
                                    "id":           self.lattice._ids[i],
                                    "text":         text,
                                    "source":       src,
                                    "similarity":   0.0,
                                    "distance_pct": 0.5,
                                    "_hybrid":      top["_hybrid"],
                                    "_intro_swap":  True,
                                }
                                raw.insert(0, intro)
                                break
        memories = raw[:5]

        # 2. Triple extraction + KG update
        new_triples = extract_triples(user_msg)
        for s, r, o in new_triples:
            self.knowledge.add_fact(s, r, o)

        # 3. KG lookup
        kg_hits = self._lookup_kg_facts(user_msg)

        # 4. Generate response — no LLM
        response = self.responder.narrate(user_msg, memories=memories,
                                              kg_hits=kg_hits)
        # Calibrate the fallback path's confidence.  Three signals:
        #   (a) Top memory's hybrid score (HDC+lexical+title+intro)
        #   (b) Whether the user's query mentioned ANY entity our
        #       corpus actually knows about (article topic OR stored
        #       claim subject)
        #   (c) Whether the query's proper-noun-shaped tokens are in
        #       the encoder vocabulary at all
        # If (b) and (c) both fail, the question is OUT-of-corpus and
        # any high "score" we got is a false-positive lexical match
        # against an unrelated sentence — hedge hard.
        # ── Confidence calibration for the fallback path ───────────
        # We want HIGH confidence only when the question is grounded
        # in something we know.  Two anchors:
        #   (a) The query mentions a proper noun matching a known
        #       entity or article topic.
        #   (b) All the query's distinctive content words are in our
        #       encoder vocabulary (i.e. we've seen them in context).
        # If neither holds, the question references something
        # genuinely outside our corpus — even a high "hybrid score"
        # is then a coincidence on common words, not real knowledge.
        q_proper = self._query_entities(user_msg)
        known_entities = set()
        for s, _, _ in self.structured.claim_triple:
            known_entities.add(s.lower())
            for tok in s.lower().split():
                known_entities.add(tok)
        title_idx = self._build_title_index()
        proper_anchored = any(
            (e.lower() in known_entities) or
            (e.lower() in title_idx) or
            any(t.lower() in title_idx for t in e.split())
            for e in q_proper
        )

        # Detect unknown content words: tokens that LOOK like a real
        # content word (>=4 chars, alphabetic) but aren't in our
        # encoder vocabulary AT ALL.  These are out-of-corpus signals.
        q_content = self._content_words(user_msg)
        # Check against the vocab snapshot taken at turn-start, NOT
        # the current vocab (which has already absorbed the new
        # query's tokens).  Otherwise "blockchain" looks "known"
        # because the encoder just learned it from this very query.
        vocab = getattr(self, "_vocab_before_turn",
                          frozenset(self.encoder.index_vectors.keys()))
        unknown_content = [
            w for w in q_content
            if len(w) >= 4
            and w.isalpha()
            and w not in vocab
            and w.lower() not in self._ENTITY_BLACKLIST
        ]

        if memories:
            base = memories[0].get("_hybrid",
                                       memories[0].get("similarity", 0.0))
        else:
            base = 0.0
        if kg_hits:
            base = max(base, 0.65)

        if (q_proper and not proper_anchored) or unknown_content:
            # Out-of-corpus signal: cap confidence below LOW so the
            # wrapper hedges or abstains.
            fallback_sim = min(base, self._CONF_LOW - 0.01)
        else:
            fallback_sim = base
        response = self._wrap_with_confidence(response, fallback_sim)

        # 5. Store the turn
        self.lattice.add(user_msg, source="user_msg", turn=len(self.turns))
        self.lattice.add(response,  source="agent_response",
                            turn=len(self.turns))
        combined = f"USER: {user_msg}\nAGENT: {response}"
        self.lattice.add(combined, source="conversation_turn",
                            turn=len(self.turns))

        # Push entities for next-turn pronoun resolution.  This is the
        # fallback path, so we use whatever entities we can find in
        # the query, the top retrieved memory, and any KG hit.
        for ent in self._query_entities(user_msg):
            self._push_entity(ent)
        if memories:
            for ent in self._query_entities(memories[0]["text"]):
                self._push_entity(ent)
        for s, _, o in kg_hits[:2]:
            self._push_entity(s)
            self._push_entity(o)

        self.turns.append({
            "ts":                 datetime.now(timezone.utc).isoformat(),
            "user":               user_msg,
            "agent":              response,
            "retrieved_memories": [m["text"] for m in memories[:3]],
            "extracted_triples":  new_triples,
            "kg_hits":            kg_hits[:3],
        })
        return response

    # ─── Self-introspection: ingest own source code ────────

    def ingest_self(self, subdirs: list[str] | None = None) -> dict:
        """Feed the agent its own source code so it can answer
        questions about the project's architecture.

        Each .py file under `subdirs` (default ['lattice']) gets parsed
        into module / class / function / method records.  Each becomes
        a lattice memory + structured claim like (X, defined_in, file)
        and (class, has_method, name) and (module, imports, dep).
        """
        from lattice.ingest_code import ingest_into_agent
        stats = ingest_into_agent(self, _TELP_ROOT,
                                       subdirs=subdirs or ["lattice"])
        # Rebuild downstream caches.
        self.structured._dirty = True
        # Note: we don't rebuild the predictor or KG since code text
        # has very different structure from prose.  The lattice +
        # structured claims are the primary access paths.
        return stats

    # ─── Image multimodal (optional, pure-HDC) ─────────────

    def _ensure_image_bank(self):
        """Lazy-load the image bank.  Pure-HDC pixel encoder, no
        neural model.  Storage: SQLite next to the lattice DB."""
        if not _IMAGES_AVAILABLE:
            raise RuntimeError("image modules unavailable (PIL not installed)")
        if self._image_bank is None:
            self._image_encoder = ImageEncoder()
            img_db = str(self.lattice.db_path).replace(".db", "_images.db")
            self._image_bank = ImageBank(img_db, encoder=self._image_encoder)

    def remember_image(self, image_path: str | Path,
                          description: str = "") -> int:
        """Store an image's hypervector + the text description in the
        bank.  The description ALSO becomes a lattice memory bound to
        the image, so 'show me the chart from yesterday' can later
        retrieve by description.
        """
        self._ensure_image_bank()
        img_id = self._image_bank.add(str(image_path), label=description)
        if description:
            self.lattice.add(f"Image: {description}",
                                source=f"image:{image_path}",
                                image_id=img_id)
        return img_id

    def recall_image_by_description(self, description: str,
                                        k: int = 3) -> list[dict]:
        """Find stored images whose descriptions best match the prompt.
        Searches the LATTICE (text descriptions of images) and returns
        the underlying image paths.
        """
        self._ensure_image_bank()
        results = self.lattice.query(description, k=k * 2)
        out = []
        for m in results:
            src = m.get("source", "")
            if src.startswith("image:"):
                out.append({
                    "image_path": src.split(":", 1)[1],
                    "description": m["text"],
                    "similarity": m.get("similarity", 0.0),
                })
                if len(out) >= k:
                    break
        return out

    def find_visually_similar(self, image_path: str | Path,
                                 k: int = 3) -> list[dict]:
        """Pure-HDC visual similarity search: encode the query image
        and find the nearest stored ones by Hamming distance."""
        self._ensure_image_bank()
        return self._image_bank.search_by_image(str(image_path), k=k)

    # ─── Inspection ────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "turns_this_session": len(self.turns),
            "lattice_total":      self.lattice.count(),
            "kg_facts":           len(self.knowledge.facts),
            "kg_entities":        len(self.knowledge.entities),
            "encoder_vocab":      self.encoder.stats()["vocab_size"],
        }


# ─── CLI ───────────────────────────────────────────────────────────


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_LATTICE_DB))
    args = ap.parse_args()

    print("=" * 64)
    print("Standalone HDC Agent  -  NO LLM, NO neural models")
    print("Encoder: corpus-trained Random Indexing")
    print("Decoder: template-based, driven by auto-discovered relations")
    print("Commands: /stats, /quit, /learn <topic>")
    print("=" * 64)

    agent = StandaloneAgent(lattice_path=Path(args.db))
    print(f"[standalone] ready. {agent.stats()}\n")

    while True:
        try:
            user_msg = input("YOU > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[standalone] goodbye. memory persisted.")
            break
        if not user_msg:
            continue
        if user_msg == "/quit":
            print(f"[standalone] final stats: {agent.stats()}")
            break
        if user_msg == "/stats":
            print(f"[standalone] {agent.stats()}")
            continue
        if user_msg.startswith("/learn "):
            topic = user_msg[len("/learn "):].strip().replace(" ", "_")
            res = agent.learn_wikipedia(topic)
            print(f"[standalone] {res}")
            continue
        try:
            response = agent.respond(user_msg)
            print(f"AGENT > {response}\n")
        except Exception as e:
            print(f"[standalone] error: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
