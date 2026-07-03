"""
autopilot/fluency.py - make Telp speak fluently without an LLM.

The StandaloneAgent already answers questions: it retrieves, extracts
claims, runs comparators, does multi-hop reasoning, generates from its
n-gram model.  What it doesn't do is *sound* fluent — its raw output
has debug-info leaks ("similarity=0.34"), canned abstention phrases
("(no retrieval hits)"), and choppy consecutive same-subject
sentences.

This module wraps the agent and gives it polish:

  1. POST-PROCESSING.  Strip the debug residue, normalize punctuation
     and casing, dedupe repeats.

  2. HEDGE CALIBRATION.  Pull the agent's internal confidence (from
     the last turn's metadata) and translate it into natural English
     hedges:
       high   → no hedge ("X.")
       med    → "I think X."
       low    → "I'm not certain, but X."
       none   → "I don't have enough to answer that."

  3. SENTENCE STITCHING.  If the agent returned multiple retrieved
     facts as separate sentences with the same subject, combine them.

  4. RETRIEVE + REPHRASE.  When the agent would return a verbatim
     retrieved sentence, optionally re-generate it through the
     n-gram model conditioned on the question — gives natural
     paraphrase grounded in the same fact.

  5. CONVERSATION CONTEXT.  The agent already tracks turns; the
     fluency layer adds light-touch continuity: "About that..."
     openers when the user follows up on the prior topic.

  6. NATURAL ABSTENTION.  Replace canned "(no retrieval hits)" with
     varied phrasings.

This is a wrapper, not a replacement.  We never change what Telp
*knows*; we change how he *says* it.

Usage:
    from mind.fluency import FluentTelp
    telp = FluentTelp()
    print(telp.respond("who was Einstein?"))
    print(telp.respond("when was he born?"))  # pronoun resolved
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path
from typing import Optional

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.standalone_agent import StandaloneAgent
from mind.qa_types import (
    classify_question, answer_matches_type, type_aware_score,
    answer_mentions_subject,
    QTYPE_GENERAL, QTYPE_DEFINITION,
)
from mind.code_synthesis import try_code_synthesis
from mind.code_writer import try_write_code, format_for_chat as _code_chat_fmt
from mind.app_composer import try_compose_app
from mind.game_composer import try_compose_game
from mind.code_corpus import CodeCorpus, try_retrieve_code
from mind.code_analogy import try_code_analogy
from mind.chart_narrative import build_chart_narrative
from mind.forward_chain import try_forward_chain
from mind.persona import PersonaStore
from mind.persona_seed import seed as seed_persona
from mind.voice import VoiceShaper
from mind.emotion import classify_emotion
from mind.user_facts import UserFactsStore


# ─── Phrase banks ──────────────────────────────────────────────────


# Varied abstention phrasings — Telp will rotate so he doesn't sound
# robotic when he doesn't know something.
_ABSTAIN_PHRASES = [
    "I don't have anything on that.",
    "I haven't been taught about that yet.",
    "Nothing in my memory matches that.",
    "I can't find anything I've learned about that.",
    "That's outside what I know — happy to learn it if you tell me.",
    "I don't have a confident answer there.",
]


# Light hedge prefixes by confidence band.
_HEDGE_HIGH = ["", "Yes — ", ""]                 # blank = no hedge
_HEDGE_MED  = ["I think ", "I believe ", "From what I know, "]
_HEDGE_LOW  = ["I'm not certain, but ", "I'm less sure here, but ",
               "Tentatively — "]


# Topic-continuity openers when the user is following up.
_FOLLOWUP_OPENERS = [
    "About that — ",
    "Following up — ",
    "On that topic, ",
    "",  # often nothing is more natural
]


# Patterns whose presence in the raw agent output signals debug leakage
# that we should strip / rephrase.
_DEBUG_LEAK_PATTERNS = [
    (re.compile(r"\(similarity=\d+\.\d+\)"),    ""),
    (re.compile(r"\(by analogy [^)]+\)"),       ""),
    (re.compile(r"\(sim=\d+\.\d+\)"),           ""),
    (re.compile(r"\(distance=\d+\)"),           ""),
    (re.compile(r"\(very low confidence:[^)]+\)"), ""),
    (re.compile(r"\(low confidence:[^)]+\)"),     ""),
    (re.compile(r"\(confidence:[^)]+\)"),         ""),
    (re.compile(r"\s{2,}"),                     " "),
    (re.compile(r"\s+([.,!?])"),                r"\1"),
]


_ABSTAIN_TRIGGERS = (
    "(no retrieval hits)",
    "i'm not sure",
    "i don't have information",
    "i haven't been taught",
    "no matches found",
    "i don't have a confident answer",
    "very low confidence",
    "closest match i found",
    "but possibly:",
)


# ─── Helpers ───────────────────────────────────────────────────────


def _confidence_band(turn: dict) -> str:
    """Map the agent's most recent turn metadata into one of
    {'high','med','low','none'}.  We look at retrieved similarity,
    claim-store hits, comparator/code_qa/arithmetic flags."""
    if not turn:
        return "none"
    # Hard signals: arithmetic / code / comparator / analogy = high
    for key in ("arithmetic", "code_qa", "compare"):
        if turn.get(key):
            return "high"
    # Some turns store the top retrieval similarity at top-level
    top_sim = turn.get("similarity")
    if top_sim is None:
        # Or as similarities list
        sims = turn.get("similarities") or []
        if sims:
            try:
                top_sim = max(float(x) for x in sims)
            except Exception:
                top_sim = None
    # Or from retrieved_memories (may be list[dict] or list[str])
    if top_sim is None:
        mems = turn.get("retrieved_memories") or []
        if mems and isinstance(mems[0], dict):
            try:
                top_sim = max(float(m.get("similarity", 0.0)) for m in mems)
            except Exception:
                top_sim = None
        elif mems:
            # List of strings — we know something was retrieved but no
            # similarity number; treat as medium.
            return "med"
    if isinstance(top_sim, (int, float)):
        if top_sim >= 0.50:
            return "high"
        if top_sim >= 0.32:
            return "med"
        if top_sim >= 0.15:
            return "low"
    # KG hits = at least medium
    if turn.get("kg_hits"):
        return "med"
    # Extracted triples = something structural was found
    if turn.get("extracted_triples"):
        return "med"
    return "none"


def _looks_like_abstention(text: str) -> bool:
    low = text.lower()
    return any(trig.lower() in low for trig in _ABSTAIN_TRIGGERS)


def _strip_debug(text: str) -> str:
    out = text
    for pat, repl in _DEBUG_LEAK_PATTERNS:
        out = pat.sub(repl, out)
    return out.strip()


def _strip_dialog_scaffolding(text: str) -> str:
    """Strip 'Human:' / 'Assistant:' / 'Bot:' / 'User:' / 'AI:'
    prefixes from retrieved memories.  Training corpora (e.g.
    instruction-tuning sets) include verbatim multi-turn transcripts —
    when we retrieve one, we don't want those role labels leaking
    into Telp's own response.

    Rules (in priority order):
      1. Q-then-A pattern: 'Human: X  Assistant: Y' → keep only 'Y'.
      2. Trailing question pattern: '...content...  Human: <question>'
         → drop the trailing 'Human:' portion entirely (the question
         is a new turn, not part of the answer).
      3. Otherwise: strip leading role labels + any inline ones.
    """
    if not text:
        return text
    # 1) Q-then-A: keep the answer
    m = re.search(
        r"(?:Human|User|Q|Question):\s*(.*?)\s*"
        r"(?:Assistant|AI|Bot|A|Answer|Telp):\s*(.+)",
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(2).strip()
    # 2) Trailing 'Human:'/'User:' with no answer following — drop it
    # along with everything after.  IMPORTANT: only drop when there's
    # actual content BEFORE the role label (otherwise we'd erase a
    # leading 'Question:' that has its own real content following).
    text = re.sub(
        r"(\S.*?)\s+(?:Human|User|Q|Question):\s*.*$",
        r"\1", text, flags=re.IGNORECASE | re.DOTALL,
    )
    # 3) Strip any leading role label.
    text = re.sub(
        r"^(?:Human|User|Assistant|AI|Bot|Telp|Agent|Q|A|Question|Answer):\s*",
        "", text, flags=re.IGNORECASE,
    )
    # 4) And drop any other inline mid-text role labels — replace with
    # a space.
    text = re.sub(
        r"\s*\b(?:Human|Assistant|Bot|AI|Telp|User):\s*",
        " ", text, flags=re.IGNORECASE,
    )
    return text.strip()


def _normalize(text: str) -> str:
    """Standard polish: trim, fix double-spaces, capitalize first
    letter, ensure ending punctuation, strip dialog scaffolding."""
    if not text:
        return ""
    s = _strip_dialog_scaffolding(text)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    # Capitalize first letter
    s = s[0].upper() + s[1:]
    # Fix standalone lowercase 'i' (common after hedge prefix)
    s = re.sub(r"\bi\b", "I", s)
    s = re.sub(r"\bi'(m|ve|d|ll)\b", lambda m: "I'" + m.group(1), s)
    # Make sure it ends with punctuation
    if s[-1] not in ".!?":
        s = s + "."
    return s


def _stitch_same_subject(sentences: list[str]) -> str:
    """Combine consecutive sentences that share a subject word.
    'Einstein was a physicist. Einstein won the Nobel Prize.' →
    'Einstein was a physicist and won the Nobel Prize.'
    Conservative: only stitches 2 sentences max."""
    if len(sentences) <= 1:
        return " ".join(sentences)
    out: list[str] = []
    i = 0
    while i < len(sentences):
        s1 = sentences[i].strip()
        if i + 1 < len(sentences):
            s2 = sentences[i + 1].strip()
            words1 = s1.split()
            words2 = s2.split()
            if (len(words1) >= 2 and len(words2) >= 2
                    and words1[0].lower() == words2[0].lower()
                    and words1[0][0].isupper()):
                # Strip subject from s2, join with " and "
                tail2 = " ".join(words2[1:]).rstrip(".!?")
                combined = s1.rstrip(".!?") + " and " + tail2 + "."
                out.append(combined)
                i += 2
                continue
        out.append(s1)
        i += 1
    return " ".join(out)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if p.strip()]


# ─── FluentTelp ────────────────────────────────────────────────────


class FluentTelp:
    """Conversational wrapper around StandaloneAgent.

    Owns the agent, intercepts respond() to apply polish, abstention
    variation, hedge calibration, and topic continuity.
    """

    def __init__(self, lattice_path: str | Path | None = None,
                   rephrase: bool = True):
        # Default: point at the lattice that has the ingested knowledge
        # (concept_bridge.db, populated by github/fred/reddit/wikipedia/
        # wisdom ingestion).
        #
        # If TELP_USE_LEARNED=1 (and the learned encoder + lattice exist),
        # switch to the GPU-trained learned lattice for far better
        # retrieval.
        import os as _os
        use_learned = _os.environ.get("TELP_USE_LEARNED", "0") == "1"

        if lattice_path is None:
            cb_path = _TELP_ROOT / "state" / "concept_bridge.db"
            learned_path = _TELP_ROOT / "state" / "concept_bridge_learned.db"
            enc_path = _TELP_ROOT / "state" / "diff_encoder.pt"
            if use_learned and learned_path.exists() and enc_path.exists():
                lattice_path = learned_path
                self._use_learned = True
            elif cb_path.exists():
                lattice_path = cb_path
                self._use_learned = False
            else:
                self._use_learned = False

        # If the learned encoder is in play, override the agent's
        # default CorpusRIEncoder with the learned one so retrieval
        # uses the same encoder that built the stored hypervectors.
        if getattr(self, "_use_learned", False):
            from lattice.diff_text_encoder import DifferentiableTextEncoder
            from lattice.learned_encoder_adapter import LearnedHDCEncoder
            diff = DifferentiableTextEncoder.load(
                _TELP_ROOT / "state" / "diff_encoder.pt")
            learned_enc = LearnedHDCEncoder(diff)
            # skip_ri_retrain=True saves ~30s — we're swapping the
            # encoder out so the RI retrain is wasted work.  Also skip
            # the n-gram retrain unless explicitly told we need it.
            self.agent = StandaloneAgent(
                lattice_path=Path(lattice_path),
                skip_ri_retrain=True,
                skip_ngram_retrain=False,   # still need text generation
            )
            # Hot-swap the encoder so query/encode use the learned vectors
            self.agent.encoder = learned_enc
            self.agent.lattice.encoder = learned_enc
            print(f"[fluency] using GPU-learned encoder "
                    f"(vocab={len(diff.vocab)}, dim={diff.dim})",
                    flush=True)
        elif lattice_path is not None:
            self.agent = StandaloneAgent(lattice_path=Path(lattice_path))
        else:
            self.agent = StandaloneAgent()
        self.rephrase = rephrase
        self.last_topic_words: set[str] = set()
        # Seed RNG for stable-ish phrase rotation
        self._rng = random.Random()
        # Sources from the most recent multi-doc answer — used by /why
        self._last_sources: list = []

        # ── Persona, voice, user-facts (personality stack) ─────────
        # Persona store — Telp's self-knowledge in the TELP_SELF subspace
        try:
            self.persona = PersonaStore(encoder=self.agent.encoder)
            n_seeded = seed_persona(self.persona)
            if n_seeded:
                print(f"[fluency] seeded {n_seeded} persona facts "
                        f"(total: {self.persona.count()})", flush=True)
        except Exception as e:
            print(f"[fluency] persona init failed: {e}", flush=True)
            self.persona = None

        # User-facts store — persistent memory of who we're talking to
        try:
            self.user_facts = UserFactsStore(encoder=self.agent.encoder)
            print(f"[fluency] user-facts store: "
                    f"{self.user_facts.count()} stored", flush=True)
        except Exception as e:
            print(f"[fluency] user_facts init failed: {e}", flush=True)
            self.user_facts = None

        # Voice shaper — Telp's characteristic phrasing
        self.voice = VoiceShaper()

        # Code corpus — curated Python snippets for retrieval-based
        # code answering when no template matches
        try:
            self.code_corpus = CodeCorpus(encoder=self.agent.encoder)
            n_seeded = self.code_corpus.seed()
            if n_seeded:
                print(f"[fluency] seeded {n_seeded} code snippets "
                        f"(corpus: {self.code_corpus.count()})", flush=True)
        except Exception as e:
            print(f"[fluency] code_corpus init failed: {e}", flush=True)
            self.code_corpus = None

    # ── Utilities ──────────────────────────────────────────────────

    def _abstain(self) -> str:
        return self._rng.choice(_ABSTAIN_PHRASES)

    def _mark_abstained(self, user_msg: str, emotion) -> None:
        """An honest miss must leave an honest turn record: no retrieved
        memories for 'how do you know?' to mis-cite later."""
        if self.agent.turns and self.agent.turns[-1].get("user") == user_msg:
            self.agent.turns[-1]["retrieved_memories"] = []
            self.agent.turns[-1]["domain"] = "abstain"
        else:
            self.agent.turns.append({
                "user": user_msg, "agent": "", "retrieved_memories": [],
                "similarity": 0.0, "domain": "abstain", "emotion": emotion,
            })

    # Common hedge openers — used to detect when the body ALREADY
    # carries a hedge so we don't stack a second one ("I believe I think").
    _HEDGE_DETECTORS = (
        "i think", "i believe", "i'm not certain", "i'm less sure",
        "from what i know", "tentatively", "based on what",
        "according to what",
    )

    def _apply_hedge(self, body: str, band: str) -> str:
        if band == "high":
            prefix = self._rng.choice(_HEDGE_HIGH)
        elif band == "med":
            prefix = self._rng.choice(_HEDGE_MED)
        elif band == "low":
            prefix = self._rng.choice(_HEDGE_LOW)
        else:
            return self._abstain()
        if not prefix:
            return body
        # ── Double-hedge guard ─────────────────────────────────
        # If body already starts with a hedge ("I believe...", "From what
        # I know..."), don't stack a second one — return body unchanged.
        low_body = body.lstrip().lower()
        if any(low_body.startswith(h) for h in self._HEDGE_DETECTORS):
            return body
        # ── First-word case adjustment ─────────────────────────
        # Normalize smart quotes so "here's" matches "here's" (otherwise
        # Unicode apostrophe vs ASCII apostrophe means the lookup fails).
        body = body.replace("’", "'").replace("‘", "'")
        # If the first WORD is a common-word verb/article/etc we can
        # lowercase it for grammatical flow.  Proper nouns and "I"
        # must stay capitalized.
        if body and " " in body:
            first_word, rest = body.split(" ", 1)
            _LOWERABLE_FIRST = {
                "the", "a", "an", "this", "that", "these", "those",
                "it", "he", "she", "they",
                "is", "was", "were", "are", "has", "have", "had",
                "in", "on", "at", "by", "for", "with", "to", "from",
                "based", "according",
                # Question/answer starters that often appear and look
                # weird capitalized mid-sentence after a hedge:
                "how", "what", "where", "when", "why", "which", "who",
                "do", "does", "did", "would", "could", "should",
                "maybe", "perhaps", "yes", "no",
                # Common sentence-starter content words:
                "here", "here's", "there", "there's", "one", "some",
                "many", "most", "every", "any", "all", "if",
                "making", "making", "going", "trying", "looking",
                # Coordinating conjunctions that look weird capitalized
                # after a hedge ("Yes — And..." → "Yes — and..."):
                "and", "but", "so", "or", "yet", "while", "because",
                "since", "though", "although", "well",
            }
            if first_word.lower() in _LOWERABLE_FIRST:
                body = first_word.lower() + " " + rest
        return prefix + body

    # ── Persona / user-facts routing heuristics ──────────────────

    _PERSONAL_QUERY_TRIGGERS = (
        "you ", "your ", "yourself", "are you", "do you", "did you",
        "can you", "could you", "would you", "have you",
        "tell me about you", "tell me about yourself",
        "what do you think", "what's your take", "whats your take",
        "what's your opinion", "whats your opinion",
        "your favorite", "your favorites", "your view",
        "you prefer", "do you prefer", "you like",
        "what are you", "who are you", "telp",
    )

    _ABOUT_USER_TRIGGERS = (
        "remember me", "remember that i", "remember when i",
        "do you remember", "what do you know about me",
        "what's my name", "whats my name", "what is my name",
        "what do i", "what did i", "what am i",
        "where do i", "where am i", "where did i",
        "what do you remember",
    )

    def _looks_personal(self, msg: str) -> bool:
        """True if the question is about Telp himself (his identity,
        opinions, preferences, style)."""
        if not msg:
            return False
        low = msg.lower()
        return any(t in low for t in self._PERSONAL_QUERY_TRIGGERS)

    def _persona_category(self, msg: str) -> Optional[str]:
        """Route a personal question to the most likely persona
        category (identity / opinion / style / value / capability).
        Returns None to disable category filtering.
        """
        low = msg.lower()
        # Identity: "who are you", "what are you", "tell me about yourself"
        if any(p in low for p in ("who are you", "what are you",
                                          "your name", "tell me about you",
                                          "tell me about yourself",
                                          "introduce yourself")):
            return "identity"
        # Opinion: "what do you think", "your take", "your opinion"
        if any(p in low for p in ("what do you think", "your take",
                                          "your opinion", "your view",
                                          "your thoughts")):
            return "opinion"
        # Capability: "can you", "what can you do"
        if any(p in low for p in ("can you", "what can you do",
                                          "are you able", "do you have")):
            return "capability"
        # Value: "your values", "what matters", "what do you believe"
        if any(p in low for p in ("your values", "what matters",
                                          "what do you believe",
                                          "what do you stand for")):
            return "value"
        # Style: "your style", "how do you", "you prefer"
        if any(p in low for p in ("your style", "how do you ",
                                          "do you prefer")):
            return "style"
        return None

    @staticmethod
    def _uf_voice(fact: str) -> str:
        """Stored third-person user fact -> second-person speech."""
        for a, b in (("User's name is", "Your name is"),
                     ("User asked me to remember:", "You asked me to remember:"),
                     ("User is a", "You're a"), ("User has", "You have"),
                     ("User likes", "You like"),
                     ("User lives in / is from", "You live in / are from"),
                     ("User works at", "You work at")):
            if fact.startswith(a):
                return b + fact[len(a):]
        return fact

    def _looks_about_user(self, msg: str) -> bool:
        """True if the question is asking about THE USER (recall of
        stored user-facts)."""
        if not msg:
            return False
        low = msg.lower()
        return any(t in low for t in self._ABOUT_USER_TRIGGERS)

    # ── Multi-document synthesis ─────────────────────────────────
    # Stopwords excluded from consensus + content-word scoring.
    _MD_STOPWORDS = frozenset("""
        a an the and or but of in on at by to from for with as is are
        was were be been being have has had do does did this that these
        those it he she they we you i me my your his her their our its
        what which who whom whose how why where when can could would
        should will shall may might must about into onto out over under
        again very just also too only own same other any all some most
        more less many such no not nor than then so if while because
        though although however moreover therefore here there said say
        says one two three first second new old good bad like get got
        make made take took use used go went see saw know knew
    """.split())

    @staticmethod
    def _md_content_words(text: str) -> set[str]:
        """Lowercased content-word tokens (len>=3, not stop)."""
        toks = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return {t for t in toks if t not in FluentTelp._MD_STOPWORDS}

    @staticmethod
    def _md_text_quality(text: str) -> float:
        """Score 0..1 — how "fact-statement-shaped" is this text?
        Low = junk (list markers, prompts, questions, code, fragments).
        High = clean declarative sentence.
        """
        if not text:
            return 0.0
        t = text.strip()
        tlow = t.lower()
        score = 1.0
        # Penalties for junk markers
        if t.startswith(("-", "*", "•", ">", "#")):
            score -= 0.3
        # Case-insensitive junk-starter check (lattice text isn't always
        # capitalized).
        for bad in ("here are", "given this", "for example",
                      "examples:", "some initial prompts",
                      "initial prompts", "example:",
                      "bullet points", "instructions:",
                      "as an ai", "as a language model",
                      "i'm sorry", "i am sorry",
                      "the following is", "here's a list"):
            if tlow.startswith(bad):
                score -= 0.5
                break
        # Many bullets / dashes — list-shaped
        if t.count(" - ") >= 2 or t.count("•") >= 2:
            score -= 0.3
        # Multiple question marks anywhere → prompt list / FAQ dump
        if t.count("?") >= 2:
            score -= 0.5
        # The memory IS a question (ends with '?') — bad answer source
        if t.rstrip().endswith("?"):
            score -= 0.6
        # Multiple "What is" / "How does" repeated → prompt list
        question_count = len(re.findall(r"\bWhat is\b|\bHow does\b|\bWhy is\b",
                                              t, re.IGNORECASE))
        if question_count >= 2:
            score -= 0.5
        # Code-shaped (lots of punctuation)
        punc = sum(1 for c in t if c in "{}[]()<>=/\\")
        if punc / max(1, len(t)) > 0.05:
            score -= 0.2
        # Starts with lowercase → fragment
        if t and t[0].islower():
            score -= 0.2
        # Math notation
        if re.search(r"\b1/c\d|\bE\s*=\s*m|\^\d", t):
            score -= 0.2
        # Penalize memories that are extremely close to a paraphrase of
        # the query itself (a stored question without an answer).
        # We can't access user_msg here statically; this is handled by
        # the caller via query_overlap > 0 check, plus the ends-with-?
        # rule above.
        return max(0.0, min(1.0, score))

    @staticmethod
    def _md_source_quality(source: str) -> float:
        """Score 0..1 — how reliable is this source?  Wikipedia, wisdom
        corpus, and FRED are high; raw web/github/reddit are mid; the
        rest are low."""
        if not source:
            return 0.4
        s = source.lower()
        if s.startswith(("wiki:", "wikipedia:")):
            return 1.0
        if s.startswith("wisdom:"):
            return 0.9
        if s.startswith("fred:"):
            return 0.85
        if s.startswith(("github:", "youtube:")):
            return 0.6
        if s.startswith(("reddit:", "web:")):
            return 0.5
        return 0.4

    def _multidoc_synthesize(self, user_msg: str,
                                  top_k: int = 20) -> dict | None:
        """Pull top-K lattice memories and synthesize a multi-source
        answer.  Quality-filters HARD because the lattice has a lot of
        web-scraped junk.

        Returns dict {text, sources, confidence_band, n_sources} or None
        if not enough quality material.
        """
        try:
            hits = self.agent.lattice.query(user_msg, k=top_k)
        except Exception:
            return None
        if not hits:
            return None

        query_words = self._md_content_words(user_msg)
        # Need at least ONE content word overlap with the query to
        # consider a memory — otherwise it's noise.
        if not query_words:
            return None
        # Classify the question's expected answer type — used to filter
        # retrievals that don't carry the right KIND of content.
        qtype = classify_question(user_msg)

        # ── Filter: drop low-quality + dialog-scaffolded ──────
        cleaned = []
        for h in hits:
            sim = float(h.get("similarity") or 0.0)
            if sim < 0.50:    # tighter floor than before
                continue
            txt = _strip_dialog_scaffolding(h.get("text") or "")
            if len(txt.split()) < 5:
                continue
            if _looks_like_abstention(txt):
                continue
            words = self._md_content_words(txt)
            # Require AT LEAST one query content word in the memory.
            if not (words & query_words):
                continue
            # Skip memories that are essentially the query itself —
            # almost all their content words come from the query (which
            # means the lattice stored the question, not an answer).
            if words and len(words - query_words) <= 1:
                continue
            text_q = self._md_text_quality(txt)
            if text_q < 0.5:   # heavy junk -> drop
                continue
            src_q = self._md_source_quality(h.get("source", "") or "")
            # Type-aware filter: when the question has a specific
            # expected answer type, the memory must carry that type
            # of content.  Skips "Einstein was a physicist" for "WHEN
            # was Einstein born?" because there's no date in it.
            type_match = type_aware_score(txt, qtype)
            if qtype != QTYPE_GENERAL and type_match < 0.5:
                continue
            # S-P-O check: the answer must mention what was asked.
            # Yes/no questions need TIGHTER subject overlap — both the
            # entity AND the property being asked about should appear.
            from mind.qa_types import QTYPE_YESNO
            min_ov = 2 if qtype == QTYPE_YESNO else 1
            if not answer_mentions_subject(txt, user_msg, min_overlap=min_ov):
                continue
            cleaned.append({
                "text":   txt,
                "sim":    sim,
                "source": h.get("source", "") or "",
                "words":  words,
                "text_q": text_q,
                "src_q":  src_q,
                "type_q": type_match,
            })
        if not cleaned:
            return None

        # ── Dedupe by word-overlap (>70% same content words) ──
        kept: list[dict] = []
        for h in cleaned:
            dup = False
            for k in kept:
                if not h["words"] or not k["words"]:
                    continue
                jacc = (len(h["words"] & k["words"]) /
                            max(1, len(h["words"] | k["words"])))
                if jacc > 0.70:
                    dup = True
                    break
            if not dup:
                kept.append(h)
            if len(kept) >= 5:
                break
        if not kept:
            return None

        # ── Consensus scoring: which content words appear in >= 2
        # of the kept memories?  Those are "agreed facts."
        word_counts: dict[str, int] = {}
        query_words = self._md_content_words(user_msg)
        for h in kept:
            for w in h["words"]:
                word_counts[w] = word_counts.get(w, 0) + 1
        consensus = {w for w, c in word_counts.items() if c >= 2}
        # Rescore each kept memory: similarity + consensus coverage +
        # query overlap + text quality + source quality + type match.
        for h in kept:
            shared = len(h["words"] & consensus)
            query_overlap = len(h["words"] & query_words)
            h["score"] = (h["sim"]
                              + 0.10 * shared
                              + 0.20 * query_overlap
                              + 0.15 * h["text_q"]
                              + 0.15 * h["src_q"]
                              + 0.20 * h.get("type_q", 0.5))
        kept.sort(key=lambda h: -h["score"])

        # ── Compose: take top 1-3 as ordered sentences.
        # The anchor is the best-scoring memory; we add 1-2 more
        # only when they bring NEW content (overlap < 50% with anchor).
        anchor = kept[0]
        composed_sentences = [_normalize(anchor["text"])]
        composed_words = anchor["words"].copy()
        sources_used = [{"source": anchor["source"], "sim": anchor["sim"]}]
        for h in kept[1:]:
            if not h["words"]:
                continue
            new = h["words"] - composed_words
            if len(new) < 3:   # not enough new info
                continue
            composed_sentences.append(_normalize(h["text"]))
            composed_words |= h["words"]
            sources_used.append({"source": h["source"], "sim": h["sim"]})
            if len(composed_sentences) >= 3:
                break

        # ── Confidence band via Bayesian-style aggregation ─────────
        # Treat each source's similarity as P(claim_correct | source).
        # Independent sources combine via noisy-OR:
        #     P_combined = 1 - prod(1 - P_i)
        # Source-quality weights each P_i: a wikipedia hit at sim=0.6
        # counts more than a reddit hit at sim=0.6.
        def _src_q_factor(src: str) -> float:
            return self._md_source_quality(src or "")

        p_disagree = 1.0
        for h in kept[:len(sources_used)]:
            p_i = h["sim"] * _src_q_factor(h["source"])
            p_i = max(0.0, min(0.95, p_i))
            p_disagree *= (1.0 - p_i)
        p_combined = 1.0 - p_disagree

        if p_combined >= 0.85:
            band = "high"
        elif p_combined >= 0.60:
            band = "med"
        elif p_combined >= 0.40:
            band = "low"
        else:
            band = "low"

        return {
            "text":        " ".join(composed_sentences),
            "sources":     sources_used,
            "band":        band,
            "n_sources":   len(sources_used),
            "consensus_n": len(consensus),
        }

    def _detect_followup(self, user_msg: str) -> bool:
        """Heuristic: if the user uses a pronoun or refers to 'that',
        'it', and we have prior topic words, treat as follow-up."""
        if not self.last_topic_words:
            return False
        toks = set(re.findall(r"\b\w+\b", user_msg.lower()))
        return bool(toks & {"it", "that", "this", "those", "they",
                              "he", "she", "him", "her", "them"})

    def _update_topic_words(self, response: str) -> None:
        """Capture the proper-noun-shaped tokens from the response as
        the rolling topic context."""
        proper = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", response)
        if proper:
            self.last_topic_words = {w.lower() for w in proper}

    def _maybe_rephrase(self, body: str, prompt: str) -> str:
        """If the agent returned a retrieved sentence verbatim and we
        have an n-gram generator, optionally rephrase it.  We only
        rephrase when it makes the body shorter or smoother — never
        replace the body if rephrasing fails or produces gibberish."""
        if not self.rephrase:
            return body
        if not hasattr(self.agent, "seq") or self.agent.seq is None:
            return body
        # Only attempt for moderate-length bodies; very short ones
        # don't benefit and generator may diverge.
        if not (20 <= len(body) <= 200):
            return body
        try:
            seed = " ".join(prompt.split()[:3])
            if hasattr(self.agent, "ensure_generator"):
                self.agent.ensure_generator()
            gen = self.agent.seq.generate(seed, n_words=18)
            # Simple sanity: must contain at least 2 anchor words from body
            body_toks = set(t.lower() for t in re.findall(r"\b\w{4,}\b", body))
            gen_toks  = set(t.lower() for t in re.findall(r"\b\w{4,}\b", gen))
            if len(body_toks & gen_toks) >= 2 and 15 < len(gen) < 220:
                return _normalize(gen)
        except Exception:
            return body
        return body

    # ── Domain-aware retrieval ───────────────────────────────────
    #
    # The lattice has BOTH encyclopedic content (wikipedia:*) and
    # trading-specific content (legacy:*, wisdom:*, github:*).  The RI
    # encoder is flat — it doesn't know that "should I trade during
    # lunch hour?" should prefer trading sources over physics.
    #
    # Before delegating to the underlying agent, we check if the query
    # looks domain-specific.  If so, we do our own source-filtered
    # lattice query and return the top hit when it's strong enough.
    # Otherwise we fall through to the agent's normal pipeline.

    _TRADING_QUERY_ANCHORS = {
        # Action verbs
        "trade", "trading", "trader", "trades", "traded",
        "buy", "buys", "buying", "bought",
        "sell", "sells", "selling", "sold",
        "long", "longs", "short", "shorts", "shorting",
        "hold", "holding", "enter", "entering", "exit", "exiting",
        # Market objects
        "stock", "stocks", "market", "markets", "ticker",
        "price", "prices", "spread", "spreads",
        "session", "sessions", "open", "close", "high", "low",
        "lunch", "morning", "afternoon", "premarket", "aftermarket",
        # Patterns / indicators
        "pattern", "patterns", "indicator", "indicators",
        "rsi", "macd", "ema", "sma", "vwap", "atr", "obv", "adx",
        "candle", "candles", "candlestick", "candlesticks",
        "hammer", "doji", "engulfing", "wedge", "flag", "triangle",
        "head", "shoulders", "channel",
        # Movement / structure
        "breakout", "breakouts", "breakdown", "reversal", "reversals",
        "pullback", "pullbacks", "retrace", "retracement",
        "rally", "selloff", "trend", "trends", "uptrend", "downtrend",
        "sideways", "consolidation", "consolidating",
        "support", "resistance", "level", "levels",
        "range", "squeeze", "chop", "choppy",
        # Risk / strategy
        "momentum", "volatility", "vix", "stop", "stops", "loss", "losses",
        "win", "wins", "winning", "lose", "losing", "lost",
        "drawdown", "risk", "reward", "kelly", "leverage", "margin",
        "size", "sizing", "position",
        # Instruments / macro
        "futures", "mes", "es", "nq", "mnq", "spx", "spy", "qqq",
        "options", "option", "etf", "etfs",
        "fed", "fomc", "rate", "rates", "interest", "inflation",
        "cpi", "pce", "earnings", "guidance",
        "bullish", "bearish",
        # Generic verbs that pair with trading queries
        "scalp", "swing", "intraday", "daily", "weekly",
    }

    _IDENTITY_QUERY_PHRASES = (
        "your name", "who are you", "what are you", "what is telp",
        "who is telp", "tell me about yourself", "yourself",
        "your brain", "your memory", "do you use",
        "are you a", "are you an", "are you ai", "are you the",
        "what's your", "whats your",
        "are you alive", "are you real", "are you human",
    )

    _TRADING_SOURCE_PREFIXES = (
        "legacy:", "wisdom:", "github:", "fred:", "reddit:", "youtube:",
    )

    _IDENTITY_SOURCE_PREFIXES = ("user_taught",)

    def _classify_query(self, msg: str) -> str:
        low = msg.lower()
        if any(p in low for p in self._IDENTITY_QUERY_PHRASES):
            return "identity"
        toks = set(re.findall(r"\b[a-z]+\b", low))
        if toks & self._TRADING_QUERY_ANCHORS:
            return "trading"
        return "general"

    def _domain_filtered_answer(self, msg: str,
                                   domain: str) -> tuple[str, float] | None:
        """Query the lattice but only consider memories whose source
        starts with one of the prefixes for `domain`.

        We pull a wide k=80 because the RI encoder produces low
        absolute similarities when one domain (wikipedia) dominates
        the corpus — the right answer might be ranked 15-50 globally
        but is still the best in-domain hit.

        For identity queries we ALSO do a literal-word fallback:
        any user_taught fact that shares a content word with the
        query is considered.

        Returns (text, similarity) for the top in-domain hit if
        similarity >= 0.20, otherwise None.
        """
        if domain == "trading":
            prefixes = self._TRADING_SOURCE_PREFIXES
            sim_floor = 0.20
        elif domain == "identity":
            prefixes = self._IDENTITY_SOURCE_PREFIXES
            sim_floor = 0.15
        else:
            return None

        # Pull a wide window so legitimate in-domain answers ranked
        # behind a swarm of unrelated Wikipedia hits still surface.
        hits = self.agent.lattice.query(msg, k=80)
        best = None
        for h in hits:
            src = h.get("source", "") or ""
            if not any(src.startswith(p) for p in prefixes):
                continue
            sim = h.get("similarity", 0.0)
            if sim >= sim_floor and (best is None or sim > best[1]):
                best = (h["text"], sim)
        if best is not None:
            return best

        # Literal-word fallback for trading: scan legacy directives and
        # notes for any whose key content words overlap the query.
        # This catches questions like "what should I do in the final
        # 15 minutes?" where the RI similarity is below the floor but
        # there's a directive that literally talks about "session
        # close" / "final minutes".
        if domain == "trading":
            stop_words = {"what", "should", "do", "in", "the", "of",
                              "during", "is", "a", "an", "are", "to",
                              "for", "this", "that", "any", "you",
                              "your", "my", "me", "we", "they", "it",
                              "i", "be", "was", "were", "have", "has",
                              "had", "and", "or", "but", "as", "by",
                              "on", "at", "from", "with", "without",
                              "remember", "tell", "about", "know",
                              "after", "before"}
            q_tokens = set(re.findall(r"\b[a-z]{3,}\b", msg.lower()))
            q_tokens -= stop_words
            if q_tokens:
                best_overlap = None
                for text, src in zip(self.agent.lattice._texts,
                                          self.agent.lattice._sources):
                    if not any(src.startswith(p) for p in prefixes):
                        continue
                    t_tokens = set(re.findall(r"\b[a-z]{3,}\b",
                                                  text.lower()))
                    overlap = len(q_tokens & t_tokens)
                    if overlap >= 2:
                        # Score = overlap / sqrt(len(text)) to prefer
                        # focused matches over long sentences that
                        # happen to mention the words.
                        score = overlap / max(1, len(text.split())) ** 0.5
                        if best_overlap is None or score > best_overlap[1]:
                            best_overlap = (text, score, overlap)
                if best_overlap is not None:
                    return (best_overlap[0],
                              min(0.55, 0.25 + 0.05 * best_overlap[2]))

        # Literal-word fallback for identity queries: scan user_taught
        # memories for one that shares a content word with the query.
        if domain == "identity":
            query_tokens = set(re.findall(r"\b[a-z]{4,}\b", msg.lower()))
            # Drop common question-words
            query_tokens -= {"what", "your", "name", "tell", "about",
                                "yourself", "this", "that", "have",
                                "does", "with", "from"}
            if not query_tokens:
                # No content words; just return the most basic identity
                # statement so "who are you?" gets answered.
                for text, src in zip(self.agent.lattice._texts,
                                          self.agent.lattice._sources):
                    if (src.startswith("user_taught") and
                            "telp" in text.lower() and
                            len(text) < 80):
                        return (text, 0.30)
                return None
            for text, src in zip(self.agent.lattice._texts,
                                      self.agent.lattice._sources):
                if not src.startswith("user_taught"):
                    continue
                text_tokens = set(re.findall(r"\b[a-z]{4,}\b",
                                                 text.lower()))
                if query_tokens & text_tokens:
                    return (text, 0.30)
        return None

    # ── Main entrypoint ───────────────────────────────────────────

    # ── Small-talk / social handler ────────────────────────────────
    # Common conversational openers should NOT hit the lattice — lattice
    # retrieval on "hey" or "how are you" returns garbage. Intercept
    # these and return brief, varied responses.
    _SMALLTALK_RESPONSES = {
        "greeting": [
            "Hey.", "Hi.", "Hello — what's up?", "Hey, what's on your mind?",
            "Hi there.", "Hey, how can I help?",
        ],
        "farewell": [
            "Later.", "See you.", "Take care.", "Bye.",
            "Talk to you soon.",
        ],
        "thanks": [
            "Anytime.", "Sure thing.", "No problem.",
            "You're welcome.", "Glad to help.",
        ],
        "howareyou": [
            "Doing fine — what are we working on?",
            "Running clean today. What's up?",
            "Online and ready. You?",
            "Good. What do you want to dig into?",
        ],
        "whoareyou": [
            "I'm Telp — your local AI. I run on HDC (hyperdimensional "
            "computing), not an LLM. I see, remember, and answer "
            "from my own lattice of memories.",
            "Telp — the HDC system you've been building. No cloud, "
            "no LLM in the loop. One mind: perception, memory, "
            "reasoning, voice — all local.",
        ],
        "areyou_ai": [
            "Yes — though built different from the LLMs you're used to. "
            "I'm built on hyperdimensional computing instead of "
            "transformers.",
            "Yeah, I'm an AI. HDC-based, runs entirely on this machine.",
        ],
        "what_can_you_do": [
            "I can see images and remember what I've seen, answer "
            "questions from my own memory, explain my reasoning, and "
            "learn from what you tell me — across sessions. Try asking "
            "about anything you've fed into the lattice, or show me "
            "a picture.",
        ],
        "feelings": [
            "That's real — thanks for telling me. Want to talk it "
            "through, or would a distraction help more?",
            "I hear you. I'm here either way — we can dig into what's "
            "going on, or just work on something else for a while.",
            "Noted, and I'm not going to pretend I know exactly how "
            "that feels — but I'm listening. What's weighing on you?",
        ],
    }

    def narrate_chart(self, reading: dict) -> str:
        """Turn an HDC chart reading dict into polished prose using
        the templated chart_narrative builder.

        The templated output lists every active conviction signal in
        structured English ("learned encoder leans down (57%)",
        "self-monitor trust is low — sizing should be dampened", etc.).
        Each sentence is deterministic and grounded in a real
        signal value — no n-gram extrapolation, no hallucinated
        content.

        For natural-language Q&A about charts ("what's MES doing
        right now?"), use `respond()` which goes through the full
        StandaloneAgent pipeline (structured_qa + multi_hop + retrieval).
        """
        narrative = build_chart_narrative(reading)
        if not narrative:
            return ""
        polished = narrative
        for pattern, repl in _DEBUG_LEAK_PATTERNS:
            polished = pattern.sub(repl, polished)
        if polished and polished[0].islower():
            polished = polished[0].upper() + polished[1:]
        return polished.strip()

    def _live_data_response(self, msg: str) -> str | None:
        """Handle questions where the answer is REAL-TIME data, not
        something in the lattice — current time, today's date, etc.

        Returns a formatted answer string, or None if the message
        isn't a live-data question.
        """
        s = msg.lower().strip().rstrip("!?.,")
        if not s:
            return None
        from datetime import datetime
        now = datetime.now().astimezone()

        # Time — %I yields 01-12 with a leading zero on Windows; strip it.
        if s in {"what time is it", "what's the time", "whats the time",
                  "what is the time", "current time", "the time",
                  "tell me the time", "time"}:
            t = now.strftime("%I:%M %p").lstrip("0")
            tz = now.strftime("%Z") or ""
            return f"It's {t} {tz}.".strip().rstrip(",")

        # Date / day
        if s in {"what day is it", "what's the date", "whats the date",
                  "what is the date", "today's date", "todays date",
                  "what's today", "what is today", "what date is it",
                  "what day of the week"}:
            return now.strftime("It's %A, %B %d, %Y.")

        if s in {"what year is it", "what's the year", "whats the year",
                  "what is the year", "current year"}:
            return now.strftime("It's %Y.")

        if s in {"what month is it", "what's the month", "whats the month",
                  "what is the month", "current month"}:
            return now.strftime("It's %B.")

        if s in {"what day of the week is it", "what's the day",
                  "what is the day"}:
            return now.strftime("It's %A.")

        return None

    def _smalltalk_category(self, msg: str) -> str | None:
        """Classify a message as smalltalk, returning the category key
        or None if it should hit the normal retrieval path."""
        s = msg.lower().strip().rstrip("!?.,")
        # Very short messages are usually smalltalk
        if not s:
            return None
        # Exact / near-exact matches
        if s in {"hi", "hello", "hey", "yo", "sup", "hey telp",
                  "hi telp", "hello telp", "good morning", "good afternoon",
                  "good evening", "morning", "evening"}:
            return "greeting"
        if s in {"bye", "goodbye", "later", "see you", "see ya",
                  "take care", "good night", "gn", "ttyl", "cya"}:
            return "farewell"
        if s in {"thanks", "thank you", "thx", "ty", "appreciate it",
                  "thanks!", "much appreciated"}:
            return "thanks"
        if s in {"how are you", "how are you doing", "how's it going",
                  "hows it going", "how you doing", "you good",
                  "you alright", "you ok"}:
            return "howareyou"
        if s in {"who are you", "what are you", "whats your name",
                  "what's your name", "what is your name", "tell me about yourself"}:
            return "whoareyou"
        if s in {"are you an ai", "are you a robot", "are you human",
                  "are you a bot", "are you real"}:
            return "areyou_ai"
        if s in {"what can you do", "what do you do", "help",
                  "what are your capabilities"}:
            return "what_can_you_do"

        # Semantic fallback (2026-07-02): paraphrase-robust category match.
        # "hey, how are you doing today?" should land on howareyou even
        # though it isn't in the exact-match sets above.
        emb_fn = getattr(self.agent.encoder, "_embed", None)
        if emb_fn is None or len(s.split()) > 9:
            return None                      # long messages = real questions
        if not hasattr(self, "_st_protos"):
            protos = {
                "greeting": ["hi there", "hello!", "hey, what's up?",
                             "good morning to you"],
                "farewell": ["goodbye", "see you later",
                             "good night, i'm heading off"],
                "thanks": ["thank you so much", "thanks a lot, really",
                           "cool, thanks!", "awesome, thank you",
                           "great, that helps"],
                "howareyou": ["how are you doing?", "how's it going today?",
                              "are you doing okay?", "how do you feel today?"],
                "whoareyou": ["who are you?", "tell me about yourself",
                              "what exactly are you?"],
                "areyou_ai": ["are you an AI?", "are you a real person or a bot?"],
                "what_can_you_do": ["what can you do?", "what are you capable of?",
                                    "how can you help me?"],
                "feelings": ["i'm feeling sad today", "i am so frustrated",
                             "i had a rough day", "i'm really happy right now",
                             "i'm tired and stressed"],
                "surprise_me": ["surprise me", "surprise me with something",
                                "tell me something interesting",
                                "teach me something new", "tell me a fun fact",
                                "share something cool", "amaze me"],
                "opinion": ["what do you think about this?",
                            "what do you think about AI?",
                            "what's your opinion on that?",
                            "what's your take on it?",
                            "do you like this?",
                            "do you ever get things wrong?",
                            "do you ever make mistakes?"],
            }
            labels, texts = [], []
            for k, exs in protos.items():
                for e in exs:
                    labels.append(k)
                    texts.append(e)
            import numpy as _np
            self._st_protos = (labels, _np.asarray(emb_fn(texts)))
        import numpy as _np
        labels, mat = self._st_protos
        q = _np.asarray(emb_fn([s]))[0]
        sims = mat @ q
        i = int(sims.argmax())
        if float(sims[i]) >= 0.60:
            return labels[i]
        return None

    # ── Age/date route: chained facts + deterministic arithmetic ──────
    _AGE_RE = re.compile(
        r"how old (?:was|is) (.+?)\s+when\s+(?:he|she|they|it)?\s*"
        r"(died|passed away)", re.I)
    _BORNDIED_RE = re.compile(
        r"when (?:was|did) (.+?)\s+(born|die|died)\b", re.I)

    def _age_route(self, user_msg: str, emotion) -> str | None:
        m = self._AGE_RE.search(user_msg)
        m2 = None if m else self._BORNDIED_RE.search(user_msg)
        if not m and not m2:
            return None
        subject = (m or m2).group(1).strip(" ?.,'\"")
        try:
            from lattice.growth import (find_life_dates, age_between,
                                        fmt_date, learn_topic)
        except Exception:
            return None
        life = find_life_dates(self.agent, subject)
        if not (life["birth"] and life["death"]):
            # the needed dates aren't in memory - go get them, then re-check
            try:
                learn_topic(self.agent, subject, force=True)
            except Exception:
                pass
            life = find_life_dates(self.agent, subject)
        body = None
        if m and life["birth"] and life["death"]:
            age = age_between(life["birth"], life["death"])
            body = (f"{subject} was {age} when he died - born "
                    f"{fmt_date(life['birth'])}, died {fmt_date(life['death'])}.")
        elif m2:
            want = m2.group(2).lower()
            if want == "born" and life["birth"]:
                body = f"{subject} was born {fmt_date(life['birth'])}."
            elif want in ("die", "died") and life["death"]:
                body = f"{subject} died {fmt_date(life['death'])}."
        if body is None:
            return None
        shaped = self.voice.shape_response(body, band="high", emotion=emotion)
        self.agent.turns.append({
            "user": user_msg, "agent": shaped,
            "retrieved_memories": life["evidence"],
            "similarity": 1.0, "domain": "age_computation", "emotion": emotion,
        })
        return shaped

    # ── Word problems: parse the story, do the arithmetic ──────────────
    _NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
                  "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                  "ten": 10, "eleven": 11, "twelve": 12, "a dozen": 12,
                  "thirteen": 13, "fourteen": 14, "fifteen": 15, "twenty": 20}
    _WM_START = r"(?:have|has|had|start(?:ed)? with|there (?:are|were))"
    _WM_MINUS = (r"(?:eat|eats|ate|eaten|lose|loses|lost|give[sn]? away|gave"
                 r"(?: away)?|sell|sells|sold|drop(?:ped)?|use[sd]?|remove[sd]?"
                 r"|break|broke|donate[sd]?|flew (?:away|off)|fl(?:y|ies) away"
                 r"|ran (?:away|off)|escap(?:ed|es)|wander(?:ed)? off)")
    # "gives ME 7 more" is a GAIN even though "gives away" is a loss
    _WM_PLUS = (r"(?:buy|buys|bought|get|gets|got|find|finds|found|receive[sd]?"
                r"|pick(?:ed)? up|gain(?:ed)?|add(?:ed)?|make|made|bake[sd]?"
                r"|g[ai]ve[sn]? (?:me|us)|hand(?:s|ed)? (?:me|us))")

    def _wm_num(self, tok: str):
        tok = tok.lower().strip()
        if tok.isdigit():
            return int(tok)
        return self._NUM_WORDS.get(tok)

    def _wm_answer(self, user_msg, emotion, body: str, steps: list) -> str:
        shaped = self.voice.shape_response(body, band="high", emotion=emotion)
        self.agent.turns.append({
            "user": user_msg, "agent": shaped,
            "retrieved_memories": steps, "similarity": 1.0,
            "domain": "word_math", "emotion": emotion,
        })
        return shaped

    def _word_math_route(self, user_msg: str, emotion) -> str | None:
        q = user_msg.lower()
        if not re.search(r"\bhow many\b.*\b(left|now|remain|total|in all"
                         r"|altogether|does .* have|do .* have"
                         r"|does each .* get|each)\b", q):
            return None
        num = r"(\d+|" + "|".join(w for w in self._NUM_WORDS if w != "a") + r")"

        # comparative: "Tom has 3 more apples than Sara, who has 5"
        m = re.search(rf"(\w+)\s+(?:has|have|had)\s+{num}\s+(more|fewer|less)"
                      rf"\s+\w+\s+than\s+(\w+),?\s+who\s+(?:has|have|had)"
                      rf"\s+{num}", q)
        if m:
            a, d, rel, b, base = (m.group(1), self._wm_num(m.group(2)),
                                  m.group(3), m.group(4),
                                  self._wm_num(m.group(5)))
            if d is not None and base is not None:
                total = base + d if rel == "more" else base - d
                op = "+" if rel == "more" else "-"
                body = (f"{a.capitalize()} has {total} - {b.capitalize()} "
                        f"has {base}, and {a.capitalize()} has {d} {rel}: "
                        f"{base} {op} {d} = {total}.")
                return self._wm_answer(user_msg, emotion, body,
                                       [f"{b}: {base}", f"{rel} {d}",
                                        f"= {total}"])

        # division: "20 candies shared equally among 4 kids"
        m = re.search(rf"{num}\s+(\w+)\s+(?:are\s+|is\s+|get\s+)?"
                      rf"(?:shared|split|divided)\s+(?:equally\s+)?"
                      rf"(?:among|between)\s+{num}", q)
        if m:
            a, thing, b = (self._wm_num(m.group(1)), m.group(2),
                           self._wm_num(m.group(3)))
            if a is not None and b:
                each, rem = divmod(a, b)
                body = (f"{each} each - {a} {thing} shared among {b}: "
                        f"{a} / {b} = {each}"
                        + (f" with {rem} left over." if rem else "."))
                return self._wm_answer(user_msg, emotion, body,
                                       [f"{a} / {b} = {each} r{rem}"])

        # multiplication: "3 boxes with 6 eggs each"
        m = re.search(rf"{num}\s+(\w+)\s+(?:with|of|containing|holding|having"
                      rf"|(?:that\s+)?(?:has|have))\s+{num}\s+(\w+)(?:\s+each)?",
                      q)
        if m:
            a, group, b, thing = (self._wm_num(m.group(1)), m.group(2),
                                  self._wm_num(m.group(3)), m.group(4))
            if a is not None and b is not None:
                body = (f"{a * b} {thing} - {a} {group} times {b} {thing} "
                        f"each: {a} * {b} = {a * b}.")
                return self._wm_answer(user_msg, emotion, body,
                                       [f"{a} * {b} = {a * b}"])

        # gain/loss chain: "had 12 cookies, gave away 4 and ate 2"
        start = re.search(self._WM_START + r"\s+" + num, q)
        if not start:
            return None
        total = self._wm_num(start.group(1))
        if total is None:
            return None
        steps = [f"start with {total}"]
        tail = q[start.end():]
        # the count can come before OR after the verb:
        # "gave away 4" / "4 flew away"
        op_re = (r"(?:" + num + r"\s+)?\b(" + self._WM_PLUS + r"|"
                 + self._WM_MINUS + r")\b(?:\s+" + num + r")?")
        for m in re.finditer(op_re, tail):
            verb = m.group(2)
            n = (self._wm_num(m.group(3)) if m.group(3)
                 else self._wm_num(m.group(1)) if m.group(1) else 1)
            if n is None:
                n = 1
            if re.fullmatch(self._WM_PLUS, verb):
                total += n
                steps.append(f"plus {n} ({verb})")
            else:
                # counts of things live in the naturals - you cannot
                # remove more than you have
                if n > total:
                    body = (f"That can't happen - you only have {total}, "
                            f"so you can't {verb} {n}. At most {total} "
                            f"can go, leaving 0.")
                    return self._wm_answer(user_msg, emotion, body,
                                           steps + [f"minus {n} impossible"])
                total -= n
                steps.append(f"minus {n} ({verb})")
        if len(steps) < 2:
            return None
        body = f"{total} left - {', '.join(steps)} -> {total}."
        return self._wm_answer(user_msg, emotion, body, steps)

    # ── Procedures: how-to answered with STEPS, fetched on miss ───────
    # Only first-person task questions ("how do I...", "how to...") — a
    # knowledge question like "how do plants make food" stays on the
    # semantic/wiki path.
    _HOWTO_TASK_RE = re.compile(
        r"^how (?:to|do i|do we|can i|should i|would i)\b", re.I)

    def _procedure_route(self, user_msg: str, emotion) -> str | None:
        q = user_msg.lower()
        # keep the promise: "...ask if you want the rest"
        if (re.search(r"\b(?:rest|remaining|more)\b", q)
                and re.search(r"\bsteps?\b|\brest\b", q)
                and getattr(self, "_last_proc", None)
                and self.agent.turns
                and self.agent.turns[-1].get("domain") == "procedure"):
            proc = self._last_proc
            title, steps = proc["title"].lower(), proc["steps"]
            rest = steps[5:]
            if not rest:
                return self.voice.shape_response(
                    f"That was all of them - {len(steps)} steps in all.",
                    band="high", emotion=emotion)
            parts = [f"{n}. {re.split(r'(?<=[.!?]) ', t.strip())[0]}"
                     for n, t in rest]
            body = f"The rest of how to {title}: " + " ".join(parts)
            shaped = self.voice.shape_response(body, band="high",
                                               emotion=emotion)
            self.agent.turns.append({
                "user": user_msg, "agent": shaped,
                "retrieved_memories":
                    [f"How to {title}, step {n}: {t}" for n, t in rest],
                "similarity": 1.0, "domain": "procedure", "emotion": emotion,
            })
            return shaped
        if not self._HOWTO_TASK_RE.match(user_msg.strip()):
            return None
        if re.search(r"\b(you|your|telp)\b", q):
            return None                     # about Telp, not a task
        try:
            from lattice.growth import procedure_steps, learn_howto
        except Exception:
            return None
        focus = [w for w in re.findall(r"[a-z]{3,}", q)
                 if w not in self._SEM_STOP]
        proc = procedure_steps(self.agent, query=user_msg)
        if proc is not None and focus:
            # same law as the facet gate: a TOPICAL procedure is not THE
            # procedure ("boil an egg" must not accept "scrambled eggs")
            try:
                # titles are near-canonical: true match >=0.9, wrong
                # facet <=0.44 (boil-vs-scrambled measured 0.438)
                cov = self.agent.encoder.focus_alignment(
                    focus, ["how to " + proc["title"].lower()],
                    reduce="min")[0]
                if cov < 0.60:
                    proc = None
            except Exception:
                pass
        learned = ""
        if proc is None:
            # a SPECIFIC task earns a fetch; vague ones fall through
            if len(focus) < 2:
                return None
            if not hasattr(self, "_howto_attempted"):
                self._howto_attempted: set = set()
            key = " ".join(sorted(focus))
            if key in self._howto_attempted:
                return None
            self._howto_attempted.add(key)
            try:
                r = learn_howto(self.agent, user_msg)
            except Exception:
                return None
            if not r.get("steps"):
                return None
            proc = {"title": r["title"],
                    "steps": list(enumerate(r["steps"], 1))}
            learned = ("I didn't know how, so I just looked it up and "
                       "learned the steps. ")
        title, steps = proc["title"].lower(), proc["steps"]
        shown = steps[:5]
        parts = []
        for n, t in shown:
            first = re.split(r"(?<=[.!?])\s+", t.strip())[0]
            parts.append(f"{n}. {first}")
        more = (f" (...and {len(steps) - 5} more steps - ask if you want "
                f"the rest.)" if len(steps) > 5 else "")
        body = f"{learned}To {title}: " + " ".join(parts) + more
        self._last_proc = proc
        shaped = self.voice.shape_response(body, band="high", emotion=emotion)
        self.agent.turns.append({
            "user": user_msg, "agent": shaped,
            "retrieved_memories":
                [f"How to {title}, step {n}: {t}" for n, t in shown],
            "similarity": 1.0, "domain": "procedure", "emotion": emotion,
        })
        return shaped

    def _howto_miss(self, question: str, answer_text: str) -> bool:
        """Procedural questions demand the HOW be covered: egg biology must not
        satisfy 'how do I boil an egg' on any answer path."""
        if not answer_text or not re.match(r"^how (do|to|can|would|should)",
                                           question.lower()):
            return False
        focus = [w for w in re.findall(r"[a-z]{3,}", question.lower())
                 if w not in self._SEM_STOP]
        if not focus:
            return False
        try:
            cov = self.agent.encoder.focus_alignment(
                focus, [answer_text], reduce="min")[0]
            return cov < 0.55
        except Exception:
            return False

    # ── Analogies: solved by LOOKING UP the dictionary, not guessing ──
    _ANALOGY_RE = re.compile(
        r"\b([a-z]+)\s+is\s+to\s+([a-z]+)\s+as\s+([a-z]+)\s+is\s+to\s+"
        r"(?:what|\?|which)", re.I)

    def _analogy_route(self, user_msg: str, emotion) -> str | None:
        m = self._ANALOGY_RE.search(user_msg.lower())
        if not m:
            return None
        a, b, c = m.group(1), m.group(2), m.group(3)
        emb_fn = getattr(self.agent.encoder, "_embed", None)
        if emb_fn is None:
            return None
        try:
            import sqlite3 as _sq
            import numpy as _np
            from lattice.dictionary_lookup import Dictionary
            if not hasattr(self, "_dict"):
                self._dict = Dictionary()
            entry_b = self._dict.lookup(b)
            gloss_b = entry_b.primary_gloss() if entry_b else None
            if not gloss_b or a not in gloss_b.lower():
                return None
            # the relationship, transported: gloss(B) with A swapped for C
            target = re.sub(rf"\b{a}s?\b", c, gloss_b.lower())
            con = _sq.connect(str(_TELP_ROOT / "state" / "wiktionary" / "dict.db"))
            # candidates must share the RELATION's key words, not just mention C
            keywords = [w for w in re.findall(r"[a-z]{5,}", gloss_b.lower())
                        if w not in (a, b) and w != c][:3]
            cands = []
            for kw in keywords or [""]:
                cands += con.execute(
                    "SELECT DISTINCT word, gloss FROM entries WHERE gloss LIKE ? "
                    "AND gloss LIKE ? AND word != ? AND LENGTH(word) < 16 LIMIT 500",
                    (f"%{c}%", f"%{kw}%", c)).fetchall()
            if len(cands) < 20:
                cands += con.execute(
                    "SELECT DISTINCT word, gloss FROM entries WHERE gloss LIKE ? "
                    "AND word != ? AND LENGTH(word) < 16 LIMIT 800",
                    (f"%{c}%", c)).fetchall()
            con.close()
            cands = list(dict.fromkeys(cands))
            if not cands:
                return None
            t_emb = _np.asarray(emb_fn([target]))[0]
            g_embs = _np.asarray(emb_fn([g for _, g in cands]))
            sims = g_embs @ t_emb
            i = int(sims.argmax())
            if float(sims[i]) < 0.55:
                return None
            word, gloss = cands[i]
        except Exception:
            return None
        body = (f"{a} is to {b} as {c} is to {word} - a {b} is "
                f"\"{gloss_b}\", and a {word} is \"{gloss}\" "
                f"(worked out from my dictionary)")
        shaped = self.voice.shape_response(body, band="high", emotion=emotion)
        self.agent.turns.append({
            "user": user_msg, "agent": shaped,
            "retrieved_memories": [f"{b}: {gloss_b}", f"{word}: {gloss}"],
            "similarity": 1.0, "domain": "analogy:dictionary", "emotion": emotion,
        })
        return shaped

    # ── Date comparisons: find both subjects' dates, COMPARE deterministically ──
    _CMP_RE = re.compile(
        r"(?:was|is|did)\s+(.+?)\s+(born|die|died)\s+(before|after|earlier than|"
        r"later than)\s+(.+?)[\s?.!]*$", re.I)
    # "who was born first, X or Y?" / "who died first, X or Y?"
    _CMP_WHO_RE = re.compile(
        r"\bwho\s+(?:was\s+)?(born|die|died)\s+(first|earlier|last|later)\b"
        r"[,:\s]+(.+?)\s+or\s+(.+?)[\s?.!]*$", re.I)

    def _compare_dates_route(self, user_msg: str, emotion) -> str | None:
        who = None
        m = self._CMP_RE.search(user_msg.strip())
        if not m:
            w = self._CMP_WHO_RE.search(user_msg.strip())
            if not w:
                return None
            event, order = w.group(1).lower(), w.group(2).lower()
            a = w.group(3).strip(" ?.,'\"")
            b = w.group(4).strip(" ?.,'\"")
            rel = "before"
            who = "first" if order in ("first", "earlier") else "last"
        else:
            a, event, rel, b = (m.group(1).strip(" ?.,'\""),
                                m.group(2).lower(), m.group(3).lower(),
                                m.group(4).strip(" ?.,'\""))
        try:
            from lattice.growth import find_life_dates, fmt_date, learn_topic
        except Exception:
            return None
        key = "birth" if event == "born" else "death"
        vals = {}
        for name in (a, b):
            life = find_life_dates(self.agent, name)
            if not life[key]:
                try:
                    learn_topic(self.agent, name, force=True)
                except Exception:
                    pass
                life = find_life_dates(self.agent, name)
            vals[name] = life[key]
        if not (vals[a] and vals[b]):
            missing = [n for n in (a, b) if not vals[n]]
            body = f"I couldn't find {event} dates for: {', '.join(missing)}."
        else:
            first_is_a = vals[a] < vals[b]
            verb = "was born" if key == "birth" else "died"
            if who is not None:
                winner = ((a if first_is_a else b) if who == "first"
                          else (b if first_is_a else a))
                body = (f"{winner} - {a} {verb} {fmt_date(vals[a])}, "
                        f"{b} {verb} {fmt_date(vals[b])}.")
            else:
                before = rel.startswith(("before", "earlier"))
                yes = first_is_a if before else not first_is_a
                body = (f"{'Yes' if yes else 'No'} - {a} {verb} "
                        f"{fmt_date(vals[a])}, {b} {verb} {fmt_date(vals[b])}, "
                        f"so {a if first_is_a else b} came first.")
        shaped = self.voice.shape_response(body, band="high", emotion=emotion)
        self.agent.turns.append({
            "user": user_msg, "agent": shaped,
            "retrieved_memories": [body], "similarity": 1.0,
            "domain": "date_comparison", "emotion": emotion,
        })
        return shaped

    # ── Definitions: the 1.75M-word offline Wiktionary answers instantly ──
    _DEFINE_RE = re.compile(
        r"^(?:define\s+|what\s+does\s+)['\"]?([a-zA-Z][a-zA-Z\-']{2,30})['\"]?"
        r"(?:\s+mean)?[\s?.!]*$|^(?:what\s+is\s+the\s+)?meaning\s+of\s+"
        r"['\"]?([a-zA-Z][a-zA-Z\-']{2,30})['\"]?[\s?.!]*$", re.I)

    def _define_route(self, user_msg: str, emotion) -> str | None:
        m = self._DEFINE_RE.match(user_msg.strip())
        if not m:
            return None
        word = (m.group(1) or m.group(2)).lower()
        try:
            from lattice.dictionary_lookup import Dictionary
            if not hasattr(self, "_dict"):
                self._dict = Dictionary()
            entry = self._dict.lookup(word)
        except Exception:
            return None
        if entry is None:
            return None
        gloss = entry.primary_gloss()
        if not gloss:
            return None
        pos = getattr(entry, "pos", None) or ""
        body = (f"{word}{f' ({pos})' if pos else ''}: {gloss} "
                f"(from my offline dictionary)")
        shaped = self.voice.shape_response(body, band="high", emotion=emotion)
        self.agent.turns.append({
            "user": user_msg, "agent": shaped,
            "retrieved_memories": [body], "similarity": 1.0,
            "domain": "dictionary", "emotion": emotion,
        })
        return shaped

    # ── Stories: dispatch to the REAL imagination engine, not n-gram babble ──
    _STORY_RE = re.compile(
        r"\b(?:tell|make\s+up|write|imagine|invent|dream\s+up)\b.{0,24}?\bstory\b"
        r"(?:.{0,20}?\babout\s+(?:a|an|the)?\s*([a-zA-Z][a-zA-Z ]{2,24}))?", re.I)

    def _story_route(self, user_msg: str, emotion) -> str | None:
        q = user_msg.lower()
        # story MEMORY: retell a remembered story / list what he's told
        if re.search(r"\b(what|which) stories\b", q):
            import sqlite3 as _sq
            con = _sq.connect(str(self.agent.lattice.db_path))
            rows = con.execute("SELECT source FROM memories WHERE source "
                               "LIKE 'story:%' ORDER BY id").fetchall()
            con.close()
            seeds = list(dict.fromkeys(r[0].split(":", 1)[1] for r in rows))
            if seeds:
                return (f"I've made up {len(seeds)} so far - about "
                        f"{', '.join(seeds)}. Want to hear one again, or "
                        f"a new one?")
            return "I haven't made up any stories yet - ask me for one!"
        if re.search(r"\b(again|you told|you made up|that story)\b", q) \
                and "story" in q:
            import sqlite3 as _sq
            con = _sq.connect(str(self.agent.lattice.db_path))
            rows = con.execute("SELECT text, source FROM memories WHERE "
                               "source LIKE 'story:%' ORDER BY id").fetchall()
            con.close()
            if rows:
                pick = rows[-1]
                for t, s in reversed(rows):     # named seed wins
                    if s.split(":", 1)[1] in q:
                        pick = (t, s)
                        break
                return pick[0]
        m = self._STORY_RE.search(user_msg)
        if not m:
            return None
        try:
            from lattice.imagination import ImaginationEngine
            if not hasattr(self, "_imagination"):
                print("[fluency] waking the imagination engine ...", flush=True)
                self._imagination = ImaginationEngine(seed=None)
            eng = self._imagination
            seed = (m.group(1) or "").strip().split()[0].lower() if m.group(1) else None
            if not seed:
                import random
                seed = random.choice(eng.cast_pool())
            frames = eng.imagine_story(seed=seed)
            story = eng.render(frames)
        except Exception as e:
            print(f"[fluency] imagination unavailable ({e})", flush=True)
            return None
        arc = frames[0].get("_arc", "") if frames else ""
        body = f"Here's one I made up (about {seed}):\n\n{story}"
        # a told story becomes a memory like any other experience
        try:
            self.agent.lattice.add(body, source=f"story:{seed}")
        except Exception:
            pass
        self.agent.turns.append({
            "user": user_msg, "agent": body,
            "retrieved_memories": [f"imagined: seed={seed} arc={arc}"],
            "similarity": 1.0, "domain": "imagination", "emotion": emotion,
        })
        return body

    # ── Learn-on-miss: unanswered questions trigger lawful retrieval ──
    _LEARN_Q = re.compile(
        r"^(what|who|where|when|which|how|why|tell me about|is|are|was|were|does|do|did)\b",
        re.I)

    def _learn_on_miss(self, user_msg: str, emotion) -> str | None:
        q = user_msg.strip()
        # only knowledge-shaped questions; never personal/command messages
        if not self._LEARN_Q.match(q) or len(q.split()) < 3:
            return None
        if re.search(r"\b(my|me|you|your|telp)\b", q.lower()):
            return None
        if not hasattr(self, "_learn_attempted"):
            self._learn_attempted: set = set()
        try:
            from lattice.growth import learn_topic, extract_topics, search_wiki
        except Exception:
            return None
        learned_titles = []

        def _try(topic: str) -> bool:
            key = topic.lower()
            if key in self._learn_attempted:
                return False
            self._learn_attempted.add(key)
            try:
                r = learn_topic(self.agent, topic)
            except Exception:
                return False
            if r.get("added"):
                learned_titles.append((r["title"], r["added"]))
                return True
            return False

        howto = bool(re.match(r"^how (do|to|can|would|should)", q.lower()))
        if howto:
            for title in search_wiki(q):
                if _try(title):
                    break
        if not learned_titles:
            for topic in extract_topics(q):
                if _try(topic):
                    break                  # one article per question
        if not learned_titles:
            # direct guesses known/unfetchable -> search the whole question
            # (finds facet articles like 'Invention of the telephone')
            for title in search_wiki(q):
                if _try(title):
                    break
        if not learned_titles:
            return None
        # answer from what was just learned (stack updates live)
        sem = self._semantic_answer(user_msg, emotion)
        title, n = learned_titles[0]
        if sem is None:
            # rank the NEW article's own rows against the question directly
            emb_fn = getattr(self.agent.encoder, "_embed", None)
            rows = [t for t, s in zip(self.agent.lattice._texts,
                                      self.agent.lattice._sources)
                    if s == f"wikipedia:{title}"]
            if emb_fn is not None and rows:
                try:
                    import numpy as _np
                    embs = _np.asarray(emb_fn([user_msg] + rows))
                    sims = embs[1:] @ embs[0]
                    i = int(sims.argmax())
                    # SAME FACET LAW as the semantic path: the just-learned
                    # article's best row must COVER the question, or the
                    # honest "learned it but can't answer" message wins -
                    # never "NZ's tallest mountain -> NZ is an island".
                    cov = 1.0
                    focus = [w for w in re.findall(r"[a-z0-9']+",
                                                   user_msg.lower())
                             if w not in self._SEM_STOP and len(w) > 1]
                    if focus:
                        try:
                            cov = self.agent.encoder.focus_alignment(
                                focus, [rows[i]], reduce="min")[0]
                        except Exception:
                            pass
                    if float(sims[i]) >= 0.35 and cov >= 0.42:
                        from mind.composer import simplify
                        body = simplify(rows[i])
                        self.agent.turns.append({
                            "user": user_msg, "agent": body,
                            "retrieved_memories": [rows[i]],
                            "similarity": float(sims[i]),
                            "domain": "learned_on_miss", "emotion": emotion,
                        })
                        return (f"I didn't have that, so I just looked it up "
                                f"and learned it. {body}")
                except Exception:
                    pass
            return (f"I didn't know, so I just read about '{title}' and "
                    f"remembered {n} facts - but I still can't answer that "
                    f"one directly. Ask me about {title} though.")
        return f"I didn't have that, so I just looked it up and learned it. {sem}"

    def _persona_opinion(self, user_msg: str) -> str | None:
        """Best persona fact for an opinion question, by float cosine."""
        emb_fn = getattr(self.agent.encoder, "_embed", None)
        if emb_fn is None or self.persona is None:
            return None
        if not hasattr(self, "_persona_emb"):
            import sqlite3 as _sq
            import numpy as _np
            con = _sq.connect(str(_TELP_ROOT / "state" / "persona.db"))
            texts = [r[0] for r in con.execute(
                "SELECT text FROM persona_facts").fetchall()]
            con.close()
            if not texts:
                return None
            self._persona_emb = (texts, _np.asarray(emb_fn(texts)))
        import numpy as _np
        texts, mat = self._persona_emb
        q = _np.asarray(emb_fn([user_msg]))[0]
        sims = mat @ q
        i = int(sims.argmax())
        return texts[i] if float(sims[i]) >= 0.32 else None

    # ── Forget route: "forget ..." actually erases memories ────────────
    def _forget_route(self, user_msg: str, emotion) -> str | None:
        q = user_msg.lower().strip()
        if not q.startswith("forget"):
            return None
        try:
            from lattice.vision import forget, watched
            db = self.agent.lattice.db_path
            if any(w in q for w in ("video", "watch", "movie", "clip", "film")):
                vids = watched(db)
                if not vids:
                    body = "I haven't watched any videos, so there's nothing to forget."
                else:
                    stem = Path(vids[-1]["path"]).stem      # default: most recent
                    for v in vids:                          # or a named one
                        if Path(v["path"]).stem.lower() in q:
                            stem = Path(v["path"]).stem
                            break
                    gone = forget(db, video=stem)
                    body = (f"Done - I've forgotten the video '{stem}': "
                            f"{len(gone)} memories erased.")
            elif any(w in q for w in ("image", "photo", "picture", "sight")):
                # sights: CLIP cross-modal search, gated at the recall
                # threshold (the old 0.20 floor deleted wrong memories)
                target = q.removeprefix("forget").strip(" :,.-")
                for filler in ("that you saw", "what you saw", "seeing",
                               "about", "the "):
                    target = target.replace(filler, " ").strip()
                if not target:
                    return None
                gone = forget(db, query=target, min_sim=0.26)
                self.agent.lattice._reload_from_disk()
                body = (f"Done - forgotten: "
                        f"{gone[0].removeprefix('Image: ')}." if gone else
                        "I couldn't find a sight matching that to forget.")
            else:
                target = q.removeprefix("forget").strip(" :,.-")
                for filler in ("that you saw", "what you saw",
                               "about", "the fact that", "the "):
                    target = target.replace(filler, " ").strip()
                if not target:
                    return None
                body = self._forget_semantic(target)
        except Exception:
            return None
        shaped = self.voice.shape_response(body, band="high", emotion=emotion)
        self.agent.turns.append({
            "user": user_msg, "agent": shaped, "retrieved_memories": [],
            "similarity": 1.0, "domain": "forget", "emotion": emotion,
        })
        return shaped

    def _forget_semantic(self, target: str) -> str:
        """Forget from the ONE memory (and the user-facts store), but only
        on a STRONG match - deletion never runs on a weak guess."""
        import numpy as _np
        lat = self.agent.lattice
        emb_fn = getattr(self.agent.encoder, "_embed", None)
        if emb_fn is None:
            return "I can't safely match that memory right now."
        # measured: true match 0.48-0.95, junk <=0.20. A RARE word from
        # the command appearing verbatim in a row is strong evidence on
        # its own ("forget the frelling fact" names the only frelling
        # row he has) - it relaxes the gate to 0.32.
        dfmap = getattr(self.agent.encoder, "doc_freq", {}) or {}
        rare = [w for w in re.findall(r"[a-z0-9']{4,}", target.lower())
                if dfmap.get(w, 0) <= 15]

        def _gate_for(text: str) -> float:
            tl = text.lower()
            # a rare command word appearing verbatim (WHOLE word - "astro"
            # must not match "astronaut") = strong evidence; without one,
            # deletion demands a decisively strong match ("what is your
            # name?" at 0.45 must never die for "forget ... named astro")
            if any(re.search(rf"\b{re.escape(w)}s?\b", tl) for w in rare):
                return 0.32
            return 0.55

        victims, gone_texts = [], []
        best_txt, best_cos = None, 0.0
        cands = lat.query(target, k=24)
        # conversation echoes are not what "forget X" means - the fact
        # rows are the target (echo text also survives in the
        # conversation_turn rows regardless)
        cands = [c for c in cands
                 if not str(c.get("source", "")).startswith(
                     ("user_msg", "agent_response"))]
        if cands:
            embs = _np.asarray(emb_fn([target] + [c["text"] for c in cands]))
            sims = embs[1:] @ embs[0]
            for c, s in zip(cands, sims):
                if float(s) > best_cos:
                    best_cos, best_txt = float(s), c["text"]
                if float(s) >= _gate_for(c["text"]):
                    victims.append(c["id"])
        gone_texts += lat.delete_ids(victims[:5])
        # user-taught "remember that..." facts live in their own subspace
        if self.user_facts is not None and self.user_facts.count() > 0:
            uf = self.user_facts
            embs = _np.asarray(emb_fn([target] + uf._texts))
            sims = embs[1:] @ embs[0]
            uf_ids = [i for i, s, t in zip(uf._ids, sims, uf._texts)
                      if float(s) >= _gate_for(t)][:5]
            if uf_ids:
                for fid in uf_ids:
                    idx = uf._ids.index(fid)
                    gone_texts.append(uf._texts[idx])
                    uf._con.execute("DELETE FROM user_facts WHERE id=?",
                                    (fid,))
                uf._con.commit()
                uf._reload()
            best_here = float(sims.max()) if len(sims) else 0.0
            if best_here > best_cos:
                best_cos = best_here
        if gone_texts:
            shown = "; ".join(" ".join(t.split())[:90] for t in gone_texts[:3])
            more = f" (+{len(gone_texts) - 3} more)" if len(gone_texts) > 3 else ""
            return f"Done - forgotten: {shown}{more}."
        close = (f" The closest I have is \"{best_txt[:90]}\" and it doesn't "
                 f"match well enough to delete safely." if best_txt else "")
        return (f"I couldn't find a memory matching '{target}' confidently "
                f"enough to forget.{close} Say it more specifically and "
                f"I will.")

    # ── Provenance route: "how do you know?" cites the actual sources ──
    _PROV_TRIGGERS = ("how do you know", "where did you learn", "your source",
                      "why do you say", "prove it", "where did that come from",
                      "how did you learn")

    def _provenance_route(self, user_msg: str, emotion) -> str | None:
        q = user_msg.lower()
        if not any(t in q for t in self._PROV_TRIGGERS):
            return None

        def _say(body: str) -> str:
            shaped = self.voice.shape_response(body, band="high",
                                               emotion=emotion)
            self.agent.turns.append({
                "user": user_msg, "agent": shaped, "retrieved_memories": [],
                "similarity": 1.0, "domain": "provenance", "emotion": emotion,
            })
            return shaped

        # nothing said yet / last answer was an honest miss: there is no
        # claim to cite, and citing an OLDER turn would be misleading
        if not self.agent.turns:
            return _say("Know what? You haven't asked me anything yet - "
                        "ask me a question first.")
        if self.agent.turns[-1].get("domain") == "abstain":
            return _say("That last one was an honest miss - I didn't claim "
                        "anything, so there's nothing to cite.")
        mems = []
        for turn in reversed(self.agent.turns):
            mems = turn.get("retrieved_memories") or []
            if mems:
                break
        if not mems:
            return None
        import sqlite3
        con = sqlite3.connect(str(self.agent.lattice.db_path))
        cites = []
        for m in mems[:3]:
            text = m.get("text") if isinstance(m, dict) else m
            if not text:
                continue
            row = con.execute("SELECT source, created_at FROM memories "
                              "WHERE text=?", (text,)).fetchone()
            if row:
                src = row[0] or "our conversations"
                cites.append(f"{src} (saved {str(row[1])[:10]})")
        con.close()
        if cites:
            uniq = list(dict.fromkeys(cites))
            body = ("I can show you exactly: that answer came from " +
                    "; ".join(uniq) + ".")
        else:
            body = ("That one was composed from this conversation itself - "
                    "no single stored memory behind it.")
        shaped = self.voice.shape_response(body, band="high", emotion=emotion)
        self.agent.turns.append({
            "user": user_msg, "agent": shaped, "retrieved_memories": mems,
            "similarity": 1.0, "domain": "provenance", "emotion": emotion,
        })
        return shaped

    # ── Vision route: questions about seeing answer from sight memory ──
    _VISION_WORDS = ("see", "saw", "seen", "look", "looked", "watch",
                     "watched", "image", "picture", "photo")

    def _vision_route(self, user_msg: str, emotion) -> str | None:
        q = user_msg.lower()
        if not any(w in q.split() or w in q for w in self._VISION_WORDS):
            return None
        # only questions about TELP's own seeing - "my video looks blurry"
        # is the user's problem, not a sight-memory query
        import re as _re
        if not _re.search(r"\b(you|your|telp)\b", q):
            return None
        if _re.search(r"\bmy\b", q) and not _re.search(r"\bdid you\b|\bhave you\b", q):
            return None
        try:
            from lattice.vision import sights, recall_semantic
            rows = sights(self.agent.lattice.db_path)
        except Exception:
            return None
        if not rows:
            return None
        # enumerate ("what have you seen?") vs search ("did you see an animal?")
        video_words = {"video", "videos", "clip", "clips", "movie", "film",
                       "footage"}
        stop = set(self._VISION_WORDS) | video_words | {
            "what", "have", "has", "had", "you", "your", "did", "do", "does",
            "the", "a", "an", "any", "me", "show", "tell", "about", "of",
            "recently", "today", "ever", "i", "we", "at", "in", "on", "to",
            "happened", "happens", "happening", "going", "there",
            "just", "earlier", "was", "is", "were"}
        content = [w for w in q.replace("?", " ").replace(",", " ").split()
                   if w not in stop]
        # "what did you watch?" / "what happened in the video?" -> watch summary
        if not content and (video_words & set(q.split()) or "watch" in q):
            try:
                from lattice.vision import watched
                vids = watched(self.agent.lattice.db_path)
            except Exception:
                vids = []
            if vids:
                shaped = self.voice.shape_response(vids[-1]["summary"],
                                                   band="high", emotion=emotion)
                self.agent.turns.append({
                    "user": user_msg, "agent": shaped,
                    "retrieved_memories": [vids[-1]["summary"]],
                    "similarity": 1.0, "domain": "vision", "emotion": emotion,
                })
                return shaped
        if content and ({"text", "sign", "written", "words", "title",
                         "subtitle", "screen"} & set(content)):
            # on-screen text: answer from the OCR rows directly (targeted scan)
            try:
                from lattice.vision import screen_texts
                rows = screen_texts(self.agent.lattice.db_path)
            except Exception:
                rows = []
            if rows:
                uniq = list(dict.fromkeys(r["text"] for r in rows))
                body = "Here's the on-screen text I've read: " + " | ".join(
                    uniq[-4:])
                shaped = self.voice.shape_response(body, band="high",
                                                   emotion=emotion)
                self.agent.turns.append({
                    "user": user_msg, "agent": shaped,
                    "retrieved_memories": uniq[-4:],
                    "similarity": 1.0, "domain": "vision:ocr",
                    "emotion": emotion,
                })
                return shaped
            return None
        if content:
            hits = recall_semantic(self.agent.lattice.db_path, " ".join(content), k=1)
            # true matches score ~0.25+; sub-0.23 is the CLIP floor for unrelated
            if hits and hits[0]["similarity"] >= 0.23:
                h = hits[0]
                body = (f"Yes - I saw that: {h['caption'].removeprefix('Image: ')} "
                        f"({Path(h['path']).name})")
            else:
                body = "I don't think I've seen that."
        else:
            recent = "; ".join(r["caption"].removeprefix("Image: ")
                               for r in rows[-3:])
            body = f"I have seen {len(rows)} image(s). Most recently: {recent}."
        shaped = self.voice.shape_response(body, band="high", emotion=emotion)
        self.agent.turns.append({
            "user": user_msg, "agent": shaped,
            "retrieved_memories": [r["caption"] for r in rows[-3:]],
            "similarity": 1.0, "domain": "vision", "emotion": emotion,
        })
        return shaped

    # ── Semantic-first knowledge answer ─────────────────────────────
    _SEM_FLOOR = 0.30      # below: not about anything we know - fall through
    _SEM_HIGH = 0.45       # above: answer plainly
    _SEM_STOP = frozenset(
        "what which who whom whose where when why how do does did is are was "
        "were will would can could should the a an of in on at to for from "
        "about with and or not no any some tell me you your it its they them "
        "i we us this that these those there here "
        "say said says speaker talk talked mention mentioned "
        "many much number have has".split())

    def _semantic_answer(self, user_msg: str, emotion) -> str | None:
        """Answer a knowledge question from meaning-based retrieval, with
        SENTENCE-LEVEL ANSWER SELECTION: among the close retrieval cohort,
        pick the sentence whose words best cover the question's focus by
        word-level meaning ('eat' -> 'eating/omnivorous', not 'habitats').
        Returns None (fall through to the legacy cascade) when nothing in
        memory is close enough."""
        # COMPUTE-shaped questions need the deterministic QA machinery
        # (compare / aggregate / analogy / multi-hop chains) - a fuzzy
        # sentence must not preempt an exact computation. Let them fall
        # through to agent.respond()'s structured cascade.
        import re as _re
        ql = user_msg.lower()
        if _re.search(r"\bhow (old|long|far|much longer)\b.*\b(when|after|before|until)\b", ql):
            return None
        if _re.search(r"\b(born|die|died|invented|founded)\b.*\b(before|after|earlier|later)\b"
                      r"|\b(older|younger|earlier|later)\s+than\b", ql):
            return None                    # compare_qa territory
        if _re.search(r"^(list|name)\s+(all|the|some)\b", ql):
            return None
        # counting needs NUMERIC EVIDENCE, not prose about the topic -
        # but a blanket refusal starved "how many moons does Jupiter
        # have?" of the stored "115 known moons" row. Filter, don't bail.
        how_many = bool(_re.search(r"\bhow many\b", ql))
        if _re.search(r"\bis to\b.*\bas\b.*\bis to\b", ql):
            return None                    # analogy_qa territory
        try:
            # query WIDE then filter: conversation echoes of the same
            # question (user_msg rows at sim 1.0) must not crowd the
            # knowledge rows out of the candidate set
            hits = self.agent.lattice.query(user_msg, k=48)
        except Exception:
            return None
        hits = [h for h in hits
                if not str(h.get("source", "")).startswith(
                    ("user_msg", "agent_response", "conversation_turn",
                     "image:", "video:"))][:14]
        if not hits:
            return None
        top_sim = float(hits[0].get("similarity", 0.0))
        if top_sim < self._SEM_FLOOR:
            return None

        # cohort of near-equals, reranked by question-focus alignment
        cohort = [h for h in hits
                  if float(h.get("similarity", 0)) >= max(self._SEM_FLOOR * 0.8,
                                                          0.7 * top_sim)]
        import re as _re
        if how_many:
            num_re = _re.compile(
                r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|"
                r"ten|eleven|twelve|dozen|hundred|thousand|million|"
                r"billion)\b", _re.I)
            cohort = [h for h in cohort if num_re.search(h["text"])]
            if not cohort:
                return None            # no numeric evidence -> honest miss
        focus = [w for w in _re.findall(r"[a-z0-9']+", user_msg.lower())
                 if w not in self._SEM_STOP and len(w) > 1]
        # NAMED ENTITIES answer by MENTION, not vibe. A rare question word
        # (low doc-freq) is an entity facet: "largest lake in Mongolia" can
        # never be satisfied by a row that names only Azerbaijan.
        ent_focus: list = []
        dfmap = getattr(self.agent.encoder, "doc_freq", None)
        if dfmap:
            n_docs = max(1, getattr(self.agent.encoder, "n_docs", 1))
            df_max = max(3, n_docs // 500)
            ent_focus = [w for w in focus
                         if len(w) >= 4 and dfmap.get(w, 0) <= df_max]

        def _mentions_all(text: str) -> bool:
            tl = text.lower()
            return all((w[:5] if len(w) > 5 else w) in tl for w in ent_focus)

        lead_words = [w for w in focus if len(w) >= 4]

        def _entity_lead(text: str) -> bool:
            toks = text.split()
            if not (lead_words and toks):
                return False
            t0 = _re.sub(r"[^a-z0-9']", "", toks[0].lower())
            return any(t0.startswith(w[:5]) for w in lead_words)

        if ent_focus:
            named = [h for h in cohort if _mentions_all(h["text"])]
            if not named:
                return None        # memory never NAMES it -> honest miss
            cohort = named
        best = cohort[0]
        if focus and len(cohort) > 1 and hasattr(self.agent.encoder,
                                                 "focus_alignment"):
            try:
                aligns = self.agent.encoder.focus_alignment(
                    focus, [h["text"] for h in cohort])
                # subject-position entity leads beat passing mentions:
                # "Galileo ... was an astronomer" > "Life of Galileo is a
                # play" - and a COUNT question wants the definite count,
                # not an estimate ("expected to have about 100...")
                hedge_re = _re.compile(
                    r"\b(?:expected|estimated|about|around|approximately"
                    r"|predicted|thought to|may have|up to)\b")
                # (no lead bonus on counts: "Jupiter Icy Moons Explorer
                # is..." must not outrank "There are 115 known moons...")
                scored = [(float(h.get("similarity", 0)) + 0.35 * a
                           + (0.35 if not how_many
                              and _entity_lead(h["text"]) else 0.0)
                           - (0.2 if how_many
                              and hedge_re.search(h["text"].lower())
                              else 0.0), h)
                          for h, a in zip(cohort, aligns)]
                scored.sort(key=lambda x: -x[0])
                best = scored[0][1]
            except Exception:
                pass
        sim = float(best.get("similarity", 0.0))
        # FACET-AWARE MISS (2026-07-02): a topical answer is not an answering
        # answer. Every focus word must be covered in meaning-space - "who
        # invented the telephone?" must not be satisfied by what a telephone IS.
        if focus and sim < 0.7 and hasattr(self.agent.encoder, "focus_alignment"):
            try:
                cov = self.agent.encoder.focus_alignment(
                    focus, [best["text"]], reduce="min")[0]
                # procedural how-to questions demand tight coverage: egg BIOLOGY
                # must not satisfy "how do I boil an egg" - miss -> lookup
                bar = 0.55 if re.match(r"^how (do|to|can|would|should)",
                                       user_msg.lower()) else 0.42
                if cov < bar:
                    return None            # uncovered facet -> honest miss
            except Exception:
                pass
        # NEGATION IS A FACET the embedding cannot see: "what do raccoons
        # NOT eat?" retrieves the same rows as the positive question. A
        # positive fact must never be served as if it answered the negative
        # - state the positive honestly and say so.
        neg_re = _re.compile(r"\b(?:not|never|don'?t|doesn'?t|didn'?t|"
                             r"won'?t|can'?t|cannot|isn'?t|aren'?t)\b")
        if neg_re.search(ql) and not neg_re.search(best["text"].lower()):
            from mind.composer import simplify
            body = ("You're asking for a negative, and what I actually hold "
                    "is the positive: " + simplify(best["text"]) +
                    " Beyond that I can't honestly rule things in or out.")
            shaped = self.voice.shape_response(body, band="med",
                                               emotion=emotion)
            self.agent.turns.append({
                "user": user_msg, "agent": shaped,
                "retrieved_memories": [{"text": best["text"],
                                        "similarity": sim}],
                "similarity": sim, "domain": "semantic:negation",
                "emotion": emotion,
            })
            return shaped

        # ── THE COMPOSED VOICE (2026-07-03): speak a paragraph composed
        # from several diverse true facts, rewritten plain - instead of
        # quoting one encyclopedia sentence. Falls back to the single best
        # fact (still cleaned) when there's too little to compose.
        used = [best]
        emb_fn = getattr(self.agent.encoder, "_embed", None)
        composed = None
        # a count wants ONE precise number, not a composed paragraph
        if emb_fn is not None and not how_many:
            try:
                from mind.composer import compose_answer, simplify
                # compose from the reranked cohort, best fact guaranteed in -
                # and EVERY member must cover the question's facets, not just
                # the lead ("how do birds fly" must not stitch anatomy trivia)
                pool = [h for h in cohort if h is not best][:7]
                if focus and pool:
                    try:
                        covs = self.agent.encoder.focus_alignment(
                            focus, [h["text"] for h in pool], reduce="min")
                        pool = [h for h, c in zip(pool, covs) if c >= 0.45]
                    except Exception:
                        pass
                pool = [best] + pool
                composed = compose_answer(user_msg, pool, emb_fn)
            except Exception:
                composed = None
        if composed is not None:
            body, used = composed
        else:
            try:
                from mind.composer import simplify
                body = simplify(best["text"])
            except Exception:
                body = best["text"].strip()
        band = "high" if max(sim, top_sim) >= self._SEM_HIGH else "med"
        shaped = self.voice.shape_response(body, band=band, emotion=emotion)
        self.agent.turns.append({
            "user": user_msg, "agent": shaped,
            "retrieved_memories": [{"text": h["text"],
                                    "similarity": float(h.get("similarity", 0))}
                                   for h in used[:4]],
            "similarity": sim, "domain": "semantic:composed" if composed
                                         else "semantic", "emotion": emotion,
        })
        return shaped

    def _track_entities(self, question: str, reply: str) -> None:
        """Feed the agent's recent-entity stack after every answered turn, so
        pronoun follow-ups resolve ('tell me about Jupiter' -> 'how many moons
        does IT have?'). Question entities outrank answer entities; answer
        entities are used only when the question named nothing (e.g. 'what is
        the biggest planet?' -> the answer's 'Jupiter' becomes the topic)."""
        try:
            from lattice.growth import extract_topics
        except Exception:
            return
        q_ents = extract_topics(question, max_topics=2)
        if q_ents:
            for e in reversed(q_ents[1:]):
                self.agent._push_entity(e)
            self.agent._push_entity(q_ents[0])      # main subject = most recent
        elif reply:
            for e in extract_topics(reply.split(".")[0], max_topics=1):
                self.agent._push_entity(e)

    def respond(self, user_msg: str, creativity: float = 0.30) -> str:
        self._last_q_resolved = user_msg
        reply = self._respond_routes(user_msg, creativity)
        try:
            self._track_entities(self._last_q_resolved or user_msg,
                                 reply or "")
        except Exception:
            pass
        return reply

    def _respond_routes(self, user_msg: str, creativity: float = 0.30) -> str:
        """Apply fluency wrapping around the underlying agent.

        creativity is a continuous dial in [0, 1] from the query router:
          * < 0.25  direct        — trim to the most essential sentence
          *  0.25–0.5 synthesized — default composition, no extension
          *  0.5–0.75 extrapolated— extend the body with a short
                                     transition-chain continuation
          *  > 0.75 imagined      — extend with a longer continuation,
                                     looser repetition penalty

        The substrate (lattice retrieval) is always the source — the
        dial only varies how much the response is composed beyond the
        nearest stored memory.
        """
        creativity = max(0.0, min(1.0, float(creativity)))

        # ── Capture user-facts from the message ─────────────────────
        # ("my name is X", "I have two cats", "remember I prefer Y")
        if self.user_facts is not None:
            try:
                added = self.user_facts.capture(user_msg)
                if added:
                    print(f"[fluency] captured {len(added)} user-fact(s)",
                            flush=True)
                    # acknowledge the teaching instead of hunting for an answer
                    if not user_msg.rstrip().endswith("?"):
                        ack = "Got it - I'll remember that."
                        self.agent.turns.append({
                            "user": user_msg, "agent": ack,
                            "retrieved_memories": [], "similarity": 1.0,
                            "domain": "user_facts:capture",
                        })
                        return ack
            except Exception:
                pass

        # ── Detect user's emotion (for response coloring) ───────────
        try:
            emotion = classify_emotion(user_msg, encoder=self.agent.encoder)
        except Exception:
            emotion = None

        # ── Memory discipline: forgetting + provenance ──────────────
        f = self._forget_route(user_msg, emotion)
        if f is not None:
            return f
        p = self._provenance_route(user_msg, emotion)
        if p is not None:
            return p

        # ── The OPERATIVE QUESTION: a long ramble that ends in a real
        # question is answered on that question, not on the embedding of
        # the whole story about the cousin's neighbor.
        q_base = user_msg
        if len(user_msg.split()) > 25 and user_msg.rstrip().endswith("?"):
            m_op = re.search(
                r"(?:^|[.!?,:;]\s+)((?:what|who|where|when|which|how|why"
                r"|do|does|did|is|are|was|were|can)\b[^.!?]{3,140}\?)\s*$",
                user_msg, re.I)
            if m_op:
                q_base = m_op.group(1).strip()

        # ── Entity-aware follow-ups: resolve pronouns ONCE for every route
        # below ("how many moons does IT have?" -> "...does Jupiter have?").
        # The stack is fed by _track_entities after each answered turn.
        q_res = q_base
        try:
            q_res, _did = self.agent._resolve_pronouns(q_base)
        except Exception:
            pass
        # the tracker must see what the routes saw, not the raw pronouns
        self._last_q_resolved = q_res

        # ── User-facts recall FIRST (before persona).  "What's my name?"
        # / "what do you know about me?" must hit user_facts even if
        # the message superficially looks personal (contains "you").
        if self.user_facts is not None and self.user_facts.count() > 0:
            uf_low = user_msg.lower()
            uf_body, uf_used = None, []
            # broad recall lists the facts - no similarity gate to defeat
            if (self._looks_about_user(user_msg)
                    and re.search(r"\b(remember|know)\b.*\bme\b", uf_low)):
                uf_used = self.user_facts.all_facts()[-6:]
                uf_body = ("Here's what I know about you: "
                           + " ".join(self._uf_voice(t) for t in uf_used))
            # targeted first-person questions check the store by float
            # meaning (the bound-subspace similarity is too coarse a gate)
            elif (re.search(r"\bmy\b|\bmine\b|\b(?:do|am|did|was) i\b",
                            uf_low)
                    and re.match(r"^(what|where|when|who|which|how|do|does"
                                 r"|did|am|is|are|was)\b", uf_low)):
                emb_fn = getattr(self.agent.encoder, "_embed", None)
                uf = self.user_facts
                if emb_fn is not None:
                    import numpy as _np
                    embs = _np.asarray(emb_fn([user_msg] + uf._texts))
                    sims = embs[1:] @ embs[0]
                    i = int(sims.argmax())
                    if float(sims[i]) >= 0.45:
                        uf_used = [uf._texts[i]]
                        uf_body = self._uf_voice(uf._texts[i])
            if uf_body:
                shaped = self.voice.shape_response(
                    uf_body, band="high", emotion=emotion,
                )
                self.agent.turns.append({
                    "user":               user_msg,
                    "agent":              shaped,
                    "retrieved_memories": uf_used,
                    "similarity":         1.0,
                    "domain":             "user_facts",
                    "emotion":            emotion,
                })
                return shaped

        # ── Vision: "what did you see?" answers from sight memory ──────
        v = self._vision_route(q_res, emotion)
        if v is not None:
            return v

        # ── Ages and dates: retrieve the facts, COMPUTE the answer ─────
        ag = self._age_route(q_res, emotion)
        if ag is not None:
            return ag

        # ── Word problems: parse the story, do the arithmetic ──────────
        wm = self._word_math_route(q_res, emotion)
        if wm is not None:
            return wm

        # ── How-to: answer with ordered steps, learn from wikiHow on miss
        pr = self._procedure_route(q_res, emotion)
        if pr is not None:
            return pr

        # ── Analogies: look up the relationship in the dictionary ──────
        an = self._analogy_route(q_res, emotion)
        if an is not None:
            return an

        # ── Date comparisons: retrieve both sides, compute the answer ──
        cd = self._compare_dates_route(q_res, emotion)
        if cd is not None:
            return cd

        # ── Word definitions from the offline dictionary ───────────────
        df = self._define_route(q_res, emotion)
        if df is not None:
            return df

        # ── Stories from the imagination engine ────────────────────────
        st = self._story_route(q_res, emotion)
        if st is not None:
            return st

        # ── Persona: identity / opinions / preferences.  Use a CATEGORY
        # hint based on the question so we don't return random persona
        # facts for every personal query.
        if (self.persona is not None and self.persona.count() > 0
                and self._looks_personal(user_msg)):
            # Route to the right persona category by question intent.
            category = self._persona_category(user_msg)
            hits = self.persona.query(user_msg, k=5, category=category)
            if not hits:
                # Try without the category filter as a fallback
                hits = self.persona.query(user_msg, k=5)
            if hits and hits[0]["similarity"] >= 0.20:
                # If the top hit is too close to a generic stock answer
                # (we returned this same one to a different question
                # recently), shuffle to the next one.
                top = hits[0]["text"]
                if top == getattr(self, "_last_persona_text", None) and len(hits) > 1:
                    top = hits[1]["text"]
                self._last_persona_text = top
                shaped = self.voice.shape_response(
                    top, band="high", emotion=emotion,
                )
                self.agent.turns.append({
                    "user":               user_msg,
                    "agent":              shaped,
                    "retrieved_memories": [h["text"] for h in hits],
                    "similarity":         hits[0]["similarity"],
                    "domain":             "persona",
                    "category":           category,
                    "emotion":            emotion,
                })
                self._update_topic_words(shaped)
                return shaped

        # ── Forward-chaining inference ──────────────────────────────
        # Multi-fact questions like "how old was X when Y happened?"
        # need symbolic chaining — neither HDC retrieval nor code
        # synthesis handles them alone.
        chain = try_forward_chain(user_msg, self.agent)
        if chain is not None:
            answer = chain["answer"]
            self.agent.turns.append({
                "user":               user_msg,
                "agent":              answer,
                "retrieved_memories": chain.get("chain", []),
                "similarity":         0.8,
                "domain":             "forward_chain",
                "chain":              chain.get("chain"),
            })
            self.last_topic_words.clear()
            return answer

        # ── Game composer short-circuit ─────────────────────────────
        # "Build me a game" / "build me hangman" → playable terminal
        # game with game loop + state + I/O.  Must come BEFORE the
        # app composer because games aren't CRUD.
        composed_game = try_compose_game(user_msg, run=True)
        if composed_game is not None:
            reply = _code_chat_fmt(composed_game)
            self.agent.turns.append({
                "user":               user_msg,
                "agent":              reply,
                "retrieved_memories": [],
                "similarity":         1.0,
                "domain":             "game_composer",
                "game":               composed_game.get("game"),
                "ran":                composed_game.get("ran"),
            })
            self.last_topic_words.clear()
            return reply

        # ── App composer short-circuit ──────────────────────────────
        # "Build me a {X} app/tracker/journal" → full CLI app
        # generated from entity + schema + CRUD + main loop.
        # Handles arbitrary entity types via generic fallback.
        composed_app = try_compose_app(user_msg, run=True)
        if composed_app is not None:
            reply = _code_chat_fmt(composed_app)
            self.agent.turns.append({
                "user":               user_msg,
                "agent":              reply,
                "retrieved_memories": [],
                "similarity":         1.0,
                "domain":             "app_composer",
                "entity":             composed_app.get("entity"),
                "ran":                composed_app.get("ran"),
            })
            self.last_topic_words.clear()
            return reply

        # ── Code WRITING short-circuit ──────────────────────────────
        # "Write me a function that...", "give me python code for...",
        # etc.  Pulls a template, fills slots, runs sandboxed, returns
        # code + output.
        code_write = try_write_code(user_msg, run=True)
        if code_write is not None:
            reply = _code_chat_fmt(code_write)
            self.agent.turns.append({
                "user":               user_msg,
                "agent":              reply,
                "retrieved_memories": [],
                "similarity":         1.0,
                "domain":             "code_writer",
                "template":           code_write.get("template"),
                "ran":                code_write.get("ran"),
            })
            self.last_topic_words.clear()
            return reply

        # ── Code corpus fallback: when no template matches but the
        # request looks code-shaped, try retrieval from the curated
        # snippet store (Layer 4).
        if self.code_corpus is not None:
            low = user_msg.lower()
            code_triggers = (
                "write", "code", "function", "python", "give me",
                "show me", "how do i", "how can i", "snippet",
                "implement",
            )
            if any(t in low for t in code_triggers):
                retrieved = try_retrieve_code(user_msg, self.code_corpus)
                if retrieved is not None:
                    reply = _code_chat_fmt(retrieved)
                    self.agent.turns.append({
                        "user":               user_msg,
                        "agent":              reply,
                        "retrieved_memories": [],
                        "similarity":         retrieved.get("similarity", 0.0),
                        "domain":             "code_corpus",
                        "template":           retrieved.get("template"),
                    })
                    self.last_topic_words.clear()
                    return reply
                # ── Layer 6: code-analogy fallback ───────────────
                # "quicksort for linked lists" etc. — HDC analogy
                # operator over the corpus.
                analog = try_code_analogy(user_msg, self.code_corpus,
                                                self.agent.encoder)
                if analog is not None:
                    reply = _code_chat_fmt(analog)
                    if analog.get("analogy"):
                        a = analog["analogy"]
                        reply += (f"\n\n_(Analogical match — adapted "
                                    f"from {a['base']!r} via "
                                    f"{', '.join(a['swaps'])})_")
                    self.agent.turns.append({
                        "user":               user_msg,
                        "agent":              reply,
                        "retrieved_memories": [],
                        "similarity":         analog.get(
                            "analogy", {}).get("target_sim", 0.0),
                        "domain":             "code_analogy",
                        "template":           analog.get("template"),
                    })
                    self.last_topic_words.clear()
                    return reply

        # ── Code synthesis short-circuit (math + live data) ─────────
        # Arithmetic and live-price questions get DETERMINISTIC answers
        # via code synthesis — no retrieval, no hedge, no noise.
        code_ans = try_code_synthesis(user_msg)
        if code_ans is not None:
            self.agent.turns.append({
                "user":               user_msg,
                "agent":              code_ans,
                "retrieved_memories": [],
                "similarity":         1.0,
                "domain":             "code_synthesis",
            })
            self.last_topic_words.clear()
            return code_ans

        # ── Live-data short-circuit (time, date, etc.) ──────────────
        # Questions about REAL-TIME state can't be answered by lattice
        # retrieval — they need the actual system clock / live data.
        live = self._live_data_response(user_msg)
        if live is not None:
            self.agent.turns.append({
                "user":               user_msg,
                "agent":              live,
                "retrieved_memories": [],
                "similarity":         1.0,
                "domain":             "live_data",
            })
            self.last_topic_words.clear()
            return live

        # ── Smalltalk short-circuit (before any retrieval) ──────────
        # Greetings, "how are you", "who are you", etc. shouldn't hit
        # the lattice — they get garbage retrievals.  Return a brief,
        # varied response and update turn history so follow-ups work.
        cat = self._smalltalk_category(user_msg)
        if cat == "opinion":
            # opinions come from the persona, matched by float-cosine MEANING
            # (the trait-bound HV query collapses similarities; 54 facts is
            # small enough to embed directly)
            body = self._persona_opinion(user_msg)
            if body is not None:
                shaped = self.voice.shape_response(body, band="med",
                                                   emotion=emotion)
                self.agent.turns.append({
                    "user": user_msg, "agent": shaped,
                    "retrieved_memories": [body],
                    "similarity": 0.5, "domain": "persona:opinion",
                })
                return shaped
            cat = None                       # no persona hit: fall through
        if cat == "surprise_me":
            # share a random remembered fact - the organism showing its memory
            try:
                import sqlite3 as _sq
                con = _sq.connect(str(self.agent.lattice.db_path))
                row = con.execute(
                    "SELECT text FROM memories WHERE source LIKE 'wikipedia:%' "
                    "AND LENGTH(text) BETWEEN 60 AND 240 "
                    "ORDER BY RANDOM() LIMIT 1").fetchone()
                con.close()
                if row:
                    reply = f"Here's one from my memory: {row[0]}"
                    self.agent.turns.append({
                        "user": user_msg, "agent": reply,
                        "retrieved_memories": [row[0]],
                        "similarity": 1.0, "domain": "smalltalk:surprise_me",
                    })
                    return reply
            except Exception:
                pass
        if cat is not None:
            pool = self._SMALLTALK_RESPONSES.get(cat) or []
            if pool:
                reply = self._rng.choice(pool)
                self.agent.turns.append({
                    "user":               user_msg,
                    "agent":              reply,
                    "retrieved_memories": [],
                    "similarity":         1.0,
                    "domain":             f"smalltalk:{cat}",
                })
                self.last_topic_words.clear()
                return reply

        # ── Semantic-first answer (2026-07-02): with the MiniLM encoder,
        # raw retrieval similarity is a TRUSTWORTHY meaning signal - the
        # legacy rerank/multidoc machinery was built to compensate for the
        # RI encoder and fights it. Scale: verbatim 1.0, true match
        # 0.37-0.55, unrelated <=0.15.
        sem = self._semantic_answer(q_res, emotion)
        if sem is not None:
            self._update_topic_words(sem)
            return self._shape_by_creativity(sem, creativity)

        # Domain-aware pre-retrieval: if the query is clearly trading
        # or identity, try to answer from the right source first.
        domain = self._classify_query(user_msg)
        if domain in ("trading", "identity"):
            dom_hit = self._domain_filtered_answer(user_msg, domain)
            if dom_hit is not None:
                text, sim = dom_hit
                band = "high" if sim >= 0.45 else (
                       "med"  if sim >= 0.32 else "low")
                body = self._apply_hedge(_normalize(text), band)
                self._update_topic_words(body)
                self.agent.turns.append({
                    "user":               user_msg,
                    "agent":              body,
                    "retrieved_memories": [text],
                    "similarity":         sim,
                    "domain":             domain,
                })
                # Apply creativity shaping even to domain hits
                return self._shape_by_creativity(body, creativity)

        # ── Multi-document synthesis (top-K consensus) ─────────────
        # For general queries, pull top-K and fuse them.  Single-doc
        # retrieval is noisy — one bad-luck top hit can ruin a real
        # answer.  Multi-doc averages over the top hits + uses content
        # overlap to surface the consensus answer.
        md = self._multidoc_synthesize(user_msg, top_k=12)
        if (md is not None and md["band"] in ("high", "med")
                and not self._howto_miss(user_msg, md.get("text", ""))):
            # Apply voice shaping (emotion + band) instead of the
            # legacy hedge — gives Telp personality through this path
            # too.
            body = self.voice.shape_response(
                _normalize(md["text"]),
                band=md["band"],
                emotion=emotion,
                is_followup=self._detect_followup(user_msg),
            )
            body = _normalize(body)
            # Stash source info on the agent turn so /why can show it
            self._last_sources = md["sources"]
            self.agent.turns.append({
                "user":               user_msg,
                "agent":              body,
                "retrieved_memories": [s["source"] for s in md["sources"]],
                "similarity":         md["sources"][0]["sim"] if md["sources"] else 0.0,
                "domain":             "multidoc",
                "n_sources":          md["n_sources"],
            })
            if self._detect_followup(user_msg):
                opener = self._rng.choice(_FOLLOWUP_OPENERS)
                if opener:
                    body = opener + body[0].lower() + body[1:]
            self._update_topic_words(body)
            return self._shape_by_creativity(body, creativity)

        # Delegate to the underlying agent
        try:
            raw = self.agent.respond(user_msg)
        except Exception as e:
            return f"Something broke when I tried to think about that ({e})."

        raw = _strip_debug(raw or "")

        last_turn = self.agent.turns[-1] if self.agent.turns else {}
        band = _confidence_band(last_turn)

        # procedural questions: if even the fallback answer doesn't cover the
        # HOW (boil), it's a miss - trigger the lookup instead of settling
        force_miss = self._howto_miss(user_msg, raw)

        if _looks_like_abstention(raw) or band == "none" or force_miss:
            # LEARN-ON-MISS (2026-07-02): before giving up, go find it -
            # fetch the topic (Wikipedia, attributed source), grow the
            # memory, answer from what was just learned.
            learned = self._learn_on_miss(q_res, emotion)
            if learned is not None:
                self._update_topic_words(learned)
                return learned
            self.last_topic_words.clear()
            self._mark_abstained(user_msg, emotion)
            return self._abstain()

        # Quality check on the raw agent output — block junk (prompt
        # lists, FAQs, lowercase fragments) from leaking through this
        # single-doc fallback path the same way multi-doc filters them.
        if self._md_text_quality(raw) < 0.5:
            self.last_topic_words.clear()
            self._mark_abstained(user_msg, emotion)
            return self._abstain()

        sentences = _split_sentences(raw)
        stitched = _stitch_same_subject(sentences)
        body = _normalize(stitched)
        # Voice-shape with emotion (legacy _apply_hedge replaced).
        body = self.voice.shape_response(
            body, band=band, emotion=emotion,
            is_followup=self._detect_followup(user_msg),
        )
        body = _normalize(body)

        if self._detect_followup(user_msg):
            opener = self._rng.choice(_FOLLOWUP_OPENERS)
            if opener:
                body = opener + body[0].lower() + body[1:]

        self._update_topic_words(body)

        # Apply the creativity dial to shape the final body
        return self._shape_by_creativity(body, creativity)

    # ── Creativity-driven shaping ─────────────────────────────────

    def _shape_by_creativity(self, body: str, creativity: float) -> str:
        """Vary the response composition by the creativity dial.

        Substrate is always the lattice; this controls how much of the
        substrate is woven into the response.
        """
        if not body:
            return body
        c = max(0.0, min(1.0, creativity))

        # Direct: take only the first sentence — most essential statement
        if c < 0.25:
            sents = _split_sentences(body)
            if sents:
                return _normalize(sents[0])
            return body

        # Synthesized: default — return composed body as-is
        if c < 0.50:
            return body

        # Extrapolated / imagined: extend via the agent's HDC generator
        # using the tail of the body as the seed.  More creativity ->
        # longer continuation, looser repetition penalty.
        try:
            seq = getattr(self.agent, "seq", None)
            if seq is None or not hasattr(seq, "generate"):
                return body
            # Seed = last 5-8 words of the current body
            seed_words = body.rstrip(".!?").strip().split()[-7:]
            seed = " ".join(seed_words)
            if not seed:
                return body
            # Scale: 8-32 extra words from creativity 0.5 to 1.0
            n_words = int(8 + (c - 0.5) * 48)
            if hasattr(self.agent, "ensure_generator"):
                self.agent.ensure_generator()
            ext = seq.generate(seed, n_words=n_words)
            if not ext:
                return body
            # The generator returns "seed + generated tokens" as one
            # string; strip the seed so we only append the new part.
            tail = ext[len(seed):].strip() if ext.startswith(seed) else ext
            if not tail:
                return body
            # Light cleanup: capitalize, trim repeated dangling words
            tail = _normalize(tail).rstrip(",;")
            # Avoid the extension ending mid-thought
            if tail and tail[-1] not in ".!?":
                tail = tail.rstrip(",") + "."
            return body.rstrip() + " " + tail
        except Exception:
            return body
