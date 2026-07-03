"""
lattice/dictionary_chat.py — Phase 15: HDC-native conversation engine.

Pure retrieval + composition.  No sampling, no learned distribution,
no LLM, no hallucination.  Every word in every response is traceable
either to:

  - a dictionary entry in state/wiktionary/dict.db (definitions,
    relations, etymology), OR
  - a frame Telp parsed and stored in lattice.line_events (the
    stories and corpora he has read)

When asked something he has no knowledge for, he says so plainly
instead of making up a plausible-sounding answer.  This is a feature
LLMs structurally cannot match.

PIPELINE
--------
  1. Parse user input into an event frame (lattice.event_frames)
  2. Classify intent: definition / relation / story / unknown
  3. Retrieve the relevant facts from dict.db + line_events
  4. Compose a multi-sentence response from those facts via the
     same render_frame() machinery the story transformer uses
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

_TELP_ROOT = Path(__file__).resolve().parents[1]
if str(_TELP_ROOT) not in sys.path:
    sys.path.insert(0, str(_TELP_ROOT))

from lattice.event_frames import parse_event, ARTICLES, PRONOUNS, WH_WORDS
from lattice.dictionary_lookup import Dictionary, Entry
from lattice.self_model import SelfModel
from lattice.conversation_knowledge import (
    ConversationKnowledge, ConversationContext,
)


# ─── Helpers ─────────────────────────────────────────────────────


_STOPWORDS = (ARTICLES | PRONOUNS | WH_WORDS |
                    {"is", "are", "was", "were", "am", "be", "been",
                     "do", "does", "did", "of", "in", "on", "at", "to",
                     "from", "by", "with", "for", "about", "an"})


def _content_words(text: str) -> list[str]:
    """Strip articles + stopwords; return content word tokens."""
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


# Words that start with a vowel LETTER but a consonant SOUND
# ("u" pronounced /y/, "o" pronounced /w/) — take "a".
_VOWEL_LETTER_CONSONANT_SOUND = {
    "unicorn", "university", "universe", "universal", "uniform",
    "union", "unique", "united", "unit", "useful", "user", "usual",
    "usurp", "utility", "ubiquitous", "european", "euphemism",
    "one", "once",
}
# Words that start with a consonant LETTER (h) but a vowel SOUND
# (silent h) — take "an".
_SILENT_H = {"hour", "honest", "honor", "honorable", "heir", "heirloom"}


def _article_for(word: str) -> str:
    """Pick a/an based on initial SOUND, not just letter.

    English rule is phonetic: "a unicorn" (yoo-), "an hour" (silent h).
    Without an IPA lookup we approximate via a small exception list
    plus the default vowel-letter heuristic.
    """
    if not word:
        return "a"
    w = word.lower()
    if w in _VOWEL_LETTER_CONSONANT_SOUND:
        return "a"
    if w in _SILENT_H:
        return "an"
    return "an" if w[0] in "aeiou" else "a"


def _trim_gloss(gloss: str, max_chars: int = 240) -> str:
    """Shorten a long Wiktionary gloss to one sentence's worth."""
    g = (gloss or "").strip()
    if not g:
        return ""
    # Cut at first sentence boundary or comma after a clause
    cuts = [m.start() for m in re.finditer(r"[.;](\s|$)", g)]
    if cuts and cuts[0] < max_chars:
        return g[:cuts[0] + 1].strip()
    if len(g) > max_chars:
        return g[:max_chars].rsplit(" ", 1)[0] + "..."
    if not g.endswith("."):
        g += "."
    return g


# ─── Intent classification ────────────────────────────────────────


# Lead-in phrases to strip BEFORE picking the content word.  Order
# matters — match longest first.
_STRIP_PREFIXES = (
    "tell me a story about ", "tell me about ", "do you know about ",
    "do you know ", "what about ", "what's the ", "what is the ",
    "what are the ", "definition of ", "define ",
    "what does the ", "what does ", "what is ", "what's ", "whats ",
    "what are ", "who is ", "who's ", "who are ",
    "explain ", "explain the ", "what means ",
    "synonyms of ", "synonym of ", "antonyms of ", "antonym of ",
    "opposite of ", "opposites of ", "kinds of ", "types of ",
    "examples of ", "related to ", "similar to ",
)
# Connecting words inside a target phrase — extract what's AFTER them
# when present.  "etymology of dictionary" -> target = "dictionary"
_OF_PREFIX = re.compile(
    r"^(etymology|origin|meaning|definition|history|sound|pronunciation|"
    r"synonym|synonyms|antonym|antonyms|opposite|opposites|"
    r"kind|kinds|type|types|example|examples)s?\s+of\s+",
    re.IGNORECASE)


def _strip_to_target(text: str) -> str:
    """Strip question/command lead-ins and return the substring that
    actually names the thing being asked about."""
    s = text.strip().lower().rstrip("?!.")
    # Strip the longest matching lead-in prefix
    for p in _STRIP_PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
            break
    # Strip a leading article on the remainder
    for art in ("a ", "an ", "the "):
        if s.startswith(art):
            s = s[len(art):]
            break
    # If what remains is "X of Y", focus on Y
    m = _OF_PREFIX.match(s)
    if m:
        s = s[m.end():]
        for art in ("a ", "an ", "the "):
            if s.startswith(art):
                s = s[len(art):]
                break
    return s.strip()


def _first_content_word(s: str) -> Optional[str]:
    words = _content_words(s)
    return words[0] if words else None


def classify_intent(text: str, frame: dict) -> dict:
    """Decide what kind of query this is.

    Returns {kind, target, ...}.  kind is one of:

      define         — "what is X" / "define X" / "X means what"
      relation       — "synonyms of X" / "types of X"
      explain        — "why ..." / "how ..." (causal / process)
      story          — "tell me about X" / "do you know X"
      greeting       — "hi" / "hello"
      unknown        — fallback
    """
    low = text.strip().lower().rstrip("?!.")

    if low in {"hi", "hello", "hey", "yo", "sup", "yo telp", "hi telp"}:
        return {"kind": "greeting", "target": None}

    # Relation queries
    if (low.startswith(("synonym", "antonym", "opposite",
                                "kinds of", "types of", "examples of",
                                "related to", "similar to"))):
        tail = _strip_to_target(text)
        target = _first_content_word(tail) or _first_content_word(low)
        rel = ("antonym" if "antonym" in low or "opposite" in low
                  else "synonym" if "synonym" in low or "similar" in low
                  else "hyponym" if "kind" in low or "type" in low
                                          or "example" in low
                  else "related")
        return {"kind": "relation", "target": target, "relation": rel}

    # Etymology queries (special-cased so they route to explain)
    if "etymology" in low or "origin of" in low:
        tail = _strip_to_target(text)
        target = _first_content_word(tail)
        return {"kind": "explain", "target": target,
                  "aspect": "etymology"}

    # Story / recall queries
    if low.startswith(("tell me about", "tell me a story",
                              "do you know", "do you remember",
                              "what about")):
        tail = _strip_to_target(text)
        target = _first_content_word(tail)
        return {"kind": "story", "target": target}

    # Why / how — explanation queries
    if low.startswith(("why ", "how come", "explain ")):
        tail = _strip_to_target(text)
        target = _first_content_word(tail)
        return {"kind": "explain", "target": target}

    # Definition queries ("what is X", "define X", ...)
    if low.startswith(("define ", "definition of ", "what does ",
                              "what is ", "what's ", "whats ",
                              "what are ", "who is ", "who's ",
                              "what means ")):
        tail = _strip_to_target(text)
        target = _first_content_word(tail)
        return {"kind": "define", "target": target}

    # Frame-based fallback: parser-detected wh-question
    if frame.get("_intent") == "question":
        cand = (frame.get("attribute") or frame.get("patient")
                  or frame.get("agent"))
        if cand and cand not in WH_WORDS:
            return {"kind": "define", "target": cand}

    # Default: try to define the first content word
    target = _first_content_word(low)
    return {"kind": "unknown", "target": target}


# ─── Response composers ──────────────────────────────────────────


# Priority order when choosing which sense to lead with.  kaikki
# orders senses arbitrarily within a POS, so without this you get
# "A bear is: someone who believes prices will fall" because the
# financial sense happens to come first in the dump.
_POS_PRIORITY = ["noun", "verb", "adj", "adv", "pron", "prep",
                       "conj", "intj", "particle", "num", "phrase",
                       "proverb", "name"]


def _rank_sense(s: "Sense") -> tuple:
    """Smaller = better.  Prefer noun > verb > adj > ...; within a
    POS prefer untagged senses (no 'rare', 'alt-of', 'figurative',
    'slang' tags) over tagged ones; then by sense index."""
    pos_rank = (_POS_PRIORITY.index(s.pos)
                       if s.pos in _POS_PRIORITY else 99)
    # Tags that mark a sense as non-prototypical.  "alt-of" /
    # "alternative" are critical: kaikki sometimes lists "alternative
    # spelling of X" as the first sense (e.g. "bear" -> alt of "bere"),
    # which is a terrible lead for "what is a bear?".
    bad_tags = {"obsolete", "archaic", "rare", "dialectal", "slang",
                       "vulgar", "figurative", "humorous", "informal",
                       "dated", "alt-of", "alternative", "abbreviation",
                       "initialism", "acronym", "misspelling",
                       "nonstandard"}
    tag_penalty = sum(1 for t in (s.tags or [])
                              if str(t).lower() in bad_tags)
    # Glosses that read like cross-references ("Alternative form of",
    # "Misspelling of", "Plural of") shouldn't lead either.
    g = (s.gloss or "").lower()
    cross_ref_penalty = (10 if g.startswith((
        "alternative", "misspelling", "plural of",
        "abbreviation of", "initialism of", "acronym of",
        "obsolete form", "archaic form", "rare form",
        "synonym of", "see ", "(see ",
    )) else 0)
    return (pos_rank, tag_penalty + cross_ref_penalty, s.sense_idx)


def _compose_definition(entry: Entry, max_senses: int = 2) -> str:
    """Compose a definition response from a dictionary entry.

    Picks the most prototypical sense (noun > verb > adj, untagged
    over tagged) as the lead, then optionally adds one extra POS.
    """
    if not entry.senses:
        return f"I do not have a definition for {entry.word}."
    sorted_senses = sorted(entry.senses, key=_rank_sense)
    s0 = sorted_senses[0]
    g0 = _trim_gloss(s0.gloss)
    art = _article_for(entry.word)
    sentences = [f"{art.capitalize()} {entry.word} "
                       f"({s0.pos}) is: {g0}"]
    # Up to N additional senses across distinct POS
    seen_pos = {s0.pos}
    extras = 0
    for s in sorted_senses[1:]:
        if extras >= max_senses - 1:
            break
        if s.pos in seen_pos:
            continue
        seen_pos.add(s.pos)
        g = _trim_gloss(s.gloss)
        sentences.append(f"As {_article_for(s.pos)} {s.pos}, "
                                  f"{entry.word} means: {g}")
        extras += 1
    return " ".join(sentences)


def _compose_relations(entry: Entry, relation: str,
                                limit: int = 8) -> str:
    """Compose a list response naming related words."""
    targets = entry.related.get(relation) or []
    if not targets:
        # Try a soft fallback to nearby relations
        if relation == "synonym":
            targets = entry.related.get("related") or []
        elif relation == "hyponym":
            targets = entry.related.get("derived") or []
        if not targets:
            return (f"I do not know any {relation}s of "
                        f"{entry.word}.")
    targets = targets[:limit]
    if len(targets) == 1:
        return f"One {relation} of {entry.word} is {targets[0]}."
    return (f"Some {relation}s of {entry.word} are: "
                f"{', '.join(targets[:-1])}, and {targets[-1]}.")


def _compose_etymology(entry: Entry) -> str:
    if not entry.etymology:
        return ""
    pos, text = next(iter(entry.etymology.items()))
    short = _trim_gloss(text, max_chars=240)
    return f"The word {entry.word} comes from: {short}"


def _compose_story_recall(target: str, line_events: dict) -> Optional[str]:
    """If Telp has read frames involving this target, recall them."""
    if not line_events:
        return None
    hits = []
    t = target.lower()
    for raw, frame in line_events.items():
        for role in ("agent", "patient", "goal", "location", "attribute"):
            v = frame.get(role)
            if v and t in str(v).lower():
                hits.append(raw)
                break
        if len(hits) >= 4:
            break
    if not hits:
        return None
    return (f"I remember reading about {target}. "
                + " ".join(hits[:4]))


# ─── ConversationEngine ──────────────────────────────────────────


class ConversationEngine:
    """Stateful chat over the Wiktionary dictionary + reading lattice.

    The "state" is a rolling list of recent (user, telp) turn pairs,
    used only for context display (a future extension would bundle
    them into a context HV for retrieval biasing).
    """

    def __init__(self,
                      dict_path: Optional[Path] = None,
                      lattice_pickle: Optional[Path] = None,
                      verbose: bool = False):
        self.dictionary = Dictionary(dict_path) if dict_path \
                                else Dictionary()
        self.verbose = verbose
        self.turns: list[tuple[str, str]] = []
        # Try to load reading lattice if pickle exists
        self.line_events: dict = {}
        try:
            from lattice.reading_lattice import ReadingLattice, LATTICE_PATH
            p = lattice_pickle or LATTICE_PATH
            if Path(p).exists():
                lat = ReadingLattice.load(p)
                self.line_events = getattr(lat, "line_events", {}) or {}
        except Exception as e:
            if verbose:
                print(f"  [chat] lattice not loaded: {e}")
        # Phase 19: self-model for first/second-person handling.
        self.self_model = SelfModel()
        # Phase 19: lazy ImaginationEngine — only instantiated when
        # the user actually asks for a story (avoids pool-building
        # cost on chat startup).
        self._imagination = None
        # Phase 20: editable knowledge document about how to converse.
        # Loaded once, queried on every response, gives Telp his
        # conversational shape (acknowledge + body + connect + invite).
        self.conv_knowledge = ConversationKnowledge()
        self.conv_context   = ConversationContext()

    def _get_imagination(self):
        """Lazy-init the imagination engine the first time it's needed."""
        if self._imagination is None:
            from lattice.imagination import ImaginationEngine
            self._imagination = ImaginationEngine(
                dictionary=self.dictionary)
        return self._imagination

    # ── Pre-handlers (run BEFORE the define pipeline) ────────────

    def _try_imagine(self, text: str) -> Optional[str]:
        """If the user wants a NEW story, call the imagination engine."""
        low = text.strip().lower().rstrip("?!.")
        # Patterns that route to imagination (not retrieval)
        seed = None
        triggered = False
        for pat in (
            r"make up a story about (?:a |an |the )?(\w+)",
            r"tell me a (?:new |original )?story about (?:a |an |the )?(\w+)",
            r"invent a story about (?:a |an |the )?(\w+)",
            r"imagine a story about (?:a |an |the )?(\w+)",
            r"dream me a story about (?:a |an |the )?(\w+)",
            r"give me a story about (?:a |an |the )?(\w+)",
            r"can you make up a story about (?:a |an |the )?(\w+)",
        ):
            m = re.search(pat, low)
            if m:
                seed = m.group(1)
                triggered = True
                break
        # General request without a seed -> use a random seed
        if not triggered:
            for pat in (
                r"^make up a story\b", r"^tell me a story\b",
                r"^invent a story\b", r"^imagine a story\b",
                r"^dream me a story\b", r"^give me a story\b",
                r"^make a story\b",
            ):
                if re.search(pat, low):
                    triggered = True
                    break
        if not triggered:
            return None

        eng = self._get_imagination()
        if seed is None:
            # Pick a random concrete noun from the cast pool
            import random
            seed = random.choice(eng.cast_pool())
        frames = eng.imagine_story(seed=seed)
        return eng.render(frames)

    def _try_recall(self, text: str) -> Optional[str]:
        """If the user asks about something Telp has READ, search
        line_events directly.  ("do you remember the bear?" should
        produce a recall, not a definition of 'remember'.)"""
        low = text.strip().lower().rstrip("?!.")
        target = None
        for pat in (
            r"do you remember (?:a |an |the )?(\w+)",
            r"have you read about (?:a |an |the )?(\w+)",
            r"what do you remember about (?:a |an |the )?(\w+)",
            r"what have you read about (?:a |an |the )?(\w+)",
        ):
            m = re.search(pat, low)
            if m:
                target = m.group(1)
                break
        if target is None:
            return None
        recall = _compose_story_recall(target, self.line_events)
        if recall:
            return recall
        return (f"I do not remember reading about {target}.  My "
                   f"memory holds {len(self.line_events)} parsed lines "
                   f"from the corpora I have read.")

    # ── Fluency wrapper helper ────────────────────────────────

    def _wrap(self, raw: str, user_text: str, intent: str,
                  topic: Optional[str] = None,
                  include_invite: bool = True) -> str:
        """Pass a raw response through the conversation_knowledge
        composer with related-word lookup for connections."""
        related = None
        if topic:
            entry = self.dictionary.lookup(topic)
            if entry:
                # Prefer hypernym (sits naturally in "related to X")
                for rel in ("hypernym", "synonym", "related",
                                  "hyponym", "meronym"):
                    if entry.related.get(rel):
                        related = entry.related[rel][0]
                        # Avoid suggesting "related to itself"
                        if related != topic:
                            break
                        related = None
        return self.conv_knowledge.compose(
            raw_answer=raw, user_text=user_text, intent=intent,
            context=self.conv_context, topic=topic, related=related,
            include_invite=include_invite,
        )

    def respond(self, user_text: str) -> str:
        # ── Phase 20: handle "tell me more" follow-ups FIRST
        # so they continue the previous topic instead of triggering
        # the define handler on the word "more".
        if self.conv_context.is_followup(user_text):
            last_topic = self.conv_context.current_topic()
            if last_topic:
                entry = self.dictionary.lookup(last_topic)
                if entry is not None:
                    # Try recall first, fall back to extra sense
                    story = _compose_story_recall(last_topic,
                                                                  self.line_events)
                    if story:
                        raw = story
                    else:
                        raw = _compose_definition(entry, max_senses=2)
                    composed = self._wrap(raw, user_text, "recall",
                                                      last_topic)
                    self.conv_context.push(user_text, composed,
                                                      last_topic)
                    self.turns.append((user_text, composed))
                    return composed
            # No previous topic — graceful fallback
            reply = ("I am not sure what to tell more about.  What "
                       "were we just discussing?")
            self.conv_context.push(user_text, reply, None)
            self.turns.append((user_text, reply))
            return reply

        # ── Phase 19: pre-handlers, run BEFORE the define pipeline.
        # Order matters: self-model first (so "thank you" doesn't get
        # parsed as a request to define "thank"), then imagine (so
        # "make up a story about a cat" doesn't define "make"), then
        # recall (so "do you remember the bear" finds the bear).

        # 1. Self / social acts — emitted RAW (no fluency wrap; self
        # responses already have the right shape).
        self_reply = self.self_model.respond(user_text)
        if self_reply is not None:
            self.conv_context.push(user_text, self_reply, None)
            self.turns.append((user_text, self_reply))
            return self_reply

        # 2. Story imagination — light wrap (opener + closer, no
        # mid-sentence invitation since the story IS the body).
        try:
            imagine_reply = self._try_imagine(user_text)
            if imagine_reply is not None:
                topic = (self.conv_knowledge._extract_topic(user_text)
                              or "this one")
                opener_pat = self.conv_knowledge._pick(
                    self.conv_knowledge.ack_patterns, "ack")
                try:
                    opener = opener_pat.format(topic=topic) \
                                if opener_pat else ""
                except KeyError:
                    opener = opener_pat or ""
                wrapped = (f"{opener}\n\n{imagine_reply}\n\n"
                              f"That was one path.  Want a different one?")
                self.conv_context.push(user_text, wrapped, topic)
                self.turns.append((user_text, wrapped))
                return wrapped
        except Exception as e:
            if self.verbose:
                print(f"  [chat] imagine failed: {e}")

        # 3. Recall from reading — wrap fully
        try:
            recall_reply = self._try_recall(user_text)
            if recall_reply is not None:
                topic = self.conv_knowledge._extract_topic(user_text)
                composed = self._wrap(recall_reply, user_text,
                                                  "recall", topic)
                self.conv_context.push(user_text, composed, topic)
                self.turns.append((user_text, composed))
                return composed
        except Exception as e:
            if self.verbose:
                print(f"  [chat] recall failed: {e}")

        # 4. Original define/relation/story/explain pipeline
        frame  = parse_event(user_text)
        intent = classify_intent(user_text, frame)
        if self.verbose:
            print(f"  [debug] intent={intent}  frame={frame}")

        kind = intent["kind"]
        target = intent.get("target")

        if kind == "greeting":
            reply = ("Hello. I am Telp. I can define words, name "
                       "related words, and recall what I have read.")

        elif kind in ("define", "unknown") and target:
            entry = self.dictionary.lookup(target)
            if entry is not None:
                reply = _compose_definition(entry)
                story = _compose_story_recall(target, self.line_events)
                if story:
                    reply += " " + story
            else:
                reply = (f"I do not know the word {target!r}. "
                            f"It is not in my dictionary.")

        elif kind == "relation" and target:
            entry = self.dictionary.lookup(target)
            if entry is None:
                reply = (f"I do not know the word {target!r}. "
                            f"It is not in my dictionary.")
            else:
                reply = _compose_relations(entry, intent["relation"])

        elif kind == "story" and target:
            story = _compose_story_recall(target, self.line_events)
            entry = self.dictionary.lookup(target)
            parts = []
            if entry is not None:
                parts.append(_compose_definition(entry, max_senses=1))
            if story:
                parts.append(story)
            elif entry is None:
                parts.append(f"I do not know about {target}.")
            reply = " ".join(parts)

        elif kind == "explain" and target:
            entry = self.dictionary.lookup(target)
            if entry is not None:
                parts = [_compose_definition(entry, max_senses=1)]
                ety = _compose_etymology(entry)
                if ety:
                    parts.append(ety)
                reply = " ".join(parts)
            else:
                reply = (f"I cannot explain {target}. "
                            f"I do not know that word.")

        else:
            reply = ("I am not sure what to say.  Try: 'what is X?', "
                       "'synonyms of X?', or 'tell me about X'.")
            # Don't wrap fallback help
            self.conv_context.push(user_text, reply, None)
            self.turns.append((user_text, reply))
            return reply

        # ── Phase 20: wrap content-engine responses in fluent shape ──
        composed = self._wrap(reply, user_text, kind, target,
                                          include_invite=(kind != "greeting"))
        self.conv_context.push(user_text, composed, target)
        self.turns.append((user_text, composed))
        return composed


# ─── CLI ─────────────────────────────────────────────────────────


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true",
                          help="print parsed intent + frame")
    ap.add_argument("--ask", action="append", default=[],
                          help="ask one question and exit (repeatable)")
    args = ap.parse_args()

    eng = ConversationEngine(verbose=args.debug)
    s = eng.dictionary.stats()
    print(f"Telp dictionary chat — "
              f"{s['distinct_words']:,} words, "
              f"{s['entries']:,} senses, "
              f"{s['related']:,} relations, "
              f"{len(eng.line_events):,} story frames")
    print()

    if args.ask:
        for q in args.ask:
            print(f"You:  {q}")
            print(f"Telp: {eng.respond(q)}")
            print()
        return

    print("Type a question.  Ctrl+C to exit.")
    print()
    while True:
        try:
            q = input("You:  ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "bye"}:
            print("Telp: Goodbye.")
            break
        print(f"Telp: {eng.respond(q)}")
        print()


if __name__ == "__main__":
    main()
