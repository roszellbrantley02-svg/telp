"""
lattice/conversation_knowledge.py — Phase 20: Telp learns fluency.

WHY
---
LLMs achieve conversational fluency through pretraining on billions
of tokens of dialogue.  Telp doesn't get that, and pretraining is
the wrong tool for him: pretraining compensates for forgetting, and
Telp doesn't forget.  What he needs is to UNDERSTAND THE TASK — once.

This module loads a plain-English document
(state/books/fluent_conversation.txt) that teaches Telp what fluent
conversation IS:
  - Principles: the rules
  - Patterns:   reusable surface templates with {slot} variables
  - Examples:   sample exchanges showing principles in use

The document is read once, parsed into queryable knowledge, and
applied at response time.  To improve Telp's conversational ability,
edit the document.  No code changes needed.

HOW
---
At response time, after the content engine (define, recall, imagine)
produces a raw answer, ConversationKnowledge.compose() wraps it in
a fluent shape:

  [ACKNOWLEDGMENT]  [BODY: raw answer]  [CONNECTION?]  [INVITATION?]

Acknowledgment, connection, and invitation are drawn from the
loaded patterns with a tiny RNG; recent picks are avoided so the
same opening doesn't fire two turns in a row.  Topic continuity
is tracked via a rolling list of recent content words.

The content engines (define / imagine / recall / self) stay
untouched.  This is purely a SURFACE-SHAPING layer.  Same content,
different delivery.
"""
from __future__ import annotations

import random
import re
import sys
from collections import deque
from pathlib import Path
from typing import Optional

_TELP_ROOT = Path(__file__).resolve().parents[1]
if str(_TELP_ROOT) not in sys.path:
    sys.path.insert(0, str(_TELP_ROOT))


# Default location of the editable knowledge document
DEFAULT_PATH = _TELP_ROOT / "state" / "books" / "fluent_conversation.txt"


# ─── Document loader ─────────────────────────────────────────────


class ConversationKnowledge:
    """Loads, parses, and queries fluent_conversation.txt.

    Layout assumed in the document:
      ## Principles                  - prose statements
      ## Acknowledgment patterns     - templates with {topic} / {related}
      ## Connection patterns         - templates
      ## Invitation patterns         - templates
      ## Brevity patterns            - short bare responses
      ## Example exchanges           - User:/Telp: pairs

    Sections are detected by `## Title` markers.  Each section's body
    is the lines until the next `##` heading or EOF.
    """

    def __init__(self,
                      path: Path | str = DEFAULT_PATH,
                      seed: Optional[int] = None):
        self.path = Path(path)
        self.rng = random.Random(seed)
        # Parsed buckets
        self.principles: list[str]              = []
        self.ack_patterns: list[str]            = []
        self.connect_patterns: list[str]        = []
        self.invite_patterns: list[str]         = []
        self.brevity_patterns: list[str]        = []
        self.example_exchanges: list[dict]      = []
        # Anti-repetition: last N picks per category
        self._recent: dict[str, deque] = {
            k: deque(maxlen=3) for k in
            ("ack", "connect", "invite")
        }
        if self.path.exists():
            self._load(self.path.read_text(encoding="utf-8"))
        else:
            print(f"[conversation_knowledge] no document at "
                      f"{self.path} — using empty defaults",
                      file=sys.stderr)

    # ── Parsing ────────────────────────────────────────────────

    def _load(self, text: str) -> None:
        sections = self._split_sections(text)
        for title, body in sections.items():
            t = title.lower()
            if t.startswith("principles"):
                self.principles = self._lines(body)
            elif t.startswith("acknowledgment patterns"):
                self.ack_patterns = self._lines(body)
            elif t.startswith("connection patterns"):
                self.connect_patterns = self._lines(body)
            elif t.startswith("invitation patterns"):
                self.invite_patterns = self._lines(body)
            elif t.startswith("brevity patterns"):
                self.brevity_patterns = self._lines(body)
            elif t.startswith("example exchanges"):
                self.example_exchanges = self._parse_exchanges(body)

    @staticmethod
    def _split_sections(text: str) -> dict[str, str]:
        """Split on '##' headings; return {title: body} preserving order."""
        sections: dict[str, str] = {}
        cur_title = None
        cur_body: list[str] = []
        for raw in text.splitlines():
            stripped = raw.strip()
            if stripped.startswith("##"):
                if cur_title is not None:
                    sections[cur_title] = "\n".join(cur_body).strip()
                cur_title = stripped.lstrip("#").strip()
                cur_body = []
            elif cur_title is not None:
                cur_body.append(raw)
        if cur_title is not None:
            sections[cur_title] = "\n".join(cur_body).strip()
        return sections

    @staticmethod
    def _lines(body: str) -> list[str]:
        """Split a section body into clean lines, drop comments + blanks."""
        out = []
        for ln in body.splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
        return out

    @staticmethod
    def _parse_exchanges(body: str) -> list[dict]:
        """Parse 'User: X\nTelp: Y' blocks into {user, telp} dicts."""
        pairs = []
        user_line = None
        for ln in body.splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            if s.lower().startswith("user:"):
                user_line = s[5:].strip()
            elif s.lower().startswith("telp:") and user_line is not None:
                telp_line = s[5:].strip()
                pairs.append({"user": user_line, "telp": telp_line})
                user_line = None
        return pairs

    # ── Pickers (with anti-repetition) ────────────────────────

    def _pick(self, pool: list[str], cat: str) -> Optional[str]:
        if not pool:
            return None
        recent = self._recent[cat]
        # Try up to 6 times to find a non-recent pick
        for _ in range(6):
            cand = self.rng.choice(pool)
            if cand not in recent:
                recent.append(cand)
                return cand
        # All recent — just take one and recycle
        cand = self.rng.choice(pool)
        recent.append(cand)
        return cand

    # ── Composition ────────────────────────────────────────────

    def compose(self,
                      raw_answer: str,
                      user_text: str,
                      intent: str,
                      context: "ConversationContext",
                      topic: Optional[str] = None,
                      related: Optional[str] = None,
                      memory_hit: Optional[str] = None,
                      include_invite: bool = True) -> str:
        """Wrap a raw answer in fluent conversation shape.

        intent           — define / story / explain / recall / imagine /
                            relation / unknown / greeting / etc.
        topic            — main subject word (used in acknowledgments)
        related          — adjacent concept (used in connections)
        memory_hit       — line Telp recalled (used in connection)
        include_invite   — set False for closings (goodbye/thanks)

        Some intents are NOT wrapped:
          - greeting / farewell / thanks / acknowledgment-only:
            those came from SelfModel and are already social acts;
            wrapping them would be silly.
        """
        # Bypass for short social acts — self_model output already
        # has the right shape.
        if intent in ("greeting", "farewell", "thanks", "social"):
            return raw_answer
        # Bypass if the raw answer is itself very short and definitive
        # (e.g. brevity-pattern style).
        if len(raw_answer) < 16 and "?" not in raw_answer:
            return raw_answer

        # Extract topic from user_text if not given
        if topic is None:
            topic = self._extract_topic(user_text)

        # Build the wrapped response
        parts: list[str] = []

        # 1. Acknowledgment (if we have a topic to anchor it).
        # Capitalize at sentence start; drop the article if the
        # template starts with "A " for an abstract / mass noun
        # ("Loneliness?" reads better than "A loneliness?").
        ack_pat = self._pick(self.ack_patterns, "ack")
        if ack_pat and topic:
            try:
                ack = ack_pat.format(topic=topic)
            except KeyError:
                ack = ack_pat   # pattern referenced unknown slot
            # Capitalize first letter
            if ack and ack[0].islower():
                ack = ack[0].upper() + ack[1:]
            # "A loneliness?" -> "Loneliness?"  Heuristic: if the topic
            # ends in -ness / -ity / -ship / -hood / -ment / -tion or
            # is plural-ish, drop a leading "A "/"An ".
            abstract_suffixes = ("ness", "ity", "ship", "hood",
                                            "ment", "tion", "sion",
                                            "ance", "ence")
            if (any(topic.lower().endswith(s)
                          for s in abstract_suffixes)
                    and (ack.startswith("A ") or ack.startswith("An "))):
                ack = ack.split(" ", 1)[1]
                ack = ack[0].upper() + ack[1:]
            parts.append(ack)

        # 2. Body — the raw answer.  If it starts with a definitional
        # opener ("A frog (noun) is: ..."), trim the part-of-speech tag
        # AND lowercase the first letter of what follows so the wrapped
        # form flows naturally: "Frog — yes.  A frog is any of various
        # small amphibians." rather than the broken "A frog is Any of
        # various...".
        body = raw_answer
        m = re.match(
            r"^(A|An|The)\s+(\w+)\s+\(\w+\)\s+is:?\s*(.*)",
            body, flags=re.DOTALL)
        if m:
            art, headword, rest = m.group(1), m.group(2), m.group(3)
            if rest and rest[0].isupper():
                rest = rest[0].lower() + rest[1:]
            body = f"{art} {headword} is {rest}".rstrip()
        # Also handle "As a verb, X means: Y" -> "As a verb, X means y"
        body = re.sub(
            r"As (a|an) (verb|noun|adj|adverb), (\w+) means:\s*([A-Z])",
            lambda m: (f"As {m.group(1)} {m.group(2)}, {m.group(3)} "
                              f"means {m.group(4).lower()}"),
            body)
        parts.append(body)

        # 3. Connection (if we have something to connect to)
        if related and self.connect_patterns:
            connect_pat = self._pick(self.connect_patterns, "connect")
            if connect_pat:
                try:
                    connect = connect_pat.format(related=related)
                    parts.append(connect)
                except KeyError:
                    pass
        elif memory_hit and "remember" not in body.lower():
            # Memory recall already in body — skip
            pass

        # 4. Invitation (last)
        if include_invite and self.invite_patterns:
            invite_pat = self._pick(self.invite_patterns, "invite")
            if invite_pat:
                try:
                    invite = invite_pat.format(
                        related=related or topic or "more")
                    parts.append(invite)
                except KeyError:
                    parts.append(invite_pat)

        # Join with spaces; smooth double punctuation
        wrapped = "  ".join(p.rstrip() for p in parts if p)
        wrapped = re.sub(r"\s+", " ", wrapped)
        wrapped = re.sub(r"\s*([.,!?])", r"\1", wrapped)
        wrapped = re.sub(r"([.!?])\s+", r"\1  ", wrapped)
        return wrapped.strip()

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _extract_topic(user_text: str) -> Optional[str]:
        """Pull the first content word from the user's question."""
        low = user_text.lower().rstrip("?!.")
        # Strip common question prefixes
        for prefix in ("tell me about a ", "tell me about an ",
                              "tell me about the ", "tell me about ",
                              "what is a ", "what is an ", "what is the ",
                              "what is ", "what's a ", "what's an ",
                              "what's the ", "what's ", "whats ",
                              "what are ", "who is ", "who's ",
                              "define ", "explain ", "do you know ",
                              "do you remember ", "make up a story about ",
                              "tell me a story about ", "why does ",
                              "why do ", "how do "):
            if low.startswith(prefix):
                low = low[len(prefix):]
                break
        # Strip leading article on the remainder
        for art in ("a ", "an ", "the "):
            if low.startswith(art):
                low = low[len(art):]
                break
        words = re.findall(r"[a-z]+", low)
        if not words:
            return None
        # Skip stopwords; return first content word
        stop = {"is", "are", "was", "were", "do", "does", "did",
                       "of", "in", "on", "at", "to", "for", "with",
                       "and", "or", "but", "you", "your", "i", "me"}
        for w in words:
            if w not in stop:
                # Topic is usually a noun; pluralize lightly for natural
                # acknowledgment ("Frogs — yes." vs "Frog — yes.")
                return w
        return None

    def stats(self) -> dict:
        return {
            "principles":          len(self.principles),
            "ack_patterns":        len(self.ack_patterns),
            "connect_patterns":    len(self.connect_patterns),
            "invite_patterns":     len(self.invite_patterns),
            "brevity_patterns":    len(self.brevity_patterns),
            "example_exchanges":   len(self.example_exchanges),
            "path":                str(self.path),
        }


# ─── Rolling conversational context ──────────────────────────────


class ConversationContext:
    """Tracks the last few turns so the engine knows what's being
    discussed.  Used for topic continuity ('tell me more').
    """

    def __init__(self, max_turns: int = 4):
        self.turns: deque = deque(maxlen=max_turns)
        self.topics: deque = deque(maxlen=max_turns)

    def push(self, user_text: str, telp_response: str,
                  topic: Optional[str] = None) -> None:
        self.turns.append({"user": user_text, "telp": telp_response})
        if topic:
            self.topics.append(topic)

    def current_topic(self) -> Optional[str]:
        return self.topics[-1] if self.topics else None

    def previous_topic(self) -> Optional[str]:
        if len(self.topics) < 2:
            return None
        return self.topics[-2]

    def is_followup(self, user_text: str) -> bool:
        """True if user_text is a follow-up like 'tell me more'."""
        low = user_text.lower().strip().rstrip("?!.")
        return low in {
            "tell me more", "more", "go on", "continue",
            "keep going", "and?", "and then?", "more please",
            "tell more", "more please", "elaborate",
        } or low.startswith(("more about", "tell me more about",
                                       "and what about", "what about that"))


# ─── CLI smoke ────────────────────────────────────────────────────


if __name__ == "__main__":
    ck = ConversationKnowledge()
    print(f"Loaded: {ck.stats()}")
    print()
    print(f"Principles ({len(ck.principles)}):")
    for p in ck.principles[:5]:
        print(f"  - {p}")
    print()
    print(f"Acknowledgment patterns ({len(ck.ack_patterns)}):")
    for p in ck.ack_patterns[:5]:
        print(f"  - {p}")
    print()
    print(f"Example exchanges ({len(ck.example_exchanges)}):")
    for e in ck.example_exchanges[:3]:
        print(f"  User: {e['user']}")
        print(f"  Telp: {e['telp']}")
        print()

    # Try a composition
    ctx = ConversationContext()
    print("Composition demo:")
    raw = "A frog is a small tailless amphibian that hops."
    composed = ck.compose(
        raw_answer=raw,
        user_text="what is a frog?",
        intent="define",
        context=ctx,
        related="amphibian",
    )
    print(f"  raw:      {raw}")
    print(f"  composed: {composed}")
