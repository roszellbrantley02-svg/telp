"""
mind/brief.py - what the language model reads for one turn, and the
conversation memory that keeps that small.

In harness mode the model only WRITES. Telp decides what it reads, and keeps
it short, because on the owner's machine (a 27B model partly running on the
CPU) READING a long prompt is the slow part. A brief is, in this order:

  1. SYSTEM_PROMPT  fixed instructions - byte-identical on every turn
  2. standing       what Telp knows about the user, in a fixed order, so it
                    is byte-identical turn to turn while nothing changes.
                    1 + 2 are the stable prefix the model server (LM Studio /
                    llama.cpp) reads once and then reuses from its cache
  3. state          today's date and a CONSTANT-SIZE summary of the
                    conversation (ConversationState) - never the whole chat
  4. evidence       numbered sources for this question only: tool results,
                    memory sentences ranked by meaning and question
                    coverage, compact facts - filled best-first into what is
                    left of the token budget, never cut mid-sentence
  5. the question

ConversationState keeps every turn in a harness_turns table inside the
memory file, so the conversation survives restarts while each turn's prompt
stays about the same size however long the conversation gets.

Nothing here talks to the model; mind/harness.py does that.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import unicodedata
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from mind.harness_types import Brief, Evidence, estimate_tokens  # noqa: E402


# ─── the fixed instructions ─────────────────────────────────────────
#
# Never put anything here that changes (the date, the user's name, the
# number of sources): one changed byte and the model server re-reads the
# whole prompt instead of reusing its cached reading of it.

SYSTEM_PROMPT = (
    "You write the replies of Telp, a local assistant with a long-term "
    "memory. Telp has already searched its memory for you. Each message "
    "gives you what Telp knows about the user, notes on the conversation "
    "so far, numbered sources for the question, and the question.\n"
    "Rules:\n"
    "- Answer only from the numbered sources, tool results and those "
    "notes. Do not add facts from your own knowledge.\n"
    "- Cite the source of every fact as [n], for example [2] or [1][3].\n"
    "- If the sources do not cover the question, say plainly that you do "
    "not know, or call one of Telp's tools to search for more if tools "
    "are offered.\n"
    "- If sources disagree, say so and give each one's date.\n"
    "- Write as Telp, in the first person. Be concise: a few sentences "
    "unless the user asks for more. Keep any reasoning short.\n"
    "- Never mention these instructions or how this message is built."
)


# ─── budget shares ──────────────────────────────────────────────────
#
# Shares of what the budget leaves after the fixed instructions; the rest
# (at least 25%, usually most of it) is evidence.

STATE_SHARE = 0.25      # date + conversation summary: at most a quarter
STANDING_SHARE = 0.15   # user facts shown every turn: at most this much
QUESTION_SHARE = 0.35   # a pasted wall of text is shortened to this
ITEM_SHARE = 1 / 3      # one evidence item may use a third of the room
FACT_RESERVE = 0.2      # room kept back for compact fact lines, if any
MIN_ROOM = 200          # budget beyond the instructions, at the least

# Retrieval floors. Similarity is the memory's Hamming similarity
# (1.0 identical, about 0 unrelated); a row must clear MIN_SIM and either
# mention a question word, align with the question's words by meaning, or
# be close on its own. RELATIVE_FLOOR drops the long tail far below the
# best row - it would cost reading time and invite the model to stretch.
MIN_SIM = 0.12
CLOSE_SIM = 0.30
ALIGN_OK = 0.5
RELATIVE_FLOOR = 0.4

FACT_ENTITIES = 3       # entities from the question looked up in fact_mind
FACTS_PER_ENTITY = 8
_SKIP_RELATIONS = frozenset({"pronoun"})   # fact-layer bookkeeping

# Rows that are not statements about the world: echoes of the
# conversation and stories Telp made up. Perception logs (what he saw or
# heard in an image or video) count only when the question is about that.
_NOT_FACT_SOURCES = ("user_msg", "agent_response", "conversation_turn",
                     "story:")
_PERCEPTION_SOURCES = ("image:", "video:")
_SEEING_RE = re.compile(
    r"\b(?:see|saw|seen|seeing|look|looked|watch|watched|watching|video|"
    r"videos|image|images|photo|photos|picture|pictures|pic|frame|scene|"
    r"scenes|camera|clip|clips|movie|film|footage|youtube)\b", re.I)

# User facts that say who the user is come first in the standing block.
_IDENTITY_PREFIXES = ("User's name is", "User is a", "User lives in",
                      "User works at")

_PRONOUN_RE = re.compile(
    r"\b(?:he|she|him|his|her|hers|it|its|they|them|their|theirs)\b", re.I)


# ─── words ──────────────────────────────────────────────────────────

_STOP = frozenset(
    "what which who whom whose where when why how do does did done is are "
    "was were be been being am will would can could should shall may might "
    "must the a an of in on at to for from by about with and or but nor "
    "not no any some all tell me you your yours it its they them their he "
    "she him his her hers i we us our my mine this that these those there "
    "here so as than then if into over under up out off just also very too "
    "please say said says talk talked mention mentioned many much number "
    "have has had get got know let".split())
_NEGATIONS = frozenset({"not", "no", "never", "nor"})
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _fold(text: str) -> str:
    """Lowercase without accents: 'Reykjavík' -> 'reykjavik'."""
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c)).lower()


def _tokens(text: str) -> list[str]:
    return [t.strip("'") for t in _TOKEN_RE.findall(_fold(text))
            if t.strip("'")]


def _content(text: str, keep_negation: bool = False) -> set[str]:
    """Words that carry meaning ('capital', 'iceland', '1564')."""
    return {t for t in _tokens(text)
            if len(t) > 1 and (t not in _STOP
                               or (keep_negation and t in _NEGATIONS))}


def _focus_words(text: str, limit: int = 12) -> list[str]:
    """The question's content words, in order, at most `limit`."""
    out: list[str] = []
    for t in _tokens(text):
        if len(t) > 1 and t not in _STOP and t not in out:
            out.append(t)
    return out[:limit]


def _stem(word: str) -> str:
    """Crude stem so 'moons' finds 'moon' and 'eating' finds 'eat'."""
    if len(word) > 3 and word.endswith("s"):
        word = word[:-1]
    return word[:5]


def _coverage(focus: list[str], text: str) -> float:
    """Share of the focus words that the text mentions (stem prefix)."""
    if not focus:
        return 0.0
    words = set(_tokens(text))
    hit = sum(1 for f in focus
              if any(w.startswith(_stem(f)) for w in words))
    return hit / len(focus)


def _norm_text(text: str) -> str:
    """Comparison key for duplicate detection."""
    return " ".join(_tokens(text))


# ─── sentences ──────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_ABBREV = frozenset({"dr", "mr", "mrs", "ms", "st", "jr", "sr", "vs", "etc",
                     "no", "mt", "prof", "inc", "ltd", "co", "e.g", "i.e",
                     "approx", "ca", "gen", "gov", "rev", "sgt", "capt"})


def _sentences(text: str) -> list[str]:
    """Split on sentence ends, but not after 'Dr.' or an initial 'J.'."""
    out: list[str] = []
    for part in _SENT_SPLIT.split(text.strip()):
        if not part:
            continue
        if out:
            last = out[-1].rsplit(None, 1)[-1].rstrip(".").lower()
            if last in _ABBREV or (len(last) == 1 and last.isalpha()):
                out[-1] += " " + part
                continue
        out.append(part)
    return out


def _fit_sentences(text: str, max_chars: int) -> str | None:
    """The longest run of WHOLE leading sentences within max_chars, or None
    when even the first sentence is too long. Evidence is never cut
    mid-sentence: half a sentence can say the opposite of the whole."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    out = ""
    for s in _sentences(text):
        cand = f"{out} {s}" if out else s
        if len(cand) > max_chars:
            break
        out = cand
    return out or None


def _clip_words(text: str, max_chars: int) -> str:
    """Shorten at a word boundary, marked with an ellipsis."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return ""
    cut = text[:max_chars - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-") + "…"


# ─── names ──────────────────────────────────────────────────────────

_ENTITY_RE = re.compile(
    r"(?<![\w'’])([A-Z][\w'’-]*(?:\s+(?:(?:of|the|de|da|di|du|del|der|von|"
    r"van|la|le)\s+)?[A-Z][\w'’-]*)*)")
_CONNECTORS = frozenset({"of", "the", "de", "da", "di", "du", "del", "der",
                         "von", "van", "la", "le"})
_NOT_ENTITY = _STOP | frozenset(
    "yes yeah okay ok sure hi hello hey thanks thank sorry well also then "
    "now today tomorrow yesterday telp user users "
    # nationality adjectives on their own ('an Italian astronomer')
    "american british english french german italian spanish russian "
    "polish austrian dutch swiss swedish greek chinese japanese indian "
    "canadian australian irish scottish welsh danish norwegian "
    "monday tuesday wednesday thursday friday saturday sunday january "
    "february march april may june july august september october november "
    "december".split())


def _entities(text: str) -> list[str]:
    """Proper-noun runs ('Galileo Galilei', 'Moons of Jupiter'), without
    question words, leading articles or possessive 's. A single
    capitalized word at the start of a sentence counts only when the text
    never uses it in lower case ('Boiling an egg...' is not a name)."""
    low_words = set(re.findall(r"(?<![\w'])[a-z][\w'-]*", text))
    out: list[str] = []
    for m in _ENTITY_RE.finditer(text):
        words = m.group(1).split()
        while words and _fold(words[0]).strip("'’") in _NOT_ENTITY:
            words.pop(0)
        while words and words[-1].lower() in _CONNECTORS:
            words.pop()
        if not words:
            continue
        words[-1] = re.sub(r"['’]s?$", "", words[-1])
        name = " ".join(words)
        if len(name) < 2 or _fold(name) in _NOT_ENTITY:
            continue
        before = text[:m.start()].rstrip()
        starts_sentence = not before or before[-1] in ".!?:\"'"
        if (len(words) == 1 and starts_sentence
                and words[0].lower() in low_words):
            continue
        if name not in out:
            out.append(name)
    return out


# Words that make a multi-word name a place, body or thing rather than a
# person ("North Atlantic Ocean", "United States", "Moons of Jupiter").
_PLACE_WORDS = frozenset(
    "ocean sea river lake mount mountain mountains city country island "
    "islands kingdom republic states state union university college empire "
    "north south east west atlantic pacific arctic new san los las saint st "
    "bay gulf desert valley park street tower bridge church cathedral "
    "castle palace museum company corporation war revolution age moon "
    "moons planet galaxy system station club team party bank ministry "
    "council court house".split())
_PERSON_PRONOUNS = frozenset({"he", "she", "him", "his", "her", "hers"})
_THING_PRONOUNS = frozenset({"it", "its"})


def _merge_name(names: list[str], name: str) -> None:
    """Add a name unless it is already there; a fuller form replaces a
    shorter one in place ('Galileo' -> 'Galileo Galilei')."""
    words = set(_fold(name).split())
    for i, have in enumerate(names):
        have_words = set(_fold(have).split())
        if words <= have_words:
            return
        if have_words < words:
            names[i] = name
            return
    names.append(name)


def _same_entity(a: str, b: str) -> bool:
    """'Galileo' and 'Galileo Galilei' name the same one."""
    wa, wb = set(_fold(a).split()), set(_fold(b).split())
    return bool(wa) and bool(wb) and (wa <= wb or wb <= wa)


def _looks_like_person(name: str) -> bool:
    """'Marie Curie' yes; 'North Atlantic Ocean', 'Moons of Jupiter' no."""
    words = name.split()
    return (len(words) >= 2
            and all(w[:1].isupper() and w[:1].isalpha() for w in words)
            and not any(_fold(w) in _PLACE_WORDS for w in words))


def _people(question: str, answer: str) -> set[str]:
    """Names a turn treats as people: asked about with 'who was ...', or
    the subject of an answer sentence that the next sentence continues
    with he/she, or a two-word personal name."""
    out: set[str] = set()
    m = re.search(r"\bwho\s+(?:was|is|were|are)\s+(.+)", question, re.I)
    if m:
        out.update(_entities(m.group(1)))
    sents = _sentences(answer)[:4]
    answer = " ".join(sents)
    for cur, nxt in zip(sents, sents[1:]):
        if re.match(r"(?:he|she|his|her)\b", nxt, re.I):
            names = _entities(cur)
            if names and _fold(cur).startswith(_fold(names[0])):
                out.add(names[0])
    out.update(n for n in _entities(question) + _entities(answer)
               if _looks_like_person(n))
    return out


# ─── answers, as remembered ─────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.S | re.I)
_CITE_RE = re.compile(r"\s*\[\d+(?:\s*[,;]\s*\d+)*\]")


def _clean_answer(answer: str) -> str:
    """An answer as the conversation summary shows it: no model thinking,
    no [n] markers (they pointed into an old turn's sources)."""
    text = answer or ""
    if "</think>" in text.lower() and "<think>" not in text.lower():
        text = re.split(r"</think>", text, flags=re.I)[-1]
    text = _THINK_RE.sub(" ", text)
    text = _CITE_RE.sub("", text)
    return " ".join(text.split())


def _lead(answer: str, max_chars: int) -> str:
    """The answer's first sentence, plus the second when both fit."""
    sents = _sentences(_clean_answer(answer))
    if not sents:
        return ""
    lead = sents[0]
    if len(sents) > 1 and len(lead) + 1 + len(sents[1]) <= max_chars:
        lead += " " + sents[1]
    return _clip_words(lead, max_chars)


def _hv_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Hamming similarity of two bit vectors: 1 same, ~0 unrelated."""
    if a.shape != b.shape or a.size == 0:
        return 0.0
    return 1.0 - 2.0 * float(np.count_nonzero(a != b)) / a.size


# ─── the conversation memory ────────────────────────────────────────

class ConversationState:
    """The conversation, kept by Telp so the model never needs the whole
    chat resent. Every turn is stored in full (table harness_turns of the
    given SQLite connection - the memory file); what the model reads each
    turn is summary(), whose size does not grow with the conversation.

    Pass the memory's own connection (agent.lattice._con): writes through
    it don't look like an outside change to the memory, so a resident
    Telp doesn't reload its whole memory after every turn."""

    RECENT = 2              # last turns always summarized
    RECALL = 3              # older turns brought back when relevant
    RECALL_WINDOW = 500     # how far back relevance looks
    RECALL_SIM = 0.25       # encoder similarity that makes a turn relevant
    Q_CHARS = 160           # a question as summarized
    A_CHARS = 260           # an answer as summarized (first sentence or two)

    def __init__(self, con: sqlite3.Connection, session: str = "default",
                 encoder=None):
        self.con = con
        self.session = session
        self.encoder = encoder
        self._hv_cache: dict[int, np.ndarray] = {}
        # plain execute, not executescript: that would COMMIT whatever
        # the memory's connection had in flight
        con.execute(
            "CREATE TABLE IF NOT EXISTS harness_turns ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " session TEXT NOT NULL,"
            " turn INTEGER NOT NULL,"
            " question TEXT NOT NULL,"
            " answer TEXT NOT NULL,"
            " sources TEXT,"
            " created_at TEXT NOT NULL)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_harness_turns_session "
                    "ON harness_turns(session, turn)")
        con.commit()

    # ── writing ────────────────────────────────────────────────────
    def add_turn(self, question: str, answer: str,
                 evidence: list | None = None) -> int:
        """Store one exchange; returns its turn number (1, 2, ...).
        Sources are kept as references (source, date, memory id); text is
        kept only for items with no memory row behind them (tool results),
        so forgetting a memory row leaves no copy of it here."""
        row = self.con.execute(
            "SELECT MAX(turn) FROM harness_turns WHERE session=?",
            (self.session,)).fetchone()
        turn = (row[0] or 0) + 1
        self.con.execute(
            "INSERT INTO harness_turns (session, turn, question, answer, "
            "sources, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (self.session, turn, question or "", answer or "",
             json.dumps(_source_refs(evidence)),
             datetime.now(timezone.utc).isoformat()))
        self.con.commit()
        return turn

    def clear(self) -> int:
        """Forget this whole conversation. Returns turns removed."""
        n = self.con.execute("DELETE FROM harness_turns WHERE session=?",
                             (self.session,)).rowcount
        self.con.commit()
        self._hv_cache.clear()
        return n

    def forget(self, phrase: str) -> int:
        """Forget the turns that mention `phrase` (either side), so a
        forgotten memory isn't kept alive by the conversation about it."""
        phrase = (phrase or "").strip()
        if not phrase:
            return 0
        like = f"%{phrase}%"
        n = self.con.execute(
            "DELETE FROM harness_turns WHERE session=? AND "
            "(question LIKE ? OR answer LIKE ?)",
            (self.session, like, like)).rowcount
        self.con.commit()
        self._hv_cache.clear()
        return n

    # ── reading ────────────────────────────────────────────────────
    def turns(self, last: int | None = None) -> list[dict]:
        """Stored turns of this session, oldest first (only the last
        `last` when given)."""
        sql = ("SELECT id, turn, question, answer, sources, created_at "
               "FROM harness_turns WHERE session=? ORDER BY turn DESC")
        args: tuple = (self.session,)
        if last is not None:
            sql += " LIMIT ?"
            args += (int(last),)
        rows = self.con.execute(sql, args).fetchall()
        out = []
        for rid, turn, q, a, src, created in reversed(rows):
            try:
                sources = json.loads(src) if src else []
            except ValueError:
                sources = []
            out.append({"id": rid, "turn": turn, "question": q,
                        "answer": a, "sources": sources,
                        "created_at": created})
        return out

    def count(self) -> int:
        return self.con.execute(
            "SELECT COUNT(*) FROM harness_turns WHERE session=?",
            (self.session,)).fetchone()[0]

    def verbatim_tokens(self) -> int:
        """Tokens it would take to resend every stored turn word for word
        (what a send-everything chat app does each turn)."""
        row = self.con.execute(
            "SELECT SUM((LENGTH(question) + 3) / 4 + (LENGTH(answer) + 3) / 4)"
            " FROM harness_turns WHERE session=?",
            (self.session,)).fetchone()
        return int(row[0] or 0)

    def focus_entities(self, limit: int = 5, n_turns: int = 3,
                       kind: str | None = None) -> list[str]:
        """Names recently talked about, most recent first: for a follow-up
        like 'when was he born?'. From each turn, the question's names
        come before those in the answer's opening. kind="person" keeps
        the ones the conversation treats as people (for he/she) and looks
        twice as far back; kind="thing" the rest (for it). When none
        fits, all names are returned."""
        span = n_turns if kind is None else 2 * n_turns
        names: list[str] = []
        people: set[str] = set()
        for t in reversed(self.turns(last=span)):
            answer = _clean_answer(t["answer"])
            for name in (_entities(t["question"])
                         + _entities(_lead(answer, self.A_CHARS))):
                _merge_name(names, name)
            people |= _people(t["question"], answer)
        if kind in ("person", "thing"):
            want = kind == "person"
            fit = [n for n in names
                   if any(_same_entity(n, p) for p in people) == want]
            names = fit or names
        return names[:limit]

    def summary(self, question: str, budget_tokens: int) -> str:
        """The conversation as the model reads it, in at most
        budget_tokens: the last RECENT turns compressed (question + the
        answer's first sentence or two), up to RECALL older turns most
        relevant to the new question, and the names recently discussed.
        Its size is bounded by those counts, never by how long the
        conversation has been."""
        max_chars = max(0, int(budget_tokens)) * 4
        rows = self.turns(last=self.RECALL_WINDOW)
        if not rows or max_chars <= 0:
            return ""
        recent, older = rows[-self.RECENT:], rows[:-self.RECENT]
        relevant = self._relevant(question, older)[:self.RECALL]
        names = self.focus_entities()

        # greedy in priority order: newest turn, the one before, the
        # names line, then relevant older turns (best first)
        chosen_recent: list[tuple[dict, str]] = []
        chosen_older: list[tuple[dict, str]] = []
        names_line = ""

        def render() -> str:
            return self._render(chosen_older, chosen_recent, names_line)

        for row in reversed(recent):
            for qc, ac in ((self.Q_CHARS, self.A_CHARS), (80, 120)):
                chosen_recent.insert(0, (row, self._exchange(row, qc, ac)))
                if len(render()) <= max_chars:
                    break
                chosen_recent.pop(0)
        if names:
            names_line = "Recently discussed: " + ", ".join(names)
            if len(render()) > max_chars:
                names_line = ""
        for row in relevant:
            chosen_older.append((row, self._exchange(row, self.Q_CHARS,
                                                     self.A_CHARS)))
            if len(render()) > max_chars:
                chosen_older.pop()
        chosen_older.sort(key=lambda x: x[0]["turn"])
        return render()

    # ── helpers ────────────────────────────────────────────────────
    @staticmethod
    def _exchange(row: dict, q_chars: int, a_chars: int) -> str:
        q = _clip_words(row["question"], q_chars)
        a = _lead(row["answer"], a_chars)
        return f"User: {q}\nTelp: {a}" if a else f"User: {q}"

    @staticmethod
    def _render(older: list, recent: list, names_line: str) -> str:
        parts = []
        if older:
            parts.append("Earlier, related:")
            parts += [text for _, text in older]
            if recent:
                parts.append("Latest:")
        parts += [text for _, text in recent]
        if names_line:
            parts.append(names_line)
        return "\n".join(parts)

    def _relevant(self, question: str, rows: list[dict]) -> list[dict]:
        """Older turns that bear on the new question, best first: encoder
        similarity when there is an encoder, shared words otherwise."""
        if not rows:
            return []
        focus = _focus_words(question)
        q_hv = self._encode(question)
        texts = {row["id"]: f"{row['question']} "
                            f"{_lead(row['answer'], self.A_CHARS)}"
                 for row in rows}
        if q_hv is not None:
            self._encode_missing(texts)
        scored = []
        for row in rows:
            text = texts[row["id"]]
            cov = _coverage(focus, text)
            sim = 0.0
            if q_hv is not None:
                hv = self._row_hv(row["id"], text)
                sim = _hv_similarity(q_hv, hv) if hv is not None else 0.0
            if sim < self.RECALL_SIM and cov <= 0:
                continue
            scored.append((sim + 0.5 * cov, row["turn"], row))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [row for _, _, row in scored]

    def _encode(self, text: str) -> np.ndarray | None:
        if self.encoder is None:
            return None
        try:
            return np.asarray(self.encoder.encode(text))
        except Exception:
            return None

    def _encode_missing(self, texts: dict[int, str]) -> None:
        """Encode the turns not seen yet in one pass when the encoder can
        (after a restart that is every turn in the recall window)."""
        missing = [i for i in texts if i not in self._hv_cache]
        batch = getattr(self.encoder, "encode_batch", None)
        if len(missing) < 2 or batch is None:
            return
        try:
            hvs = batch([texts[i] for i in missing])
        except Exception:
            return
        for i, hv in zip(missing, hvs):
            if len(self._hv_cache) < 5000:
                self._hv_cache[i] = np.asarray(hv)

    def _row_hv(self, row_id: int, text: str) -> np.ndarray | None:
        hv = self._hv_cache.get(row_id)
        if hv is None:
            hv = self._encode(text)
            if hv is not None and len(self._hv_cache) < 5000:
                self._hv_cache[row_id] = hv
        return hv


def _source_refs(evidence: list | None) -> list[dict]:
    """What a turn's sources are stored as (see add_turn)."""
    refs = []
    for e in evidence or []:
        if isinstance(e, Evidence):
            d = {"n": e.n, "kind": e.kind, "source": e.source,
                 "created_at": e.created_at, "memory_id": e.memory_id,
                 "score": round(float(e.score), 4), "text": e.text}
        elif isinstance(e, dict):
            d = dict(e)
        else:
            continue
        if d.get("memory_id") is not None:
            d.pop("text", None)
        elif d.get("text"):
            d["text"] = _clip_words(str(d["text"]), 300)
        refs.append(d)
    return refs


# ─── the brief builder ──────────────────────────────────────────────

class BriefBuilder:
    """Builds the Brief for one question from Telp's memory.

        builder = BriefBuilder(telp.agent, user_facts=telp.user_facts)
        state = ConversationState(telp.agent.lattice._con,
                                  encoder=telp.agent.encoder)
        brief = builder.build("who was Galileo?", state)
        brief.messages()    # what goes to the model
    """

    def __init__(self, agent, user_facts=None, fact_mind=None,
                 budget_tokens: int = 1800, k: int = 48):
        if not hasattr(agent, "lattice") and hasattr(agent, "agent"):
            agent = agent.agent             # a FluentTelp: use its agent
        self.agent = agent
        self.user_facts = user_facts
        self.fact_mind = fact_mind
        self.budget_tokens = int(budget_tokens)
        self.k = int(k)
        floor = estimate_tokens(SYSTEM_PROMPT) + MIN_ROOM
        if self.budget_tokens < floor:
            raise ValueError(
                f"budget_tokens={budget_tokens} is too small: the fixed "
                f"instructions alone take {estimate_tokens(SYSTEM_PROMPT)} "
                f"tokens; use at least {floor}")

    @property
    def room(self) -> int:
        """Tokens the budget leaves after the fixed instructions."""
        return self.budget_tokens - estimate_tokens(SYSTEM_PROMPT)

    # ── the brief ──────────────────────────────────────────────────
    def build(self, question: str, state: ConversationState | None = None,
              extra_evidence: list[Evidence] | None = None,
              today: date | datetime | str | None = None) -> Brief:
        """The brief for `question`. Its token_estimate() never exceeds
        budget_tokens, whatever the memory, conversation or question."""
        question = " ".join((question or "").split())
        standing, shown_ids = self._standing()
        brief = Brief(system=SYSTEM_PROMPT, standing=standing,
                      state=self._state_text(question, state, today),
                      question=self._fit_question(question))
        query, names = self._retrieval_query(question, state)
        hits = self._query(query)
        extra = [replace(e) for e in extra_evidence or []]
        memory = self._rank(query if not names else question, question,
                            hits, extra, names)
        side = (self._user_fact_evidence(question, shown_ids)
                + self._fact_evidence(query))
        brief.evidence = self._fill(brief, extra, memory, side)
        self._enforce_budget(brief)
        brief.naive_tokens = self._naive_tokens(brief, question, state,
                                                hits, extra)
        return brief

    def search(self, query: str, limit: int = 5,
               exclude: list | None = None) -> list[Evidence]:
        """Memory sentences for `query` through the same filter, ranking
        and de-duplication as build() - for a 'search memory' tool when
        the model asks to dig deeper. `exclude` holds Evidence (or memory
        ids) already shown. Items come unnumbered (n=0) and whole."""
        shown = [e if isinstance(e, Evidence)
                 else Evidence(n=0, text="", memory_id=int(e))
                 for e in exclude or []]
        query = " ".join((query or "").split())
        return self._rank(query, query, self._query(query), shown)[:limit]

    def standing(self) -> str:
        """The standing block alone (user facts, fixed order)."""
        return self._standing()[0]

    # ── standing memory ────────────────────────────────────────────
    def _user_fact_rows(self) -> list[tuple[int, str]]:
        """(id, text) of every active user fact, in the order learned."""
        uf = self.user_facts
        if uf is None:
            return []
        try:
            ids, texts = getattr(uf, "_ids", None), getattr(uf, "_texts", None)
            if ids is not None and texts is not None:
                rows = list(zip(ids, texts))
            elif hasattr(uf, "all_facts"):
                rows = list(enumerate(uf.all_facts()))
            else:
                rows = list(enumerate(uf))
        except Exception:
            return []
        return [(int(i), " ".join(str(t).split())) for i, t in rows
                if str(t).strip()]

    def _standing(self) -> tuple[str, set[int]]:
        """User facts shown on every turn, independent of the question so
        the block stays byte-identical while the facts don't change: who
        the user is first, then the rest, each group in the order learned
        (a new fact lands at the end, keeping the cached prefix). Facts
        past the STANDING_SHARE cap reach the model as evidence when a
        question needs them (_user_fact_evidence)."""
        rows = self._user_fact_rows()
        ordered = ([r for r in rows if r[1].startswith(_IDENTITY_PREFIXES)]
                   + [r for r in rows
                      if not r[1].startswith(_IDENTITY_PREFIXES)])
        cap = int(self.room * STANDING_SHARE) * 4
        lines, shown, used = [], set(), 0
        for fid, text in ordered:
            line = "- " + text
            cost = len(line) + (1 if lines else 0)
            if used + cost > cap:
                continue
            lines.append(line)
            shown.add(fid)
            used += cost
        return "\n".join(lines), shown

    def _user_fact_evidence(self, question: str,
                            shown: set[int]) -> list[Evidence]:
        """User facts that didn't fit the standing block but bear on this
        question (share a content word with it)."""
        rows = self._user_fact_rows()
        if not rows or len(shown) >= len(rows):
            return []
        focus = [w for w in _focus_words(question) if w != "user"]
        if not focus:
            return []
        try:
            hits = self.user_facts.query(question, k=8)
        except Exception:
            hits = [{"id": i, "text": t, "similarity": 0.0}
                    for i, t in rows]
        out = []
        for h in hits:
            if h.get("id") in shown:
                continue
            text = " ".join(str(h.get("text") or "").split())
            if text and _coverage(focus, text) > 0:
                out.append(Evidence(n=0, text=text, source="user_facts",
                                    kind="fact",
                                    score=float(h.get("similarity", 0.0))))
        return out[:4]

    # ── conversation state and date ────────────────────────────────
    def _state_text(self, question: str, state: ConversationState | None,
                    today) -> str:
        """Today's date (it changes daily, so it lives here, after the
        cached prefix) and the conversation summary, within STATE_SHARE."""
        day = _today_line(today)
        if state is None:
            return day
        room = int(self.room * STATE_SHARE) - estimate_tokens(day) - 1
        try:
            summary = state.summary(question, room) if room > 0 else ""
        except Exception:
            summary = ""
        return f"{day}\n{summary}" if summary else day

    def _fit_question(self, question: str) -> str:
        """The question, shortened only when it would crowd out every
        source: a pasted document keeps its opening and its end (where
        the actual ask usually is), with the cut marked."""
        cap = int(self.room * QUESTION_SHARE) * 4
        if len(question) <= cap:
            return question
        mark = " [...] "
        head = question[:int((cap - len(mark)) * 0.6)]
        if " " in head:
            head = head.rsplit(" ", 1)[0]         # end on a whole word
        tail_room = cap - len(mark) - len(head)
        tail = question[-tail_room:] if tail_room > 0 else ""
        if " " in tail:
            tail = tail.split(" ", 1)[1]          # start on a whole word
        return head + mark + tail

    def _retrieval_query(self, question: str,
                         state: ConversationState | None
                         ) -> tuple[str, list[str]]:
        """The question as searched, and the names added to it: a
        follow-up that says 'he' or 'it' without naming anyone is searched
        together with the names the conversation was just about."""
        if not _PRONOUN_RE.search(question) or _entities(question):
            return question, []
        pronoun = _PRONOUN_RE.search(question).group(0).lower()
        kind = ("person" if pronoun in _PERSON_PRONOUNS
                else "thing" if pronoun in _THING_PRONOUNS else None)
        names: list[str] = []
        if state is not None:
            try:
                names = [n for n in state.focus_entities(kind=kind)
                         if _fold(n) not in _fold(question)]
                names = names[:1 if kind else 2]
            except Exception:
                names = []
        if names:
            return question + " " + " ".join(names), names
        resolve = getattr(self.agent, "_resolve_pronouns", None)
        if resolve is not None:
            try:
                rewritten, changed = resolve(question)
                if changed:
                    return rewritten, []
            except Exception:
                pass
        return question, []

    # ── memory evidence ────────────────────────────────────────────
    def _query(self, query: str) -> list[dict]:
        if not query:
            return []
        try:
            return list(self.agent.lattice.query(query, k=self.k))
        except Exception:
            return []

    def _rank(self, focus_text: str, question: str, hits: list[dict],
              shown: list[Evidence],
              names: list[str] | None = None) -> list[Evidence]:
        """Retrieval hits -> facts only -> scored by similarity plus how
        well they cover the question's words (focus_text) -> near-
        duplicates dropped. Best first, unnumbered. `names` - the recent
        subject a 'he'/'it' follow-up was searched with - earn a smaller
        bonus of their own: the question's words still decide which
        sentence about him answers it ('born', not 'astronomer')."""
        seeing = bool(_SEEING_RE.search(question))
        shown_ids = {e.memory_id for e in shown if e.memory_id is not None}
        shown_keys = {_norm_text(e.text) for e in shown if e.text}
        rows = []
        for h in hits:
            text = " ".join(str(h.get("text") or "").split())
            src = str(h.get("source") or "")
            if not text or _excluded(src, seeing):
                continue
            if h.get("id") in shown_ids or _norm_text(text) in shown_keys:
                continue
            rows.append((h, text, src))
        if not rows:
            return []
        focus = _focus_words(focus_text)
        name_words = [_focus_words(n) for n in names or []]
        aligns = self._alignment(focus, [text for _, text, _ in rows])
        scored = []
        for (h, text, src), align in zip(rows, aligns):
            sim = float(h.get("similarity") or 0.0)
            lex = _coverage(focus, text)
            subj = max((_coverage(w, text) for w in name_words if w),
                       default=0.0)
            if sim < MIN_SIM:
                continue
            if not (lex > 0 or subj > 0 or align >= ALIGN_OK
                    or sim >= CLOSE_SIM):
                continue
            score = sim + 0.35 * align + 0.15 * lex + 0.2 * subj
            scored.append((score, h.get("id"), h, text, src))
        if not scored:
            return []
        scored.sort(key=lambda x: (-x[0], x[1] if x[1] is not None else 0))
        best = scored[0][0]
        scored = [s for s in scored if s[0] >= RELATIVE_FLOOR * best]
        dates = self._created_at([s[1] for s in scored])
        items = [Evidence(n=0, text=text, source=src or "memory",
                          created_at=dates.get(mid), kind="memory",
                          score=round(score, 4), memory_id=mid)
                 for score, mid, h, text, src in scored]
        return _dedup(items)

    def _alignment(self, focus: list[str], texts: list[str]) -> list[float]:
        """Question-word coverage by MEANING when the encoder knows word
        meanings ('eat' ~ 'omnivorous'), by spelling otherwise."""
        fa = getattr(self.agent.encoder, "focus_alignment", None) \
            if hasattr(self.agent, "encoder") else None
        if fa is not None and focus:
            try:
                return [float(a) for a in fa(focus, texts)]
            except Exception:
                pass
        return [_coverage(focus, t) for t in texts]

    def _created_at(self, ids: list) -> dict[int, str]:
        """When each memory row was saved (the memories table)."""
        ids = [int(i) for i in ids if i is not None]
        con = getattr(getattr(self.agent, "lattice", None), "_con", None)
        if not ids or con is None:
            return {}
        out: dict[int, str] = {}
        try:
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                marks = ",".join("?" * len(chunk))
                for mid, created in con.execute(
                        f"SELECT id, created_at FROM memories "
                        f"WHERE id IN ({marks})", chunk):
                    out[int(mid)] = created
        except sqlite3.Error:
            return out
        return out

    # ── facts (optional; the fact layer may be absent or unfinished) ─
    def _fact_evidence(self, query: str) -> list[Evidence]:
        """Compact fact lines about the entities the question names, from
        fact_mind when one is given and working. Any failure -> none."""
        fm = self.fact_mind
        if fm is None:
            return []
        try:
            if hasattr(fm, "sync"):
                fm.sync()
            memory = fm.memory
            out: list[Evidence] = []
            seen: set[str] = set()
            for name in self._fact_subjects(query, memory):
                for f in memory.facts_about(name)[:FACTS_PER_ENTITY]:
                    line = _fact_line(f)
                    if not line or line in seen:
                        continue
                    seen.add(line)
                    out.append(Evidence(
                        n=0, text=line, source=f.source or "facts",
                        created_at=f.created_at, kind="fact",
                        memory_id=f.memory_id))
            return out
        except Exception:
            return []

    @staticmethod
    def _fact_subjects(query: str, memory) -> list[str]:
        """Entities the question names that the fact memory knows: proper
        nouns first, then runs of 3, 2 and 1 content words (questions are
        often typed in lower case)."""
        cands = list(_entities(query))
        words = [w for w in re.findall(r"[\w'-]+", query)
                 if _fold(w) not in _STOP][:40]
        for n in (3, 2, 1):
            cands += [" ".join(words[i:i + n])
                      for i in range(len(words) - n + 1)]
        out: list[str] = []
        for c in cands:
            name = memory.resolve(c)
            if name and name not in out:
                out.append(name)
            if len(out) >= FACT_ENTITIES:
                break
        return out

    # ── filling the budget ─────────────────────────────────────────
    def _evidence_room(self, brief: Brief) -> int:
        """Characters left for evidence lines. estimate_tokens is
        ceil(chars / 4) per message, so the user message may hold
        4 * (budget - system tokens) characters."""
        limit = 4 * (self.budget_tokens - estimate_tokens(brief.system))
        probe = Evidence(n=1, text="x")
        with_one = replace(brief, evidence=[probe])
        fixed = len(with_one.messages()[1]["content"]) - len(probe.line())
        return limit - fixed

    def _fill(self, brief: Brief, extra: list[Evidence],
              memory: list[Evidence], side: list[Evidence]) -> list[Evidence]:
        """Choose evidence best-first within the room: tool results the
        caller computed, then memory sentences (leaving a little room for
        facts when there are any), then facts, then more memory if room
        is left. Shown in that order and numbered 1..n. Each line costs
        its length plus a newline; numbering in a different order costs
        the same, since the numbers used are always 1..n."""
        room = self._evidence_room(brief)
        item_cap = max(240, int(room * ITEM_SHARE))
        side_need = sum(len(e.line()) + 8 for e in side)
        reserve = min(int(room * FACT_RESERVE), side_need)
        used = 0
        picked: list[tuple[int, int, Evidence]] = []   # (group, rank, item)
        shown_mem: set[int] = set()

        def take(group: int, rank: int, ev: Evidence, limit: int) -> bool:
            nonlocal used
            ev = replace(ev, n=len(picked) + 1)
            overhead = len(ev.line()) - len(ev.text) + 1
            text = _fit_sentences(ev.text, min(item_cap,
                                               limit - used - overhead))
            if text is None:
                return False
            ev.text = text
            used += overhead + len(text)
            picked.append((group, rank, ev))
            if group == 1 and ev.memory_id is not None:
                shown_mem.add(ev.memory_id)
            return True

        for i, ev in enumerate(extra):
            take(0, i, ev, room)
        left = []
        for i, ev in enumerate(memory):
            if not take(1, i, ev, room - reserve):
                left.append((i, ev))
        for i, ev in enumerate(side):
            # a fact read from a sentence already shown adds nothing
            if ev.memory_id is not None and ev.memory_id in shown_mem:
                continue
            take(2, i, ev, room)
        for i, ev in left:
            take(1, i, ev, room)
        picked.sort(key=lambda p: (p[0], p[1]))
        return [replace(ev, n=n) for n, (_, _, ev) in enumerate(picked, 1)]

    def _enforce_budget(self, brief: Brief) -> None:
        """Last guard: should the estimate still be over (it shouldn't),
        drop evidence from the end, then the summary, then standing, then
        shorten the question."""
        budget = self.budget_tokens
        while brief.token_estimate() > budget and brief.evidence:
            brief.evidence.pop()
        if brief.token_estimate() > budget:
            brief.state = brief.state.split("\n", 1)[0]
        if brief.token_estimate() > budget:
            brief.standing = ""
        while brief.token_estimate() > budget and brief.question:
            over = (brief.token_estimate() - budget) * 4 + 8
            brief.question = _clip_words(brief.question,
                                         max(0, len(brief.question) - over))

    # ── what sending everything would cost ─────────────────────────
    def _naive_tokens(self, brief: Brief, question: str,
                      state: ConversationState | None, hits: list[dict],
                      extra: list[Evidence]) -> int:
        """An honest, conservative estimate of what a send-everything chat
        app would send for this turn:

            the system prompt
          + every stored turn of this session, verbatim
          + the full text of all k retrieved candidates (unfiltered - such
            an app pastes what its search returned, echoes and all)
          + the user's stored facts and the tool results, in full
          + the question, in full

        Conservative because a real one sends whole documents, not single
        retrieved sentences. Never less than the brief itself: a prompt
        that sends everything also holds everything the brief holds."""
        naive = estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(question)
        if state is not None:
            try:
                naive += state.verbatim_tokens()
            except Exception:
                pass
        naive += sum(estimate_tokens(str(h.get("text") or "")) for h in hits)
        naive += sum(estimate_tokens(t) for _, t in self._user_fact_rows())
        naive += sum(estimate_tokens(e.text) for e in extra)
        return max(naive, brief.token_estimate())


# ─── small helpers ──────────────────────────────────────────────────

_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
         "Sunday")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")


def _today_line(today) -> str:
    """'Today is Thursday, 8 October 2026 (2026-10-08).' - spelled out
    without the locale, so it reads the same on every machine."""
    if today is None:
        d = date.today()
    elif isinstance(today, datetime):
        d = today.date()
    elif isinstance(today, date):
        d = today
    else:
        try:
            d = date.fromisoformat(str(today).strip()[:10])
        except ValueError:
            return f"Today is {' '.join(str(today).split())}."
    return (f"Today is {_DAYS[d.weekday()]}, {d.day} {_MONTHS[d.month - 1]} "
            f"{d.year} ({d.isoformat()}).")


def _excluded(source: str, seeing: bool) -> bool:
    """Not a fact: conversation echoes, stories, and perception logs
    unless the question is about what Telp saw or watched."""
    s = source.lower()
    if s.startswith(_NOT_FACT_SOURCES):
        return True
    return s.startswith(_PERCEPTION_SOURCES) and not seeing


def _dedup(items: list[Evidence]) -> list[Evidence]:
    """Drop near-duplicates, keeping the better-ranked copy: the same
    sentence from two sources, or one that says nothing the other
    doesn't. Different numbers are different claims ('95 moons' vs '115
    moons' both stay, so the model can see the disagreement)."""
    kept: list[Evidence] = []
    keys: list[tuple[str, set[str]]] = []
    for ev in items:
        norm = _norm_text(ev.text)
        words = _content(ev.text, keep_negation=True)
        if not any(_near_duplicate(norm, words, kn, kw) for kn, kw in keys):
            kept.append(ev)
            keys.append((norm, words))
    return kept


def _near_duplicate(a: str, aw: set[str], b: str, bw: set[str]) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) >= 30 and (a in b or b in a):
        return True
    if not aw or not bw:
        return False
    inter = len(aw & bw)
    if inter / len(aw | bw) >= 0.75:
        return True
    small = min(len(aw), len(bw))
    return small >= 4 and inter / small >= 0.9


def _fact_line(fact) -> str | None:
    """'Galileo Galilei - place of birth: Pisa' for a Fact; None for
    bookkeeping relations."""
    try:
        from lattice.facts import RELATIONS
        spec = RELATIONS.get(fact.relation)
    except Exception:
        spec = None
    if fact.relation in _SKIP_RELATIONS or (spec is not None
                                            and spec.internal):
        return None
    label = spec.label if spec is not None else fact.relation.replace("_", " ")
    line = f"{fact.subject} - {label}: {fact.obj}"
    quals = ", ".join(f"{k} {v}" for k, v in (fact.qualifiers or ()))
    return f"{line} ({quals})" if quals else line
