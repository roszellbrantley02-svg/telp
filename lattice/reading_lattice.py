"""
lattice/reading_lattice.py - Telp's growing language memory.

Reads text, breaks it into multi-resolution units (phonemes, syllables-
proxy, words, phrases, lines), composes HVs at each level via the
phoneme_hdc primitives, and stores them in per-level cleanup memories.

Multi-resolution = your point #3 (prediction levels).
Cleanup at each level = your point #4 (cleanup memory).
Prosody bindings = your point #1 (stress, phrase boundaries, rhythm).
Semantic hooks = your point #2 (per-word optional bindings for
concept/image/object/color/action).

STORAGE
-------
The lattice is a dict of CleanupMemory instances, one per level:
  phoneme_mem  : 39 ARPABET atoms (pre-loaded)
  word_mem     : grows as words are read
  phrase_mem   : grows as phrases (between commas/periods) are read
  line_mem     : grows as full lines are read
  ngram_mem_2  : adjacent-word bigrams
  ngram_mem_3  : adjacent-word trigrams

Plus a parallel SEMANTIC store:
  word_semantics[word] -> dict of role->HV bindings
                          (concept/image/color/object/action — empty in v0.1
                          but the slots exist so future modules can hook in)

PERSISTENCE
-----------
Pickled to state/reading_lattice.pkl.  Each call to read_text() updates
the in-memory lattice; explicit save() persists.
"""
from __future__ import annotations
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from train.v5_hdc_prototype import D, bind, bundle, hamming_distance
from lattice.phoneme_hdc import (
    phoneme_hv, compose_word, compose_sentence, position_hv, _det_hv,
    similarity,
)
from lattice.cleanup_memory import CleanupMemory
from lattice.g2p_and_prosody import (
    text_to_units, word_to_phonemes, word_to_phonemes_all,
    stress_pattern_of_line,
)


_TELP_ROOT = Path(__file__).resolve().parents[1]
LATTICE_PATH = _TELP_ROOT / "state" / "reading_lattice.pkl"


# --- Prosody role HVs (deterministic) -----------------------------


R_STRESS         = _det_hv("role::stress")
R_PHRASE_BREAK   = _det_hv("role::phrase_break")
R_RHYTHM         = _det_hv("role::rhythm")
R_LINE_POSITION  = _det_hv("role::line_position")

R_CONCEPT  = _det_hv("role::concept")    # semantic-grounding slots
R_IMAGE    = _det_hv("role::image")
R_OBJECT   = _det_hv("role::object")
R_COLOR    = _det_hv("role::color")
R_ACTION   = _det_hv("role::action")

R_PREV_WORD = _det_hv("role::prev_word")  # distributional context (Phase 12.4b)
R_NEXT_WORD = _det_hv("role::next_word")


def stress_hv(level: int) -> np.ndarray:
    return _det_hv(f"stress::{level}")


def phrase_break_hv(kind: str) -> np.ndarray:
    return _det_hv(f"phrase_break::{kind}")


def rhythm_hv(pattern: tuple) -> np.ndarray:
    return _det_hv(f"rhythm::{'-'.join(str(s) for s in pattern)}")


# --- Reading lattice ----------------------------------------------


class ReadingLattice:
    """Telp's growing word/phrase/line memory, with prosody + semantic
    hooks, multi-resolution cleanup, and persistence."""

    def __init__(self):
        # Per-level cleanup memories
        self.phoneme_mem  = CleanupMemory.from_phonemes()
        self.word_mem     = CleanupMemory()
        self.phrase_mem   = CleanupMemory()
        self.line_mem     = CleanupMemory()
        self.bigram_mem   = CleanupMemory()
        self.trigram_mem  = CleanupMemory()

        # Co-occurrence count (for prediction): bigram label -> Counter(next_word -> count)
        self.next_word_counts: dict[str, Counter] = {}
        # 3-gram (last_two_words -> Counter)
        self.next_word_counts_3: dict[tuple, Counter] = {}
        # Phrase->next phrase
        self.next_phrase_counts: dict[str, Counter] = {}

        # Per-word optional semantic bindings (role -> HV)
        # Slots exist; empty until grounded by other modules.
        self.word_semantics: dict[str, dict[str, np.ndarray]] = {}

        # Distributional context (Phase 12.4b):
        # For each word, track Counter(prev_word) and Counter(next_word)
        # across all observations.  Used to compute context_hvs() later
        # via bundle of bind(R_PREV/NEXT, neighbor_hv).
        self.prev_word_obs: dict[str, Counter] = {}
        self.next_word_obs: dict[str, Counter] = {}
        # Materialized context HV per word, computed lazily / on demand
        self.context_hvs: dict[str, np.ndarray] = {}

        # Total observation counts (for diagnostics)
        self.word_freq: Counter = Counter()
        self.n_lines_read: int = 0
        self.n_words_read: int = 0

        # Phase 12.6: event frames per stored line.
        # Maps line_label -> {role: value, ..., "_intent": str}
        self.line_events: dict[str, dict] = {}

    # ── Word HV: phonemes + spelling + morphemes + prosody ────

    def _compose_word_with_prosody(self, unit: dict) -> np.ndarray:
        """Word identity HV = bundle of:
          1. PHONEMES  (sound — same as v0.1)
          2. SPELLING  (letters — Phase 12.3, distinguishes homophones)
          3. MORPHEMES (root+affixes — Phase 12.3, lets running/runs cluster)
          4. STRESS    (prosody — same as v0.1)
        """
        # Lazy imports to avoid circulars
        from lattice.grapheme_hdc import spelling_hv
        from lattice.morphology import morpheme_hv

        phon_seq_with_stress = unit["phonemes"]
        word = unit["word"]

        components = []

        # 1. Phoneme composition — bundle ALL CMU pronunciation variants
        # (Phase 12.4a).  potato pə-TAY-toh and pə-TAH-toh both snap to
        # the same word identity because both phoneme sequences contribute
        # to the same phoneme component.
        all_variants = word_to_phonemes_all(word)
        if len(all_variants) > 1:
            variant_hvs = [compose_word([p for p, _ in v])
                                for v in all_variants]
            components.append(bundle(variant_hvs))
        else:
            components.append(compose_word(
                [p for p, _ in phon_seq_with_stress]))

        # 2. Spelling composition (NEW — Phase 12.3)
        # This is what lets `bear` and `bare` differ: same phonemes
        # produce identical phoneme component, different letters
        # produce different spelling component -> different identity HVs
        components.append(spelling_hv(word))

        # 3. Morpheme composition (NEW — Phase 12.3)
        # Words sharing a ROOT cluster via this component:
        # morpheme_hv("running") and morpheme_hv("runs") share the
        # R_ROOT-bound `run` part
        components.append(morpheme_hv(word))

        # 4. Stress per phoneme position
        for i, (_, s) in enumerate(phon_seq_with_stress):
            if s > 0:
                components.append(
                    bind(position_hv(i),
                          bind(R_STRESS, stress_hv(s))))

        return bundle(components)

    # ── Read text ─────────────────────────────────────────────

    def read_text(self, text: str, verbose: bool = False) -> dict:
        """Ingest a passage.  Updates all per-level memories + co-occurrence.

        Returns stats: n_new_words, n_new_phrases, n_new_lines.
        """
        n_new_words = 0
        n_new_phrases = 0
        n_new_lines = 0

        units = text_to_units(text)
        line_buf: list[dict] = []
        phrase_buf: list[dict] = []
        prev_words: list[str] = []  # for n-gram capture

        def _flush_phrase():
            nonlocal n_new_phrases
            if not phrase_buf:
                return
            word_hvs = [u["_hv"] for u in phrase_buf if "word" in u]
            words   = [u["word"] for u in phrase_buf if "word" in u]
            if not word_hvs:
                phrase_buf.clear()
                return
            phrase_hv = compose_sentence(word_hvs)
            phrase_label = " ".join(words)
            if not self.phrase_mem.has(phrase_label):
                self.phrase_mem.add(phrase_label, phrase_hv)
                n_new_phrases += 1
            # Phrase->next phrase co-occurrence happens at flush time;
            # tracked at the line level below.
            phrase_buf.clear()

        def _flush_line():
            nonlocal n_new_lines
            if not line_buf:
                return
            line_words = [u for u in line_buf if "word" in u]
            if not line_words:
                line_buf.clear()
                return
            word_hvs = [u["_hv"] for u in line_words]
            line_hv = compose_sentence(word_hvs)
            # Bind in the rhythm pattern (stress sequence)
            rhythm = stress_pattern_of_line(line_buf)
            if rhythm:
                line_hv = bundle([line_hv,
                                    bind(R_RHYTHM, rhythm_hv(rhythm))])
            line_label = " ".join(u["word"] for u in line_words)
            if not self.line_mem.has(line_label):
                self.line_mem.add(line_label, line_hv)
                n_new_lines += 1
                # Phase 12.6: parse the line as an event frame and
                # store role assignments + intent.  The line's HV
                # gets the event_hv bundled in so role queries can
                # use the line memory directly.
                try:
                    from lattice.event_frames import (
                        parse_event, event_hv as _event_hv,
                    )
                    frame = parse_event(line_label)
                    self.line_events[line_label] = frame
                    e_hv = _event_hv(frame)
                    if e_hv.any():
                        # Bundle the event HV into the line memory
                        # so a single line carries both surface form
                        # and structural form
                        merged = bundle([line_hv, e_hv])
                        # Replace the stored bundle for this line
                        idx = self.line_mem._labels.index(line_label)
                        self.line_mem._hvs[idx] = merged
                except Exception:
                    pass
            self.n_lines_read += 1
            line_buf.clear()

        for unit in units:
            if "word" in unit:
                # Compose word HV with prosody
                wh = self._compose_word_with_prosody(unit)
                unit["_hv"] = wh
                w = unit["word"]
                # Add/update word memory
                if not self.word_mem.has(w):
                    self.word_mem.add(w, wh)
                    n_new_words += 1
                    # Initialize empty semantic-binding slots
                    self.word_semantics.setdefault(w, {})
                self.word_freq[w] += 1
                self.n_words_read += 1

                # Distributional context observation (Phase 12.4b):
                # record (current_word -> prev_word) and look back to
                # also record (prev_word -> current as next)
                if len(prev_words) >= 1:
                    prev1 = prev_words[-1]
                    # current word's PREV is prev1
                    self.prev_word_obs.setdefault(w, Counter())[prev1] += 1
                    # prev1's NEXT is current word
                    self.next_word_obs.setdefault(prev1, Counter())[w] += 1

                # n-gram observations
                if len(prev_words) >= 1:
                    prev1 = prev_words[-1]
                    self.next_word_counts.setdefault(prev1, Counter())[w] += 1
                    # bigram memory: bundle of two consecutive words
                    bg_hv = compose_sentence(
                        [self.word_mem.get_hv(prev1), wh])
                    bg_label = f"{prev1} {w}"
                    if not self.bigram_mem.has(bg_label):
                        self.bigram_mem.add(bg_label, bg_hv)
                if len(prev_words) >= 2:
                    prev2 = (prev_words[-2], prev_words[-1])
                    self.next_word_counts_3.setdefault(prev2, Counter())[w] += 1
                    tg_hv = compose_sentence([
                        self.word_mem.get_hv(prev2[0]),
                        self.word_mem.get_hv(prev2[1]),
                        wh,
                    ])
                    tg_label = f"{prev2[0]} {prev2[1]} {w}"
                    if not self.trigram_mem.has(tg_label):
                        self.trigram_mem.add(tg_label, tg_hv)
                prev_words.append(w)
                if len(prev_words) > 8:
                    prev_words.pop(0)

                line_buf.append(unit)
                phrase_buf.append(unit)
            elif "phrase_break" in unit:
                kind = unit["phrase_break"]
                if kind in ("comma", "semicolon", "colon"):
                    _flush_phrase()
                elif kind in ("period", "exclamation", "question"):
                    _flush_phrase()
                    _flush_line()
                    prev_words.clear()
                elif kind == "line":
                    _flush_phrase()
                    _flush_line()
                    prev_words.clear()
                elif kind == "stanza":
                    _flush_phrase()
                    _flush_line()
                    prev_words.clear()

        # Final flushes
        _flush_phrase()
        _flush_line()

        stats = {
            "n_new_words":   n_new_words,
            "n_new_phrases": n_new_phrases,
            "n_new_lines":   n_new_lines,
            "total_words_read": self.n_words_read,
            "vocab_size":    len(self.word_mem),
        }
        if verbose:
            print(f"[lattice] {stats}", file=sys.stderr)
        return stats

    # ── Distributional context HVs (Phase 12.4b) ───────────────

    def compute_context_hvs(self,
                                       min_observations: int = 2,
                                       verbose: bool = False) -> int:
        """For each word, bundle bind(R_PREV, prev_word_hv) and
        bind(R_NEXT, next_word_hv) across every observed neighbor.

        Words appearing in SIMILAR contexts (similar neighbors before
        and after) will have similar context_hvs.  This is the LEARNED
        morphology mechanism — ran and runs end up similar because
        they appear in parallel slots in "Yesterday the X ran" /
        "Today the X runs" templates.  Same for mice/cats (plural
        noun position) and yeah/no (interjection position).

        Returns count of words for which context_hvs were computed.
        """
        from train.v5_hdc_prototype import bundle as _bundle
        n_computed = 0
        for w in self.word_mem.labels():
            prev_neighbors = self.prev_word_obs.get(w, Counter())
            next_neighbors = self.next_word_obs.get(w, Counter())
            if (sum(prev_neighbors.values())
                    + sum(next_neighbors.values()) < min_observations):
                continue
            components = []
            for neighbor, count in prev_neighbors.items():
                if not self.word_mem.has(neighbor):
                    continue
                n_hv = self.word_mem.get_hv(neighbor)
                # Repeat the binding `count` times in the bundle to
                # weight frequent neighbors more heavily.  Cap at 8
                # to avoid one super-frequent stop word dominating.
                rep = min(count, 8)
                for _ in range(rep):
                    components.append(bind(R_PREV_WORD, n_hv))
            for neighbor, count in next_neighbors.items():
                if not self.word_mem.has(neighbor):
                    continue
                n_hv = self.word_mem.get_hv(neighbor)
                rep = min(count, 8)
                for _ in range(rep):
                    components.append(bind(R_NEXT_WORD, n_hv))
            if not components:
                continue
            self.context_hvs[w] = _bundle(components)
            n_computed += 1
            if verbose and n_computed % 50 == 0:
                print(f"[context] computed {n_computed} word context HVs",
                        file=sys.stderr)
        return n_computed

    def distributional_neighbors(self, word: str,
                                                top_k: int = 8) -> list[tuple[str, float]]:
        """Find words whose CONTEXT HV is most similar to `word`'s.

        Captures distributional relatedness (ran ≈ runs because they
        appear in similar slots).  Disjoint from concept similarity
        (which uses WordNet) and from phoneme/spelling similarity.
        """
        target = self.context_hvs.get(word)
        if target is None:
            return []
        from lattice.phoneme_hdc import similarity as _sim
        scored = []
        for w, hv in self.context_hvs.items():
            if w == word:
                continue
            scored.append((w, _sim(target, hv)))
        scored.sort(key=lambda kv: -kv[1])
        return scored[:top_k]

    # ── Semantic-grounding hooks ────────────────────────────────

    def bind_semantic(self, word: str, role: str, hv: np.ndarray) -> bool:
        """Attach a semantic binding to a word.  Returns True if attached.

        role: "concept" | "image" | "object" | "color" | "action" | custom
        hv:   the 10000-D HV from whatever module is grounding this word
              (image gallery, trading reasoner, etc.)
        """
        if not self.word_mem.has(word):
            return False
        self.word_semantics.setdefault(word, {})[role] = hv
        return True

    def get_grounded_word_hv(self, word: str) -> Optional[np.ndarray]:
        """Word HV bundled with ALL its semantic bindings.  Returns the
        plain word HV if no bindings exist."""
        base = self.word_mem.get_hv(word)
        if base is None:
            return None
        sem = self.word_semantics.get(word, {})
        if not sem:
            return base
        components = [base]
        for role, hv in sem.items():
            components.append(bind(_det_hv(f"role::{role}"), hv))
        return bundle(components)

    # ── Persistence ────────────────────────────────────────────

    def save(self, path: Path | str = LATTICE_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path | str = LATTICE_PATH) -> "ReadingLattice":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Diagnostics ────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "phonemes":  len(self.phoneme_mem),
            "words":     len(self.word_mem),
            "phrases":   len(self.phrase_mem),
            "lines":     len(self.line_mem),
            "bigrams":   len(self.bigram_mem),
            "trigrams":  len(self.trigram_mem),
            "n_words_read": self.n_words_read,
            "n_lines_read": self.n_lines_read,
            "top_words":  dict(self.word_freq.most_common(10)),
        }


# --- CLI -----------------------------------------------------------


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(
        description="Read text into Telp's growing language lattice")
    p.add_argument("--text", default=None,
                       help="Inline text to read")
    p.add_argument("--file", default=str(_TELP_ROOT / "state" / "books"
                                                   / "nursery_rhymes.txt"))
    p.add_argument("--no-save", action="store_true",
                       help="Don't persist the lattice to disk")
    p.add_argument("--reset", action="store_true",
                       help="Start with empty lattice (default: append)")
    args = p.parse_args()

    if args.reset or not LATTICE_PATH.exists():
        lattice = ReadingLattice()
        print(f"[lattice] starting fresh", file=sys.stderr)
    else:
        lattice = ReadingLattice.load()
        print(f"[lattice] loaded existing: {lattice.stats()}",
                file=sys.stderr)

    if args.text:
        stats = lattice.read_text(args.text, verbose=True)
    else:
        text = Path(args.file).read_text(encoding="utf-8")
        # Strip the comment lines starting with #
        text = "\n".join(ln for ln in text.splitlines()
                                  if not ln.strip().startswith("#"))
        stats = lattice.read_text(text, verbose=True)

    print(json.dumps(lattice.stats(), indent=2, default=str))

    if not args.no_save:
        lattice.save()
        print(f"[lattice] saved -> {LATTICE_PATH}", file=sys.stderr)
