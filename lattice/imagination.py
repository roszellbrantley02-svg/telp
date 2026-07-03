"""
lattice/imagination.py — Phase 16: generative imagination.

THESIS
------
LLMs need to see "jump" in 500K contexts to use it.  A child sees
it a handful of times and uses it forever.  The difference: the
child has a HANDLE for the word — sound, spelling, meaning, the
roles it can play — and creativity is recombining handles into
new structures.

Telp has 1.33M handles now (Phase 15).  Imagination is the act of
ALGEBRAICALLY recombining them into frames that didn't exist in
any training data.  Three operations:

  1. Combinatorial binding — bind(R_AGENT, dragon_hv) +
     bind(R_GOAL, teapot_hv) makes a new frame whose binding is
     valid, even though no human ever wrote it.

  2. HV-distance steering — pick a filler whose HV is mid-distant
     from the protagonist.  Too near = boring (typical).  Too far
     = nonsense.  Mid-distant = surprising-but-coherent.

  3. Arc-constrained generation — pre-pick an emotional arc
     (lonely -> curious -> brave -> happy), generate one frame per
     beat whose attribute slot matches.  This gives the output
     plot SHAPE.

OUTPUT
------
A list of frames Telp built by recombination.  Rendered via the
existing render_frame() machinery to English.  Every word in the
output is traceable to one of:

  - a noun in dict.db whose hypernym is animal/place/etc.
  - a verb retrieved from a pool keyed to an emotional beat
  - a structural slot fixed by the frame template

No LLM, no sampling from a learned distribution, no hallucination.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Optional

_TELP_ROOT = Path(__file__).resolve().parents[1]
if str(_TELP_ROOT) not in sys.path:
    sys.path.insert(0, str(_TELP_ROOT))

from lattice.dictionary_lookup import Dictionary
from lattice.story_transform import render_frame
from lattice.role_typing import RoleFillerTable
from lattice.justification import Justifier


# Novelty threshold for attaching a justification clause.
# >= this score means the filler is unusual enough in this role
# that we should ground it via its dictionary definition.
_NOVELTY_THRESHOLD = 0.85


# ─── Emotional arcs ──────────────────────────────────────────────


ARCS: dict[str, list[str]] = {
    "lonely_to_loved": [
        "lonely", "sad", "curious", "afraid", "brave", "happy", "loved"
    ],
    "small_to_brave": [
        "small", "afraid", "curious", "brave", "strong", "proud"
    ],
    "lost_to_found": [
        "lost", "afraid", "curious", "hopeful", "found", "happy"
    ],
    "ordinary_to_magical": [
        "bored", "curious", "surprised", "amazed", "delighted"
    ],
    "stranger_to_friend": [
        "alone", "shy", "curious", "warm", "trusting", "friend"
    ],
    "sleepy_to_awake": [
        "sleepy", "curious", "surprised", "playful", "tired", "loved"
    ],
}


# ─── Action verbs by emotional beat ──────────────────────────────


ACTIONS_BY_BEAT: dict[str, list[str]] = {
    "lonely":    ["walked", "sat", "looked", "wandered", "waited"],
    "sad":       ["sighed", "rested", "thought", "looked"],
    "small":     ["walked", "looked", "watched", "wondered"],
    "alone":     ["walked", "sat", "watched", "rested"],
    "bored":     ["watched", "yawned", "wandered", "looked"],
    "lost":      ["walked", "looked", "searched", "called"],
    "sleepy":    ["yawned", "rested", "dreamed", "slept"],

    "curious":   ["walked", "peeked", "looked", "asked", "explored"],
    "shy":       ["watched", "waited", "smiled"],
    "afraid":    ["hid", "ran", "watched", "trembled", "froze"],
    "scared":    ["hid", "ran", "watched", "trembled"],
    "surprised": ["gasped", "stared", "stopped", "watched"],
    "hopeful":   ["walked", "looked", "called", "smiled"],

    "brave":     ["walked", "stood", "spoke", "smiled"],
    "strong":    ["walked", "lifted", "carried", "stood"],
    "warm":      ["smiled", "spoke", "sat"],
    "amazed":    ["watched", "smiled", "wondered", "danced"],
    "playful":   ["played", "danced", "laughed", "jumped"],
    "trusting":  ["smiled", "spoke", "sat", "walked"],
    "delighted": ["played", "laughed", "danced", "smiled"],

    "happy":     ["smiled", "laughed", "played", "danced"],
    "loved":     ["smiled", "rested", "hugged", "thanked"],
    "found":     ["smiled", "ran", "called", "hugged"],
    "proud":     ["stood", "smiled", "walked"],
    "friend":    ["played", "smiled", "spoke", "walked"],
    "tired":     ["yawned", "rested", "smiled"],
}


# Fallback when an arc beat has no entry above.
_DEFAULT_ACTIONS = ["walked", "watched", "smiled", "rested"]


# ─── Fantastical settings (always available, augment dict pool) ──


# Settings that aren't always tagged as "place" in Wiktionary but
# read beautifully as story locations.  Hand-curated from words
# Telp already knows; not hardcoded vocabulary, just a hint that
# these surface forms work as locations.
_FANTASTICAL_SETTINGS = [
    "moon", "sky", "cloud", "mountain", "dream", "song", "story",
    "memory", "shadow", "river", "ocean", "island", "garden",
    "library", "kitchen", "attic", "tower", "bridge", "cave",
    "valley", "meadow", "forest", "thunderstorm", "rainbow",
    "mirror", "puddle", "rooftop", "wind",
]


# ─── ImaginationEngine ───────────────────────────────────────────


class ImaginationEngine:
    """Combine handles Telp already has into original stories.

    Builds three pools from dict.db:
      cast     - characters (animals, creatures, beings)
      settings - places to visit (rooms, landforms, fantastical)
      objects  - things to encounter (optional, used as patients)

    Then walks an emotional arc, building one frame per beat by
    binding (protagonist + arc_action + sampled_filler).  Rendered
    via the existing render_frame() machinery.
    """

    def __init__(self,
                      dictionary: Optional[Dictionary] = None,
                      seed: Optional[int] = None,
                      lattice_pickle: Optional[Path] = None):
        self.dict = dictionary or Dictionary()
        self.rng = random.Random(seed)
        self._cast_pool: Optional[list[str]] = None
        self._setting_pool: Optional[list[str]] = None
        self._object_pool: Optional[list[str]] = None

        # Phase 17: load observed role-filler distributions from
        # the reading lattice so we can detect novelty per role,
        # and stand up a Justifier to ground unusual choices via
        # their dictionary definition.
        line_events = {}
        try:
            from lattice.reading_lattice import ReadingLattice, LATTICE_PATH
            p = lattice_pickle or LATTICE_PATH
            if Path(p).exists():
                lat = ReadingLattice.load(p)
                line_events = getattr(lat, "line_events", {}) or {}
        except Exception:
            pass
        self.role_table = RoleFillerTable(line_events)
        self.justifier  = Justifier(self.dict)

    # ── Pool building (lazy) ───────────────────────────────────

    def _pool_from_hypernyms(self,
                                       hypernyms: list[str],
                                       limit_per: int = 2000,
                                       max_len: int = 10,
                                       min_len: int = 4,
                                       extras: list[str] = ()) -> list[str]:
        """Sample words whose hypernym is in the given list.

        Filters out niche / awkward / region-specific entries that
        read badly in stories — words with vulgar slang senses, very
        short abbreviations, or rare compound forms.
        """
        # Avoid words whose first dictionary sense reads as obscure,
        # vulgar, technical, or a cross-reference.
        skip_words = {
            "ass", "boob", "tit", "cock", "dick", "bitch",
            "watercock", "shitass", "ftp", "ppv", "vcr", "lsm",
            "ipo", "ngo", "ceo", "cfo", "atm", "diy", "etc",
            "milkshed", "frontcountry", "upcountry", "backwoods",
            "barbel", "cuspid", "autosome", "anchovy", "cofferdam",
            "boottree", "quercitron", "caprine", "furbearer",
            "barnroom", "sheepyard", "borderland", "interior",
            "scrubland", "lakefront", "creekfront", "riverfront",
        }
        seen = set()
        out = []
        for h in hypernyms:
            for word, _ in self.dict.reverse_related(
                    h, "hypernym", limit=limit_per):
                wl = word.lower().strip()
                if (wl and wl.isalpha() and " " not in wl
                          and min_len <= len(wl) <= max_len
                          and wl not in seen
                          and wl not in skip_words):
                    seen.add(wl)
                    out.append(wl)
        for w in extras:
            wl = w.lower().strip()
            if wl not in seen:
                seen.add(wl)
                out.append(wl)
        return out

    def cast_pool(self) -> list[str]:
        if self._cast_pool is None:
            self._cast_pool = self._pool_from_hypernyms(
                ["animal", "creature", "mammal", "bird", "fish",
                  "insect", "reptile", "amphibian", "being"],
                extras=["dragon", "unicorn", "phoenix", "fox", "owl",
                          "rabbit", "frog", "bear", "wolf", "deer",
                          "mouse", "cat", "dog", "sparrow", "raven",
                          "beetle", "fairy", "robot", "ghost"])
        return self._cast_pool

    def setting_pool(self) -> list[str]:
        if self._setting_pool is None:
            self._setting_pool = self._pool_from_hypernyms(
                ["place", "location", "room", "structure", "building",
                  "landform"],
                extras=_FANTASTICAL_SETTINGS)
        return self._setting_pool

    def object_pool(self) -> list[str]:
        if self._object_pool is None:
            self._object_pool = self._pool_from_hypernyms(
                ["object", "tool", "vehicle", "food", "beverage",
                  "tree", "flower", "vessel"],
                extras=["teapot", "feather", "key", "candle",
                          "letter", "lantern", "ribbon", "book",
                          "map", "thread", "bell", "stone"])
        return self._object_pool

    # ── Generation ─────────────────────────────────────────────

    def _pick_action(self, beat: str) -> str:
        pool = ACTIONS_BY_BEAT.get(beat, _DEFAULT_ACTIONS)
        return self.rng.choice(pool)

    def _pick(self, pool: list[str], exclude: set[str] = None) -> str:
        exclude = exclude or set()
        for _ in range(10):
            cand = self.rng.choice(pool)
            if cand not in exclude:
                return cand
        return self.rng.choice(pool)

    def _maybe_justify(self, frame: dict, role: str, action: str,
                              filler: str) -> dict:
        """If `filler` is novel in this (role, action) slot, attach
        a justification clause from its dictionary entry.  Modifies
        `frame` in place and returns it.

        Doesn't filter — every unusual choice STAYS.  Justification
        just gives it a reason to be there.
        """
        n = self.role_table.novelty(role, action, filler)
        if n < _NOVELTY_THRESHOLD:
            return frame
        clause = self.justifier.justify(filler)
        if clause:
            frame["_justification"] = clause
            frame["_novelty"] = round(n, 2)
        return frame

    def imagine_story(self,
                            seed: str,
                            arc_name: Optional[str] = None,
                            n_beats: int = 6) -> list[dict]:
        """Generate an original story as a sequence of frames.

        seed       : protagonist word (e.g. "dragon", "owl", "robot")
        arc_name   : key into ARCS, or None for a random arc
        n_beats    : how many emotional beats to traverse (auto-trimmed
                       to len(arc))
        """
        protagonist = seed.lower().strip()
        if arc_name is None:
            arc_name = self.rng.choice(list(ARCS.keys()))
        arc = ARCS[arc_name]
        n_beats = min(n_beats, len(arc))
        beats = arc[:n_beats]

        # Need enough unique partners to fill the middle beats —
        # n_beats - 2 (excluding establish + resolve).  Sample
        # without replacement so partners don't repeat.
        n_partners = max(3, n_beats - 1)
        cast_avail = [c for c in self.cast_pool() if c != protagonist]
        cast = self.rng.sample(cast_avail,
                                          k=min(n_partners, len(cast_avail)))
        settings = self.rng.sample(self.setting_pool(),
                                              k=min(max(4, n_beats),
                                                          len(self.setting_pool())))

        frames: list[dict] = []
        used = {protagonist}

        # ── Opening: "In the morning the X woke up."
        frames.append({
            "_intent": "statement",
            "location": "morning",
            "agent": protagonist,
            "action": "woke",
        })

        # ── Establish initial feeling
        frames.append({
            "_intent": "statement",
            "agent": protagonist, "action": "felt",
            "attribute": beats[0],
        })

        # ── Want statement (gives the story a goal).  Some wants
        # are concrete count nouns (friend, way, answer) -> take an
        # article via the renderer.  Abstract mass nouns (courage,
        # rest, peace, adventure) read better bare.
        wants_by_beat = {
            "lonely": "friend", "alone": "friend",
            "lost": "way", "sleepy": "rest",
            "bored": "adventure", "small": "courage",
        }
        want = wants_by_beat.get(beats[0], "answer")
        abstract_wants = {"courage", "rest", "peace", "adventure",
                                "hope", "strength", "wisdom", "love",
                                "joy"}
        want_frame = {
            "_intent": "statement",
            "agent": protagonist, "action": "wanted",
        }
        if want in abstract_wants:
            # Use attribute slot so renderer emits "wanted courage"
            # without an article.
            want_frame["attribute"] = want
        else:
            want_frame["patient"] = want
        frames.append(want_frame)

        # ── Walk → encounter → react → move loop, one per beat
        for i in range(1, len(beats) - 1):
            beat = beats[i]
            setting = settings[(i - 1) % len(settings)]
            partner = cast[(i - 1) % len(cast)]
            used.add(partner)

            # Move to a setting — if the setting is novel for a
            # "walked" frame, attach a justification clause.
            move_frame = {
                "_intent": "statement",
                "agent": protagonist, "action": "walked",
                "goal": setting,
            }
            self._maybe_justify(move_frame, "goal", "walked", setting)
            frames.append(move_frame)

            # Encounter someone there — partner can also be novel.
            encounter_frame = {
                "_intent": "statement",
                "location": setting,
                "agent": protagonist, "action": "saw",
                "patient": partner,
            }
            self._maybe_justify(encounter_frame, "patient", "saw",
                                          partner)
            frames.append(encounter_frame)

            # Greeting
            frames.append({
                "_intent": "statement",
                "agent": protagonist, "action": "said",
                "patient": "hi", "goal": partner,
            })

            # Partner's reaction (state)
            frames.append({
                "_intent": "statement",
                "agent": partner, "action": "was",
                "attribute": beat,
            })

            # Partner's action  (if scared, hide; if friend, smile)
            partner_action = self._pick_action(beat)
            partner_frame = {
                "_intent": "statement",
                "agent": partner, "action": partner_action,
            }
            # Add a sensible complement
            if partner_action in {"hid", "ran"}:
                partner_frame["location"] = self._pick(
                    self.setting_pool(), exclude=used)
            elif partner_action in {"smiled", "laughed", "danced",
                                                  "played"}:
                pass   # bare verb reads naturally here
            else:
                pass
            frames.append(partner_frame)

            # Protagonist's reaction
            frames.append({
                "_intent": "statement",
                "agent": protagonist, "action": "felt",
                "attribute": beats[min(i + 1, len(beats) - 1)],
            })

        # ── Resolution: protagonist + last ACTUALLY-USED partner
        # together.  Middle beats used cast[0..n_beats-3]; the last
        # one is the partner who paired with the final-but-one beat.
        final_beat = beats[-1]
        last_partner_idx = max(0, (len(beats) - 2) - 1)
        last_partner = (cast[last_partner_idx]
                              if last_partner_idx < len(cast)
                              else "friend")
        last_setting = settings[-1]

        frames.append({
            "_intent": "statement",
            "agent": f"{protagonist} and {last_partner}",
            "action": "played",
            "location": last_setting,
        })

        # Optional object interaction (gives the ending texture).
        # Objects can be very novel ("buick", "ribbon", "lantern") so
        # justification is especially welcome here.
        if self.object_pool():
            obj = self._pick(self.object_pool())
            obj_frame = {
                "_intent": "statement",
                "agent": "they", "action": "watched",
                "patient": obj,
            }
            self._maybe_justify(obj_frame, "patient", "watched", obj)
            frames.append(obj_frame)

        frames.append({
            "_intent": "statement",
            "agent": "they", "action": "were",
            "attribute": "friends",
        })

        # ── Closing: "In the evening the X walked home.  The X felt FINAL."
        frames.append({
            "_intent": "statement",
            "location": "evening",
            "agent": protagonist, "action": "walked",
            "patient": "home",
        })
        frames.append({
            "_intent": "statement",
            "agent": protagonist, "action": "felt",
            "attribute": final_beat,
        })

        # Annotate frames with arc + cast for trace/debug
        for f in frames:
            f.setdefault("_arc", arc_name)

        return frames

    # ── Convenience ────────────────────────────────────────────

    def render(self, frames: list[dict]) -> str:
        return "\n".join(render_frame(f) for f in frames)


# ─── CLI smoke ───────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="dragon",
                          help="protagonist (default: dragon)")
    ap.add_argument("--arc",  default=None,
                          help="arc name (default: random)")
    ap.add_argument("--rng",  type=int, default=None,
                          help="RNG seed for reproducibility")
    args = ap.parse_args()

    eng = ImaginationEngine(seed=args.rng)
    frames = eng.imagine_story(seed=args.seed, arc_name=args.arc)
    print(f"# arc = {frames[0].get('_arc')}")
    print(f"# protagonist = {args.seed}")
    print()
    print(eng.render(frames))
