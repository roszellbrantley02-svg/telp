# Telp — a memory-first artificial mind

**Telp is not a language model.** He is a local, auditable assistant built around one
persistent memory: every sentence he knows is stored with its source and date, encoded as a
10,000-bit hypervector (the sign bits of a fixed random projection of a MiniLM sentence
embedding), and found again by meaning. Neural networks perceive and rank; rules, templates
and deterministic computation do the rest.

Every LLM is language-first — a text predictor with memory bolted on. Telp is built the
other way around. At his center is one memory store, and **everything he does is a memory
operation**:

| act | mechanism |
|---|---|
| **Perceiving** | CLIP eyes, Whisper ears, OCR reading — filing timestamped experience |
| **Answering** | retrieval by meaning, facet-checked, *cited* |
| **Reasoning** | deterministic computation over retrieved facts (math, dates, comparisons) |
| **Speaking** | composition of known truths — never prediction of likely words |
| **Growing** | fetches Wikipedia / wikiHow / URLs / YouTube himself when he doesn't know |
| **Forgetting** | surgical, claim-level deletion on command, provably |
| **Believing** | conflicting counts and roles surfaced with sources + dates; newest wins, history kept |

The consequence: **Telp doesn't invent facts.** A factual answer is a stored sentence he
can cite (`how do you know that?` → *"from wikipedia:Iceland, saved 2026-07-02"*), a
computation he can show, or an honest "I don't know". That is not the same as always being
right - he can faithfully repeat a source that is wrong or out of date, or pick the wrong
memory - which is exactly why every answer carries its source and date. (Stories he makes
up, his persona lines and small talk are not facts and are never served as answers.)

## What talking to him looks like

From a memory that has already learned a few hundred topics; the exact wording depends on
what he has learned.

```
> what is the capital of Iceland?
I didn't have that, so I just looked it up and learned it.
Iceland's capital and largest city is Reykjavík...

> how do you know that?
I can show you exactly: that answer came from wikipedia:Iceland (saved 2026-07-02).

> how old was Galileo when he died?
Galileo was 77 when he died - born February 15, 1564, died January 8, 1642.

> tell me about Jupiter
Jupiter is the fifth planet from the Sun, and the largest in the Solar System.
Also, Jupiter is the third-brightest natural object in the Earth's night sky...
On top of that, Jupiter's diameter is 11 times that of Earth.
   (three separate memories, composed into one voice - each one provable)

> remember that Jupiter has 95 confirmed moons     (true in 2023!)
Got it - I'll remember that.

> how many moons does Jupiter have?
There are 115 known moons of the planet Jupiter as of 9 April 2026.
(Careful - my memories disagree here: you told me 95, while the newest -
wikipedia:Moons of Jupiter says 115. Both may have been right when
written - these counts change.)
   (conflicting truths are SURFACED with sources and dates, never
    silently resolved; corrections supersede: "my favorite color is
    green" after "...blue" -> "Got it - updated what I believe.
    (Previously: blue.)")

> how do I boil an egg?
I didn't know how, so I just looked it up and learned the steps.
To boil eggs: 1. Place 3-6 eggs in a pot and cover them with cold water.
2. Bring the pot to boil, then turn off the heat and let it sit for 10 minutes. ...
   (procedures come from wikiHow as ordered steps - and "boil an egg" will
    never be answered with the scrambled-eggs recipe: the facet gate
    rejects topical-but-wrong procedures)

> forget the video you watched
Done - I've forgotten the video 'sintel_480': 29 memories erased.
```

He also **watches video** (scene cuts → keyframes labelled by CLIP from a 1,300-word
vocabulary), **hears** captionless audio (local Whisper), **reads text inside frames**
(OCR), fuses sight+speech into single memories (*"at 12s, while showing an elephant, the
speaker says..."*), writes small programs from a library of templates (17 code templates,
6 terminal games, SQLite CRUD apps) and runs them in a separate time-limited process, makes
up short stories by filling a story template with characters and places from a Wiktionary
dictionary, and remembers *you* across sessions.

## Quickstart

```bash
pip install -r requirements.txt
python telp.py                       # talk (REPL)
python telp.py serve                 # or: run him as a RESIDENT MIND -
                                     # boot once, then every ask answers in
                                     # ~0.1s and he resyncs his memory when
                                     # other processes teach/learn/forget
python telp.py ask "hello, who are you?"
python telp.py see photo.jpg         # he looks, names, remembers
python telp.py watch video.mp4       # scenes + speech + on-screen text
python telp.py watch "https://www.youtube.com/watch?v=..."
python telp.py learn "Photosynthesis"          # grow from Wikipedia
python telp.py learn "https://any.url/doc"     # or any page
python telp.py teach "My dog's name is Astro." # or from you
python telp.py forget the video      # selective forgetting
```

Models (MiniLM, CLIP, Whisper, EasyOCR) download from Hugging Face on first use to
`TELP_MODEL_DIR` (default `~/.cache/telp`). ffmpeg on PATH (or `TELP_FFMPEG_DIR`) enables
video watching. Everything Telp remembers lives in `state/` (or `$TELP_STATE_DIR`), in one
memory file shared by every command - chat, ask, teach, learn, see, watch and forget.
A fresh Telp starts nearly empty and **grows by living**: seed him at scale with
`python -m lattice.educate --corpus your_corpus.jsonl --target 20000` (any JSONL with
`{"text": ...}` rows), point `lattice/wiktionary_ingest.py` at a
[kaikki.org](https://kaikki.org) Wiktionary dump for the offline dictionary +
imagination lane, or just let learn-on-miss fill him in as you talk.

## Running the tests

```bash
pip install -r requirements-dev.txt     # numpy, scipy, Pillow, pytest
python -m pytest
```

The suite runs offline in a few seconds: it swaps MiniLM for a small stand-in sentence
model (`tests/fake_minilm.py`), points `TELP_STATE_DIR` at a temporary directory, and
drives the real code paths - routing, memory, teach/ask/learn/forget through `telp.py`,
and the resident daemon. CI runs it on Python 3.10-3.12 for every push.

## Architecture

```
telp.py                     the door: chat / ask / see / watch / learn / teach / forget,
                            plus `serve`, a resident daemon on 127.0.0.1:7580
  mind/                     the cascade: mind/fluency.py tries ~30 handlers in order
                            (teach, facts about you, forget, provenance, dates, ages,
                            word problems, how-to, definitions, essays, stories, code,
                            arithmetic, meaning search, learn-on-miss), plus persona,
                            voice, user memory and the code/app/game composers
  lattice/                  the substrate: the memory store (SQLite + in-RAM bit matrix,
                            brute-force Hamming search), the semantic encoder
                            (MiniLM -> SHA-seeded projection -> 10,000 sign bits),
                            vision, hearing, OCR, growth (learn-on-miss), the story
                            engine, the offline dictionary, lattice/paths.py (state dir)
  train/v5_hdc_prototype.py the HDC primitive layer: bind (XOR), bundle (majority),
                            Hamming similarity (Kanerva's VSA)
  tests/                    the offline test suite
```

About half of `lattice/` is research the app doesn't load: earlier encoders, HDC
experiments (feed-forward layers, graph message passing, image codecs) and their
`test_*.py` experiment scripts, which print results rather than assert them. The running
app's hypervectors come from the semantic encoder; binding and bundling are used in a few
places (analogy fallback, phrase bundles, a small fact base) rather than for retrieval.

Design laws:
1. **No LLM in the loop.** Neural nets perceive (light/sound/text → vectors) and rank
   which memory fits; they never write what he says.
2. **One memory.** Everything he knows about the world lives in one store, in one
   encoding, with source and date on every row (his persona, facts about you and the code
   snippets have small stores of their own).
3. **A topical answer is not an answering answer.** Facet-coverage gates reject confident
   wrong-facet matches on every answer path ("how do I boil an egg" must cover *boiling*,
   not just *eggs*) - misses trigger lawful retrieval instead.
4. **Generation is composition of known truths.** The composed voice selects diverse
   facts (embedding MMR), simplifies them by deterministic rules (dropping pronunciations
   and life dates, never qualifying clauses), and joins them - every sentence it says is
   a stored memory.

## Honest limitations

- He composes and retrieves; he does not do open-ended novel reasoning, planning, or
  freeform essay writing. Ask him something he can't do and he says so.
- Conversation depth is retrieval-bounded; pronoun carry-over across long threads is
  imperfect.
- Knowledge = what he has been fed plus what he looks up. He starts small.
- Conflicting memories are surfaced for counts ("how many...") and roles ("who is the X
  of Y"); for other questions the newest or best match wins without a note.
- Routing is rule-based: each handler is picked by patterns, so unusual phrasings can
  land on the wrong one. The test suite pins the known cases.
- Code runs in a separate process with a time limit, no keyboard input, a throwaway
  working directory and (on POSIX) memory/CPU caps - but it is not a security sandbox.
  Only Telp's own templates are run.
- Word-problem arithmetic covers gain/loss chains, multiplication ("3 boxes
  of 6 eggs"), sharing/division, and comparatives ("Tom has 3 more than
  Sara"); multi-step rate problems are not parsed yet.

These are documented boundaries, not bugs: the trade for a mind that never bluffs.

## Lineage

Built on Pentti Kanerva's hyperdimensional computing / Vector Symbolic Architectures —
the cognitive-architecture road largely bypassed when LLMs took off. Telp is a working
argument that the road still leads somewhere: a complete perceive → remember → reason →
speak loop on one PC, no cloud, no API key, fully auditable.

## License

MIT
