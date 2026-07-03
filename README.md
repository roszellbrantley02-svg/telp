# Telp — a memory-first artificial mind

**Telp is not a language model.** He is a local, auditable cognitive organism built on
hyperdimensional computing (HDC/VSA): one persistent memory of 10,000-bit hypervectors,
neural networks used strictly as *sense organs*, and deterministic reasoning on top.

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
| **Forgetting** | surgical deletion on command, provably |

The consequence: **Telp cannot hallucinate.** Structurally, not by policy. Every sentence
he says is a memory he can cite (`how do you know that?` → *"from wikipedia:Iceland,
saved 2026-07-02"*), a computation he can show, or an honest refusal.

## What talking to him looks like

```
> what is the capital of Iceland?
I didn't have that, so I just looked it up and learned it.
Iceland: Its capital and largest city is Reykjavík...

> how do you know that?
I can show you exactly: that answer came from wikipedia:Iceland (saved 2026-07-02).

> how old was Galileo when he died?
Galileo was 77 when he died - born February 15, 1564, died January 8, 1642.

> tell me about Jupiter
Jupiter is the fifth planet from the Sun, and the largest in the Solar System.
Also, Jupiter is the third-brightest natural object in the Earth's night sky...
On top of that, Jupiter's diameter is 11 times that of Earth.
   (three separate memories, composed into one voice - each one provable)

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

He also **watches video** (scene detection → CLIP-named timestamped keyframes), **hears**
captionless audio (local Whisper), **reads text inside frames** (OCR), fuses sight+speech
into single bound memories (*"at 12s, while showing an elephant, the speaker says..."*),
writes and runs code, composes original stories by algebra (the imagination engine — no
sampling, every word dictionary-justified), and remembers *you* across sessions.

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

Models (MiniLM, CLIP, Whisper, EasyOCR — a few hundred MB total) download on first use
to `TELP_MODEL_DIR` (default `~/.cache/telp`). ffmpeg on PATH enables video watching.
A fresh Telp starts nearly empty and **grows by living**: seed him at scale with
`python -m lattice.educate --corpus your_corpus.jsonl --target 20000` (any JSONL with
`{"text": ...}` rows), point `lattice/wiktionary_ingest.py` at a
[kaikki.org](https://kaikki.org) Wiktionary dump for the offline dictionary +
imagination lane, or just let learn-on-miss fill him in as you talk.

## Architecture

```
telp.py                     the door: chat / ask / see / watch / learn / teach / forget
  mind/                     the cascade: fluency, persona, voice, emotion, user memory,
                            composed voice, code/app/game composers, Q&A typing
  lattice/                  the substrate: the Lattice memory store, semantic encoder
                            (MiniLM -> SHA-seeded LSH -> 10,000-bit hypervectors,
                            deterministic forever), vision, hearing, OCR, growth
                            (learn-on-miss), imagination, offline dictionary
  train/v5_hdc_prototype.py the HDC primitive layer: bind (XOR), bundle (majority),
                            Hamming similarity (Kanerva's VSA)
```

Design laws:
1. **No LLM in the loop.** Neural nets transduce (light/sound/text → meaning vectors);
   they never decide what he says.
2. **One memory.** Everything he knows lives in one store, in one encoding, with source
   and date on every row.
3. **A topical answer is not an answering answer.** Facet-coverage gates reject
   confident wrong-facet matches ("how do I boil an egg" must cover *boiling*, not just
   *eggs*) — misses trigger lawful retrieval instead.
4. **Generation is composition of known truths.** The composed voice selects diverse
   facts (embedding MMR), simplifies them by deterministic rules, and joins them — it
   cannot say anything it cannot prove.

## Honest limitations

- He composes and retrieves; he does not do open-ended novel reasoning, planning, or
  freeform essay writing. Ask him something he can't do and he says so.
- Conversation depth is retrieval-bounded; pronoun carry-over across long threads is
  imperfect.
- Knowledge = what he has been fed plus what he looks up. He starts small.
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
