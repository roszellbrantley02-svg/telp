"""End-to-end behaviour of the answer pipeline on a fresh memory."""
import sqlite3

import pytest

from tests.conftest import REAL_GROWTH


def _last(telp):
    return telp.agent.turns[-1] if telp.agent.turns else {}


# ─── facts about the user ──────────────────────────────────────────

def test_request_is_answered_not_filed_as_a_fact(telp):
    reply = telp.respond("I have a question about Rome.")
    assert not reply.startswith("Got it")
    assert telp.user_facts.count() == 0


def test_name_and_occupation_are_parsed_cleanly(telp):
    telp.respond("my name is Eric and I'm a developer")
    texts = set(telp.user_facts._texts)
    assert "User's name is Eric." in texts
    assert "User is a developer." in texts


def test_a_mood_does_not_replace_your_occupation(telp):
    telp.respond("I'm a developer")
    reply = telp.respond("I'm a little tired today")
    assert "Previously" not in reply
    assert "User is a developer." in telp.user_facts._texts


def test_new_workplace_supersedes_the_old_one(telp):
    telp.respond("I work at Acme")
    reply = telp.respond("I work at Globex")
    assert "Previously: User works at Acme" in reply
    assert "User works at Acme." not in telp.user_facts._texts


def test_forget_is_not_captured_as_a_new_fact(telp):
    telp.respond("I have two cats")
    n = telp.user_facts.count()
    telp.respond("forget that I have a pair of cats")
    assert telp.user_facts.count() <= n
    assert _last(telp).get("domain") != "user_facts:capture"


# ─── routes that used to grab the wrong messages ───────────────────

def test_photosynthesis_is_not_a_vision_question(telp):
    telp.agent.lattice.add("Image: an image showing a cat on a sofa",
                           source="image:/tmp/cat.png")
    telp.respond("can you explain photosynthesis")
    assert _last(telp).get("domain") != "vision"


def test_how_do_you_know_if_is_a_question_not_provenance(telp):
    telp.respond("how do you know if an egg is bad")
    assert _last(telp).get("domain") != "provenance"


def test_how_do_you_know_that_cites_the_source(telp):
    telp.respond("remember that the Zorblax river flows through Quendia")
    telp.respond("where does the Zorblax river flow?")
    reply = telp.respond("how do you know that?")
    assert _last(telp).get("domain") == "provenance"
    assert "user_taught" in reply or "told me" in reply


def test_invented_stories_never_come_back_as_facts(telp):
    story = ("Here's one I made up (about dragon):\n\nIn the morning the "
             "dragon woke. The dragon felt alone and flew to the purple "
             "volcano of Zorg.")
    telp.agent.lattice.add(story, source="story:dragon")
    reply = telp.respond("where did the dragon fly?")
    assert "volcano of Zorg" not in reply


def test_no_memories_gives_an_honest_miss_not_an_echo(telp):
    reply = telp.respond("what do raccoons eat?")
    assert reply.strip().lower() not in ("raccoons eat", "raccoons", "raccoon")


# ─── creativity shaping ────────────────────────────────────────────

def test_terse_answers_keep_the_conflict_warning(telp):
    body = ("There are 115 known moons of Jupiter. (Careful - my memories "
            "disagree here: you told me 95, while the newest says 115. "
            "Both may have been right when written.)")
    out = telp._shape_by_creativity(body, 0.0)
    assert out.startswith("There are 115 known moons of Jupiter.")
    assert "(Careful" in out and out.rstrip().endswith(")")


def test_high_creativity_never_appends_generated_words(telp):
    body = "Venus and Mars are the planets next to Earth."
    assert telp._shape_by_creativity(body, 1.0) == body


# ─── forgetting takes effect immediately ───────────────────────────

def test_forgetting_a_youtube_video_removes_all_its_rows(fresh_state):
    from lattice.store import Lattice
    from lattice.semantic_encoder import SemanticEncoder
    from lattice import vision

    db = fresh_state / "concept_bridge.db"
    lat = Lattice(db, encoder=SemanticEncoder())
    vid, title = "abcdefghijk", "Big Buck Bunny"
    lat.add(f"Image: At 12s in video '{title}': an image showing a rabbit",
            source="image:/tmp/sights/Big Buck Bunny/t0012s.png")
    lat.add(f"In the video '{title}' at 12s the screen shows the text: "
            f"\"Bunny\"", source=f"video:{title}")
    lat.add(f"In the video '{title}' at 12s, while showing a rabbit, the "
            f"speaker says: \"hello\"", source=f"youtube:{vid}")
    lat.add(f"In the video '{title}' (at 30s): a long passage of speech",
            source=f"youtube:{vid}")
    lat.add(f"Telp watched the YouTube video '{title}': 1 scenes seen, "
            f"1 moments, 1 spoken passages remembered.",
            source=f"video:youtube:{vid}")
    lat.add("Iceland's capital is Reykjavik.", source="wikipedia:Iceland")
    lat.close()

    gone = vision.forget(db, video=f"youtube:{vid}")
    assert len(gone) == 5
    con = sqlite3.connect(str(db))
    left = [r[0] for r in con.execute("SELECT text FROM memories")]
    con.close()
    assert left == ["Iceland's capital is Reykjavik."]


# ─── learning from Wikipedia ───────────────────────────────────────

def test_learned_sentences_keep_their_subject(fresh_state, monkeypatch):
    import lattice.fetch_wiki as fw
    from lattice.standalone_agent import StandaloneAgent
    extract = ("The Beatles were an English rock band formed in Liverpool "
               "in 1960. The core lineup of the band comprised John Lennon, "
               "Paul McCartney, George Harrison and Ringo Starr. The group "
               "was integral to the development of 1960s counterculture.")
    monkeypatch.setattr(fw, "fetch_full_lead",
                        lambda t: {"title": "The Beatles", "extract": extract})
    agent = StandaloneAgent(skip_ngram_retrain=True)
    r = REAL_GROWTH["learn_topic"](agent, "The Beatles")
    assert r["added"] == 3
    rows = agent.lattice._texts
    assert "The Beatles: The group was integral to the development of " \
           "1960s counterculture." in rows
    assert ("The Beatles were an English rock band formed in Liverpool "
            "in 1960.") in rows


def test_topical_but_wrong_facet_is_a_miss_on_every_path(telp):
    telp.agent.lattice.add(
        "A telephone is a telecommunications device that lets two people "
        "talk to each other over a distance.", source="wikipedia:Telephone")
    reply = telp.respond("who invented the telephone?")
    assert "telecommunications device" not in reply


# ─── from the code review ──────────────────────────────────────────

def test_please_remember_a_fact_is_kept(telp):
    telp.respond("please remember that my birthday is May 5")
    assert "User's birthday is May 5." in telp.user_facts._texts


@pytest.mark.parametrize("msg,fact", [
    ("My name is Eric. Can you help me with my essay?",
     "User's name is Eric."),
    ("I work at Acme, could you write me a resignation letter",
     "User works at Acme."),
    ("I am a nurse, please explain triage", "User is a nurse."),
])
def test_fact_with_a_request_is_kept_and_the_request_answered(telp, msg,
                                                              fact):
    reply = telp.respond(msg)
    assert fact in telp.user_facts._texts
    assert not reply.startswith("Got it")


@pytest.mark.parametrize("ask", [
    "what are your sources?", "Really? How do you know that?",
    "where did you learn that from?", "how do you know that's true?",
    "can you cite your sources?",
])
def test_provenance_phrasings(telp, ask):
    telp.respond("remember that the Zorblax river flows through Quendia")
    telp.respond("where does the Zorblax river flow?")
    telp.respond(ask)
    assert _last(telp).get("domain") == "provenance"


def test_vision_still_hears_ing_forms(telp):
    telp.agent.lattice.add("Image: an image showing a cat on a sofa",
                           source="image:/tmp/cat.png")
    telp.respond("what have you been watching?")
    assert _last(telp).get("domain") == "vision"


def test_old_persona_lines_are_retired(fresh_state):
    from mind.persona import PersonaStore
    from mind.persona_seed import seed, RETIRED_PERSONA_TEXTS, PERSONA_FACTS
    from lattice.semantic_encoder import SemanticEncoder
    store = PersonaStore(encoder=SemanticEncoder())
    store.add(RETIRED_PERSONA_TEXTS[3], trait="full", category="opinion")
    store.add("I'm Telp.", category="identity")
    seed(store)
    assert not set(RETIRED_PERSONA_TEXTS) & set(store._texts)
    assert {t for t, _, _ in PERSONA_FACTS} <= set(store._texts)


def test_old_identity_seed_is_migrated(fresh_state):
    from lattice.standalone_agent import StandaloneAgent
    from mind.seed_identity import (seed_if_needed, IDENTITY_FACTS,
                                    RETIRED_IDENTITY_FACTS)
    agent = StandaloneAgent()
    for fact in [IDENTITY_FACTS[0], RETIRED_IDENTITY_FACTS[0]]:
        agent.lattice.add(fact, source="user_taught", tags="identity")
    seed_if_needed(agent)
    rows = dict(zip(agent.lattice._texts, agent.lattice._sources))
    assert RETIRED_IDENTITY_FACTS[0] not in rows
    assert set(IDENTITY_FACTS) <= set(rows)
    assert {rows[f] for f in IDENTITY_FACTS} == {"identity"}


@pytest.mark.parametrize("title,extract,head_ok", [
    ("Python (programming language)",
     "Python is a high-level programming language. It was created by "
     "Guido van Rossum and first released in 1991.",
     "Python is a high-level programming language."),
    ("Émile Zola",
     "Émile Zola was a French novelist and journalist. Zola was a "
     "major figure in the political liberalization of France.",
     "Émile Zola was a French novelist and journalist."),
])
def test_title_anchoring_on_tricky_titles(fresh_state, monkeypatch, title,
                                          extract, head_ok):
    import lattice.fetch_wiki as fw
    from lattice.standalone_agent import StandaloneAgent
    monkeypatch.setattr(fw, "fetch_full_lead",
                        lambda t: {"title": title, "extract": extract})
    agent = StandaloneAgent(skip_ngram_retrain=True)
    REAL_GROWTH["learn_topic"](agent, title)
    assert head_ok in agent.lattice._texts
    # a forced re-fetch doesn't duplicate anything
    n = agent.lattice.count()
    REAL_GROWTH["learn_topic"](agent, title, force=True)
    assert agent.lattice.count() == n
