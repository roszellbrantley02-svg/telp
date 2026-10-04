"""Rewrite rules, persona, emotion, dates, and the code/game tools."""
import time

import pytest


# ─── simplify keeps meaning ────────────────────────────────────────

from mind.composer import simplify


@pytest.mark.parametrize("raw,expected", [
    ("Marie Curie: She was a Polish physicist.",
     "Marie Curie was a Polish physicist."),
    ("Einstein: He was a physicist.", "Einstein was a physicist."),
    ("Galileo Galilei (15 February 1564 – 8 January 1642) was an Italian "
     "astronomer.", "Galileo Galilei was an Italian astronomer."),
    ("Iceland (Icelandic: Ísland; pronounced [ˈistlant]) is a Nordic "
     "island country.", "Iceland is a Nordic island country."),
    ("Reykjavik (; ) is the capital.", "Reykjavik is the capital."),
])
def test_simplify_removes_noise(raw, expected):
    assert simplify(raw) == expected


@pytest.mark.parametrize("raw,must_keep", [
    ("Acetaminophen is a common pain reliever (overdose can cause fatal "
     "liver damage) and is sold over the counter.", "fatal liver damage"),
    ("Pluto was the ninth planet (until it was reclassified as a dwarf "
     "planet in 2006).", "until it was reclassified"),
    ("The study found the drug effective in a large randomized trial "
     "across many hospitals and patient groups in several countries over "
     "a decade of follow-up, but later researchers argued the trial "
     "design was flawed.", "design was flawed"),
])
def test_simplify_keeps_meaning(raw, must_keep):
    assert must_keep in simplify(raw)


def test_essays_have_no_filler_sentences():
    import mind.writer as writer
    src = open(writer.__file__, encoding="utf-8").read()
    assert "There is more to it." not in src
    assert "verified to preserve the original meaning" not in src


# ─── persona and identity ──────────────────────────────────────────

def test_every_persona_fact_can_be_found(fresh_state):
    from mind.persona import PersonaStore
    from mind.persona_seed import seed, PERSONA_FACTS
    from lattice.semantic_encoder import SemanticEncoder
    store = PersonaStore(encoder=SemanticEncoder())
    seed(store)
    for text, trait, _cat in PERSONA_FACTS:
        top = store.query(text, k=1)[0]
        assert top["text"] == text and top["similarity"] > 0.9, (text, trait)


def test_identity_facts_are_not_credited_to_the_user(fresh_state):
    from lattice.standalone_agent import StandaloneAgent
    from mind.seed_identity import seed_if_needed, IDENTITY_FACTS
    agent = StandaloneAgent()
    assert seed_if_needed(agent)["seeded"]
    srcs = {s for t, s in zip(agent.lattice._texts, agent.lattice._sources)
            if t in IDENTITY_FACTS}
    assert srcs == {"identity"}
    assert not any("E drive" in f for f in IDENTITY_FACTS)
    assert not seed_if_needed(agent)["seeded"]      # once per memory


# ─── emotion: whole words only ─────────────────────────────────────

from mind.emotion import classify_emotion


@pytest.mark.parametrize("msg", [
    "what is einstein known for", "explain photosynthesis",
    "tell me about the great wall", "what is snow",
    "Alexander the Great", "why did Rome fail",
])
def test_ordinary_questions_carry_no_emotion_keyword(msg):
    assert classify_emotion(msg) is None


@pytest.mark.parametrize("msg,emotion", [
    ("ugh this is broken", "frustrated"),
    ("wow that is amazing", "excited"),
    ("huh? I don't understand", "confused"),
    ("tldr please", "urgent"),
])
def test_clear_emotions_still_detected(msg, emotion):
    assert classify_emotion(msg) == emotion


# ─── answer types ──────────────────────────────────────────────────

from mind.qa_types import answer_matches_type, classify_question, QTYPE_DATE


@pytest.mark.parametrize("text,ok", [
    ("The Battle of Hastings was fought in 1066.", True),
    ("It was fought on 14 October 1066.", True),
    ("Rome was founded in 753 BC.", True),
    ("Pompeii was destroyed in AD 79.", True),
    ("He was born on March 14, 1879.", True),
    ("The ad campaign failed.", False),
    ("Einstein was a physicist.", False),
])
def test_date_answers(text, ok):
    assert answer_matches_type(text, QTYPE_DATE) is ok


def test_how_old_is_a_quantity_question():
    assert classify_question("how old is the universe?") == "quantity"


# ─── year arithmetic wording ───────────────────────────────────────

class _FakeLattice:
    def __init__(self, rows):
        self.rows = rows

    def query(self, q, k=20):
        return [{"text": t, "similarity": 0.9} for t in self.rows]


class _FakeAgent:
    def __init__(self, rows):
        self.lattice = _FakeLattice(rows)


def test_age_at_event_keeps_the_users_wording():
    from mind.forward_chain import _handler_age_at_event
    agent = _FakeAgent(["Abraham Lincoln was born in 1809.",
                        "The civil war started in 1861."])
    r = _handler_age_at_event(agent, "Lincoln", "the civil war started")
    assert r["answer"] == ("Lincoln was about 52 years old when the civil "
                           "war started (1861).")
    assert "published" not in r["answer"]


# ─── code and games ────────────────────────────────────────────────

def test_running_code_never_waits_for_the_keyboard():
    from mind.code_writer import _safe_run
    t0 = time.time()
    ok, out = _safe_run("try:\n    input('> ')\nexcept EOFError:\n"
                        "    print('no stdin')\n", timeout=5.0)
    assert ok and "no stdin" in out
    assert time.time() - t0 < 4


def test_game_demo_runs_without_stalling():
    from mind.game_composer import try_compose_game
    t0 = time.time()
    r = try_compose_game("build me a guess the number game")
    assert r["ran"] and "demo run" in r["output"]
    assert "Attempt 1: " not in r["output"].split("---")[0]
    assert time.time() - t0 < 5


@pytest.mark.parametrize("msg", [
    "what are the rules of tic-tac-toe",
    "who invented rock paper scissors",
    "is hangman a good game for kids",
    "what is the probability of heads or tails",
])
def test_mentioning_a_game_is_not_a_request_for_one(msg):
    from mind.game_composer import detect_game_intent
    assert detect_game_intent(msg) is None


@pytest.mark.parametrize("msg,game", [
    ("build me a game", "guess_the_number"),
    ("make me a hangman game", "hangman"),
    ("let's play rps", "rock_paper_scissors"),
    ("tic tac toe", "tic_tac_toe"),
    ("can we play tic tac toe?", "tic_tac_toe"),
])
def test_game_requests(msg, game):
    from mind.game_composer import detect_game_intent
    assert detect_game_intent(msg) == game


def test_chained_code_keeps_its_imports():
    from mind.code_writer import try_compose
    r = try_compose("sort [3,1,2,3] then count")
    assert r["ran"], r["output"]
    assert "from collections import Counter" in r["code"]


def test_words_in_the_request_are_not_list_data():
    from mind.code_writer import extract_literal_list
    assert extract_literal_list(
        "write a function to sort a list then reverse it") is None
    assert extract_literal_list("sort the list 5, 3, 9") == [5, 3, 9]
    assert extract_literal_list("sort the numbers 5 3 9") == [5, 3, 9]
