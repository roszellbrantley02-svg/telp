"""Each message reaches the handler it was meant for."""
import pytest

from mind.fluency import FluentTelp
from mind.app_composer import detect_app_intent


# ─── arithmetic doesn't grab dates and phone numbers ───────────────

@pytest.mark.parametrize("msg,bad", [
    ("meeting on 2024-10-15", "1999"),
    ("call 555-1234", "-679"),
    ("who won the game 3-2?", "3-2 = 1"),
])
def test_numbers_in_sentences_are_not_arithmetic(telp, msg, bad):
    assert bad not in telp.respond(msg)


def test_real_arithmetic_still_answers(telp):
    assert "= 14" in telp.respond("what is 2*(3+4)?")


# ─── code requests get code, not a database app ────────────────────

@pytest.mark.parametrize("msg", [
    "write a python script to reverse a string",
    "write a bash script to back up my files",
    "make a shopping list",
    "write a python script that sorts a list",
])
def test_code_and_list_requests_are_not_apps(msg):
    assert detect_app_intent(msg) is None


@pytest.mark.parametrize("msg,entity", [
    ("build me a todo list", "todo"),
    ("build me a note app", "note"),
    ("build a habit tracker", "habit"),
    ("create a contact book", "contact"),
    ("build me a recipe app", "note"),
])
def test_real_app_requests_still_build_apps(msg, entity):
    assert detect_app_intent(msg)["entity"] == entity


def test_python_script_request_returns_reverse_code(telp):
    reply = telp.respond("write a python script to reverse a string")
    assert "CREATE TABLE" not in reply
    assert "def " in reply and "reverse" in reply.lower()


# ─── stories are about what you asked for ──────────────────────────

@pytest.mark.parametrize("msg,seed", [
    ("tell me a story about an owl", "owl"),
    ("write a story about an elephant", "elephant"),
    ("tell me a story about apples", "apples"),
    ("tell me a story about a purple elephant who loves jam", "elephant"),
    ("make up a story about the dragon named Bob", "dragon"),
])
def test_story_topic(msg, seed):
    m = FluentTelp._STORY_RE.search(msg)
    assert m and FluentTelp._story_seed(m.group(1)) == seed


def test_story_without_topic_has_no_seed():
    m = FluentTelp._STORY_RE.search("tell me a story")
    assert m and FluentTelp._story_seed(m.group(1)) is None
