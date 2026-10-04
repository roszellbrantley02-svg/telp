"""Arithmetic answers sums - and only sums."""
import pytest

from mind.code_synthesis import try_code_synthesis
from lattice.arithmetic_qa import detect_and_eval


@pytest.mark.parametrize("msg,expected", [
    ("what is 12 * 7", "12 * 7 = 84"),
    ("what's 23 * 47", "23 * 47 = 1081"),
    ("2*(3+4)", "2*(3+4) = 14"),
    ("what is 2*(3+4)?", "2*(3+4) = 14"),
    ("5 squared", "5**2 = 25"),
    ("what is 2^10", "2**10 = 1024"),
    ("what is 25 percent of 80", "25 * 0.01 * 80 = 20"),
    ("what is the square root of 144", "sqrt(144) = 12"),
    ("3 x 4", "3 * 4 = 12"),
    ("calculate 100 - 22 / 7", "100 - 22 / 7 = 96.8571"),
    ("-5 + 3", "-5 + 3 = -2"),
])
def test_real_math_is_answered(msg, expected):
    assert try_code_synthesis(msg) == expected


@pytest.mark.parametrize("msg", [
    "meeting on 2024-10-15",
    "call 555-1234",
    "what happened on 9/11?",
    "who won the game 3-2?",
    "I have 3 cats and 2 dogs",
    "what is einstein known for",
    "the 2010-2014 season",
    "what is 7/0",
])
def test_non_math_with_numbers_is_left_alone(msg):
    assert try_code_synthesis(msg) is None


@pytest.mark.parametrize("msg", ["what is 9**9**9?", "what is 2**20000"])
def test_huge_powers_are_refused_not_hung(msg):
    assert try_code_synthesis(msg) is None
    assert detect_and_eval(msg) is None


def test_agent_arithmetic_still_works():
    r = detect_and_eval("what is 2 ** 10?")
    assert r is not None and r.value == 1024


# ─── from the code review ──────────────────────────────────────────

@pytest.mark.parametrize("msg", ["what is pow(9, 9**9)", "pow(10, 10**8)",
                                 "pow(2, 3, 5)"])
def test_pow_cannot_bypass_the_size_guard(msg):
    assert try_code_synthesis(msg) is None


@pytest.mark.parametrize("msg,expected", [
    ("What’s 23 * 47?", "23 * 47 = 1081"),
    ("can you calculate 4*4 for me", "4*4 = 16"),
    ("what does 5 + 5 equal?", "5 + 5 = 10"),
    ("what is 500-1000", "500-1000 = -500"),
    ("what is -3 squared", "(-3)**2 = 9"),
    ("what is pow(2, 10)", "pow(2, 10) = 1024"),
])
def test_more_phrasings_of_real_math(msg, expected):
    assert try_code_synthesis(msg) == expected


def test_hex_is_not_multiplication():
    assert try_code_synthesis("0x10") is None
