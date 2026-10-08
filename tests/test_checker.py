"""The answer checker: in harness mode Telp verifies every sentence the model
wrote against the evidence Telp supplied - numbers, dates and names exactly,
other words loosely - with no language model involved, and shows the user
which sentences it couldn't back up."""
import time

import pytest

from mind.checker import annotate, check_answer, split_sentences, summarize
from mind.harness_types import Evidence, SentenceCheck

EVIDENCE = [
    Evidence(1, "Reykjavík is the capital and largest city of Iceland.",
             "wikipedia:Iceland", "2026-07-02T10:00:00"),
    Evidence(2, "Galileo Galilei was an Italian astronomer and physicist, "
                "born in Pisa on 15 February 1564.",
             "wikipedia:Galileo Galilei", "2026-07-02"),
    Evidence(3, "Galileo discovered the four largest moons of Jupiter in 1610.",
             "wikipedia:Galileo Galilei", "2026-07-03"),
    Evidence(4, "Galileo died on 8 January 1642 in Arcetri.",
             "wikipedia:Galileo Galilei", "2026-07-03"),
    Evidence(5, "The universe is about 13.8 billion years old.",
             "wikipedia:Universe", "2026-08-01"),
    Evidence(6, "Marie Curie was born in Warsaw in 1867.",
             "wikipedia:Marie Curie", "2026-08-02"),
    Evidence(7, "Bananas are rich in potassium and grow in tropical climates.",
             "wikipedia:Banana", None),
    Evidence(8, "Þingvellir is a national park in Iceland, about 40 km from "
                "Reykjavík.", "wikipedia:Thingvellir", None),
    Evidence(9, "In 2020 the city had a population of 390,000.",
             "wikipedia:Example City", None),
    Evidence(10, "Pluto is no longer considered a planet.",
             "wikipedia:Pluto", None),
    Evidence(11, "The user's dog is named Astro.", "user_taught",
             "2026-09-30", kind="memory"),
    Evidence(12, "1642 - 1564 = 78", "tool:calculate", None, kind="tool"),
]


def _checks(answer: str, evidence=EVIDENCE, **kw) -> list[SentenceCheck]:
    return check_answer(answer, evidence, **kw)


def _one(sentence: str, evidence=EVIDENCE, **kw) -> SentenceCheck:
    checks = check_answer(sentence, evidence, **kw)
    assert len(checks) == 1, [c.sentence for c in checks]
    return checks[0]


# ─── splitting into sentences ───────────────────────────────────────

def test_abbreviations_do_not_end_sentences():
    text = ("Dr. Smith lived c. 1500 in Rome. He climbed Mt. Everest, e.g. "
            "twice, i.e. often. See No. 5 on Jan. 8, 1642. Apples, pears, "
            "etc. The end.")
    assert split_sentences(text) == [
        "Dr. Smith lived c. 1500 in Rome.",
        "He climbed Mt. Everest, e.g. twice, i.e. often.",
        "See No. 5 on Jan. 8, 1642.",
        "Apples, pears, etc.",
        "The end.",
    ]


def test_decimals_and_initials_stay_inside_their_sentence():
    text = ("The universe is 13.8 billion years old [5]. J. R. R. Tolkien "
            "served in the U.S. Navy. He moved to the U.S. The move went "
            "well. Version 2.0 shipped.")
    assert split_sentences(text) == [
        "The universe is 13.8 billion years old [5].",
        "J. R. R. Tolkien served in the U.S. Navy.",
        "He moved to the U.S.",
        "The move went well.",
        "Version 2.0 shipped.",
    ]


def test_citations_stay_with_the_sentence_they_follow():
    text = ("Reykjavik is the capital.[1] It is in Iceland. [1][8] "
            "Galileo was Italian [2, 3]. Pluto [10]\n[10]\nNext.")
    assert split_sentences(text) == [
        "Reykjavik is the capital.[1]",
        "It is in Iceland. [1][8]",
        "Galileo was Italian [2, 3].",
        "Pluto [10]\n[10]",
        "Next.",
    ]
    checks = _checks(text)
    assert checks[0].cites == [1] and checks[1].cites == [1, 8]
    assert checks[2].cites == [2, 3] and checks[3].cites == [10]


def test_lists_newlines_headings_tables_and_code():
    text = ("## Galileo\n"
            "**Key facts**\n"
            "His discoveries include:\n"
            "- the four largest moons of Jupiter [3]\n"
            "* born in Pisa [2]\n"
            "1. Died in 1642 [4]\n"
            "2) Italian [2]\n"
            "Steps: 1. Boil water. 2. Add eggs.\n"
            "Galileo was born\n"
            "in Pisa [2]. He was Italian [2].\n"
            "\n"
            "| Name | Born |\n"
            "|---|---|\n"
            "| Galileo | 1564 |\n"
            "```python\nprint('not. checked')\n```\n"
            "Done.")
    assert split_sentences(text) == [
        "## Galileo", "**Key facts**", "His discoveries include:",
        "the four largest moons of Jupiter [3]", "born in Pisa [2]",
        "Died in 1642 [4]", "Italian [2]",
        "Steps: 1. Boil water.", "2. Add eggs.",
        "Galileo was born\nin Pisa [2].", "He was Italian [2].",
        "| Name | Born |", "| Galileo | 1564 |", "Done.",
    ]


def test_thinking_is_never_checked():
    for raw in ("<think>The capital is Akureyri?</think>Reykjavik is the "
                "capital [1].",
                "Reykjavik is the capital [1].<think>but maybe Akureyri",
                "maybe Akureyri</think>\n\nReykjavik is the capital [1]."):
        checks = _checks(raw)
        assert [c.sentence for c in checks] == ["Reykjavik is the capital [1]."]
        assert checks[0].status == "supported"


# ─── sentences that claim nothing ───────────────────────────────────

@pytest.mark.parametrize("sentence", [
    "Hello!", "Hi there, Roszell!", "Great question.", "Thanks for asking!",
    "Sure!", "You're welcome.", "I hope this helps!",
    "Let me know if you need anything else.",
    "I don't know.", "I'm not sure about that.",
    "The sources don't say when Galileo married.",
    "My memory doesn't mention his wife.",
    "I couldn't find anything about Galileo's children.",
    "There is no information about his marriage in my sources.",
    "Would you like me to search for more?",
    "What year do you mean?",
    "Did he die in 1642?",
    "If you want, I can look it up.",
    "I can tell you more about Galileo if you like.",
    "Here's what I found:",
    "Here is a quick summary of what my memory says.",
    "That's all I have.",
])
def test_no_claim_sentences_are_left_alone(sentence):
    check = _one(sentence)
    assert check.status == "no_claim", check.note
    assert check.note


def test_the_claim_after_an_admission_is_still_checked():
    good = _one("I don't know when he married, but he was born in 1564 [2].")
    assert good.status == "supported"
    bad = _one("I'm not sure, but he was born in 1570 [2].")
    assert bad.status == "unsupported" and "1570" in bad.note


def test_did_you_know_is_a_claim_in_question_form():
    assert _one("Did you know that Galileo died in 1642 [4]?").status \
        == "supported"
    assert _one("Did you know that Galileo died in 1650 [4]?").status \
        == "unsupported"


def test_a_no_claim_sentence_citing_a_missing_source_is_flagged():
    check = _one("Here's what I found [42]:")
    assert check.status == "unsupported"
    assert check.note.startswith("bad citation") and "[42]" in check.note


# ─── supported: the sources say it, however it is worded ────────────

@pytest.mark.parametrize("sentence", [
    "Reykjavik is the capital of Iceland [1].",
    "Iceland's capital city is Reykjavík [1].",
    "The largest city in Iceland, Reykjavik, also serves as its capital [1].",
    "Galileo was an astronomer and physicist from Italy [2].",
    "He discovered Jupiter's four largest moons in 1610. [3]",
    "Galileo discovered the four biggest moons of Jupiter in 1610 [3].",
    "Galileo died in 1642 [4].",
    "The universe is 13.8 billion years old [5].",
    "Bananas contain a lot of potassium [7].",
    "Pluto is not considered a planet anymore [10].",
    "Your dog is called Astro [11].",
    "**Reykjavík** is the capital of Iceland [1].",
    "Reykjavik is the capital of Iceland (source 1).",
    "According to [1], Reykjavik is the capital of Iceland.",
])
def test_paraphrases_of_a_source_are_supported(sentence):
    check = _one(sentence)
    assert check.status == "supported", check.note
    assert check.best_evidence == check.cites[0]
    assert check.score > 0.5


def test_an_uncited_claim_is_checked_against_the_best_matching_source():
    check = _one("Galileo was born in Pisa in 1564.")
    assert check.status == "supported"
    assert check.cites == [] and check.best_evidence == 2
    assert "not cited" in check.note and "[2]" in check.note
    wrong = _one("Galileo was born in Pisa in 1570.")
    assert wrong.status == "unsupported"
    assert "no source mentions 1570" in wrong.note


def test_an_uncited_sentence_needs_enough_to_tie_it_to_a_source():
    # "big" is in [1] ("largest city") - one shared word says nothing
    check = _one("It is very big.")
    assert check.status == "unsupported"
    assert "too little" in check.note


def test_a_fake_citation_on_a_true_sentence_says_where_it_is():
    check = _one("Reykjavik is the capital of Iceland [99].")
    assert check.status == "unsupported"
    assert check.note == ("bad citation: there is no source [99]; "
                          "[1] says it")


def test_a_tool_result_backs_a_computed_number():
    assert _one("Galileo was 78 when he died [4][12].").status == "supported"
    assert _one("1642 minus 1564 is 78 [12].").status == "supported"
    assert _one("Galileo was 77 when he died [4][12].").status == "unsupported"


# ─── unsupported: what no source says ───────────────────────────────

def test_wrong_number_is_unsupported_and_named():
    check = _one("The universe is 14 billion years old [5].")
    assert check.status == "unsupported"
    assert "14 billion" in check.note
    assert _one("The universe is 13.8 million years old [5].").status \
        == "unsupported"
    assert _one("In 2020 the city had a population of 400,000 [9].").status \
        == "unsupported"


def test_wrong_year_and_wrong_day_are_unsupported():
    year = _one("Galileo died in 1650 [4].")
    assert year.status == "unsupported" and "1650" in year.note
    day = _one("Galileo died on January 9, 1642 [4].")
    assert day.status == "unsupported" and "January 9, 1642" in day.note


def test_invented_name_is_unsupported_and_named():
    check = _one("Galileo Smith discovered the moons of Jupiter [3].")
    assert check.status == "unsupported"
    assert "Smith" in check.note
    other = _one("Galileo was born in Florence [2].")
    assert other.status == "unsupported" and "Florence" in other.note


def test_invented_detail_without_numbers_or_names_is_unsupported():
    check = _one("Galileo was a famous Italian painter [2].")
    assert check.status == "unsupported" and "painter" in check.note
    assert _one("Bananas are blue [7].").status == "unsupported"


def test_fake_citation_is_a_failure_with_a_note():
    check = _one("Reykjavik is the capital of Iceland [99].")
    assert check.status == "unsupported"
    assert check.note.startswith("bad citation")
    assert "[99]" in check.note
    # one real and one invented source: still a broken citation
    mixed = _one("Reykjavik is the capital of Iceland [1][99].")
    assert mixed.status == "unsupported" and "[99]" in mixed.note
    assert mixed.best_evidence == 1


def test_citing_the_wrong_source_fails_but_says_which_one_has_it():
    check = _one("Galileo died in 1642 [6].")
    assert check.status == "unsupported"
    assert "[4] does say it" in check.note


def test_a_flipped_negation_is_unsupported():
    assert _one("Galileo was not born in Pisa [2].").status == "unsupported"
    assert _one("Pluto is a planet [10].").status == "unsupported"


def test_words_that_only_look_like_negations():
    assert _one("No, Galileo died in 1642 [4].").status == "supported"
    assert _one("Galileo was not only an astronomer but also a physicist "
                "[2].").status == "supported"
    assert _one("Pluto is no longer considered a planet [10].").status \
        == "supported"


def test_claims_without_sources_are_unsupported():
    check = _one("Reykjavik is the capital of Iceland [1].", evidence=[])
    assert check.status == "unsupported"
    assert _one("Reykjavik is the capital of Iceland.",
                evidence=[]).note == "there were no sources to check it against"


# ─── several citations on one sentence ──────────────────────────────

@pytest.mark.parametrize("cites", ["[2][3]", "[2, 3]", "[2,3]", "[2-3]",
                                   "[2][3][7]"])
def test_one_sentence_backed_by_several_sources_together(cites):
    check = _one("Galileo was born in Pisa in 1564 and discovered Jupiter's "
                 f"moons in 1610 {cites}.")
    assert check.status == "supported", check.note
    assert 2 in check.cites and 3 in check.cites


def test_inline_citations_for_each_part_of_a_sentence():
    check = _one("Galileo was born in Pisa [2] and died in Arcetri in 1642 [4].")
    assert check.status == "supported" and check.cites == [2, 4]


def test_multiple_citations_cannot_be_mixed_into_a_new_fact():
    # [2] has Galileo's birth year, [6] is about Marie Curie
    check = _one("Marie Curie was born in 1564 [2][6].")
    assert check.status == "unsupported" and "mixes" in check.note
    # Pisa is from [2], 1610 is from [3] - but [3] isn't about a birth
    assert _one("Galileo was born in Pisa [2] in 1610 [3].").status \
        == "unsupported"


def test_one_of_several_cited_sources_is_enough():
    check = _one("Galileo died in 1642 [3][4].")
    assert check.status == "supported" and check.best_evidence == 4
    assert check.note == "backed by [4]"


# ─── spelled differently, said the same ─────────────────────────────

@pytest.mark.parametrize("date", [
    "8 January 1642", "January 8, 1642", "January 8 1642",
    "the 8th of January, 1642", "Jan. 8, 1642", "1642-01-08", "8/1/1642",
    "January 1642", "1642",
])
def test_dates_written_any_way_match(date):
    check = _one(f"Galileo died on {date} [4].")
    assert check.status == "supported", check.note


def test_accents_and_old_letters_fold():
    assert _one("Reykjavik is the capital of Iceland [1].").status \
        == "supported"
    ev = [Evidence(1, "Reykjavik is the capital of Iceland.", "wikipedia:Iceland")]
    assert _one("Reykjavík is the capital of Iceland [1].",
                evidence=ev).status == "supported"
    assert _one("Thingvellir is a national park about 40 kilometres from "
                "Reykjavik [8].").status == "supported"
    assert _one("Þingvellir lies 40 km from Reykjavík [8].").status \
        == "supported"


@pytest.mark.parametrize("number", ["390,000", "390000", "390 thousand"])
def test_number_formats_match(number):
    check = _one(f"In 2020 the city had a population of {number} [9].")
    assert check.status == "supported", check.note


def test_number_words_match_digits():
    ev = [Evidence(1, "Jupiter has 4 large moons found by Galileo.", "x")]
    assert _one("Galileo found four large moons of Jupiter [1].",
                evidence=ev).status == "supported"
    assert _one("Galileo found five large moons of Jupiter [1].",
                evidence=ev).status == "unsupported"


# ─── a whole answer ─────────────────────────────────────────────────

LIST_ANSWER = """Sure! Here's what my memory says about Galileo:

- He was born in Pisa in 1564 [2].
- He discovered the four largest moons of Jupiter in 1610 [3].
- He died in 1650 [4].
- He invented the telescope [3].

Would you like to know more?"""


def test_a_list_answer_is_checked_item_by_item():
    checks = _checks(LIST_ANSWER)
    assert [c.status for c in checks] == [
        "no_claim", "no_claim",
        "supported", "supported", "unsupported", "unsupported",
        "no_claim"]
    assert "1650" in checks[4].note
    assert "telescope" in checks[5].note


def test_empty_answers():
    for empty in ("", "   \n  ", "<think>thinking only</think>"):
        assert check_answer(empty, EVIDENCE) == []
        assert annotate(empty, [], EVIDENCE) == ""
    assert summarize([])["sentences"] == 0
    assert summarize([])["verdict"] == "no_claims"


def test_a_long_answer_is_checked_quickly():
    evidence = [Evidence(i + 1,
                         f"The town of Place{i} had {1000 + 37 * i:,} people "
                         f"in {1900 + i} and lies on the river Flow{i}.",
                         f"wikipedia:Place{i}", "2026-07-02")
                for i in range(30)]
    sentences, expected = [], []
    for k in range(50):
        i = k % 30
        if k % 3 == 0:
            sentences.append(f"Place{i} had {1000 + 37 * i} people in "
                             f"{1900 + i} [{i + 1}].")
            expected.append("supported")
        elif k % 3 == 1:
            sentences.append(f"The town of Place{i} lies on the river Flow{i}.")
            expected.append("supported")
        else:
            sentences.append(f"Place{i} had {5000 + i} people in {1800 + i}.")
            expected.append("unsupported")
    answer = " ".join(sentences)
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        checks = check_answer(answer, evidence)
        best = min(best, time.perf_counter() - t0)
    assert [c.status for c in checks] == expected
    assert best < 0.2, f"{best * 1000:.0f} ms"


# ─── meaning, when an encoder is given ──────────────────────────────

class _SynonymEncoder:
    """A stand-in for SemanticEncoder.focus_alignment that knows that
    'spotted' means 'discovered' - and nothing else."""

    def __init__(self):
        self.calls = 0

    def focus_alignment(self, focus_words, texts, reduce="mean"):
        self.calls += 1
        return [0.95 if focus_words == ["spotted"] and "discovered" in t
                else 0.1 for t in texts]


def test_an_encoder_lets_meaning_match_a_different_word():
    sentence = "Galileo spotted Jupiter's four largest moons in 1610 [3]."
    assert _one(sentence).status == "unsupported"
    enc = _SynonymEncoder()
    assert _one(sentence, encoder=enc).status == "supported"
    assert enc.calls >= 1
    # meaning never excuses a wrong number or name
    assert _one("Galileo spotted Jupiter's four largest moons in 1611 [3].",
                encoder=enc).status == "unsupported"
    assert _one("Galileo spotted the moons of Saturn in 1610 [3].",
                encoder=enc).status == "unsupported"
    # and works for uncited claims too
    assert _one("Galileo spotted Jupiter's four largest moons in 1610.",
                encoder=enc).status == "supported"


def test_a_failing_encoder_falls_back_to_words():
    class Broken:
        def focus_alignment(self, *a, **k):
            raise RuntimeError("model not loaded")
    checks = _checks("Galileo died in 1642 [4]. Galileo spotted the moons of "
                     "Jupiter in 1610 [3].", encoder=Broken())
    assert [c.status for c in checks] == ["supported", "unsupported"]
    assert _one("Galileo died in 1642 [4].", encoder=object()).status \
        == "supported"


def test_the_real_semantic_encoder_plugs_in(fresh_state):
    from lattice.semantic_encoder import SemanticEncoder
    enc = SemanticEncoder()             # the fake MiniLM in tests
    answer = ("Reykjavik is the capital of Iceland [1]. Galileo died in "
              "1650 [4]. Galileo Smith discovered moons [3].")
    assert [c.status for c in _checks(answer, encoder=enc)] == \
        ["supported", "unsupported", "unsupported"]


# ─── what the user sees ─────────────────────────────────────────────

def test_annotate_marks_unverified_sentences_and_lists_sources():
    answer = ("Reykjavik is the capital of Iceland [1]. Galileo died in "
              "1650 [4]. Galileo died in 1642. [4]")
    checks = _checks(answer)
    shown = annotate(answer, checks, EVIDENCE)
    body, note, sources = shown.split("\n\n")
    assert body == ("Reykjavik is the capital of Iceland [1]. Galileo died in "
                    "1650 [4] [unverified]. Galileo died in 1642. [4]")
    assert "[unverified]" in note
    assert sources.splitlines() == [
        "Sources:",
        "[1] wikipedia:Iceland, 2026-07-02",
        "[4] wikipedia:Galileo Galilei, 2026-07-03",
    ]


def test_annotate_cites_the_source_behind_an_uncited_sentence():
    answer = "Galileo was born in Pisa in 1564."
    checks = _checks(answer)
    shown = annotate(answer, checks, EVIDENCE)
    assert shown.startswith("Galileo was born in Pisa in 1564 [2].")
    assert "[2] wikipedia:Galileo Galilei, 2026-07-02" in shown
    plain = annotate(answer, checks, EVIDENCE, cite_matches=False)
    assert plain == "Galileo was born in Pisa in 1564."


def test_annotate_says_plainly_when_nothing_could_be_verified():
    answer = "Galileo died in 1650 [4]. He was 90."
    shown = annotate(answer, _checks(answer), EVIDENCE)
    assert "1650 [4] [unverified]." in shown and "90 [unverified]." in shown
    assert "couldn't verify any of this" in shown
    no_sources = annotate(answer, check_answer(answer, []), [])
    assert "no sources" in no_sources and "Sources:" not in no_sources


def test_annotate_leaves_a_claimless_answer_alone():
    answer = "I don't know. The sources don't say when he married."
    assert annotate(answer, _checks(answer), EVIDENCE) == answer


def test_annotate_replaces_the_models_own_source_list():
    answer = ("Reykjavik is the capital of Iceland [1].\n\nSources:\n"
              "[1] Wikipedia - Iceland\n[2] Wikipedia - Galileo")
    checks = _checks(answer)
    assert [c.sentence for c in checks] == [
        "Reykjavik is the capital of Iceland [1]."]
    shown = annotate(answer, checks, EVIDENCE)
    assert shown == ("Reykjavik is the capital of Iceland [1].\n\nSources:\n"
                     "[1] wikipedia:Iceland, 2026-07-02")


def test_annotate_marks_bad_citations_and_skips_them_in_sources():
    answer = "Reykjavik is the capital of Iceland [1][99]."
    shown = annotate(answer, _checks(answer), EVIDENCE)
    assert shown.startswith("Reykjavik is the capital of Iceland [1][99] "
                            "[unverified].")
    assert "[99] " not in shown.split("Sources:")[1]


def test_annotate_marks_list_items():
    shown = annotate(LIST_ANSWER, _checks(LIST_ANSWER), EVIDENCE)
    assert "- He died in 1650 [4] [unverified]." in shown
    assert "- He invented the telescope [3] [unverified]." in shown
    assert "- He was born in Pisa in 1564 [2]." in shown
    assert shown.rstrip().endswith("[4] wikipedia:Galileo Galilei, 2026-07-03")


# ─── counts ─────────────────────────────────────────────────────────

def test_summarize_counts_the_verdicts():
    answer = ("Hello! Reykjavik is the capital of Iceland [1]. Galileo died "
              "in 1650 [4]. Bananas are rich in potassium. Pluto is a moon "
              "[77].")
    s = summarize(_checks(answer))
    assert s["sentences"] == 5 and s["claims"] == 4 and s["no_claim"] == 1
    assert s["supported"] == 2 and s["unsupported"] == 2
    assert s["supported_share"] == 0.5
    assert s["uncited"] == 1 and s["bad_citations"] == 1
    assert s["verdict"] == "partly_verified"
    assert s["unsupported_sentences"] == ["Galileo died in 1650 [4].",
                                          "Pluto is a moon [77]."]
    assert summarize(_checks("Galileo died in 1642 [4]."))["verdict"] \
        == "verified"
    assert summarize(_checks("Galileo died in 1650 [4]."))["verdict"] \
        == "unverified"
    assert summarize(_checks("I don't know."))["verdict"] == "no_claims"
