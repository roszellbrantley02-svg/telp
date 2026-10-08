"""The brief: what the language model reads in harness mode - small, cited,
inside a token budget, with a stable prefix and a constant-size memory of
the conversation."""
import sqlite3
from dataclasses import dataclass
from datetime import date

import pytest

from mind.brief import SYSTEM_PROMPT, BriefBuilder, ConversationState
from mind.harness_types import Evidence, estimate_tokens

TODAY = date(2026, 10, 8)

WORLD = [
    ("Reykjavik is the capital and largest city of Iceland.",
     "wikipedia:Iceland"),
    ("Iceland is an island country in the North Atlantic Ocean.",
     "wikipedia:Iceland"),
    ("Galileo Galilei was an Italian astronomer and physicist.",
     "wikipedia:Galileo Galilei"),
    ("Galileo Galilei was born in Pisa in 1564.",
     "wikipedia:Galileo Galilei"),
    ("Galileo discovered the four largest moons of Jupiter in 1610.",
     "wikipedia:Galileo Galilei"),
    ("Marie Curie was born in Warsaw in 1867.", "wikipedia:Marie Curie"),
    ("Bananas are rich in potassium and grow in tropical climates.",
     "wikipedia:Banana"),
    ("The Pacific Ocean is the largest ocean on Earth.",
     "wikipedia:Pacific Ocean"),
]


def _teach_world(telp) -> dict:
    return {text: telp.agent.lattice.add(text, source=src)
            for text, src in WORLD}


def _builder(telp, **kw) -> BriefBuilder:
    return BriefBuilder(telp.agent, user_facts=telp.user_facts, **kw)


def _state(telp, session: str = "default") -> ConversationState:
    return ConversationState(telp.agent.lattice._con, session=session,
                             encoder=telp.agent.encoder)


# ─── evidence from memory ───────────────────────────────────────────

def test_relevant_memories_come_first_numbered_and_dated(telp):
    ids = _teach_world(telp)
    brief = _builder(telp).build("what is the capital of Iceland?",
                                 today=TODAY)
    ev = brief.evidence
    assert ev and ev[0].text.startswith("Reykjavik is the capital")
    assert [e.n for e in ev] == list(range(1, len(ev) + 1))
    top = ev[0]
    assert top.source == "wikipedia:Iceland" and top.kind == "memory"
    assert top.memory_id == ids[top.text]
    assert top.created_at and top.created_at[:4].isdigit()
    assert top.score > 0
    # precision: nothing about bananas, oceans or astronomers
    shown = " ".join(e.text for e in ev)
    assert "Bananas" not in shown and "Galileo" not in shown
    assert "Pacific" not in shown
    user = brief.messages()[1]["content"]
    assert "[1] Reykjavik is the capital" in user
    assert user.endswith("Question: what is the capital of Iceland?")


def test_chat_echoes_stories_and_unasked_sights_are_not_evidence(telp):
    lat = telp.agent.lattice
    lat.add("Reykjavik is the capital and largest city of Iceland.",
            source="wikipedia:Iceland")
    lat.add("what is the capital of Iceland?", source="user_msg")
    lat.add("The capital of Iceland is Reykjavik, I believe.",
            source="agent_response")
    lat.add("User: capital of Iceland? Telp: the capital is Reykjavik.",
            source="conversation_turn")
    lat.add("Once upon a time the capital of Iceland was ruled by a dragon.",
            source="story:dragon")
    lat.add("Image: a photo of the capital of Iceland at night.",
            source="image:/tmp/reykjavik.jpg")
    builder = _builder(telp)
    brief = builder.build("what is the capital of Iceland?", today=TODAY)
    assert {e.source for e in brief.evidence} == {"wikipedia:Iceland"}
    # a question about what he saw may use what he saw - and still no
    # echoes or stories
    seen = builder.build("what did you see in the photo of Iceland?",
                         today=TODAY)
    sources = [e.source for e in seen.evidence]
    assert any(s.startswith("image:") for s in sources)
    assert not any(s in ("user_msg", "agent_response", "conversation_turn")
                   or s.startswith("story:") for s in sources)


def test_near_duplicates_are_dropped_but_disagreements_stay(telp):
    lat = telp.agent.lattice
    lat.add("Reykjavik is the capital and largest city of Iceland.",
            source="wikipedia:Iceland")
    lat.add("Reykjavik is the capital and largest city of Iceland.",
            source="user_taught")
    lat.add("Reykjavík is the capital and the largest city of Iceland!",
            source="web:example")
    lat.add("Jupiter has 95 confirmed moons.", source="user_taught")
    lat.add("Jupiter has 115 confirmed moons.",
            source="wikipedia:Moons of Jupiter")
    builder = _builder(telp)
    brief = builder.build("what is the capital of Iceland?", today=TODAY)
    assert len([e for e in brief.evidence if "capital" in e.text]) == 1
    moons = builder.build("how many moons does Jupiter have?", today=TODAY)
    counts = sorted(e.text for e in moons.evidence if "moons" in e.text)
    assert counts == ["Jupiter has 115 confirmed moons.",
                      "Jupiter has 95 confirmed moons."]


def test_tool_results_come_first_and_the_callers_items_are_untouched(telp):
    _teach_world(telp)
    tool = Evidence(n=99, text="17 * 23 = 391", source="tool:calculate",
                    kind="tool")
    brief = _builder(telp).build("what is the capital of Iceland?",
                                 extra_evidence=[tool], today=TODAY)
    assert brief.evidence[0].text == "17 * 23 = 391"
    assert brief.evidence[0].n == 1 and brief.evidence[0].kind == "tool"
    assert brief.evidence[1].text.startswith("Reykjavik")
    assert tool.n == 99


def test_empty_memory_says_none_found(telp):
    brief = _builder(telp).build("what is the capital of Atlantis?",
                                 today=TODAY)
    assert brief.evidence == []
    assert "Sources for this question: none found." in \
        brief.messages()[1]["content"]
    assert brief.standing == ""
    assert brief.state == "Today is Thursday, 8 October 2026 (2026-10-08)."
    assert brief.token_estimate() <= 1800


def test_search_digs_deeper_without_repeating_what_was_shown(telp):
    _teach_world(telp)
    builder = _builder(telp)
    brief = builder.build("who was Galileo Galilei?", today=TODAY)
    shown = [e for e in brief.evidence if "Galileo" in e.text]
    assert shown
    more = builder.search("Galileo Galilei", limit=10, exclude=shown)
    assert all(m.memory_id not in {e.memory_id for e in shown}
               for m in more)
    every = builder.search("Galileo Galilei", limit=10)
    assert {e.memory_id for e in every} >= \
        {e.memory_id for e in shown} | {m.memory_id for m in more}


# ─── the budget ─────────────────────────────────────────────────────

def test_budget_holds_for_huge_inputs_and_never_cuts_mid_sentence(telp):
    lat = telp.agent.lattice
    long_text = " ".join(
        f"Hekla in Iceland erupted in the year {1100 + i} and covered the "
        f"farms around it in ash." for i in range(40))
    stored = []
    for i in range(60):
        stored.append(f"Iceland volcano note {i}. " + long_text)
        lat.add(stored[-1], source=f"wikipedia:Hekla {i}")
    for i in range(60):
        stored.append(f"The Hekla volcano in Iceland erupted violently in "
                      f"the year {1000 + i}.")
        lat.add(stored[-1], source="wikipedia:Hekla")
    for i in range(60):
        telp.user_facts.add(f"User's hobby number {i} is watching Iceland "
                            f"volcano videos.")
    state = _state(telp)
    for i in range(30):
        state.add_turn("tell me about the volcanoes of Iceland " * 20,
                       long_text)
    huge_question = ("Here is my essay about Iceland. " + long_text * 5
                     + " So when did Hekla erupt?")
    extra = [Evidence(n=0, text=long_text, source="tool:search_memory",
                      kind="tool")]
    for budget in (500, 1800, 6000):
        builder = BriefBuilder(telp.agent, user_facts=telp.user_facts,
                               budget_tokens=budget)
        for q in ("did Hekla erupt in Iceland?", huge_question):
            brief = builder.build(q, state=state, extra_evidence=extra,
                                  today=TODAY)
            assert brief.token_estimate() <= budget, (budget, q[:30])
            assert brief.evidence, budget
            for e in brief.evidence:
                # whole sentences only: a leading run of the stored text
                assert e.text.endswith((".", "!", "?"))
                if e.kind == "memory":
                    assert any(s.startswith(e.text) for s in stored)
            if q is huge_question:
                assert len(brief.question) < len(q)
                assert brief.question.endswith("So when did Hekla erupt?")
                assert brief.question.startswith("Here is my essay")
            assert brief.naive_tokens >= brief.token_estimate()


def test_a_budget_smaller_than_the_instructions_is_refused(telp):
    with pytest.raises(ValueError):
        BriefBuilder(telp.agent, budget_tokens=100)


# ─── the stable prefix ──────────────────────────────────────────────

def test_system_and_standing_are_byte_identical_across_turns(telp):
    _teach_world(telp)
    telp.user_facts.add("User likes astronomy.")
    telp.user_facts.add("User's name is Eric.")
    builder = _builder(telp)
    state = _state(telp)
    prefixes = []
    questions = ["what is the capital of Iceland?", "who was Galileo?",
                 "when was he born?", "which ocean is the largest?",
                 "what do bananas contain?"]
    for i, q in enumerate(questions):
        brief = builder.build(q, state=state, today=TODAY)
        system, user = (m["content"] for m in brief.messages())
        prefixes.append((system, brief.standing,
                         user.split("Conversation so far:")[0]))
        state.add_turn(q, f"Answer number {i} [1].", brief.evidence)
    assert all(p == prefixes[0] for p in prefixes)
    system, standing, _ = prefixes[0]
    assert system == SYSTEM_PROMPT
    # who the user is comes first, whatever order it was learned in
    assert standing == "- User's name is Eric.\n- User likes astronomy."

    # the date changes daily, so it lives after the cached prefix
    later = builder.build("who was Galileo?", state=state,
                          today=date(2026, 10, 9))
    assert later.messages()[0]["content"] == SYSTEM_PROMPT
    assert "2026" not in SYSTEM_PROMPT
    assert "Friday, 9 October 2026" in later.state

    # a new fact extends the block at its end: the old prefix still holds
    telp.user_facts.add("User has two cats.")
    grown = builder.build("who was Galileo?", state=state, today=TODAY)
    assert grown.standing.startswith(standing)
    assert grown.standing.endswith("- User has two cats.")


def test_standing_is_capped_and_overflow_facts_come_when_relevant(telp):
    for i in range(40):
        telp.user_facts.add(f"User's note number {i} is about gardening "
                            f"tomatoes.")
    telp.user_facts.add("User's dog is named Astro.")
    builder = _builder(telp)
    brief = builder.build("what is my dog called?", today=TODAY)
    assert estimate_tokens(brief.standing) <= 0.15 * builder.room + 1
    assert "Astro" not in brief.standing
    hits = [e for e in brief.evidence if "Astro" in e.text]
    assert hits and hits[0].source == "user_facts" and hits[0].kind == "fact"
    # an unrelated question doesn't drag it in
    other = builder.build("what is the capital of Iceland?", today=TODAY)
    assert not any("Astro" in e.text for e in other.evidence)
    assert other.standing == brief.standing


# ─── the conversation ───────────────────────────────────────────────

def test_follow_up_pronouns_search_the_recent_subject(telp):
    _teach_world(telp)
    builder = _builder(telp)
    state = _state(telp)
    first = builder.build("who was Galileo Galilei?", state=state,
                          today=TODAY)
    state.add_turn("who was Galileo Galilei?",
                   "Galileo Galilei was an Italian astronomer [1].",
                   first.evidence)
    assert state.focus_entities()[0] == "Galileo Galilei"
    follow = builder.build("when was he born?", state=state, today=TODAY)
    texts = [e.text for e in follow.evidence]
    assert texts[0] == "Galileo Galilei was born in Pisa in 1564."
    curie = "Marie Curie was born in Warsaw in 1867."
    assert curie not in texts or texts.index(curie) > 0
    assert "Recently discussed: Galileo Galilei" in follow.state
    assert "User: who was Galileo Galilei?" in follow.state
    # the old turn's [1] pointed at old sources - not shown again
    assert "[1]" not in follow.state

    # 'he' skips the places and planets discussed since for the person
    for q, a in (("what is the capital of Iceland?",
                  "Reykjavik is the capital of Iceland [1]."),
                 ("how many moons does Jupiter have?",
                  "Jupiter has many moons [1].")):
        state.add_turn(q, a, builder.build(q, state=state,
                                           today=TODAY).evidence)
    later = builder.build("where was he born?", state=state, today=TODAY)
    assert later.evidence[0].text == \
        "Galileo Galilei was born in Pisa in 1564."
    # ... and 'it' the most recent thing
    big = builder.build("how big is it?", state=state, today=TODAY)
    assert big.evidence == [] or "Galileo" not in big.evidence[0].text


def test_conversation_summary_stays_constant_size_over_50_turns(telp):
    state = _state(telp)
    topics = ["Iceland", "Galileo", "Jupiter", "volcanoes", "bananas"]
    sizes = []
    for i in range(50):
        topic = topics[i % 5]
        state.add_turn(
            f"tell me more about {topic}, part {i}",
            f"<think>the user wants {topic}</think>{topic} part {i} is a "
            f"long story [1]. It has many details worth knowing [2]. "
            + "More words follow here. " * 40)
        summary = state.summary("what else about Galileo?", 450)
        sizes.append(estimate_tokens(summary))
    assert max(sizes) <= 450
    assert max(sizes[15:]) - min(sizes[15:]) <= 8      # flat, not growing
    assert sizes[-1] <= max(sizes[5:15]) + 4
    last = state.summary("what else about Galileo?", 450)
    assert "User: tell me more about bananas, part 49" in last
    assert "Telp: bananas part 49 is a long story. It has many details " \
        "worth knowing." in last
    assert "Earlier, related:" in last and "Galileo, part" in last
    assert "<think>" not in last and "[1]" not in last
    # a small budget is a hard ceiling
    assert estimate_tokens(state.summary("Galileo?", 60)) <= 60
    assert state.summary("Galileo?", 0) == ""


def test_summary_hides_model_thinking_and_old_citation_numbers(fresh_state):
    con = sqlite3.connect(str(fresh_state / "concept_bridge.db"))
    state = ConversationState(con)
    state.add_turn("who was Galileo?",
                   "<think>look at [1]</think>Galileo was an astronomer [1].")
    state.add_turn("where was he born?",
                   "the user means him</think>He was born in Pisa [1][2].")
    state.add_turn("and Kepler?", "Kepler was German [1]. <think>cut off")
    text = state.summary("tell me more about Galileo", 300)
    assert "Telp: Galileo was an astronomer." in text
    assert "Telp: He was born in Pisa." in text
    assert "Telp: Kepler was German." in text
    assert "think" not in text and "[1]" not in text
    assert "user means" not in text
    con.close()


def test_conversation_persists_across_instances_and_sessions(fresh_state):
    db = fresh_state / "concept_bridge.db"
    con = sqlite3.connect(str(db))
    state = ConversationState(con, session="chat-1")
    evidence = [
        Evidence(n=1, text="Galileo Galilei was an Italian astronomer.",
                 source="wikipedia:Galileo Galilei", memory_id=7,
                 created_at="2026-07-02T10:00:00+00:00"),
        Evidence(n=2, text="6 * 7 = 42", source="tool:calculate",
                 kind="tool")]
    assert state.add_turn("who was Galileo?",
                          "Galileo Galilei was an Italian astronomer [1].",
                          evidence) == 1
    assert state.add_turn("where was he born?",
                          "He was born in Pisa [1].") == 2
    summary = state.summary("when did he die?", 300)
    con.close()

    con2 = sqlite3.connect(str(db))
    again = ConversationState(con2, session="chat-1")
    turns = again.turns()
    assert [t["turn"] for t in turns] == [1, 2]
    refs = turns[0]["sources"]
    # memory rows are kept by reference (forgetting them leaves no copy);
    # tool results, which have no row, keep their text
    assert refs[0]["memory_id"] == 7 and "text" not in refs[0]
    assert refs[0]["source"] == "wikipedia:Galileo Galilei"
    assert refs[1]["text"] == "6 * 7 = 42"
    assert again.summary("when did he die?", 300) == summary
    names = again.focus_entities()
    assert names[0] == "Pisa" and "Galileo Galilei" in names
    assert "Galileo" not in names               # merged into the full name
    # he/she want a person, it a thing
    assert again.focus_entities(kind="person")[0] == "Galileo Galilei"
    assert again.focus_entities(kind="thing")[0] == "Pisa"
    assert again.add_turn("when did he die?", "In 1642.") == 3

    other = ConversationState(con2, session="chat-2")
    assert other.turns() == [] and other.summary("anything", 300) == ""
    assert other.add_turn("hello", "Hi.") == 1

    assert again.forget("Pisa") == 1
    assert [t["turn"] for t in again.turns()] == [1, 3]
    assert again.clear() == 2 and again.count() == 0
    assert other.count() == 1


def test_naive_tokens_grow_with_the_conversation_while_the_brief_stays_flat(
        telp):
    _teach_world(telp)
    builder = _builder(telp)
    state = _state(telp)
    questions = ["who was Galileo?", "what is the capital of Iceland?",
                 "where was Galileo born?", "what did Galileo discover?"]
    sent, naive = [], []
    for i in range(40):
        q = questions[i % 4]
        brief = builder.build(q, state=state, today=TODAY)
        sent.append(brief.token_estimate())
        naive.append(brief.naive_tokens)
        assert brief.naive_tokens >= brief.token_estimate()
        state.add_turn(q, "Galileo Galilei was an Italian astronomer born "
                       "in Pisa [1]. He found moons of Jupiter [2]. "
                       + "He did a great many other things too. " * 12,
                       brief.evidence)
    assert all(naive[i + 4] > naive[i] for i in range(len(naive) - 4))
    assert naive[-1] > 5 * naive[0]
    assert max(sent) <= builder.budget_tokens
    assert max(sent[8:]) - min(sent[8:]) <= 0.1 * builder.budget_tokens
    assert sent[-1] < naive[-1] / 4


# ─── facts (optional) ───────────────────────────────────────────────

@dataclass
class _Fact:
    subject: str
    relation: str
    obj: str
    source: str = ""
    created_at: str | None = None
    memory_id: int | None = None
    qualifiers: tuple = ()


class _FactMemory:
    def __init__(self, facts):
        self.facts = facts

    def resolve(self, name):
        return ("Galileo Galilei"
                if name.lower() in ("galileo", "galileo galilei") else None)

    def facts_about(self, name):
        return self.facts if name == "Galileo Galilei" else []


class _FactMind:
    def __init__(self, facts):
        self.memory = _FactMemory(facts)

    def sync(self):
        pass


class _BrokenFactMind:
    def sync(self):
        raise RuntimeError("the fact layer is unfinished")


def test_fact_lines_are_optional_and_a_broken_fact_layer_is_ignored(telp):
    _teach_world(telp)
    q = "who was Galileo?"
    plain = BriefBuilder(telp.agent).build(q, today=TODAY)
    broken = BriefBuilder(telp.agent, fact_mind=_BrokenFactMind()).build(
        q, today=TODAY)
    assert [e.text for e in broken.evidence] == \
        [e.text for e in plain.evidence]

    facts = [_Fact("Galileo Galilei", "born_in", "Pisa", "wikidata:Q307"),
             _Fact("Galileo Galilei", "died_year", "1642", "wikidata:Q307"),
             _Fact("Galileo Galilei", "pronoun", "he", "wikidata:Q307")]
    rich = BriefBuilder(telp.agent, fact_mind=_FactMind(facts)).build(
        q, today=TODAY)
    lines = [e for e in rich.evidence if e.kind == "fact"]
    assert len(lines) == 2                      # bookkeeping is not shown
    assert lines[0].text.startswith("Galileo Galilei - ")
    assert lines[0].text.endswith(": Pisa")
    assert lines[1].text.endswith(": 1642")
    assert lines[0].source == "wikidata:Q307"
    assert [e.n for e in rich.evidence] == \
        list(range(1, len(rich.evidence) + 1))
    assert rich.evidence[-1].kind == "fact"     # after the memory sentences
