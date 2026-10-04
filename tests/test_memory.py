"""The one memory: every command shares one file, nothing gets orphaned."""
import sqlite3

import numpy as np


def _texts(db):
    con = sqlite3.connect(str(db))
    try:
        return [r[0] for r in con.execute("SELECT text FROM memories")]
    finally:
        con.close()


def test_chat_and_learn_share_one_memory(fresh_state):
    from mind.fluency import FluentTelp
    from lattice.standalone_agent import StandaloneAgent
    from lattice import vision

    t = FluentTelp()
    t.respond("remember that the Zorblax river flows through Quendia")
    assert t.agent.lattice.db_path == str(vision.CHAT_LATTICE)

    # what `telp learn/see/watch` do before fetching anything
    StandaloneAgent(lattice_path=vision.CHAT_LATTICE, skip_ngram_retrain=True)

    t2 = FluentTelp()
    assert any("Zorblax" in x for x in t2.agent.lattice._texts)
    assert not (fresh_state / "standalone_lattice.db").exists()


def test_legacy_second_memory_is_recovered(fresh_state):
    from lattice.store import Lattice
    from lattice.standalone_agent import StandaloneAgent
    from lattice.semantic_encoder import SemanticEncoder

    enc = SemanticEncoder()
    legacy = fresh_state / "standalone_lattice.db"
    old = Lattice(legacy, encoder=enc)
    old.add("Astro is the name of my dog.", source="user_taught")
    old.close()
    main = Lattice(fresh_state / "concept_bridge.db", encoder=enc)
    main.add("Iceland's capital is Reykjavik.", source="wikipedia:Iceland")
    main.close()

    agent = StandaloneAgent()
    assert "Astro is the name of my dog." in agent.lattice._texts
    assert "Iceland's capital is Reykjavik." in agent.lattice._texts
    assert not legacy.exists()
    assert (fresh_state / "standalone_lattice.db.merged").exists()

    # runs once: a second boot doesn't duplicate anything
    agent2 = StandaloneAgent()
    assert agent2.lattice._texts.count("Astro is the name of my dog.") == 1


def test_listing_sights_does_not_create_a_memory_file(fresh_state):
    from lattice import vision
    db = fresh_state / "concept_bridge.db"
    assert vision.sights(db) == []
    assert vision.watched(db) == []
    assert not db.exists()


def test_appends_keep_stack_consistent_and_search_works(fresh_state):
    from lattice.store import Lattice
    from lattice.semantic_encoder import SemanticEncoder

    lat = Lattice(fresh_state / "m.db", encoder=SemanticEncoder())
    texts = [f"fact number {i} about topic {i % 7}" for i in range(150)]
    for x in texts[:100]:
        lat.add(x, source="t")
    enc = lat.encoder
    lat.add_many([(x, enc.encode(x), "t") for x in texts[100:]])
    assert lat._stack.shape == (150, enc.dim)
    for i in (0, 99, 100, 149):
        assert np.array_equal(lat._stack[i], enc.encode(texts[i]))
    assert lat.query(texts[123], k=1)[0]["text"] == texts[123]


def test_daemon_notices_writes_from_other_processes(fresh_state):
    from lattice.store import Lattice
    from lattice.semantic_encoder import SemanticEncoder

    enc = SemanticEncoder()
    a = Lattice(fresh_state / "m.db", encoder=enc)
    a.add("first", source="t")
    assert not a.changed_on_disk()          # own writes don't count
    b = Lattice(fresh_state / "m.db", encoder=enc)
    ids = b._ids[:]
    b.delete_ids(ids)                       # forget one ...
    b.add("second", source="t")             # ... learn one: same row count
    assert a.changed_on_disk()
    a._reload_from_disk()
    assert a._texts == ["second"]
    assert not a.changed_on_disk()


def test_outside_commit_before_our_own_write_is_still_noticed(fresh_state):
    from lattice.store import Lattice
    from lattice.semantic_encoder import SemanticEncoder
    enc = SemanticEncoder()
    a = Lattice(fresh_state / "m.db", encoder=enc)
    b = Lattice(fresh_state / "m.db", encoder=enc)
    a.add("x", source="t")
    b.add("taught elsewhere", source="t")
    a.add("own", source="t")
    assert a.changed_on_disk()


def test_legacy_merge_runs_once_even_if_rename_fails(fresh_state):
    import shutil
    from lattice.store import Lattice, merge_legacy_memory
    from lattice.semantic_encoder import SemanticEncoder
    enc = SemanticEncoder()
    legacy, target = fresh_state / "legacy.db", fresh_state / "main.db"
    old = Lattice(legacy, encoder=enc)
    old.add("Astro is the name of my dog.", source="user_taught")
    old.close()
    backup = fresh_state / "copy.db"
    shutil.copy(legacy, backup)
    assert merge_legacy_memory(target, legacy, encoder=enc) == 1
    # the user forgets the fact; the legacy file reappears (failed rename)
    main = Lattice(target, encoder=enc)
    main.delete_ids(main._ids[:])
    main.close()
    shutil.copy(backup, legacy)
    assert merge_legacy_memory(target, legacy, encoder=enc) == 0
    assert Lattice(target, encoder=enc).count() == 0


def test_legacy_merge_reencodes_old_vectors(fresh_state):
    import sqlite3
    from lattice.store import Lattice, merge_legacy_memory, _SCHEMA
    from lattice.semantic_encoder import SemanticEncoder
    enc = SemanticEncoder()
    legacy, target = fresh_state / "legacy.db", fresh_state / "main.db"
    con = sqlite3.connect(str(legacy))
    con.executescript(_SCHEMA)
    con.execute("INSERT INTO memories (created_at, text, hv, source) "
                "VALUES ('2026-01-01', 'Astro is my dog.', ?, 'user_taught')",
                (np.zeros(enc.dim, dtype=np.int8).tobytes(),))
    con.commit()
    con.close()
    merge_legacy_memory(target, legacy, encoder=enc)
    lat = Lattice(target, encoder=enc)
    assert np.array_equal(lat._stack[0], enc.encode("Astro is my dog."))
