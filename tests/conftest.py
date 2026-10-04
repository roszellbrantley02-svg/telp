"""
Test harness: offline, deterministic, isolated.

* TELP_STATE_DIR points at a throwaway directory BEFORE any Telp module is
  imported, so no test can ever touch real memories in state/.
* sentence_transformers is replaced by tests/fake_minilm.py, a hashed
  bag-of-words model - no network, no 80MB MiniLM download, no torch.
* `fresh_state` re-points every module-level store path at a per-test
  directory, so each test starts with an empty mind.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["TELP_STATE_DIR"] = tempfile.mkdtemp(prefix="telp-test-state-")
os.environ.pop("TELP_RI", None)
os.environ.pop("TELP_USE_LEARNED", None)


# ─── fake MiniLM ────────────────────────────────────────────────────

from tests.fake_minilm import install as _install_fake_minilm  # noqa: E402

_install_fake_minilm()


# ─── per-test state ─────────────────────────────────────────────────

REAL_GROWTH: dict = {}

@pytest.fixture
def fresh_state(tmp_path, monkeypatch):
    """Point every store at an empty per-test state directory."""
    import lattice.paths as paths
    monkeypatch.setattr(paths, "STATE_DIR", tmp_path)
    monkeypatch.setattr(paths, "MEMORY_DB", tmp_path / "concept_bridge.db")
    monkeypatch.setattr(paths, "LEGACY_MEMORY_DB",
                        tmp_path / "standalone_lattice.db")
    patches = {
        "lattice.standalone_agent": {
            "DEFAULT_LATTICE_DB": tmp_path / "concept_bridge.db",
            "MEMORY_DB": tmp_path / "concept_bridge.db",
            "LEGACY_MEMORY_DB": tmp_path / "standalone_lattice.db"},
        "lattice.vision": {"CHAT_LATTICE": tmp_path / "concept_bridge.db",
                           "SIGHTS_DIR": tmp_path / "sights"},
        "mind.fluency": {"MEMORY_DB": tmp_path / "concept_bridge.db"},
        "mind.persona": {"PERSONA_DB": tmp_path / "persona.db"},
        "mind.user_facts": {"USER_FACTS_DB": tmp_path / "user_facts.db"},
        "mind.code_corpus": {"CODE_CORPUS_DB": tmp_path / "code_corpus.db"},
        "mind.seed_identity": {"_MARKER": tmp_path / ".identity_seeded"},
        "mind.restyle": {"_DICT_DB": tmp_path / "wiktionary" / "dict.db"},
    }
    import importlib
    for mod_name, attrs in patches.items():
        mod = importlib.import_module(mod_name)
        for attr, val in attrs.items():
            if hasattr(mod, attr):
                monkeypatch.setattr(mod, attr, val)
    # StandaloneAgent's default argument was bound at import time
    import lattice.standalone_agent as sa
    defaults = list(sa.StandaloneAgent.__init__.__defaults__)
    defaults[0] = tmp_path / "concept_bridge.db"
    monkeypatch.setattr(sa.StandaloneAgent.__init__, "__defaults__",
                        tuple(defaults))
    # learn-on-miss must never reach the network from tests (tests that
    # exercise the real learners stub the fetchers and use REAL_GROWTH)
    import lattice.growth as growth
    REAL_GROWTH.setdefault("learn_topic", growth.learn_topic)
    monkeypatch.setattr(growth, "learn_topic",
                        lambda *a, **k: {"title": "offline", "added": 0,
                                         "error": "offline in tests"})
    monkeypatch.setattr(growth, "learn_howto",
                        lambda *a, **k: {"title": "offline", "added": 0,
                                         "error": "offline in tests"})
    monkeypatch.setattr(growth, "search_wiki", lambda *a, **k: [])
    return tmp_path


@pytest.fixture
def telp(fresh_state):
    """A freshly booted FluentTelp on an empty per-test memory."""
    from mind.fluency import FluentTelp
    t = FluentTelp()
    # deterministic voice: no random openers in assertions
    if t.voice is not None:
        t.voice._rng.seed(0) if hasattr(t.voice, "_rng") else None
    return t
