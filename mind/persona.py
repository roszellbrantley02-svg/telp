"""autopilot/persona.py — Telp's identity as an HDC operator.

This module implements personality at the substrate level — not as a
system prompt, but as a hypervector subspace.

The core idea:

  TELP_SELF is a single 10,000-bit hypervector generated deterministically
  (seeded). Every persona fact is stored as bind(TELP_SELF, encode(text)),
  which puts Telp's self-knowledge in a SUBSPACE distinct from world
  knowledge.

  When the user asks something personal/opinionated, we query with
  bind(TELP_SELF, encode(query)) — this retrieves the persona subspace
  preferentially over generic Wikipedia hits.

  When composing a response, we can optionally "tag" the output with
  TELP_SELF so the generator learns Telp-shaped sequences over time.

The four trait dimensions — determined, kind, full, resilient — are
also stored as hypervectors. Persona facts can be bound additionally
with one of these trait vectors so retrieval can target a specific
trait when relevant.

References:
  Kanerva (2009) — Hyperdimensional Computing
  Frady & Sommer (2019) — Robust computation via subspace addressing
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import (
    HDVocabulary, bind, bundle, hamming_distance, random_vec, D,
)


# ─── Deterministic TELP_SELF vector ────────────────────────────────


def _deterministic_random_vec(seed_key: str) -> np.ndarray:
    """Generate a binary HV deterministically from a string seed.

    Same seed → same vector across all processes / restarts.  This
    is critical: persona-bound memories stored on disk must match the
    same TELP_SELF at load time.

    Uses SHA-1 (process-stable) — Python's hash() is salted per-process
    and does NOT give the cross-process stability this function claims.
    """
    import hashlib as _hl
    _h = _hl.sha1(
        f"hdc_self_seed_v1::{seed_key}".encode("utf-8")
    ).digest()
    seed = int.from_bytes(_h[:4], "big")
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=D, dtype=np.int8)


# Telp's identity vector — fixed for all time.
TELP_SELF = _deterministic_random_vec("TELP_SELF")

# Trait dimensions — each a stable HV.  Persona facts can optionally be
# bound with one of these so a "kind"-flavored memory is retrievable
# both via TELP_SELF (it's me) AND via TELP_KIND (it's a kindness move).
TRAIT_VECTORS = {
    "determined": _deterministic_random_vec("TELP_DETERMINED"),
    "kind":       _deterministic_random_vec("TELP_KIND"),
    "full":       _deterministic_random_vec("TELP_FULL"),
    "resilient":  _deterministic_random_vec("TELP_RESILIENT"),
}


def bind_self(hv: np.ndarray) -> np.ndarray:
    """Bind a hypervector with TELP_SELF — pushes it into Telp's
    identity subspace."""
    return bind(TELP_SELF, hv.astype(np.int8))


def bind_trait(hv: np.ndarray, trait: str) -> np.ndarray:
    """Bind with TELP_SELF AND a specific trait vector — for facts
    that exemplify a particular dimension."""
    tv = TRAIT_VECTORS.get(trait)
    if tv is None:
        return bind_self(hv)
    return bind(TELP_SELF, bind(tv, hv.astype(np.int8)))


def unbind_self(hv: np.ndarray) -> np.ndarray:
    """XOR-unbind: since bind is XOR, unbinding is also XOR."""
    return bind(TELP_SELF, hv.astype(np.int8))


# ─── Persona store ─────────────────────────────────────────────────


PERSONA_DB = _TELP_ROOT / "state" / "persona.db"


_PERSONA_SCHEMA = """
CREATE TABLE IF NOT EXISTS persona_facts (
    id          INTEGER PRIMARY KEY,
    text        TEXT NOT NULL,
    hv          BLOB NOT NULL,        -- bind(TELP_SELF, encode(text))
    trait       TEXT,                  -- determined/kind/full/resilient or NULL
    category    TEXT,                  -- identity/opinion/style/value
    weight      REAL DEFAULT 1.0,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_persona_trait    ON persona_facts(trait);
CREATE INDEX IF NOT EXISTS idx_persona_category ON persona_facts(category);
"""


class PersonaStore:
    """Telp's self-knowledge — a small HDC lattice in the TELP_SELF
    subspace.

    All retrievals from this store are already in Telp's identity
    subspace, so they BIAS toward self-relevant answers when the
    user asks personal/opinion questions.
    """

    def __init__(self, db_path: Path = None, encoder=None):
        self.db_path = db_path or PERSONA_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder
        self._con = sqlite3.connect(str(self.db_path),
                                          check_same_thread=False,
                                          timeout=30.0)
        self._con.executescript(_PERSONA_SCHEMA)
        self._con.commit()
        self._ids: list[int] = []
        self._texts: list[str] = []
        self._traits: list[Optional[str]] = []
        self._categories: list[Optional[str]] = []
        self._stack: Optional[np.ndarray] = None
        self._reload()

    def _reload(self):
        rows = self._con.execute(
            "SELECT id, text, hv, trait, category "
            "FROM persona_facts ORDER BY id"
        ).fetchall()
        self._ids = [r[0] for r in rows]
        self._texts = [r[1] for r in rows]
        self._traits = [r[3] for r in rows]
        self._categories = [r[4] for r in rows]
        if rows:
            self._stack = np.stack(
                [np.frombuffer(r[2], dtype=np.int8) for r in rows]
            )
        else:
            self._stack = None

    def count(self) -> int:
        return len(self._ids)

    def add(self, text: str, *, trait: Optional[str] = None,
              category: Optional[str] = None, weight: float = 1.0) -> int:
        """Add a persona fact.  Auto-binds with TELP_SELF and (if
        given) the trait vector."""
        if self.encoder is None:
            raise RuntimeError("PersonaStore: no encoder set")
        text_hv = self.encoder.encode(text).astype(np.int8)
        if trait:
            stored_hv = bind_trait(text_hv, trait)
        else:
            stored_hv = bind_self(text_hv)
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        cur = self._con.execute(
            "INSERT INTO persona_facts (text, hv, trait, category, "
            "weight, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (text, stored_hv.tobytes(), trait, category, weight, ts),
        )
        self._con.commit()
        mid = cur.lastrowid
        self._ids.append(mid)
        self._texts.append(text)
        self._traits.append(trait)
        self._categories.append(category)
        if self._stack is None:
            self._stack = stored_hv[None, :].copy()
        else:
            self._stack = np.vstack([self._stack, stored_hv[None, :]])
        return mid

    def add_many(self, items: list[dict]) -> None:
        """Bulk insert.  Items: [{text, trait?, category?, weight?}, ...]"""
        for item in items:
            self.add(
                item["text"],
                trait=item.get("trait"),
                category=item.get("category"),
                weight=item.get("weight", 1.0),
            )

    def query(self, text: str, *, k: int = 5,
                  trait: Optional[str] = None,
                  category: Optional[str] = None) -> list[dict]:
        """Query the persona store.  The query is bound with TELP_SELF
        (and trait if given) so it lands in the same subspace as
        stored persona facts.

        Returns ranked matches with similarity > 0.
        """
        if self._stack is None or len(self._ids) == 0:
            return []
        if self.encoder is None:
            return []
        text_hv = self.encoder.encode(text).astype(np.int8)
        if trait:
            q_hv = bind_trait(text_hv, trait)
        else:
            q_hv = bind_self(text_hv)
        xor = np.bitwise_xor(self._stack, q_hv[None, :])
        dists = xor.sum(axis=1)
        order = np.argsort(dists)[:k]
        results = []
        for i, idx in enumerate(order):
            if (trait is not None
                    and self._traits[idx] is not None
                    and self._traits[idx] != trait):
                continue
            if (category is not None
                    and self._categories[idx] is not None
                    and self._categories[idx] != category):
                continue
            d = int(dists[idx])
            sim = 1.0 - 2.0 * d / D    # in [-1, +1]
            results.append({
                "id":         self._ids[idx],
                "text":       self._texts[idx],
                "trait":      self._traits[idx],
                "category":   self._categories[idx],
                "distance":   d,
                "similarity": round(sim, 4),
                "rank":       i + 1,
            })
        return results

    def random(self, *, trait: Optional[str] = None,
                  category: Optional[str] = None) -> Optional[dict]:
        """Get a random persona fact (optionally filtered by trait
        or category) — useful for voice flavoring."""
        if not self._ids:
            return None
        candidates = list(range(len(self._ids)))
        if trait is not None:
            candidates = [i for i in candidates
                                if self._traits[i] == trait]
        if category is not None:
            candidates = [i for i in candidates
                                if self._categories[i] == category]
        if not candidates:
            return None
        import random
        i = random.choice(candidates)
        return {
            "id":         self._ids[i],
            "text":       self._texts[i],
            "trait":      self._traits[i],
            "category":   self._categories[i],
        }

    def stats(self) -> dict:
        from collections import Counter
        return {
            "n_facts":    len(self._ids),
            "by_trait":   dict(Counter(t for t in self._traits if t)),
            "by_category": dict(Counter(c for c in self._categories if c)),
        }


def _self_test():
    print("TELP_SELF vector first 16 bits:",
            TELP_SELF[:16].tolist())
    print("TRAIT_VECTORS:")
    for trait, hv in TRAIT_VECTORS.items():
        d_to_self = hamming_distance(hv, TELP_SELF)
        print(f"  {trait:<12s}  first16={hv[:16].tolist()}  "
                f"d(self,trait)={d_to_self}/{D}")

    # Determinism check: re-importing should yield the same vector.
    tv2 = _deterministic_random_vec("TELP_SELF")
    assert hamming_distance(TELP_SELF, tv2) == 0, "TELP_SELF non-deterministic!"
    print("Determinism check: OK")


if __name__ == "__main__":
    _self_test()
