"""
lattice/store.py - persistent append-only memory store.

A Lattice instance is the user-facing object. It owns:
  - A text encoder (sentence -> hypervector)
  - A SQLite-backed memory table
  - An in-memory stack of all hypervectors for fast batch search
  - The HDC algebra (bind, unbind, bundle, permute)

Lifecycle
---------
  Lattice("memory.db")  -> opens or creates DB, loads all memories
  .add(text, **meta)     -> persists + appends to in-memory stack
  .query(text, k=5)      -> KNN search by paraphrased meaning
  .compose(*texts)       -> bundle several memories into one hypervector
  .analogy(a, b, c)      -> algebraic solve a:b :: c:?
  .counterfactual(mid, swap={'cat':'dog'})  -> what-if surgery on a memory

Persistence is append-only: nothing is ever modified once stored.
That gives the Lattice its identity — it grows monotonically with
your life. Memories accumulate; they don't get rewritten.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import D, bind, bundle, hamming_distance  # noqa: E402
from lattice.text_encoder import TextEncoder, hv_distance_pct          # noqa: E402


# ─── Schema ────────────────────────────────────────────────────────


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    text        TEXT NOT NULL,
    hv          BLOB NOT NULL,
    metadata    TEXT,
    source      TEXT,
    tags        TEXT
);
CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_source     ON memories(source);
"""


# ─── Lattice ───────────────────────────────────────────────────────


class Lattice:
    """The personal mind: append-only HD memory store + algebra.

    GPU acceleration: set ``device="cuda"`` (or via env var
    ``LATTICE_DEVICE=cuda``) to mirror the in-memory stack on the GPU.
    Hamming-distance queries then use torch.bitwise_xor on the GPU
    tensor — ~10-50× speedup for the 600K-memory lattice.
    """

    def __init__(self, db_path: str | Path,
                  encoder: Optional[TextEncoder] = None,
                  device: str | None = None):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder or TextEncoder()
        # check_same_thread=False so a FastAPI worker thread can read
        # from a connection opened in the startup thread. Lattice is
        # append-only and writes go through a single-writer path so the
        # SQLite locking is fine. Set timeout to avoid lock contention.
        self._con = sqlite3.connect(self.db_path,
                                          check_same_thread=False,
                                          timeout=30.0)
        self._con.executescript(_SCHEMA)
        self._con.commit()
        # In-memory parallel structures for fast batch search.
        self._ids: list[int] = []
        self._texts: list[str] = []
        self._sources: list[str] = []
        self._stack: Optional[np.ndarray] = None   # shape (N, D)
        # GPU mirror — populated by _reload_from_disk + _add_with_hv
        # when device != "cpu".
        import os as _os
        self.device = (device or
                          _os.environ.get("LATTICE_DEVICE", "cpu")).lower()
        self._stack_t = None   # torch.int8 tensor on cuda when active
        self._reload_from_disk()

    # ─── persistence ────────────────────────────────────────

    def _reload_from_disk(self):
        rows = self._con.execute(
            "SELECT id, text, hv, source FROM memories ORDER BY id"
        ).fetchall()
        self._ids = [r[0] for r in rows]
        self._texts = [r[1] for r in rows]
        self._sources = [r[3] or "" for r in rows]
        if rows:
            vecs = [np.frombuffer(r[2], dtype=np.int8) for r in rows]
            self._stack = np.stack(vecs)
        else:
            self._stack = None
        self._push_to_device()

    def _push_to_device(self) -> None:
        """Mirror the in-memory stack on GPU as a torch.int8 tensor.

        No-op when device == "cpu" OR torch is unavailable OR _stack is
        empty.  Idempotent — drops previous tensor before re-uploading.
        """
        if self.device == "cpu" or self._stack is None:
            self._stack_t = None
            return
        try:
            import torch
            self._stack_t = torch.from_numpy(self._stack).to(
                device=self.device, dtype=torch.int8, non_blocking=True
            )
        except Exception:
            # GPU unavailable -> silently fall back to CPU path.
            self._stack_t = None

    def count(self) -> int:
        return len(self._ids)

    # ─── add ───────────────────────────────────────────────

    def add(self, text: str, *, source: str = "",
              tags: str = "", **metadata) -> int:
        """Store a memory. Returns its id."""
        hv = self.encoder.encode(text)
        return self._add_with_hv(text, hv, source=source, tags=tags,
                                    metadata=metadata)

    def add_with_hv(self, text: str, hv: np.ndarray, *,
                     source: str = "", tags: str = "", **metadata) -> int:
        """Store a pre-encoded memory. For cases where the hypervector
        came from algebra (counterfactual / composed) rather than a
        sentence."""
        return self._add_with_hv(text, hv, source=source, tags=tags,
                                    metadata=metadata)

    def _add_with_hv(self, text, hv, *, source, tags, metadata) -> int:
        assert hv.dtype == np.int8 and hv.shape == (D,)
        ts = datetime.now(timezone.utc).isoformat()
        cur = self._con.execute(
            "INSERT INTO memories (created_at, text, hv, metadata, source, tags) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ts, text, hv.tobytes(),
              json.dumps(metadata) if metadata else None,
              source or None, tags or None)
        )
        self._con.commit()
        mid = cur.lastrowid
        # Append to in-memory stack
        self._ids.append(mid)
        self._texts.append(text)
        self._sources.append(source or "")
        if self._stack is None:
            self._stack = hv[None, :].copy()
        else:
            self._stack = np.vstack([self._stack, hv[None, :]])
        # Append to GPU mirror too (when active).
        if self._stack_t is not None:
            try:
                import torch
                row = torch.from_numpy(hv.astype(np.int8)).to(
                    device=self.device, dtype=torch.int8, non_blocking=True
                ).unsqueeze(0)
                self._stack_t = torch.cat([self._stack_t, row], dim=0)
            except Exception:
                # If something goes wrong, drop the mirror and fall back
                # to CPU — never crash the lattice add path.
                self._stack_t = None
        return mid

    def add_many(self, items: "list[tuple[str, np.ndarray, str]]") -> list[int]:
        """Bulk insert.  items = [(text, hv_int8_D, source), ...].

        Single SQLite transaction (no fsync per row), and the
        in-memory _stack is rebuilt once at the end via a single
        np.vstack instead of O(N) vstacks.  ~100-1000x faster than
        calling add() per item.
        """
        if not items:
            return []
        ts = datetime.now(timezone.utc).isoformat()
        ids: list[int] = []
        rows = []
        for text, hv, source in items:
            assert hv.dtype == np.int8 and hv.shape == (D,)
            rows.append((ts, text, hv.tobytes(), None, source or None, None))
        # Single transaction
        cur = self._con.cursor()
        cur.execute("BEGIN")
        try:
            for row in rows:
                cur.execute(
                    "INSERT INTO memories (created_at, text, hv, metadata, "
                    "source, tags) VALUES (?, ?, ?, ?, ?, ?)", row)
                ids.append(cur.lastrowid)
            self._con.commit()
        except Exception:
            self._con.rollback()
            raise
        # Update in-memory parallel structures in bulk
        new_texts = [t for t, _, _ in items]
        new_srcs  = [s or "" for _, _, s in items]
        new_hvs   = np.stack([hv for _, hv, _ in items])
        self._ids.extend(ids)
        self._texts.extend(new_texts)
        self._sources.extend(new_srcs)
        if self._stack is None:
            self._stack = new_hvs
        else:
            self._stack = np.vstack([self._stack, new_hvs])
        # Refresh GPU mirror after bulk insert (cheaper to re-upload
        # than incrementally cat() one row at a time).
        if self._stack_t is not None:
            self._push_to_device()
        return ids

    # ─── query ─────────────────────────────────────────────

    def query(self, text: str, k: int = 5,
                threshold: Optional[float] = None) -> list[dict]:
        """KNN by paraphrased meaning. threshold = max ham-distance fraction
        (e.g. 0.4 means 'skip results more than 40% different')."""
        hv = self.encoder.encode(text)
        return self.query_vector(hv, k=k, threshold=threshold)

    def query_vector(self, hv: np.ndarray, k: int = 5,
                       threshold: Optional[float] = None) -> list[dict]:
        if self._stack is None or len(self._ids) == 0:
            return []
        # GPU path: use torch.bitwise_xor on the cuda mirror when active.
        # Tile the stack so a 603K x 10000 XOR doesn't try to allocate
        # ~24 GB all at once.  Chunk size tuned so peak VRAM stays
        # under ~2 GB for the temporary XOR buffer.
        if self._stack_t is not None:
            import torch
            n = self._stack_t.shape[0]
            tile = 50_000
            q = torch.from_numpy(hv.astype(np.int8)).to(
                device=self.device, dtype=torch.int8, non_blocking=True
            )
            dists_t = torch.empty(n, dtype=torch.int32,
                                       device=self.device)
            for s in range(0, n, tile):
                e = min(s + tile, n)
                chunk = self._stack_t[s:e]
                xor = torch.bitwise_xor(chunk, q.unsqueeze(0))
                # int16 holds up to 32767 — D=10000 fits.  Saves 2× memory
                # vs int32 in the intermediate sum.
                dists_t[s:e] = xor.to(torch.int16).sum(dim=1).to(torch.int32)
                del xor
            order_t = torch.argsort(dists_t, stable=True)[:k]
            order = order_t.detach().cpu().numpy()
            dists = dists_t.detach().cpu().numpy()
        else:
            xor = np.bitwise_xor(self._stack, hv[None, :])
            dists = xor.sum(axis=1)
            order = np.argsort(dists, kind="stable")[:k]
        out = []
        for rank, idx in enumerate(order):
            d = int(dists[idx])
            pct = d / D
            if threshold is not None and pct > threshold:
                continue
            out.append({
                "rank": rank + 1,
                "id": self._ids[idx],
                "text": self._texts[idx],
                "source": self._sources[idx],
                "distance": d,
                "distance_pct": pct,
                "similarity": 1.0 - 2.0 * pct,
            })
        return out

    def get(self, memory_id: int) -> Optional[dict]:
        row = self._con.execute(
            "SELECT id, created_at, text, hv, metadata, source, tags "
            "FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "created_at": row[1],
            "text": row[2],
            "hv": np.frombuffer(row[3], dtype=np.int8),
            "metadata": json.loads(row[4]) if row[4] else {},
            "source": row[5] or "",
            "tags": row[6] or "",
        }

    # ─── algebra ───────────────────────────────────────────

    def compose(self, *texts: str) -> np.ndarray:
        """Bundle several text fragments into one hypervector via majority
        vote. The result represents the combined meaning."""
        vecs = [self.encoder.encode(t) for t in texts]
        return bundle(vecs)

    def counterfactual(self, memory_id: int, swap: dict) -> np.ndarray:
        """Surgical concept swap: take a memory, XOR out the old concepts,
        XOR in the new ones. Returns a new hypervector you can query
        with .query_vector(...).

        Example:
            hv = lattice.counterfactual(memory_id=42,
                                        swap={'coffee': 'tea'})
            results = lattice.query_vector(hv)
        """
        mem = self.get(memory_id)
        if not mem:
            raise KeyError(memory_id)
        hv = mem["hv"].copy()
        for old, new in swap.items():
            hv_old = self.encoder.encode(old)
            hv_new = self.encoder.encode(new)
            hv = np.bitwise_xor(np.bitwise_xor(hv, hv_old), hv_new)
        return hv

    def analogy(self, a: str, b: str, c: str) -> np.ndarray:
        """Solve a:b :: c:?  via XOR algebra.

        Example: analogy('king', 'queen', 'man') should produce a vector
        near 'woman'. (Word-level analogies work best when the encoder
        embeds atomic concepts cleanly.)
        """
        h_a = self.encoder.encode(a)
        h_b = self.encoder.encode(b)
        h_c = self.encoder.encode(c)
        # ? = c XOR a XOR b   (relation from a->b applied to c)
        return np.bitwise_xor(np.bitwise_xor(h_c, h_a), h_b)

    # ─── narration (LLM decoder) ───────────────────────────

    def narrate(self, query: str, k: int = 5,
                  threshold: Optional[float] = None) -> str:
        """Query the Lattice and have the LLM decoder narrate the result.

        Returns a natural-language string. Loads the decoder lazily on
        first call (~1GB download the first time, then cached).
        """
        results = self.query(query, k=k, threshold=threshold)
        # Lazy import — only pay the cost if narration is actually used
        from lattice.decoder import get_decoder
        return get_decoder().narrate(query, results)

    # ─── inspection ────────────────────────────────────────

    def stats(self) -> dict:
        if self._stack is None:
            return {"n_memories": 0}
        # Distribution of pairwise NN distances on a small sample
        rng = np.random.default_rng(0)
        n = len(self._ids)
        sample_idx = rng.choice(n, size=min(n, 50), replace=False)
        nn_dists = []
        for i in sample_idx:
            xor = np.bitwise_xor(self._stack, self._stack[i][None, :])
            d = xor.sum(axis=1)
            d[i] = D  # exclude self
            nn_dists.append(int(d.min()))
        return {
            "n_memories": n,
            "db_path": self.db_path,
            "sample_nn_distance_pct_mean": float(np.mean(nn_dists)) / D,
            "sample_nn_distance_pct_min":  float(np.min(nn_dists))  / D,
            "sample_nn_distance_pct_max":  float(np.max(nn_dists))  / D,
        }

    def __len__(self) -> int:
        return self.count()

    def close(self):
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
