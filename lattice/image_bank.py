"""
lattice/image_bank.py - persistent image memory with HDC index.

Stores: (image_path, hypervector, metadata)
Search: by visual similarity (HDC Hamming distance) OR by text prompt
        (CLIP cross-modal) — returns file path(s) to the actual image(s).

This is how HDC "stores" images: the hypervector is the SEARCH KEY,
the actual pixels stay in a file. Retrieve by similarity, open the
file to display. This is also how every real production image search
system works (Google Images, Pinterest, etc.) — they don't reconstruct,
they index.

Persistence: SQLite, same pattern as the Lattice text store.
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

from train.v5_hdc_prototype import D, hamming_distance


_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    image_path  TEXT NOT NULL,
    hv          BLOB NOT NULL,
    metadata    TEXT,
    label       TEXT,
    source      TEXT
);
CREATE INDEX IF NOT EXISTS idx_label ON images(label);
"""


class ImageBank:
    """Persistent HDC-indexed image memory."""

    def __init__(self, db_path: str | Path, encoder=None):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(self.db_path)
        self._con.executescript(_SCHEMA)
        self._con.commit()
        self.encoder = encoder    # ImageEncoder or CLIPImageEncoder
        # In-memory stack for fast batch search
        self._ids: list[int] = []
        self._paths: list[str] = []
        self._labels: list[str] = []
        self._stack: Optional[np.ndarray] = None
        self._reload()

    def _reload(self):
        rows = self._con.execute(
            "SELECT id, image_path, label, hv FROM images ORDER BY id"
        ).fetchall()
        self._ids = [r[0] for r in rows]
        self._paths = [r[1] for r in rows]
        self._labels = [r[2] or "" for r in rows]
        if rows:
            vecs = [np.frombuffer(r[3], dtype=np.int8) for r in rows]
            self._stack = np.stack(vecs)
        else:
            self._stack = None

    def count(self) -> int:
        return len(self._ids)

    # ─── Add ────────────────────────────────────────────────

    def add(self, image_path: str | Path, label: str = "",
              source: str = "", **metadata) -> int:
        if self.encoder is None:
            raise RuntimeError("ImageBank needs an encoder to add images")
        hv = self.encoder.encode(image_path)
        return self._add_with_hv(image_path, hv, label, source, metadata)

    def add_many(self, paths: list[str | Path], labels: list[str] = None,
                   source: str = "") -> list[int]:
        if self.encoder is None:
            raise RuntimeError("ImageBank needs an encoder")
        labels = labels or [""] * len(paths)
        hvs = self.encoder.encode_many(paths)
        ids = []
        for path, hv, label in zip(paths, hvs, labels):
            ids.append(self._add_with_hv(path, hv, label, source, {}))
        return ids

    def _add_with_hv(self, image_path, hv, label, source, metadata):
        assert hv.dtype == np.int8 and hv.shape == (D,)
        ts = datetime.now(timezone.utc).isoformat()
        cur = self._con.execute(
            "INSERT INTO images (created_at, image_path, hv, metadata, label, source) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ts, str(image_path), hv.tobytes(),
              json.dumps(metadata) if metadata else None,
              label or None, source or None),
        )
        self._con.commit()
        iid = cur.lastrowid
        self._ids.append(iid)
        self._paths.append(str(image_path))
        self._labels.append(label or "")
        if self._stack is None:
            self._stack = hv[None, :].copy()
        else:
            self._stack = np.vstack([self._stack, hv[None, :]])
        return iid

    # ─── Search ─────────────────────────────────────────────

    def search_by_image(self, query_path: str | Path, k: int = 5) -> list[dict]:
        if self._stack is None:
            return []
        hv = self.encoder.encode(query_path)
        return self._search(hv, k)

    def search_by_text(self, text: str, k: int = 5) -> list[dict]:
        """Cross-modal search: text -> CLIP -> HDC -> image."""
        if self._stack is None:
            return []
        if not hasattr(self.encoder, "encode_text"):
            raise RuntimeError("encoder doesn't support text encoding")
        hv = self.encoder.encode_text(text)
        return self._search(hv, k)

    def _search(self, hv: np.ndarray, k: int) -> list[dict]:
        xor = np.bitwise_xor(self._stack, hv[None, :])
        dists = xor.sum(axis=1)
        order = np.argsort(dists)[:k]
        return [{
            "rank": i + 1,
            "id": self._ids[idx],
            "path": self._paths[idx],
            "label": self._labels[idx],
            "distance": int(dists[idx]),
            "distance_pct": float(dists[idx]) / D,
        } for i, idx in enumerate(order)]

    # ─── Copy ───────────────────────────────────────────────

    def copy_image(self, query_path: str | Path, dest_path: str | Path) -> dict:
        """Find the nearest stored image to query, and copy it to dest_path.

        This is "make a copy of a picture if we give it to it" — HDC retrieves
        the most similar known image and we return that file.
        """
        import shutil
        results = self.search_by_image(query_path, k=1)
        if not results:
            return {"copied": False, "reason": "bank is empty"}
        match = results[0]
        shutil.copy(match["path"], str(dest_path))
        return {
            "copied": True,
            "dest_path": str(dest_path),
            "matched_path": match["path"],
            "matched_label": match["label"],
            "distance_pct": match["distance_pct"],
        }

    def close(self):
        self._con.close()
