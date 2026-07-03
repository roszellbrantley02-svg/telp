"""
lattice/educate.py - bulk education: feed the one memory from the prepared
corpus (E:/telp_corpus, fetched May 2026), encoded with THE semantic encoder
on the GPU. This is the May ingestion fleet's purpose, repointed at the live
store (the old CLIs wrote to the retired learned lattice).

    python -m lattice.educate --target 20000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

CORPUS = Path(r"E:\telp_corpus\jsonl_text\wikipedia_en_clean.jsonl")
_PRON = ("it ", "its ", "they ", "their ", "he ", "she ", "his ", "her ",
         "this ", "these ")
_TITLE_RE = re.compile(r"^(.{2,48}?)\s+(?:is|was|are|were)\b")


def _quality(s: str) -> bool:
    if not (40 < len(s) < 300):
        return False
    if not s[0].isupper() or s[-1] not in ".!?":
        return False
    letters = sum(c.isalpha() or c == " " for c in s)
    return letters >= len(s) * 0.8


def educate(target: int = 20000, corpus: Path = CORPUS, batch: int = 2048):
    from lattice.standalone_agent import StandaloneAgent
    from lattice.vision import CHAT_LATTICE
    agent = StandaloneAgent(lattice_path=CHAT_LATTICE, skip_ngram_retrain=True)
    enc = agent.encoder
    if not hasattr(enc, "encode_batch"):
        raise SystemExit("educate requires the semantic encoder")
    existing = set(agent.lattice._texts)
    pend_texts: list[str] = []
    pend_src: list[str] = []
    n_added = 0
    t0 = time.time()

    def flush():
        nonlocal n_added
        if not pend_texts:
            return
        hvs = enc.encode_batch(pend_texts)
        agent.lattice.add_many(list(zip(pend_texts, hvs, pend_src)))
        n_added += len(pend_texts)
        el = time.time() - t0
        print(f"  {n_added}/{target} remembered  ({n_added/el:.0f} facts/s)",
              flush=True)
        pend_texts.clear()
        pend_src.clear()

    with open(corpus, encoding="utf-8") as f:
        for line in f:
            if n_added + len(pend_texts) >= target:
                break
            try:
                text = json.loads(line)["text"]
            except Exception:
                continue
            first = text.split(".", 1)[0]
            m = _TITLE_RE.match(first)
            title = (m.group(1) if m else first[:40]).strip()
            for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")):
                s = s.strip()
                if not _quality(s):
                    continue
                if s.lower().startswith(_PRON):
                    s = f"{title}: {s}"
                if s in existing:
                    continue
                existing.add(s)
                pend_texts.append(s)
                pend_src.append(f"wikipedia:{title}")
                if len(pend_texts) >= batch:
                    flush()
                if n_added + len(pend_texts) >= target:
                    break
    flush()
    print(f"DONE: {n_added} new memories in {time.time()-t0:.0f}s; "
          f"lattice now {agent.lattice.count()} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=20000)
    ap.add_argument("--corpus", default=str(CORPUS))
    a = ap.parse_args()
    educate(target=a.target, corpus=Path(a.corpus))
