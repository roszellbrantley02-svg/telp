"""
lattice/fetch_wiki_bulk.py - fetch a larger Wikipedia corpus.

Given a set of seed categories, expand to ~1000 article titles via
the MediaWiki API and fetch each one's lead section.  Skips articles
already in state/wiki_corpus.json.  Conservative rate limit (0.3s/req)
to stay under the 429 threshold.

Run:
    python -m lattice.fetch_wiki_bulk --target 1000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_PATH = _TELP_ROOT / "state" / "wiki_corpus.json"
_UA = {"User-Agent": "telp-lattice-research/0.1 (educational use)"}


# Seed categories.  Each maps to ~50-200 articles typically.
_SEED_CATEGORIES = [
    # Geography
    "Capitals_in_Europe",
    "Capitals_in_Asia",
    "Capitals_in_Africa",
    "Capitals_in_South_America",
    "Member_states_of_the_United_Nations",
    "Mountain_ranges_by_country",
    # People
    "20th-century_American_writers",
    "Nobel_laureates_in_Physics",
    "Nobel_laureates_in_Chemistry",
    "20th-century_British_novelists",
    "Ancient_Greek_philosophers",
    "Renaissance_painters",
    "Classical-era_composers",
    "Heads_of_state_of_the_United_States",
    "American_jazz_musicians",
    # Science
    "Chemical_elements",
    "Astronomical_objects",
    "Mammals_of_Africa",
    "Inventions",
    # History
    "World_War_II",
    "Ancient_Roman_emperors",
]


def category_members(cat: str, limit: int = 200,
                      timeout: float = 10.0) -> list[str]:
    """Return article titles in the given category (page namespace only)."""
    params = {
        "action": "query", "list": "categorymembers",
        "cmtitle": f"Category:{cat}",
        "cmtype": "page",
        "cmlimit": str(limit),
        "format": "json", "formatversion": "2",
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        members = data.get("query", {}).get("categorymembers", [])
        return [m["title"].replace(" ", "_") for m in members]
    except Exception as e:
        print(f"[bulk] category {cat!r} failed: {e}", flush=True)
        return []


def fetch_lead(topic: str, timeout: float = 10.0) -> dict | None:
    """Same as fetch_wiki.fetch_full_lead, inlined to avoid circular dep."""
    params = {
        "action": "query", "prop": "extracts", "exintro": "1",
        "explaintext": "1", "redirects": "1",
        "titles": topic, "format": "json", "formatversion": "2",
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", [])
        if not pages or "missing" in pages[0]:
            return {"topic": topic, "error": "missing"}
        page = pages[0]
        return {
            "topic":       topic,
            "title":       page.get("title", topic),
            "extract":     page.get("extract", ""),
            "description": "",
        }
    except Exception as e:
        return {"topic": topic, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1000,
                      help="Target number of articles in the corpus")
    ap.add_argument("--rate", type=float, default=0.3,
                      help="Seconds between fetches")
    args = ap.parse_args()

    # Load existing corpus.
    corpus = (json.loads(_CORPUS_PATH.read_text(encoding="utf-8"))
                if _CORPUS_PATH.exists() else [])
    existing = {e["topic"] for e in corpus}
    print(f"[bulk] starting with {len(existing)} topics already in corpus")

    # 1) Expand categories to topic list.
    topics: list[str] = []
    seen: set[str] = set(existing)
    for cat in _SEED_CATEGORIES:
        if len(topics) + len(existing) >= args.target:
            break
        mem = category_members(cat, limit=100)
        for t in mem:
            if t in seen:
                continue
            seen.add(t)
            topics.append(t)
        print(f"[bulk] category {cat}: +{len([t for t in mem if t in seen])} "
                f"(running total candidates: {len(topics)})", flush=True)
        time.sleep(args.rate)

    print(f"[bulk] {len(topics)} new topics to fetch")
    n_ok = 0; n_err = 0
    for i, t in enumerate(topics):
        if len(existing) + n_ok >= args.target:
            break
        r = fetch_lead(t)
        if r and r.get("extract") and len(r["extract"]) > 100:
            corpus.append(r)
            n_ok += 1
        else:
            n_err += 1
        if (i + 1) % 25 == 0:
            print(f"[bulk] {i+1}/{len(topics)} attempted, +{n_ok} ok, "
                    f"{n_err} skipped", flush=True)
            # Flush corpus periodically so we don't lose progress.
            _CORPUS_PATH.write_text(
                json.dumps(corpus, indent=1, ensure_ascii=False),
                encoding="utf-8")
        time.sleep(args.rate)

    _CORPUS_PATH.write_text(json.dumps(corpus, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
    print(f"[bulk] DONE. corpus now has {len(corpus)} topics "
            f"(+{n_ok} new this run)")


if __name__ == "__main__":
    main()
