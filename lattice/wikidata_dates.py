"""
lattice/wikidata_dates.py - fetch birth/death years from Wikidata.

Our Wikipedia REST/lead corpus rarely includes structured life dates
("X was born in 1879" is in the infobox, not the lead text).  Wikidata
has these properties cleanly: P569 = date of birth, P570 = date of
death.  Hit Wikidata's REST API for entities we already know about,
extract the year, and emit (entity, born_year, YYYY) /
(entity, died_year, YYYY) claims.

Lookup chain:
  Wikipedia page title  ->  MediaWiki API  ->  Wikidata Q-ID
  Wikidata Q-ID         ->  Wikidata REST  ->  P569/P570 values

Cached locally in state/wikidata_dates.json so we don't refetch.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
_CACHE_PATH = _TELP_ROOT / "state" / "wikidata_dates.json"


_UA = {"User-Agent": "telp-lattice-research/0.1 (educational use)"}


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(cache, indent=1, ensure_ascii=False),
                                encoding="utf-8")


def _wikipedia_title_to_qid(title: str, timeout: float = 8.0) -> str | None:
    """Resolve a Wikipedia article title to its Wikidata Q-ID."""
    params = {
        "action": "query", "prop": "pageprops", "ppprop": "wikibase_item",
        "redirects": "1", "titles": title,
        "format": "json", "formatversion": "2",
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return None
        return pages[0].get("pageprops", {}).get("wikibase_item")
    except Exception:
        return None


_DATE_RE = re.compile(r"([+-])(\d{1,4})-(\d{2})-(\d{2})")


def _extract_year(claim_block: list) -> int | None:
    """Pull the year out of a Wikidata claim block (P569 / P570)."""
    if not claim_block:
        return None
    for cl in claim_block:
        try:
            ts = cl["mainsnak"]["datavalue"]["value"]["time"]
        except (KeyError, TypeError):
            continue
        m = _DATE_RE.match(ts)
        if m:
            sign, year, _, _ = m.groups()
            y = int(year)
            return -y if sign == "-" else y
    return None


def _wikidata_dates(qid: str, timeout: float = 8.0) -> dict:
    """Return {'born': YYYY, 'died': YYYY} (either may be missing)."""
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        entity = data.get("entities", {}).get(qid, {})
        claims = entity.get("claims", {})
        out = {}
        born = _extract_year(claims.get("P569"))
        died = _extract_year(claims.get("P570"))
        if born is not None: out["born"] = born
        if died is not None: out["died"] = died
        return out
    except Exception:
        return {}


def fetch_dates_for_topic(topic: str) -> dict:
    """Public entry: returns {'born': year, 'died': year} for a
    Wikipedia article topic name.  Uses local cache."""
    cache = _load_cache()
    if topic in cache:
        return cache[topic]
    title = topic.replace("_", " ")
    qid = _wikipedia_title_to_qid(title)
    if not qid:
        cache[topic] = {}
        _save_cache(cache)
        return {}
    dates = _wikidata_dates(qid)
    dates["qid"] = qid
    cache[topic] = dates
    _save_cache(cache)
    return dates


def enrich_corpus(topics: list[str], rate_limit_s: float = 0.3) -> dict:
    """Batch-fetch dates for a list of topic names.  Returns the
    cumulative {topic -> {born, died, qid}} mapping after fetch.
    """
    cache = _load_cache()
    fetched = 0
    for t in topics:
        if t in cache and cache[t]:
            continue
        title = t.replace("_", " ")
        qid = _wikipedia_title_to_qid(title)
        if not qid:
            cache[t] = {}
            continue
        dates = _wikidata_dates(qid)
        dates["qid"] = qid
        cache[t] = dates
        fetched += 1
        if rate_limit_s > 0:
            time.sleep(rate_limit_s)
    _save_cache(cache)
    print(f"[wikidata] fetched dates for {fetched} new topics; "
            f"cache now has {len(cache)} entries", flush=True)
    return cache


if __name__ == "__main__":
    # Smoke test
    for t in ["Albert_Einstein", "Marie_Curie", "Plato",
               "Wolfgang_Amadeus_Mozart"]:
        d = fetch_dates_for_topic(t)
        print(f"  {t}: {d}")
