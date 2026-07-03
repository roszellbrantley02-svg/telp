"""
lattice/fetch_wiki.py - fetch Wikipedia summaries in bulk via urllib.

Uses Wikipedia's REST API. Saves to JSON for offline use.

Usage:
    python -m lattice.fetch_wiki                   # fetch the default 100-topic set
    python -m lattice.fetch_wiki --topics t1 t2    # specific topics
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = _TELP_ROOT / "state" / "wiki_corpus.json"


# Curated 100+ topic list across diverse categories
TOPICS = [
    # Countries (15)
    "Germany","France","Japan","United_States","United_Kingdom","Italy","Spain",
    "China","India","Brazil","Canada","Australia","Russia","Egypt","South_Africa",
    # Cities (10)
    "Paris","London","Tokyo","Berlin","New_York_City","Rome","Madrid","Cairo",
    "Mumbai","Beijing",
    # Animals (10)
    "Lion","Tiger","Elephant","Wolf","Eagle","Penguin","Shark","Dolphin","Octopus",
    "Kangaroo",
    # Scientists (12)
    "Albert_Einstein","Isaac_Newton","Marie_Curie","Charles_Darwin","Galileo_Galilei",
    "Nikola_Tesla","Stephen_Hawking","Alan_Turing","Richard_Feynman","Carl_Sagan",
    "Rosalind_Franklin","Niels_Bohr",
    # Authors (10)
    "William_Shakespeare","Leo_Tolstoy","Mark_Twain","Jane_Austen","George_Orwell",
    "Charles_Dickens","Virginia_Woolf","Franz_Kafka","Hermann_Hesse","Ernest_Hemingway",
    # Musicians/composers (10)
    "Wolfgang_Amadeus_Mozart","Ludwig_van_Beethoven","Johann_Sebastian_Bach",
    "The_Beatles","Bob_Dylan","Jimi_Hendrix","Miles_Davis","Aretha_Franklin",
    "Frédéric_Chopin","David_Bowie",
    # Inventions / things (12)
    "Telephone","Internet","Computer","Light_bulb","Steam_engine","Penicillin",
    "Printing_press","Airplane","Television","Refrigerator","DNA","Quantum_mechanics",
    # Planets / cosmic (8)
    "Sun","Moon","Mars","Jupiter","Saturn","Earth","Black_hole","Galaxy",
    # Elements (8)
    "Hydrogen","Oxygen","Carbon","Gold","Iron","Uranium","Helium","Nitrogen",
    # Philosophers (8)
    "Plato","Aristotle","Socrates","Immanuel_Kant","Friedrich_Nietzsche",
    "John_Locke","Confucius","René_Descartes",
]


def fetch_one(topic: str, timeout: float = 10.0) -> dict | None:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(topic)}"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "telp-lattice-research/0.1 (educational use)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {
            "topic": topic,
            "title": data.get("title", topic),
            "extract": data.get("extract", ""),
            "description": data.get("description", ""),
            "type": data.get("type", ""),    # "disambiguation" -> not facts
        }
    except Exception as e:
        return {"topic": topic, "error": str(e)}


def fetch_full_lead(topic: str, timeout: float = 10.0) -> dict | None:
    """Fetch the full lead section of an article via the MediaWiki
    query API.  Much richer than fetch_one() (which uses the REST
    summary endpoint).  Use this for topics where the short summary
    misses key facts.
    """
    params = {
        "action": "query", "prop": "extracts", "exintro": "1",
        "explaintext": "1", "redirects": "1",
        "titles": topic, "format": "json", "formatversion": "2",
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "telp-lattice-research/0.1 (educational use)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return {"topic": topic, "error": "no pages"}
        page = pages[0]
        if "missing" in page:
            return {"topic": topic, "error": "missing"}
        return {
            "topic":   topic,
            "title":   page.get("title", topic),
            "extract": page.get("extract", ""),
            "description": "",
        }
    except Exception as e:
        return {"topic": topic, "error": str(e)}


def fetch_all(topics: list[str], max_workers: int = 8) -> list[dict]:
    print(f"Fetching {len(topics)} Wikipedia summaries with {max_workers} workers ...")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_one, t): t for t in topics}
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            results.append(r)
            if i % 10 == 0:
                print(f"  {i+1}/{len(topics)} done")
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", nargs="*", help="Specific topics to fetch")
    ap.add_argument("--out", default=str(OUT_FILE))
    args = ap.parse_args()

    topics = args.topics if args.topics else TOPICS
    results = fetch_all(topics)
    n_ok = sum(1 for r in results if "extract" in r and r["extract"])
    n_err = sum(1 for r in results if "error" in r)
    total_text = sum(len(r.get("extract", "")) for r in results)
    print(f"\n  fetched: {n_ok} ok, {n_err} errors")
    print(f"  total text: {total_text} chars (~{total_text//5} words)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
    print(f"  saved to: {args.out}")


if __name__ == "__main__":
    main()
