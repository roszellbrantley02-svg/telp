"""autopilot/code_corpus.py — curated Python snippet store with
HDC-based retrieval.

The template system (Layer 1-3) handles ~20 canonical tasks well.
When a request doesn't match any template, this module kicks in: a
small lattice of curated Python snippets queried by semantic
similarity to the request.

Snippets cover gaps the templates don't:
  * Common stdlib idioms (json, datetime, pathlib, collections, ...)
  * Popular library skeletons (requests, sqlite3, csv, argparse, ...)
  * Patterns Telp's own codebase uses

Architecture:
  * Each snippet has a description, code, and tags.
  * We HDC-encode the description (using the agent's existing encoder).
  * Store in state/code_corpus.db.
  * At retrieval: encode the user's request, find top-3 closest
    snippets, return the best one as a "here's an example that does
    something similar" response.

This is the bridge from template matching to corpus-based code
suggestion — closer to how an LLM works, but transparent and
inspectable.
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

CODE_CORPUS_DB = _TELP_ROOT / "state" / "code_corpus.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS code_snippets (
    id          INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    code        TEXT NOT NULL,
    tags        TEXT,
    hv          BLOB NOT NULL,
    created_at  TEXT NOT NULL
);
"""


# ─── Curated snippets ────────────────────────────────────────────


# (description, code, tags)
SNIPPETS = [
    ("read a JSON file into a Python dict",
        "import json\n\n"
        "def read_json(path):\n"
        "    with open(path, 'r', encoding='utf-8') as f:\n"
        "        return json.load(f)\n\n"
        "# Example:\n"
        "# data = read_json('config.json')\n",
        "json,file,read,dict"),

    ("write a Python dict to a JSON file with pretty indenting",
        "import json\n\n"
        "def write_json(path, data, indent=2):\n"
        "    with open(path, 'w', encoding='utf-8') as f:\n"
        "        json.dump(data, f, indent=indent, ensure_ascii=False)\n\n"
        "# Example:\n"
        "# write_json('out.json', {'name': 'Eric', 'count': 5})\n",
        "json,file,write,dict"),

    ("parse command-line arguments with argparse",
        "import argparse\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser(description='example')\n"
        "    parser.add_argument('--name', required=True)\n"
        "    parser.add_argument('--count', type=int, default=1)\n"
        "    parser.add_argument('--verbose', action='store_true')\n"
        "    args = parser.parse_args()\n"
        "    print(args.name, args.count, args.verbose)\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        "argparse,cli,arguments,argv"),

    ("read a CSV file into a list of dicts",
        "import csv\n\n"
        "def read_csv(path):\n"
        "    with open(path, 'r', encoding='utf-8', newline='') as f:\n"
        "        reader = csv.DictReader(f)\n"
        "        return list(reader)\n\n"
        "# Example:\n"
        "# rows = read_csv('data.csv')\n"
        "# for r in rows: print(r['name'], r['value'])\n",
        "csv,file,read,dict"),

    ("write a list of dicts to a CSV file",
        "import csv\n\n"
        "def write_csv(path, rows):\n"
        "    if not rows: return\n"
        "    fieldnames = list(rows[0].keys())\n"
        "    with open(path, 'w', encoding='utf-8', newline='') as f:\n"
        "        writer = csv.DictWriter(f, fieldnames=fieldnames)\n"
        "        writer.writeheader()\n"
        "        writer.writerows(rows)\n\n"
        "# Example:\n"
        "# write_csv('out.csv', [{'name': 'A', 'value': 1}])\n",
        "csv,file,write,dict"),

    ("query a SQLite database",
        "import sqlite3\n\n"
        "def query_db(db_path, sql, params=()):\n"
        "    con = sqlite3.connect(db_path)\n"
        "    try:\n"
        "        rows = con.execute(sql, params).fetchall()\n"
        "    finally:\n"
        "        con.close()\n"
        "    return rows\n\n"
        "# Example:\n"
        "# rows = query_db('app.db', 'SELECT * FROM users WHERE age > ?', (18,))\n",
        "sqlite3,database,query,sql"),

    ("get the current date and time as ISO 8601",
        "from datetime import datetime, timezone\n\n"
        "def now_iso():\n"
        "    return datetime.now(timezone.utc).isoformat()\n\n"
        "# Example:\n"
        "print(now_iso())   # '2026-05-22T22:30:00+00:00'\n",
        "datetime,iso8601,timestamp,utc"),

    ("parse an ISO 8601 datetime string",
        "from datetime import datetime\n\n"
        "def parse_iso(s):\n"
        "    return datetime.fromisoformat(s.replace('Z', '+00:00'))\n\n"
        "# Example:\n"
        "dt = parse_iso('2026-05-22T22:30:00Z')\n"
        "print(dt.year, dt.month, dt.day)\n",
        "datetime,iso8601,parse"),

    ("compute days between two dates",
        "from datetime import datetime\n\n"
        "def days_between(date1, date2):\n"
        "    d1 = datetime.fromisoformat(date1)\n"
        "    d2 = datetime.fromisoformat(date2)\n"
        "    return abs((d2 - d1).days)\n\n"
        "# Example:\n"
        "print(days_between('2026-01-01', '2026-12-31'))   # 364\n",
        "datetime,days,subtract,interval"),

    ("list all files in a directory matching a pattern",
        "from pathlib import Path\n\n"
        "def find_files(folder, pattern='*'):\n"
        "    return list(Path(folder).glob(pattern))\n\n"
        "# Example:\n"
        "# py_files = find_files('src', '*.py')\n"
        "# for p in py_files: print(p)\n",
        "pathlib,glob,directory,filesystem"),

    ("walk a directory tree recursively",
        "from pathlib import Path\n\n"
        "def walk(folder):\n"
        "    for p in Path(folder).rglob('*'):\n"
        "        if p.is_file():\n"
        "            yield p\n\n"
        "# Example:\n"
        "# for path in walk('src'):\n"
        "#     print(path, path.stat().st_size)\n",
        "pathlib,walk,recursive,filesystem"),

    ("group items by a key function",
        "from collections import defaultdict\n\n"
        "def group_by(items, key_fn):\n"
        "    groups = defaultdict(list)\n"
        "    for item in items:\n"
        "        groups[key_fn(item)].append(item)\n"
        "    return dict(groups)\n\n"
        "# Example:\n"
        "result = group_by(['apple', 'ant', 'banana', 'berry'], lambda s: s[0])\n"
        "print(result)   # {'a': ['apple', 'ant'], 'b': ['banana', 'berry']}\n",
        "group,categorize,defaultdict,partition"),

    ("send an HTTP POST with JSON body using stdlib",
        "import urllib.request\n"
        "import json\n\n"
        "def http_post_json(url, body, timeout=10):\n"
        "    data = json.dumps(body).encode('utf-8')\n"
        "    req = urllib.request.Request(url, data=data, method='POST',\n"
        "                                  headers={'Content-Type': 'application/json'})\n"
        "    with urllib.request.urlopen(req, timeout=timeout) as r:\n"
        "        return json.loads(r.read().decode('utf-8'))\n\n"
        "# Example:\n"
        "# resp = http_post_json('https://api.example.com/echo', {'name': 'Eric'})\n",
        "http,post,json,api,urllib"),

    ("retry a function with exponential backoff",
        "import time\n\n"
        "def retry(fn, max_attempts=5, base_delay=0.5):\n"
        "    last_exc = None\n"
        "    for attempt in range(max_attempts):\n"
        "        try:\n"
        "            return fn()\n"
        "        except Exception as e:\n"
        "            last_exc = e\n"
        "            time.sleep(base_delay * (2 ** attempt))\n"
        "    raise last_exc\n\n"
        "# Example:\n"
        "# result = retry(lambda: risky_api_call())\n",
        "retry,backoff,exception,resilience"),

    ("measure how long a function takes to run",
        "import time\n"
        "from functools import wraps\n\n"
        "def timed(fn):\n"
        "    @wraps(fn)\n"
        "    def wrapper(*args, **kwargs):\n"
        "        t0 = time.time()\n"
        "        result = fn(*args, **kwargs)\n"
        "        elapsed = time.time() - t0\n"
        "        print(f'{fn.__name__}: {elapsed:.3f}s')\n"
        "        return result\n"
        "    return wrapper\n\n"
        "# Example:\n"
        "@timed\n"
        "def slow():\n"
        "    time.sleep(0.5)\n"
        "slow()\n",
        "timing,decorator,benchmark,profile"),

    ("cache function results to avoid recomputing",
        "from functools import lru_cache\n\n"
        "@lru_cache(maxsize=128)\n"
        "def fib(n):\n"
        "    if n < 2: return n\n"
        "    return fib(n-1) + fib(n-2)\n\n"
        "# Example:\n"
        "print(fib(50))   # 12586269025\n"
        "print(fib.cache_info())\n",
        "cache,memoize,lru_cache,decorator"),

    ("split a list into chunks of N items",
        "def chunks(items, n):\n"
        "    for i in range(0, len(items), n):\n"
        "        yield items[i:i+n]\n\n"
        "# Example:\n"
        "for chunk in chunks([1,2,3,4,5,6,7,8,9,10], 3):\n"
        "    print(list(chunk))\n"
        "# [1,2,3] [4,5,6] [7,8,9] [10]\n",
        "chunk,batch,split,list"),

    ("compute moving average over a sliding window",
        "from collections import deque\n\n"
        "def moving_average(items, window=5):\n"
        "    if window <= 0: return []\n"
        "    out = []\n"
        "    buf = deque(maxlen=window)\n"
        "    for x in items:\n"
        "        buf.append(x)\n"
        "        if len(buf) == window:\n"
        "            out.append(sum(buf) / window)\n"
        "    return out\n\n"
        "# Example:\n"
        "print(moving_average([1,2,3,4,5,6,7,8,9,10], window=3))\n"
        "# [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]\n",
        "moving_average,window,smoothing,timeseries"),

    ("compute the Pearson correlation between two lists",
        "def correlation(xs, ys):\n"
        "    n = len(xs)\n"
        "    if n != len(ys) or n < 2:\n"
        "        return None\n"
        "    mx = sum(xs) / n\n"
        "    my = sum(ys) / n\n"
        "    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))\n"
        "    sx2 = sum((x - mx) ** 2 for x in xs)\n"
        "    sy2 = sum((y - my) ** 2 for y in ys)\n"
        "    denom = (sx2 * sy2) ** 0.5\n"
        "    return cov / denom if denom else None\n\n"
        "# Example:\n"
        "print(correlation([1,2,3,4,5], [2,4,6,8,10]))   # 1.0\n",
        "correlation,statistics,pearson"),

    ("compute median of a list",
        "def median(values):\n"
        "    if not values: return None\n"
        "    sorted_vals = sorted(values)\n"
        "    n = len(sorted_vals)\n"
        "    mid = n // 2\n"
        "    if n % 2 == 1:\n"
        "        return sorted_vals[mid]\n"
        "    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2\n\n"
        "# Example:\n"
        "print(median([3, 1, 4, 1, 5, 9, 2, 6]))   # 3.5\n",
        "median,statistics,sort"),

    ("rolling standard deviation over a window",
        "def rolling_std(items, window=10):\n"
        "    if window < 2 or len(items) < window:\n"
        "        return []\n"
        "    out = []\n"
        "    for i in range(window - 1, len(items)):\n"
        "        slice_ = items[i - window + 1: i + 1]\n"
        "        mean = sum(slice_) / window\n"
        "        var = sum((x - mean) ** 2 for x in slice_) / window\n"
        "        out.append(var ** 0.5)\n"
        "    return out\n\n"
        "# Example:\n"
        "print(rolling_std([1, 2, 3, 4, 5, 6, 7, 8], window=3))\n",
        "rolling,std,deviation,statistics,window"),

    ("read a file line by line memory-efficiently",
        "def stream_lines(path):\n"
        "    with open(path, 'r', encoding='utf-8') as f:\n"
        "        for line in f:\n"
        "            yield line.rstrip()\n\n"
        "# Example:\n"
        "# for line in stream_lines('huge.txt'):\n"
        "#     process(line)\n",
        "file,stream,generator,memory"),

    ("class with constructor and string repr",
        "class Person:\n"
        "    def __init__(self, name, age):\n"
        "        self.name = name\n"
        "        self.age = age\n"
        "    def __repr__(self):\n"
        "        return f'Person(name={self.name!r}, age={self.age})'\n"
        "    def __eq__(self, other):\n"
        "        return (isinstance(other, Person)\n"
        "                and self.name == other.name and self.age == other.age)\n\n"
        "# Example:\n"
        "p = Person('Eric', 39)\n"
        "print(p)\n",
        "class,oop,object,repr,equality"),

    ("dataclass example",
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class Trade:\n"
        "    ticker: str\n"
        "    side: str       # 'buy' or 'sell'\n"
        "    qty: int\n"
        "    price: float\n"
        "    realized_pnl: float = 0.0\n\n"
        "# Example:\n"
        "t = Trade('MES=F', 'sell', 1, 7492.0)\n"
        "print(t)   # Trade(ticker='MES=F', side='sell', qty=1, price=7492.0, realized_pnl=0.0)\n",
        "dataclass,class,oop,decorator"),

    ("count occurrences with a regular dict",
        "def count(items):\n"
        "    counts = {}\n"
        "    for x in items:\n"
        "        counts[x] = counts.get(x, 0) + 1\n"
        "    return counts\n\n"
        "# Example:\n"
        "print(count('mississippi'))   # {'m': 1, 'i': 4, 's': 4, 'p': 2}\n",
        "count,frequency,dict"),

    ("sort a list of dicts by a key",
        "def sort_by_key(items, key, descending=False):\n"
        "    return sorted(items, key=lambda d: d[key], reverse=descending)\n\n"
        "# Example:\n"
        "rows = [{'name': 'A', 'score': 5}, {'name': 'B', 'score': 9}, {'name': 'C', 'score': 3}]\n"
        "print(sort_by_key(rows, 'score', descending=True))\n",
        "sort,dict,key,list"),

    ("filter a list with a condition",
        "def filter_items(items, predicate):\n"
        "    return [x for x in items if predicate(x)]\n\n"
        "# Example:\n"
        "evens = filter_items([1, 2, 3, 4, 5, 6, 7, 8], lambda x: x % 2 == 0)\n"
        "print(evens)   # [2, 4, 6, 8]\n",
        "filter,list_comp,predicate"),

    ("map a function over a list",
        "def map_items(items, fn):\n"
        "    return [fn(x) for x in items]\n\n"
        "# Example:\n"
        "squared = map_items([1, 2, 3, 4, 5], lambda x: x * x)\n"
        "print(squared)   # [1, 4, 9, 16, 25]\n",
        "map,transform,list_comp"),

    ("simple HTTP server",
        "from http.server import HTTPServer, BaseHTTPRequestHandler\n\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200)\n"
        "        self.send_header('Content-Type', 'text/plain')\n"
        "        self.end_headers()\n"
        "        self.wfile.write(b'hello from a Python HTTP server')\n\n"
        "if __name__ == '__main__':\n"
        "    HTTPServer(('localhost', 8000), Handler).serve_forever()\n",
        "http,server,web,localhost"),

    ("parse a query string from a URL",
        "from urllib.parse import urlparse, parse_qs\n\n"
        "def parse_url(url):\n"
        "    parsed = urlparse(url)\n"
        "    return {\n"
        "        'scheme': parsed.scheme,\n"
        "        'host':   parsed.netloc,\n"
        "        'path':   parsed.path,\n"
        "        'query':  parse_qs(parsed.query),\n"
        "    }\n\n"
        "# Example:\n"
        "print(parse_url('https://example.com/api?name=eric&count=5'))\n",
        "url,parse,query,urllib"),

    ("compute a SHA-256 hash of a string",
        "import hashlib\n\n"
        "def sha256(text):\n"
        "    return hashlib.sha256(text.encode('utf-8')).hexdigest()\n\n"
        "# Example:\n"
        "print(sha256('hello world'))\n",
        "hash,sha256,crypto,fingerprint"),

    ("base64 encode and decode bytes",
        "import base64\n\n"
        "def b64_encode(data: bytes) -> str:\n"
        "    return base64.b64encode(data).decode('ascii')\n\n"
        "def b64_decode(text: str) -> bytes:\n"
        "    return base64.b64decode(text)\n\n"
        "# Example:\n"
        "encoded = b64_encode(b'hello')\n"
        "print(encoded)            # 'aGVsbG8='\n"
        "print(b64_decode(encoded))   # b'hello'\n",
        "base64,encode,decode,binary"),

    ("environment variable with default",
        "import os\n\n"
        "def env(name, default=None):\n"
        "    return os.environ.get(name, default)\n\n"
        "# Example:\n"
        "print(env('HOME', '/tmp'))\n",
        "env,environ,os,config"),

    ("ensure a directory exists",
        "from pathlib import Path\n\n"
        "def ensure_dir(path):\n"
        "    Path(path).mkdir(parents=True, exist_ok=True)\n\n"
        "# Example:\n"
        "ensure_dir('out/nested/folder')\n",
        "directory,mkdir,pathlib,filesystem"),

    ("threaded function with a worker pool",
        "from concurrent.futures import ThreadPoolExecutor\n\n"
        "def parallel(fn, items, max_workers=4):\n"
        "    with ThreadPoolExecutor(max_workers=max_workers) as pool:\n"
        "        return list(pool.map(fn, items))\n\n"
        "# Example:\n"
        "def square(x): return x * x\n"
        "print(parallel(square, [1, 2, 3, 4, 5]))   # [1, 4, 9, 16, 25]\n",
        "thread,pool,concurrent,parallel"),

    ("compute first N primes using a sieve",
        "def primes(limit):\n"
        "    sieve = [True] * (limit + 1)\n"
        "    sieve[0] = sieve[1] = False\n"
        "    for i in range(2, int(limit**0.5) + 1):\n"
        "        if sieve[i]:\n"
        "            for j in range(i*i, limit + 1, i):\n"
        "                sieve[j] = False\n"
        "    return [i for i, p in enumerate(sieve) if p]\n\n"
        "# Example:\n"
        "print(primes(50))\n",
        "primes,sieve,eratosthenes,number_theory"),
]


# ─── Encoder + store ─────────────────────────────────────────────


class CodeCorpus:
    def __init__(self, db_path: Path = None, encoder=None):
        self.db_path = db_path or CODE_CORPUS_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder
        self._con = sqlite3.connect(str(self.db_path),
                                          check_same_thread=False,
                                          timeout=30.0)
        self._con.executescript(_SCHEMA)
        self._con.commit()
        self._ids: list[int] = []
        self._descriptions: list[str] = []
        self._codes: list[str] = []
        self._tags: list[str] = []
        self._stack: Optional[np.ndarray] = None
        self._reload()

    def _reload(self):
        rows = self._con.execute(
            "SELECT id, description, code, tags, hv "
            "FROM code_snippets ORDER BY id"
        ).fetchall()
        self._ids          = [r[0] for r in rows]
        self._descriptions = [r[1] for r in rows]
        self._codes        = [r[2] for r in rows]
        self._tags         = [r[3] or "" for r in rows]
        if rows:
            self._stack = np.stack(
                [np.frombuffer(r[4], dtype=np.int8) for r in rows]
            )
        else:
            self._stack = None

    def count(self) -> int:
        return len(self._ids)

    def add(self, description: str, code: str, tags: str = "") -> int:
        if self.encoder is None:
            raise RuntimeError("CodeCorpus: encoder not set")
        # Encode the description (the "what does this do" signal) + tags
        hv = self.encoder.encode(description + " " + tags).astype(np.int8)
        ts = datetime.now(timezone.utc).isoformat()
        cur = self._con.execute(
            "INSERT INTO code_snippets (description, code, tags, hv, "
            "created_at) VALUES (?, ?, ?, ?, ?)",
            (description, code, tags, hv.tobytes(), ts),
        )
        self._con.commit()
        mid = cur.lastrowid
        self._ids.append(mid)
        self._descriptions.append(description)
        self._codes.append(code)
        self._tags.append(tags)
        if self._stack is None:
            self._stack = hv[None, :].copy()
        else:
            self._stack = np.vstack([self._stack, hv[None, :]])
        return mid

    def seed(self):
        if self.count() > 0:
            return 0
        for desc, code, tags in SNIPPETS:
            self.add(desc, code, tags)
        return len(SNIPPETS)

    def query(self, msg: str, k: int = 3) -> list[dict]:
        if self._stack is None or len(self._ids) == 0 or self.encoder is None:
            return []
        q_hv = self.encoder.encode(msg).astype(np.int8)
        from train.v5_hdc_prototype import D
        xor = np.bitwise_xor(self._stack, q_hv[None, :])
        dists = xor.sum(axis=1)
        order = np.argsort(dists)[:k]
        out = []
        for rank, idx in enumerate(order):
            d = int(dists[idx])
            sim = 1.0 - 2.0 * d / D
            out.append({
                "id":          self._ids[idx],
                "description": self._descriptions[idx],
                "code":        self._codes[idx],
                "tags":        self._tags[idx],
                "distance":    d,
                "similarity":  round(sim, 4),
                "rank":        rank + 1,
            })
        return out


# ─── Try-retrieve entry ──────────────────────────────────────────


def try_retrieve_code(msg: str, corpus: "CodeCorpus",
                            min_similarity: float = 0.40) -> Optional[dict]:
    """If the corpus has a relevant snippet for this message, return
    it; otherwise None.  Used as a fallback when templates miss."""
    if corpus is None or corpus.count() == 0:
        return None
    hits = corpus.query(msg, k=3)
    if not hits or hits[0]["similarity"] < min_similarity:
        return None
    top = hits[0]
    return {
        "code":        top["code"],
        "label":       top["description"],
        "template":    f"corpus:{top['id']}",
        "ran":         False,
        "output":      "",
        "customized":  False,
        "similarity":  top["similarity"],
        "alternatives": [{"description": h["description"],
                              "similarity": h["similarity"]}
                              for h in hits[1:]],
    }


def _self_test():
    print(f"  {len(SNIPPETS)} curated snippets ready to be seeded")
    by_tag = {}
    for desc, _, tags in SNIPPETS:
        for tag in tags.split(","):
            by_tag[tag] = by_tag.get(tag, 0) + 1
    print(f"  tag distribution (top 10):")
    for tag, count in sorted(by_tag.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {tag:<15s}  {count}")


if __name__ == "__main__":
    _self_test()
