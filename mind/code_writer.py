"""autopilot/code_writer.py — simple code synthesis for chat.

Take a natural-language request ("write me a function that filters
duplicates from a list") and return a runnable Python snippet.

This is the introductory layer — common tasks with templated code.
Each template has:
  * intent_patterns: regexes that match the request
  * slot_extractors:   functions that pull parameter names from the msg
  * code:              template string with {slots}
  * example_input:     a small demo input to RUN the code against
  * example_label:     a one-line description

When a template matches, we:
  1. Generate the code by filling slots.
  2. Run it sandboxed against the example input.
  3. Return code + result so the user sees both.

Future extensions:
  * Multi-template composition (forward-chain code transformations)
  * Self-modification (Telp editing his own modules)
  * Backtest snippet synthesis (trading-specific)
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


# ─── Slot extractors ──────────────────────────────────────────────


def _extract_list_name(msg: str) -> str:
    """Pull a 'noun' phrase that probably names the list/data."""
    m = re.search(
        r"\b(?:of|in|from|the|my)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b",
        msg, re.IGNORECASE,
    )
    if m:
        return m.group(1)
    # Fallback
    return "items"


def _extract_number(msg: str, default: int = 10) -> int:
    """Pull the first integer in the message."""
    m = re.search(r"\b(\d+)\b", msg)
    return int(m.group(1)) if m else default


def _extract_string_arg(msg: str) -> str:
    """Pull a quoted string or fall back to a placeholder."""
    m = re.search(r"['\"]([^'\"]+)['\"]", msg)
    return m.group(1) if m else "your text here"


# ── Layer 2: extract the user's actual data from the message ──────


def extract_literal_list(msg: str) -> Optional[list]:
    """Find a Python-list literal in the message. Returns the parsed
    list, or None.

    Handles nested brackets by tracking bracket depth (the simple
    regex can't match nested groups).
    """
    # Bracket-balanced scan: find the longest [ ... ] block.
    start = msg.find("[")
    while start >= 0:
        depth = 0
        for i in range(start, len(msg)):
            c = msg[i]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    candidate = msg[start:i+1]
                    try:
                        value = ast.literal_eval(candidate)
                        if isinstance(value, list):
                            return value
                    except Exception:
                        pass
                    break
        # Look for the next [ after this one
        start = msg.find("[", start + 1)
    # Try comma-separated bare numbers/words after a keyword
    m = re.search(
        r"\b(?:list|numbers|values|items|of|from|the\s+list)\s*[:=]?\s*"
        r"([^.?!]*)$",
        msg, re.IGNORECASE,
    )
    if m:
        tail = m.group(1).strip().rstrip(".?!")
        # Look for at least two comma-separated tokens
        parts = [p.strip().strip("'\"") for p in re.split(r"[,\s]+", tail)
                  if p.strip()]
        if len(parts) >= 2:
            parsed = []
            for p in parts:
                try:
                    parsed.append(ast.literal_eval(p))
                except Exception:
                    parsed.append(p)
            if parsed:
                return parsed
    return None


def extract_literal_string(msg: str) -> Optional[str]:
    """Find a quoted string in the message."""
    for quote in ('"', "'"):
        m = re.search(rf"{quote}([^{quote}]+){quote}", msg)
        if m:
            return m.group(1)
    return None


def extract_literal_int(msg: str) -> Optional[int]:
    """Find a bare integer in the message. Returns the first standalone
    integer (not part of a list, not part of a year, etc.)."""
    # Take the last standalone integer (often the parameter)
    # Filter out years (4 digits >= 1900) and ints inside brackets.
    candidates = []
    for m in re.finditer(r"\b(\d+)\b", msg):
        val = int(m.group(1))
        # Skip years
        if 1900 <= val <= 2099:
            continue
        # Skip integers inside brackets
        start = m.start()
        before = msg[:start]
        after = msg[m.end():]
        if before.rfind("[") > before.rfind("]"):
            continue
        candidates.append(val)
    if not candidates:
        return None
    # Prefer the LAST integer (typically the parameter the user is
    # passing): "fibonacci of 15" → 15.
    return candidates[-1]


def extract_literal_path(msg: str) -> Optional[str]:
    """Find a file path in the message ('input.txt', './data.csv', etc.)."""
    # Quoted path wins
    quoted = extract_literal_string(msg)
    if quoted and (("." in quoted) or ("/" in quoted) or ("\\" in quoted)):
        return quoted
    # Bare path with extension
    m = re.search(r"\b([\w./\\-]+\.[a-zA-Z]{1,5})\b", msg)
    if m:
        return m.group(1)
    return None


# ─── Templates ────────────────────────────────────────────────────


@dataclass
class CodeTemplate:
    name: str
    intent_patterns: list[re.Pattern]
    code: str
    example_input: dict = field(default_factory=dict)
    example_label: str = ""
    slot_fn: Optional[Callable] = None


def _make_pat(*phrases: str) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in phrases]


def _slot_default(msg: str) -> dict:
    return {"name": _extract_list_name(msg)}


TEMPLATES: list[CodeTemplate] = [
    # ── Filter duplicates ──
    CodeTemplate(
        name="filter_duplicates",
        intent_patterns=_make_pat(
            r"\b(filter|remove|drop)\s+(duplicates?|dupes?)",
            r"\bunique\s+(items?|values?|elements?)",
            r"\bdedupe",
            r"\bdistinct\s+(items?|values?|elements?)",
        ),
        code=(
            "def remove_duplicates(items):\n"
            "    seen = set()\n"
            "    out = []\n"
            "    for x in items:\n"
            "        if x not in seen:\n"
            "            seen.add(x)\n"
            "            out.append(x)\n"
            "    return out\n"
            "\n"
            "# Example:\n"
            "result = remove_duplicates([1, 2, 2, 3, 1, 4, 3, 5])\n"
            "print(result)\n"
        ),
        example_input={"items": [1, 2, 2, 3, 1, 4, 3, 5]},
        example_label="remove duplicates while preserving order",
    ),

    # ── Sort a list ──
    CodeTemplate(
        name="sort_list",
        intent_patterns=_make_pat(
            r"\bsort\s+(?:a\s+|the\s+|my\s+)?(?:list|array|sequence|items?|numbers?)",
            r"\bsort\b.*\[",            # "sort [1, 2, 3]"
            r"\bin\s+(ascending|descending)\s+order",
        ),
        code=(
            "def sort_list(items, descending=False):\n"
            "    return sorted(items, reverse=descending)\n"
            "\n"
            "# Example:\n"
            "result = sort_list([3, 1, 4, 1, 5, 9, 2, 6])\n"
            "print(result)\n"
        ),
        example_label="sort a list ascending (or descending=True)",
    ),

    # ── Reverse ──
    CodeTemplate(
        name="reverse",
        intent_patterns=_make_pat(
            r"\breverse\s+(?:a\s+|the\s+|my\s+)?(string|list|array|items?)",
            r"\breverse\b.*['\"]",       # reverse "hello"
            r"\breverse\b.*\[",           # reverse [1, 2, 3]
            r"\bflip\s+(?:a\s+|the\s+)?(string|list)",
        ),
        code=(
            "def reverse(x):\n"
            "    return x[::-1]\n"
            "\n"
            "# Example:\n"
            "print(reverse('hello'))\n"
            "print(reverse([1, 2, 3, 4, 5]))\n"
        ),
        example_label="reverse a string or list using slice",
    ),

    # ── Count occurrences ──
    CodeTemplate(
        name="count_occurrences",
        intent_patterns=_make_pat(
            r"\bcount\s+\w+\s*(?:occurrences?|times|frequenc)",
            r"\bcount\s+(?:occurrences?|how\s+many|the\s+number)",
            r"\bcount\s+(?:word|item|letter|character)s?",
            r"\bhow\s+many\s+(?:times|of)",
            r"\bfrequency\s+of",
            r"\btally\s+(items?|words?)",
        ),
        code=(
            "from collections import Counter\n"
            "\n"
            "def count_occurrences(items):\n"
            "    return Counter(items)\n"
            "\n"
            "# Example:\n"
            "result = count_occurrences(['apple', 'orange', 'apple', 'pear', 'apple'])\n"
            "print(result)\n"
            "print('most common:', result.most_common(1))\n"
        ),
        example_label="count how often each item appears",
    ),

    # ── Sum / mean / max / min ──
    CodeTemplate(
        name="aggregate_stats",
        intent_patterns=_make_pat(
            r"\b(mean|average|avg)\s+of",
            r"\bcompute\s+(mean|average|sum|max|min|stats)",
            r"\bcompute\s+stats?",
            r"\b(?:stats?|statistics)\s+(?:for|of|on)",
            r"\bgive\s+(?:me\s+)?(?:the\s+)?(sum|mean|average|max|min|stats)",
            r"\b(sum|total|mean|average|max|min)\s+of\s+(?:a\s+|the\s+|my\s+)?(list|array|numbers?)",
        ),
        code=(
            "def stats(numbers):\n"
            "    return {\n"
            "        'sum':   sum(numbers),\n"
            "        'mean':  sum(numbers) / len(numbers) if numbers else 0,\n"
            "        'min':   min(numbers) if numbers else None,\n"
            "        'max':   max(numbers) if numbers else None,\n"
            "        'count': len(numbers),\n"
            "    }\n"
            "\n"
            "# Example:\n"
            "result = stats([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])\n"
            "print(result)\n"
        ),
        example_label="compute sum / mean / min / max for a list",
    ),

    # ── Fibonacci ──
    CodeTemplate(
        name="fibonacci",
        intent_patterns=_make_pat(
            r"\bfibonacci",
            r"\bfib\b",
        ),
        code=(
            "def fibonacci(n):\n"
            "    a, b = 0, 1\n"
            "    out = []\n"
            "    for _ in range(n):\n"
            "        out.append(a)\n"
            "        a, b = b, a + b\n"
            "    return out\n"
            "\n"
            "# Example:\n"
            "print(fibonacci(10))\n"
        ),
        example_label="generate first N Fibonacci numbers",
    ),

    # ── FizzBuzz ──
    CodeTemplate(
        name="fizzbuzz",
        intent_patterns=_make_pat(
            r"\bfizz\s*buzz",
            r"\bfizzbuzz",
        ),
        code=(
            "def fizzbuzz(n):\n"
            "    out = []\n"
            "    for i in range(1, n + 1):\n"
            "        if i % 15 == 0:\n"
            "            out.append('FizzBuzz')\n"
            "        elif i % 3 == 0:\n"
            "            out.append('Fizz')\n"
            "        elif i % 5 == 0:\n"
            "            out.append('Buzz')\n"
            "        else:\n"
            "            out.append(str(i))\n"
            "    return out\n"
            "\n"
            "# Example:\n"
            "print(fizzbuzz(20))\n"
        ),
        example_label="classic FizzBuzz",
    ),

    # ── Factorial ──
    CodeTemplate(
        name="factorial",
        intent_patterns=_make_pat(
            r"\bfactorial",
        ),
        code=(
            "def factorial(n):\n"
            "    if n < 0:\n"
            "        raise ValueError('factorial of negative')\n"
            "    result = 1\n"
            "    for i in range(2, n + 1):\n"
            "        result *= i\n"
            "    return result\n"
            "\n"
            "# Example:\n"
            "print(factorial(5))   # 120\n"
            "print(factorial(10))  # 3628800\n"
        ),
        example_label="compute n! iteratively",
    ),

    # ── Primality / prime numbers ──
    CodeTemplate(
        name="primes",
        intent_patterns=_make_pat(
            r"\bis.*\bprime",
            r"\bprime\s+numbers?",
            r"\bgenerate\s+primes?",
            r"\bfind\s+(?:all\s+)?primes?",
            r"\bprimes?\s+up\s+to",
            r"\blist\s+(?:all\s+)?primes?",
            r"\bsieve",
            r"\bcheck\s+if.*prime",
        ),
        code=(
            "def is_prime(n):\n"
            "    if n < 2: return False\n"
            "    if n < 4: return True\n"
            "    if n % 2 == 0: return False\n"
            "    i = 3\n"
            "    while i * i <= n:\n"
            "        if n % i == 0: return False\n"
            "        i += 2\n"
            "    return True\n"
            "\n"
            "def primes_up_to(n):\n"
            "    return [x for x in range(2, n + 1) if is_prime(x)]\n"
            "\n"
            "# Example:\n"
            "print(is_prime(29))         # True\n"
            "print(primes_up_to(30))     # [2,3,5,7,11,13,17,19,23,29]\n"
        ),
        example_label="check primality / list primes up to n",
    ),

    # ── Read a file ──
    CodeTemplate(
        name="read_file",
        intent_patterns=_make_pat(
            r"\bread\s+(?:a\s+|the\s+|my\s+)?file",
            r"\bopen\s+(?:a\s+|the\s+)?file",
            r"\bload\s+(?:a\s+|the\s+)?(?:text\s+)?file",
        ),
        code=(
            "def read_file(path):\n"
            "    with open(path, 'r', encoding='utf-8') as f:\n"
            "        return f.read()\n"
            "\n"
            "def read_lines(path):\n"
            "    with open(path, 'r', encoding='utf-8') as f:\n"
            "        return [line.rstrip() for line in f]\n"
            "\n"
            "# Example usage:\n"
            "# text = read_file('input.txt')\n"
            "# lines = read_lines('input.txt')\n"
        ),
        example_label="read text/lines from a file",
    ),

    # ── Write a file ──
    CodeTemplate(
        name="write_file",
        intent_patterns=_make_pat(
            r"\bwrite\s+(?:to\s+)?(?:a\s+|the\s+|my\s+)?file",
            r"\bsave\s+(?:to\s+)?(?:a\s+|the\s+)?file",
        ),
        code=(
            "def write_file(path, text):\n"
            "    with open(path, 'w', encoding='utf-8') as f:\n"
            "        f.write(text)\n"
            "\n"
            "def append_to_file(path, text):\n"
            "    with open(path, 'a', encoding='utf-8') as f:\n"
            "        f.write(text)\n"
            "\n"
            "# Example usage:\n"
            "# write_file('out.txt', 'hello world')\n"
            "# append_to_file('log.txt', 'another line\\n')\n"
        ),
        example_label="write or append text to a file",
    ),

    # ── Regex match ──
    CodeTemplate(
        name="regex_match",
        intent_patterns=_make_pat(
            r"\bregex",
            r"\bregular\s+expression",
            r"\bfind\s+all\s+(?:matches|emails?|urls?|numbers?|words?)",
        ),
        code=(
            "import re\n"
            "\n"
            "def find_all(pattern, text):\n"
            "    return re.findall(pattern, text)\n"
            "\n"
            "# Common patterns:\n"
            "EMAIL_RX  = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}'\n"
            "URL_RX    = r'https?://[^\\s]+'\n"
            "NUMBER_RX = r'-?\\d+(?:\\.\\d+)?'\n"
            "\n"
            "# Example:\n"
            "text = 'Email me at alice@example.com or bob@x.org. Visit https://example.com'\n"
            "print('emails:', find_all(EMAIL_RX, text))\n"
            "print('urls:',   find_all(URL_RX, text))\n"
        ),
        example_label="regex find_all with email/URL/number presets",
    ),

    # ── Split / join string ──
    CodeTemplate(
        name="split_string",
        intent_patterns=_make_pat(
            r"\bsplit\s+(?:a\s+|the\s+|my\s+)?string",
            r"\bsplit\s+by",
            r"\bjoin\s+(?:a\s+|the\s+)?list",
        ),
        code=(
            "def split_text(text, separator=' '):\n"
            "    return text.split(separator)\n"
            "\n"
            "def join_items(items, separator=', '):\n"
            "    return separator.join(str(x) for x in items)\n"
            "\n"
            "# Example:\n"
            "print(split_text('one,two,three', ','))\n"
            "print(join_items(['a', 'b', 'c'], ' - '))\n"
        ),
        example_label="split a string / join a list",
    ),

    # ── Zip / make dict from two lists ──
    CodeTemplate(
        name="zip_dict",
        intent_patterns=_make_pat(
            r"\bzip\s+two",
            r"\bdict(?:ionary)?\s+from\s+(?:two\s+)?lists?",
            r"\bpair\s+up\s+two",
        ),
        code=(
            "def dict_from_lists(keys, values):\n"
            "    return dict(zip(keys, values))\n"
            "\n"
            "# Example:\n"
            "result = dict_from_lists(['name', 'age', 'role'],\n"
            "                          ['Eric', 39, 'builder'])\n"
            "print(result)\n"
        ),
        example_label="zip two lists into a dict",
    ),

    # ── Flatten ──
    CodeTemplate(
        name="flatten",
        intent_patterns=_make_pat(
            r"\bflatten\s+(?:a\s+|the\s+)?(?:nested\s+)?list",
            r"\bflatten\b.*\[",       # "flatten [1, [2, 3]]"
            r"\bone-?dimensional",
        ),
        code=(
            "def flatten(nested):\n"
            "    out = []\n"
            "    for item in nested:\n"
            "        if isinstance(item, (list, tuple)):\n"
            "            out.extend(flatten(item))\n"
            "        else:\n"
            "            out.append(item)\n"
            "    return out\n"
            "\n"
            "# Example:\n"
            "print(flatten([1, [2, [3, 4], 5], [6, [7, [8]]]]))\n"
        ),
        example_label="flatten an arbitrarily-nested list",
    ),

    # ── Note-taking CLI app (SQLite-backed) ──
    CodeTemplate(
        name="note_app",
        intent_patterns=_make_pat(
            r"\bnote(?:s|book|-taking)?\s+(?:app|application|cli|tool|program)",
            r"\b(?:build|make|write|create)\s+(?:me\s+)?(?:a\s+)?notes?\b",
            r"\btodo\s+(?:app|list|application|cli)",
            r"\b(?:build|make|write|create)\s+(?:me\s+)?(?:a\s+)?todo\b",
        ),
        code=(
            '"""A tiny CLI note-taking app, SQLite-backed.\n'
            "\n"
            "Usage:\n"
            "    python notes.py add 'first note'\n"
            "    python notes.py list\n"
            "    python notes.py search 'first'\n"
            "    python notes.py delete 1\n"
            '"""\n'
            "import sqlite3\n"
            "import sys\n"
            "from datetime import datetime, timezone\n"
            "from pathlib import Path\n"
            "\n"
            "DB = Path('notes.db')\n"
            "\n"
            "_SCHEMA = '''\n"
            "CREATE TABLE IF NOT EXISTS notes (\n"
            "    id         INTEGER PRIMARY KEY,\n"
            "    text       TEXT NOT NULL,\n"
            "    created_at TEXT NOT NULL\n"
            ");\n"
            "'''\n"
            "\n"
            "def _con():\n"
            "    con = sqlite3.connect(DB)\n"
            "    con.executescript(_SCHEMA)\n"
            "    return con\n"
            "\n"
            "def add(text):\n"
            "    ts = datetime.now(timezone.utc).isoformat()\n"
            "    with _con() as c:\n"
            "        cur = c.execute(\n"
            "            'INSERT INTO notes (text, created_at) VALUES (?, ?)',\n"
            "            (text, ts),\n"
            "        )\n"
            "        print(f'added #{cur.lastrowid}')\n"
            "\n"
            "def list_notes():\n"
            "    with _con() as c:\n"
            "        rows = c.execute(\n"
            "            'SELECT id, created_at, text FROM notes ORDER BY id'\n"
            "        ).fetchall()\n"
            "    if not rows:\n"
            "        print('(no notes yet)')\n"
            "        return\n"
            "    for r in rows:\n"
            "        print(f'#{r[0]:>3}  {r[1][:19]}  {r[2]}')\n"
            "\n"
            "def search(query):\n"
            "    with _con() as c:\n"
            "        rows = c.execute(\n"
            "            'SELECT id, created_at, text FROM notes '\n"
            "            'WHERE text LIKE ? ORDER BY id',\n"
            "            (f'%{query}%',),\n"
            "        ).fetchall()\n"
            "    if not rows:\n"
            "        print(f'(no matches for {query!r})')\n"
            "        return\n"
            "    for r in rows:\n"
            "        print(f'#{r[0]:>3}  {r[1][:19]}  {r[2]}')\n"
            "\n"
            "def delete(note_id):\n"
            "    with _con() as c:\n"
            "        cur = c.execute('DELETE FROM notes WHERE id = ?', (int(note_id),))\n"
            "        if cur.rowcount:\n"
            "            print(f'deleted #{note_id}')\n"
            "        else:\n"
            "            print(f'no note #{note_id}')\n"
            "\n"
            "def main():\n"
            "    if len(sys.argv) < 2:\n"
            "        print(__doc__.strip())\n"
            "        return\n"
            "    cmd = sys.argv[1].lower()\n"
            "    args = sys.argv[2:]\n"
            "    if   cmd == 'add'    and args: add(' '.join(args))\n"
            "    elif cmd == 'list':            list_notes()\n"
            "    elif cmd == 'search' and args: search(' '.join(args))\n"
            "    elif cmd == 'delete' and args: delete(args[0])\n"
            "    else:\n"
            "        print('usage: notes.py {add <text>|list|search <q>|delete <id>}')\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
            "\n"
            "# Quick demo (executed in this run):\n"
            "import os, tempfile\n"
            "DB = Path(tempfile.gettempdir()) / 'notes_demo.db'\n"
            "if DB.exists(): DB.unlink()\n"
            "add('buy groceries')\n"
            "add('finish reading the trade journal')\n"
            "add('call mom on Sunday')\n"
            "print('--- all notes ---')\n"
            "list_notes()\n"
            "print('--- search \"trade\" ---')\n"
            "search('trade')\n"
            "print('--- delete #2 ---')\n"
            "delete(2)\n"
            "print('--- after delete ---')\n"
            "list_notes()\n"
        ),
        example_label=(
            "a working CLI note-taking app — SQLite storage, "
            "add / list / search / delete commands"
        ),
    ),

    # ── HTTP GET ──
    CodeTemplate(
        name="http_get",
        intent_patterns=_make_pat(
            r"\bhttp\s+get",
            r"\bfetch\s+(?:a\s+|the\s+)?(?:url|webpage|api)",
            r"\bdownload\s+(?:a\s+|the\s+)?(?:url|webpage|file)\s+from",
            r"\brequests?\s*\.\s*get",
        ),
        code=(
            "import urllib.request\n"
            "\n"
            "def fetch(url, timeout=10):\n"
            "    with urllib.request.urlopen(url, timeout=timeout) as r:\n"
            "        return r.read().decode('utf-8', errors='replace')\n"
            "\n"
            "# Example:\n"
            "# html = fetch('https://example.com')\n"
            "# print(html[:200])\n"
        ),
        example_label="fetch a URL with the stdlib (no requests dependency)",
    ),
]


# ─── Intent matching ──────────────────────────────────────────────


# Trigger words — at least one must appear for us to even try matching
# a template.  Avoids running every template against every chat message.
_GLOBAL_TRIGGERS = (
    "write", "code", "function", "python", "script", "snippet",
    "give me", "show me", "generate", "implement", "build me",
    "how do i", "how would i", "how can i",
)


def classify_intent(msg: str,
                          permissive: bool = False) -> Optional[CodeTemplate]:
    """Return the first template whose intent_patterns match msg, or
    None.

    Strategy: an intent pattern match alone is enough signal — the
    patterns are specific (e.g., "fibonacci", "fizzbuzz", "remove
    duplicates").  We DO require either:
      * a global code-trigger word ("write", "code", "function", ...)
        OR a literal data block in the message ("[1,2,3]" or a
        quoted string).
    This prevents pure-knowledge questions ("what is fibonacci?") from
    accidentally triggering code synthesis on every single keyword.

    `permissive=True` is used during multi-step composition for the
    SECOND-and-later steps: they shouldn't need explicit data because
    the prior step's output feeds them.  A bare verb ("sort", "reverse")
    is enough.
    """
    if not msg:
        return None
    low = msg.lower()
    has_trigger = any(t in low for t in _GLOBAL_TRIGGERS)
    has_data    = bool(extract_literal_list(msg) or
                          extract_literal_string(msg))
    has_imperative = bool(re.match(
        r"^\s*(sort|reverse|count|compute|calculate|find|filter|remove|"
        r"flatten|fibonacci|factorial|fizz|fizzbuzz|primes?|"
        r"split|join|read|write|fetch|download|"
        r"build|make|create|notes?|todo)\b", low,
    ))
    if not permissive and not (has_trigger or has_data or has_imperative):
        return None

    # In permissive mode, also accept bare verbs (no noun, no data)
    if permissive:
        bare_verb = re.match(
            r"^\s*(sort|reverse|flatten|count|filter)\s*\.?$", low,
        )
        if bare_verb:
            verb = bare_verb.group(1)
            verb_to_template = {
                "sort":     "sort_list",
                "reverse":  "reverse",
                "flatten":  "flatten",
                "count":    "count_occurrences",
                "filter":   "filter_duplicates",
            }
            tn = verb_to_template.get(verb)
            if tn:
                for tmpl in TEMPLATES:
                    if tmpl.name == tn:
                        return tmpl

    for tmpl in TEMPLATES:
        for pat in tmpl.intent_patterns:
            if pat.search(msg):
                return tmpl
    return None


# ─── Sandboxed execution ──────────────────────────────────────────


def _safe_run(code: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Run code in a fresh subprocess.  Captures stdout/stderr.
    Returns (ok, output_text).
    """
    # Quick syntax check
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    # Refuse obviously unsafe patterns
    blocklist = ("os.system", "subprocess.", "shutil.rmtree",
                  "__import__", "eval(", "exec(")
    # We allow open( (file I/O is intentional in many templates).
    # We allow input(  — subprocess has no stdin attached, so input()
    # raises EOFError immediately rather than hanging.  Game templates
    # legitimately define play() functions using input().
    for bad in blocklist:
        if bad in code:
            return False, f"refusing to run code containing {bad!r}"

    # Run in a subprocess with -I (isolated) to avoid env interference.
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8",
    )
    try:
        tmp.write(code)
        tmp.close()
        try:
            res = subprocess.run(
                [sys.executable, "-I", "-W", "ignore", tmp.name],
                capture_output=True, text=True, timeout=timeout,
                check=False,
            )
            out = (res.stdout or "")
            err = (res.stderr or "")
            if res.returncode != 0:
                return False, (out + err).strip()[:2000]
            return True, out.strip()[:2000]
        except subprocess.TimeoutExpired:
            return False, f"(execution exceeded {timeout}s)"
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except Exception:
            pass


# ─── Layer 2: parameter customization ────────────────────────────


# Each template can declare a "customize" pattern: a regex that locates
# the example-input line(s) and the substitution rule.  When the user's
# message has literal data, we replace the canonical example with the
# user's data.
#
# Format: {template_name: (find_pattern, build_replacement)}.
# build_replacement is a callable(template_str, user_inputs) -> str.

def _customize_with_list(code: str, user_list: list,
                              func_name: str, var_name: str = "result") -> str:
    """Replace the '# Example:' block in `code` with one using
    `user_list` instead of the canonical demo."""
    # Find the example block (everything from # Example: onward)
    lines = code.split("\n")
    out_lines = []
    skip_to_end = False
    for line in lines:
        if line.strip().startswith("# Example"):
            # Replace the example block with one using user_list
            out_lines.append(f"# Example (using your list):")
            out_lines.append(f"result = {func_name}({user_list!r})")
            out_lines.append("print(result)")
            skip_to_end = True
            continue
        if not skip_to_end:
            out_lines.append(line)
    return "\n".join(out_lines).rstrip() + "\n"


def _customize_with_string(code: str, user_string: str,
                                func_name: str) -> str:
    lines = code.split("\n")
    out = []
    skip = False
    for line in lines:
        if line.strip().startswith("# Example"):
            out.append("# Example (using your input):")
            out.append(f"result = {func_name}({user_string!r})")
            out.append("print(result)")
            skip = True
            continue
        if not skip:
            out.append(line)
    return "\n".join(out).rstrip() + "\n"


def _customize_with_int(code: str, user_int: int, func_name: str) -> str:
    lines = code.split("\n")
    out = []
    skip = False
    for line in lines:
        if line.strip().startswith("# Example"):
            out.append(f"# Example (n = {user_int}):")
            out.append(f"result = {func_name}({user_int})")
            out.append("print(result)")
            skip = True
            continue
        if not skip:
            out.append(line)
    return "\n".join(out).rstrip() + "\n"


# Per-template customization rules: what data type to look for and
# which function name to call with it.
_TEMPLATE_CUSTOMIZERS = {
    # name → (data_extractor, func_name_to_call, customizer_fn)
    "filter_duplicates":  ("list", "remove_duplicates", _customize_with_list),
    "sort_list":          ("list", "sort_list",          _customize_with_list),
    "reverse":            ("any",  "reverse",            None),   # handled below
    "count_occurrences":  ("list", "count_occurrences",  _customize_with_list),
    "aggregate_stats":    ("list", "stats",              _customize_with_list),
    "fibonacci":          ("int",  "fibonacci",          _customize_with_int),
    "fizzbuzz":           ("int",  "fizzbuzz",           _customize_with_int),
    "factorial":          ("int",  "factorial",          _customize_with_int),
    "flatten":            ("list", "flatten",            _customize_with_list),
}


def customize_template(template: CodeTemplate, msg: str) -> str:
    """Return the template's code customized with user-supplied data
    if any was found in `msg`.  Falls back to the canonical example.
    """
    if template.name not in _TEMPLATE_CUSTOMIZERS:
        return template.code

    data_type, func_name, customizer = _TEMPLATE_CUSTOMIZERS[template.name]

    if data_type == "list":
        user_list = extract_literal_list(msg)
        if user_list is not None and len(user_list) >= 2:
            return customizer(template.code, user_list, func_name)
    elif data_type == "int":
        user_int = extract_literal_int(msg)
        if user_int is not None and 0 < user_int <= 100:
            return customizer(template.code, user_int, func_name)
    elif data_type == "any":
        # Reverse: try string first, then list
        s = extract_literal_string(msg)
        if s:
            return _customize_with_string(template.code, s, func_name)
        l = extract_literal_list(msg)
        if l is not None and len(l) >= 2:
            return _customize_with_list(template.code, l, func_name)

    # No usable data found — return canonical example
    return template.code


# ─── Public entry ────────────────────────────────────────────────


# ─── Layer 3: composition of multiple templates ──────────────────


# Split connectors signal that the user wants a composed pipeline.
_COMPOSE_CONNECTORS = re.compile(
    r"\b(?:then|and\s+then|after\s+that|afterwards|next|"
    r"and\s+also)\b",
    re.IGNORECASE,
)


def _split_for_composition(msg: str) -> list[str]:
    """Split a message into ordered sub-requests using connectors.
    Returns at least one element (the original) if no connector found.
    """
    parts = _COMPOSE_CONNECTORS.split(msg)
    parts = [p.strip(" ,.;") for p in parts if p.strip()]
    return parts if len(parts) >= 2 else [msg]


def _compose_pipeline(template_results: list[dict]) -> str:
    """Stitch multiple templates' code into one pipeline script.

    Each template's function definitions are kept once; we deduplicate
    by function-name signature.  The example calls are chained: the
    output of step N becomes the input to step N+1.
    """
    if not template_results:
        return ""
    # 1. Collect all unique function definitions
    func_defs: list[str] = []
    seen_funcs: set[str] = set()
    for t in template_results:
        for block in _extract_function_blocks(t["code"]):
            sig = block.split("\n", 1)[0].strip()
            if sig not in seen_funcs:
                seen_funcs.add(sig)
                func_defs.append(block.rstrip())

    # 2. Build the pipeline section: extract the first function from each
    # template; chain results via a `value` variable.
    pipeline_lines = ["", "# Composed pipeline:"]
    first_func_name = _first_func_name(template_results[0]["code"])
    if first_func_name:
        # Use the customized example input from the first template
        sample_value = _extract_first_example_input(template_results[0]["code"])
        if sample_value is not None:
            pipeline_lines.append(f"value = {first_func_name}({sample_value!r})")
        else:
            pipeline_lines.append(f"value = {first_func_name}(...)")
        for t in template_results[1:]:
            fname = _first_func_name(t["code"])
            if fname:
                pipeline_lines.append(f"value = {fname}(value)")
        pipeline_lines.append("print(value)")
    return "\n\n".join(func_defs) + "\n" + "\n".join(pipeline_lines) + "\n"


def _extract_function_blocks(code: str) -> list[str]:
    """Return each top-level `def name(...):` block as its own string."""
    blocks = []
    cur = []
    in_block = False
    for line in code.split("\n"):
        if re.match(r"^def\s+\w+\s*\(", line):
            if cur and in_block:
                blocks.append("\n".join(cur))
            cur = [line]
            in_block = True
        elif in_block:
            if line.startswith(" ") or line == "":
                cur.append(line)
            else:
                # New top-level statement -> end of this function
                blocks.append("\n".join(cur))
                cur = []
                in_block = False
    if cur and in_block:
        blocks.append("\n".join(cur))
    return blocks


def _first_func_name(code: str) -> Optional[str]:
    m = re.search(r"^def\s+(\w+)\s*\(", code, re.MULTILINE)
    return m.group(1) if m else None


def _extract_first_example_input(code: str) -> Optional[object]:
    """Try to pull the literal argument of the first example call."""
    for line in code.split("\n"):
        m = re.search(r"\w+\((\[.*?\]|'[^']*'|\"[^\"]*\"|\d+)\)", line)
        if m:
            try:
                return ast.literal_eval(m.group(1))
            except Exception:
                return None
    return None


def try_compose(msg: str, run: bool = True) -> Optional[dict]:
    """If the message describes a multi-step task ("X then Y"),
    classify each sub-step, run them, and return a composed pipeline.

    Returns None when there's only one step (caller should fall back
    to try_write_code).
    """
    parts = _split_for_composition(msg)
    if len(parts) < 2:
        return None
    step_results = []
    for i, part in enumerate(parts):
        # First step needs full classification; later steps can be
        # permissive (no data needed — flows from previous step).
        tmpl = classify_intent(part, permissive=(i > 0))
        if tmpl is None:
            return None
        # Customize step 0 with literal data; later steps don't need
        # it (we'll patch their example calls in the composer).
        step_code = customize_template(tmpl, part) if i == 0 else tmpl.code
        step_results.append({"template": tmpl.name, "code": step_code,
                                "label": tmpl.example_label})
    if not step_results:
        return None

    composed_code = _compose_pipeline(step_results)
    if not composed_code:
        return None

    result = {
        "code":      composed_code,
        "label":     " → ".join(s["template"] for s in step_results),
        "template":  "composed",
        "steps":     [s["template"] for s in step_results],
        "ran":       False,
        "output":    "",
        "customized": True,
    }
    if run:
        ok, out = _safe_run(composed_code)
        result["ran"]    = ok
        result["output"] = out
    return result


def try_write_code(msg: str, run: bool = True) -> Optional[dict]:
    """Try to satisfy a code-writing request.  Returns:
        {
          'code':         "...",
          'label':        "...",
          'template':     "filter_duplicates",
          'ran':          True/False,
          'output':       "...",
        }
    or None when no template matched.
    """
    # Layer 3: multi-step composition first
    composed = try_compose(msg, run=run)
    if composed is not None:
        return composed

    tmpl = classify_intent(msg)
    if tmpl is None:
        return None
    # Layer 2: customize with the user's actual data if present
    code = customize_template(tmpl, msg)
    result = {
        "code":     code,
        "label":    tmpl.example_label,
        "template": tmpl.name,
        "ran":      False,
        "output":   "",
        "customized": code != tmpl.code,
        "refined":  False,
    }
    if run:
        ok, out = _safe_run(code)
        # ── Layer 5: refine on failure ──────────────────────────
        # If the customized code failed, try the canonical version.
        # Often the customization injected a bad/incompatible value.
        if not ok and result["customized"]:
            result["refine_log"] = (
                f"customized run failed: {out[:120]} "
                "— retrying with canonical example"
            )
            ok2, out2 = _safe_run(tmpl.code)
            if ok2:
                code = tmpl.code
                result["code"]   = code
                result["ran"]    = True
                result["output"] = out2
                result["refined"] = True
                result["customized"] = False
                return result
        result["ran"]    = ok
        result["output"] = out
    return result


def format_for_chat(payload: dict) -> str:
    """Format a code-writer payload as a chat-ready response."""
    if not payload:
        return ""
    parts = []
    if payload.get("label"):
        parts.append(f"Here's a snippet that does {payload['label']}:")
    parts.append("```python")
    parts.append(payload["code"].rstrip())
    parts.append("```")
    if payload.get("ran") and payload.get("output"):
        parts.append("Output:")
        parts.append("```")
        parts.append(payload["output"])
        parts.append("```")
    elif payload.get("output"):
        parts.append(f"(Note: {payload['output']})")
    return "\n".join(parts)


# ─── Self-test ───────────────────────────────────────────────────


def _self_test():
    cases = [
        "write me a function to remove duplicates",
        "give me python code to sort a list",
        "how do i compute fibonacci in python",
        "show me fizzbuzz",
        "write code to count word occurrences",
        "give me python code to read a file",
        "fetch a url in python",
        "factorial function in python",
        "find all primes up to n",
        "how do i flatten a nested list",
        # negatives
        "what is einstein known for",
        "hey",
    ]
    for msg in cases:
        result = try_write_code(msg, run=True)
        if result is None:
            print(f"  [-] {msg!r}: no template matched")
            continue
        ok_mark = "OK " if result["ran"] else "FAIL"
        print(f"  [{ok_mark}] {msg!r}")
        print(f"        template: {result['template']}")
        if result["ran"] and result["output"]:
            print(f"        out: {result['output'][:90]}")


if __name__ == "__main__":
    _self_test()
