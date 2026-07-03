"""autopilot/app_composer.py — generalizing planner for small CLI apps.

Breaks "build me a {THING} app" into:
  1. Entity:    what's the thing? (note, todo, trade, bookmark, ...)
  2. Schema:    what fields does it have?
  3. Storage:   SQLite table + CRUD operations
  4. CLI:       add/list/search/delete + entity-specific verbs
  5. Demo:     a runnable demo block at the bottom

This composer GENERALIZES — entity types Telp has never seen still
get a working app via the fallback `{text, created_at}` schema.

Architecture insight:  most "small data app" boils down to
ENTITY × CRUD.  By parameterizing both axes, one composer covers
the entire genre.
"""
from __future__ import annotations

import re
from typing import Optional


# ─── Entity schema library ───────────────────────────────────────


# Each entry: name → (display_singular, schema_fields, optional_verbs)
# schema_fields = list of (col_name, sql_type_with_default, is_text_searchable)
# optional_verbs = extra CLI verbs beyond add/list/search/delete

ENTITY_SCHEMAS: dict[str, dict] = {
    "note": {
        "singular": "note",
        "fields": [
            ("text",       "TEXT NOT NULL",         True),
            ("created_at", "TEXT NOT NULL",         False),
        ],
        "primary_text": "text",
        "demo_inputs": [
            "buy groceries",
            "finish reading the trade journal",
            "call mom on Sunday",
        ],
        "search_text": "trade",
    },
    "todo": {
        "singular": "todo",
        "fields": [
            ("text",       "TEXT NOT NULL",         True),
            ("done",       "INTEGER DEFAULT 0",     False),
            ("created_at", "TEXT NOT NULL",         False),
        ],
        "primary_text": "text",
        "extra_verbs": ["done", "undone"],   # toggle the done bit
        "demo_inputs": [
            "buy groceries",
            "ship the trading bot",
            "call mom on Sunday",
        ],
        "search_text": "trading",
    },
    "trade": {
        "singular": "trade",
        "fields": [
            ("ticker",  "TEXT NOT NULL",            True),
            ("side",    "TEXT NOT NULL",            False),
            ("qty",     "INTEGER NOT NULL",         False),
            ("entry",   "REAL NOT NULL",            False),
            ("exit",    "REAL",                     False),
            ("pnl",     "REAL DEFAULT 0",           False),
            ("ts",      "TEXT NOT NULL",            False),
        ],
        "primary_text": "ticker",
        "extra_verbs": ["close", "summary"],   # close <id> <exit_px>; summary = total pnl
        "demo_inputs": [
            "MES=F sell 1 7500.00 7480.00",
            "MNQ=F buy 1 29500.00 29560.00",
            "MGC=F sell 1 2050.00 2045.00",
        ],
        "search_text": "MNQ",
    },
    "bookmark": {
        "singular": "bookmark",
        "fields": [
            ("url",        "TEXT NOT NULL",         False),
            ("title",      "TEXT NOT NULL",         True),
            ("tags",       "TEXT",                  True),
            ("created_at", "TEXT NOT NULL",         False),
        ],
        "primary_text": "title",
        "demo_inputs": [
            "https://example.com  Example Domain  reference,demo",
            "https://news.example.com  Latest News  news,daily",
            "https://docs.python.org  Python Docs  python,reference",
        ],
        "search_text": "python",
    },
    "habit": {
        "singular": "habit",
        "fields": [
            ("name",        "TEXT NOT NULL",        True),
            ("streak",      "INTEGER DEFAULT 0",    False),
            ("last_logged", "TEXT",                 False),
        ],
        "primary_text": "name",
        "extra_verbs": ["log"],   # log <name> → bump streak
        "demo_inputs": [
            "morning walk",
            "read 20 pages",
            "no sugar",
        ],
        "search_text": "walk",
    },
    "book": {
        "singular": "book",
        "fields": [
            ("title",      "TEXT NOT NULL",         True),
            ("author",     "TEXT",                  True),
            ("status",     "TEXT DEFAULT 'to_read'", False),
            ("rating",     "INTEGER",               False),
            ("created_at", "TEXT NOT NULL",         False),
        ],
        "primary_text": "title",
        "demo_inputs": [
            "The Pragmatic Programmer  Hunt & Thomas",
            "Reminiscences of a Stock Operator  Edwin Lefevre",
            "The Mind of Wall Street  Leon Levy",
        ],
        "search_text": "stock",
    },
    "expense": {
        "singular": "expense",
        "fields": [
            ("description", "TEXT NOT NULL",        True),
            ("amount",      "REAL NOT NULL",        False),
            ("category",    "TEXT",                 True),
            ("created_at",  "TEXT NOT NULL",        False),
        ],
        "primary_text": "description",
        "extra_verbs": ["total"],   # total spent
        "demo_inputs": [
            "coffee  4.50  food",
            "internet bill  79.99  utilities",
            "lunch  12.00  food",
        ],
        "search_text": "food",
    },
    "contact": {
        "singular": "contact",
        "fields": [
            ("name",       "TEXT NOT NULL",         True),
            ("phone",      "TEXT",                  False),
            ("email",      "TEXT",                  True),
            ("notes",      "TEXT",                  True),
        ],
        "primary_text": "name",
        "demo_inputs": [
            "Alice Johnson  555-0101  alice@example.com  team lead",
            "Bob Smith  555-0102  bob@example.com  friend",
            "Carol Davis  555-0103  carol@example.com  family",
        ],
        "search_text": "alice",
    },
    "journal": {
        "singular": "entry",
        "fields": [
            ("title",      "TEXT",                  True),
            ("body",       "TEXT NOT NULL",         True),
            ("mood",       "TEXT",                  False),
            ("created_at", "TEXT NOT NULL",         False),
        ],
        "primary_text": "body",
        "demo_inputs": [
            "good day  shipped the composer  positive",
            "tough morning  market was choppy  meh",
            "evening reflection  family dinner was great  positive",
        ],
        "search_text": "market",
    },
    "log": {
        "singular": "entry",
        "fields": [
            ("level",      "TEXT NOT NULL",         False),
            ("message",    "TEXT NOT NULL",         True),
            ("ts",         "TEXT NOT NULL",         False),
        ],
        "primary_text": "message",
        "demo_inputs": [
            "info  startup",
            "warn  cache miss",
            "error  connection refused",
        ],
        "search_text": "cache",
    },
}


# ─── Intent detection ────────────────────────────────────────────


_BUILD_APP_RX = re.compile(
    r"\b(?:build|make|create|write|generate)\s+(?:me\s+)?"
    r"(?:a\s+|an\s+|some\s+)?"
    r"(\w+(?:[- ]\w+)?)\s+"
    r"(?:app|application|cli|tracker|tracker|manager|journal|list|"
    r"tool|program|script|system|store|book|database)",
    re.IGNORECASE,
)


# Aliases — phrases users say that map to a canonical entity name.
_ENTITY_ALIASES = {
    "task":          "todo",
    "tasks":         "todo",
    "todos":         "todo",
    "todo list":     "todo",
    "to-do":         "todo",
    "notes":         "note",
    "memo":          "note",
    "memos":         "note",
    "trades":        "trade",
    "trading":       "trade",
    "bookmarks":     "bookmark",
    "link":          "bookmark",
    "links":         "bookmark",
    "habits":        "habit",
    "books":         "book",
    "reading":       "book",
    "reading list":  "book",
    "library":       "book",
    "recipe":        "note",      # generic-ish — treat as named items
    "recipes":       "note",
    "meditation":    "habit",     # nearest fit
    "workout":       "habit",
    "workouts":      "habit",
    "exercise":      "habit",
    "music":         "note",
    "movie":         "note",
    "movies":        "note",
    "expenses":      "expense",
    "spending":      "expense",
    "budget":        "expense",
    "contacts":      "contact",
    "address book":  "contact",
    "addressbook":   "contact",
    "diary":         "journal",
    "log":           "log",
    "logs":          "log",
    "logger":        "log",
    "logging":       "log",
}


def detect_app_intent(msg: str) -> Optional[dict]:
    """Detect a 'build me an X app' intent.  Returns:
      {entity: 'note', alias: 'notes', original: 'note taking', ...}
    or None if no app-build intent.
    """
    if not msg:
        return None
    m = _BUILD_APP_RX.search(msg)
    if not m:
        return None
    raw_entity = m.group(1).lower().strip()
    # Normalize: strip trailing "taking" / "tracking" filler
    raw_clean = re.sub(r"\s+(taking|tracking|management|keeping)$",
                              "", raw_entity)
    # Look up alias / direct hit
    entity = _ENTITY_ALIASES.get(raw_clean)
    if entity is None:
        # Try the bare singular
        entity = _ENTITY_ALIASES.get(raw_clean.rstrip("s"))
    if entity is None:
        # Direct hit on schema name (singular)
        if raw_clean in ENTITY_SCHEMAS:
            entity = raw_clean
        elif raw_clean.rstrip("s") in ENTITY_SCHEMAS:
            entity = raw_clean.rstrip("s")
    return {
        "entity":   entity,                # may be None → generic fallback
        "raw":      raw_entity,
        "original": raw_clean,
    }


# ─── Code generation ─────────────────────────────────────────────


def _generic_schema(entity_name: str) -> dict:
    """Build a generic schema for an entity Telp doesn't have a
    canonical template for.  Uses {text, created_at} — works for any
    'collection of things with a label' app."""
    return {
        "singular": entity_name,
        "fields": [
            ("text",       "TEXT NOT NULL",         True),
            ("created_at", "TEXT NOT NULL",         False),
        ],
        "primary_text": "text",
        "demo_inputs": [
            f"first {entity_name}",
            f"second {entity_name}",
            f"third {entity_name}",
        ],
        "search_text": entity_name,
    }


def compose_app(entity_name: str, schema: dict = None) -> str:
    """Generate a complete, runnable CLI app file.

    Args:
      entity_name:  e.g., 'todo', 'note', 'trade'.  Used as table name +
                    file name in the generated code.
      schema:       Optional override.  If None, look up ENTITY_SCHEMAS
                    or fall back to generic.
    """
    if schema is None:
        schema = ENTITY_SCHEMAS.get(entity_name) or _generic_schema(entity_name)
    singular = schema["singular"]
    fields   = schema["fields"]
    primary  = schema["primary_text"]
    extra_verbs = schema.get("extra_verbs", [])
    demo_inputs = schema.get("demo_inputs", [])
    search_text = schema.get("search_text", "the")

    table = singular + "s"
    field_cols = [name for name, _, _ in fields]
    schema_sql = ",\n    ".join(f"{n} {t}" for n, t, _ in fields)
    text_search_cols = [n for n, _, searchable in fields if searchable]

    # Identify autogenerated columns (don't need to be passed to add())
    # created_at / ts are auto-filled with now.
    auto_cols = {"created_at", "ts"}
    user_cols = [n for n in field_cols if n not in auto_cols]

    # Build the field signature for add() and the SQL placeholders
    add_args = ", ".join(user_cols)
    insert_cols = ", ".join(field_cols)
    insert_ph = ", ".join(["?"] * len(field_cols))

    # Build the docstring usage section
    usage_lines = [
        f"    python {singular}s.py add <{primary}> [other fields...]",
        f"    python {singular}s.py list",
        f"    python {singular}s.py search <query>",
        f"    python {singular}s.py delete <id>",
    ]
    for verb in extra_verbs:
        if verb in ("done", "undone"):
            usage_lines.append(f"    python {singular}s.py {verb} <id>")
        elif verb == "close":
            usage_lines.append(
                f"    python {singular}s.py close <id> <exit_price>"
            )
        elif verb == "summary":
            usage_lines.append(f"    python {singular}s.py summary")
        elif verb == "log":
            usage_lines.append(f"    python {singular}s.py log <name>")
        elif verb == "total":
            usage_lines.append(f"    python {singular}s.py total")

    # Build the demo block — adds the demo inputs then runs list + search +
    # delete to show the full lifecycle.
    demo_calls = []
    for idx, demo in enumerate(demo_inputs, 1):
        # Split on double-space; map to add() positional args
        args = re.split(r"\s{2,}", demo)
        # Pad with empty strings if fewer args than expected
        while len(args) < len(user_cols):
            args.append("")
        # Truncate
        args = args[:len(user_cols)]
        # Build the call: add(arg1, arg2, ...) with type coercion
        coerced = []
        for col, val in zip(user_cols, args):
            # Find the field type
            col_type = next((t for n, t, _ in fields if n == col), "TEXT")
            t_upper = col_type.upper()
            if "INTEGER" in t_upper:
                coerced.append(str(int(val)) if val else "0")
            elif "REAL" in t_upper or "FLOAT" in t_upper:
                coerced.append(f"{float(val):.4f}" if val else "0.0")
            else:
                coerced.append(repr(val))
        demo_calls.append(f"add({', '.join(coerced)})")

    # Compile everything into a final program
    doc = (
        f'"""A CLI {singular} app — SQLite-backed.\n\n'
        f"Usage:\n"
        + "\n".join(usage_lines) +
        '\n"""\n'
    )

    src = []
    src.append(doc)
    src.append("import sqlite3")
    src.append("import sys")
    src.append("from datetime import datetime, timezone")
    src.append("from pathlib import Path")
    src.append("")
    src.append(f"DB = Path('{singular}s.db')")
    src.append("")
    src.append("_SCHEMA = '''")
    src.append(f"CREATE TABLE IF NOT EXISTS {table} (")
    src.append("    id INTEGER PRIMARY KEY,")
    src.append(f"    {schema_sql}")
    src.append(");")
    src.append("'''")
    src.append("")
    src.append("def _now():")
    src.append("    return datetime.now(timezone.utc).isoformat()")
    src.append("")
    src.append("def _con():")
    src.append("    con = sqlite3.connect(DB)")
    src.append("    con.executescript(_SCHEMA)")
    src.append("    return con")
    src.append("")

    # add() — takes user_cols (autogen cols filled internally)
    src.append(f"def add({add_args}):")
    for col, _, _ in fields:
        if col in auto_cols:
            continue
    # Build the actual INSERT call values
    insert_values = []
    for col in field_cols:
        if col in auto_cols:
            insert_values.append("_now()")
        else:
            insert_values.append(col)
    src.append(f"    with _con() as c:")
    src.append(f"        cur = c.execute(")
    src.append(f"            'INSERT INTO {table} ({insert_cols}) "
                  f"VALUES ({insert_ph})',")
    src.append(f"            ({', '.join(insert_values)},),")
    src.append(f"        )")
    src.append(f"        print(f'added #{{cur.lastrowid}}')")
    src.append("")

    # list_x()
    list_fn_name = f"list_{table}"
    src.append(f"def {list_fn_name}():")
    src.append(f"    with _con() as c:")
    src.append(f"        rows = c.execute('SELECT * FROM {table} "
                  f"ORDER BY id').fetchall()")
    src.append(f"    if not rows:")
    src.append(f"        print('(no {table} yet)')")
    src.append(f"        return")
    src.append(f"    for r in rows:")
    # Format each row
    src.append(f"        print('#{{:>3}}'.format(r[0]) + '  ' + "
                  f"'  '.join(str(x) for x in r[1:]))")
    src.append("")

    # search() — over all text-searchable columns
    src.append(f"def search(query):")
    if text_search_cols:
        like_clauses = " OR ".join(f"{c} LIKE ?" for c in text_search_cols)
        wildcards = "(" + ", ".join([f"f'%{{query}}%'"] *
                                              len(text_search_cols)) + ",)"
        src.append(f"    with _con() as c:")
        src.append(f"        rows = c.execute(")
        src.append(f"            'SELECT * FROM {table} "
                      f"WHERE {like_clauses} ORDER BY id',")
        src.append(f"            tuple([f'%{{query}}%'] * "
                      f"{len(text_search_cols)}),")
        src.append(f"        ).fetchall()")
        src.append(f"    if not rows:")
        src.append(f"        print(f'(no {table} matching {{query!r}})')")
        src.append(f"        return")
        src.append(f"    for r in rows:")
        src.append(f"        print('#{{:>3}}'.format(r[0]) + '  ' + "
                      f"'  '.join(str(x) for x in r[1:]))")
    else:
        src.append(f"    print('(this app has no text-searchable fields)')")
    src.append("")

    # delete()
    src.append(f"def delete({singular}_id):")
    src.append(f"    with _con() as c:")
    src.append(f"        cur = c.execute('DELETE FROM {table} "
                  f"WHERE id = ?', (int({singular}_id),))")
    src.append(f"        if cur.rowcount:")
    src.append(f"            print(f'deleted #{{{singular}_id}}')")
    src.append(f"        else:")
    src.append(f"            print(f'no {singular} #{{{singular}_id}}')")
    src.append("")

    # Extra verbs
    if "done" in extra_verbs:
        src.append(f"def mark_done({singular}_id):")
        src.append(f"    with _con() as c:")
        src.append(f"        c.execute('UPDATE {table} SET done = 1 "
                      f"WHERE id = ?', (int({singular}_id),))")
        src.append(f"        print(f'marked #{{{singular}_id}} as done')")
        src.append("")
        src.append(f"def mark_undone({singular}_id):")
        src.append(f"    with _con() as c:")
        src.append(f"        c.execute('UPDATE {table} SET done = 0 "
                      f"WHERE id = ?', (int({singular}_id),))")
        src.append(f"        print(f'marked #{{{singular}_id}} as undone')")
        src.append("")
    if "close" in extra_verbs:
        # Trade-specific: close a trade with an exit price, compute PnL
        src.append(f"def close({singular}_id, exit_price):")
        src.append(f"    exit_price = float(exit_price)")
        src.append(f"    with _con() as c:")
        src.append(f"        row = c.execute('SELECT side, qty, entry "
                      f"FROM {table} WHERE id = ?',")
        src.append(f"                          (int({singular}_id),)).fetchone()")
        src.append(f"        if not row:")
        src.append(f"            print(f'no {singular} #{{{singular}_id}}')")
        src.append(f"            return")
        src.append(f"        side, qty, entry = row")
        src.append(f"        sign = 1 if side == 'buy' else -1")
        src.append(f"        pnl = (exit_price - entry) * qty * sign")
        src.append(f"        c.execute('UPDATE {table} SET exit = ?, "
                      f"pnl = ? WHERE id = ?',")
        src.append(f"                    (exit_price, pnl, int({singular}_id)))")
        src.append(f"        print(f'closed #{{{singular}_id}} @ "
                      f"{{exit_price}}  pnl=${{pnl:+.2f}}')")
        src.append("")
    if "summary" in extra_verbs:
        src.append(f"def summary():")
        src.append(f"    with _con() as c:")
        src.append(f"        rows = c.execute('SELECT pnl FROM {table} "
                      f"WHERE pnl != 0').fetchall()")
        src.append(f"    if not rows:")
        src.append(f"        print('(no closed {table})')")
        src.append(f"        return")
        src.append(f"    total = sum(r[0] for r in rows)")
        src.append(f"    wins = sum(1 for r in rows if r[0] > 0)")
        src.append(f"    losses = sum(1 for r in rows if r[0] < 0)")
        src.append(f"    print(f'{{len(rows)}} closed  "
                      f"wins={{wins}}  losses={{losses}}  "
                      f"total=${{total:+.2f}}')")
        src.append("")
    if "log" in extra_verbs:
        src.append(f"def log_habit(name):")
        src.append(f"    with _con() as c:")
        src.append(f"        row = c.execute('SELECT streak, last_logged "
                      f"FROM {table} WHERE name = ?',")
        src.append(f"                          (name,)).fetchone()")
        src.append(f"        if not row:")
        src.append(f"            print(f'no habit named {{name!r}}')")
        src.append(f"            return")
        src.append(f"        streak, _ = row")
        src.append(f"        c.execute('UPDATE {table} SET streak = ?, "
                      f"last_logged = ? WHERE name = ?',")
        src.append(f"                    ((streak or 0) + 1, _now(), name))")
        src.append(f"        print(f'{{name!r}} streak -> {{(streak or 0) + 1}}')")
        src.append("")
    if "total" in extra_verbs:
        src.append(f"def total():")
        src.append(f"    with _con() as c:")
        src.append(f"        rows = c.execute('SELECT amount FROM {table}'"
                      f").fetchall()")
        src.append(f"    print(f'total: ${{sum(r[0] for r in rows):.2f}} "
                      f"across {{len(rows)}} entries')")
        src.append("")

    # main()
    src.append("def main():")
    src.append("    if len(sys.argv) < 2:")
    src.append("        print(__doc__.strip())")
    src.append("        return")
    src.append("    cmd = sys.argv[1].lower()")
    src.append("    args = sys.argv[2:]")
    src.append(f"    if   cmd == 'add'    and len(args) >= {len(user_cols)}: "
                  f"add(*args[:{len(user_cols)}])")
    src.append(f"    elif cmd == 'list':  {list_fn_name}()")
    src.append( "    elif cmd == 'search' and args: search(' '.join(args))")
    src.append( "    elif cmd == 'delete' and args: delete(args[0])")
    if "done" in extra_verbs:
        src.append( "    elif cmd == 'done' and args: mark_done(args[0])")
        src.append( "    elif cmd == 'undone' and args: mark_undone(args[0])")
    if "close" in extra_verbs:
        src.append( "    elif cmd == 'close' and len(args) >= 2: close(args[0], args[1])")
    if "summary" in extra_verbs:
        src.append( "    elif cmd == 'summary': summary()")
    if "log" in extra_verbs:
        src.append( "    elif cmd == 'log' and args: log_habit(' '.join(args))")
    if "total" in extra_verbs:
        src.append( "    elif cmd == 'total': total()")
    src.append("    else:")
    src.append("        print(__doc__.strip())")
    src.append("")
    src.append("if __name__ == '__main__':")
    src.append("    main()")
    src.append("")

    # Demo block — actually runs the app so output is visible
    src.append("# === Live demo (executed in this run) ===")
    src.append("import os, tempfile")
    src.append(f"DB = Path(tempfile.gettempdir()) / 'composer_demo_{singular}.db'")
    src.append("if DB.exists(): DB.unlink()")
    for call in demo_calls:
        src.append(call)
    src.append("print('--- list ---')")
    src.append(f"{list_fn_name}()")
    if search_text:
        src.append(f'print("--- search {search_text!r} ---")')
        src.append(f"search({search_text!r})")
    src.append("print('--- delete #2 ---')")
    src.append("delete(2)")
    src.append("print('--- after delete ---')")
    src.append(f"{list_fn_name}()")
    # Run trade-specific demo extras
    if "close" in extra_verbs:
        # Trades example already includes exit prices; PnL is computed
        # but the demo data feeds entry+exit at add time so close would
        # double-close. We'll skip and call summary instead.
        src.append("print('--- summary ---')")
        src.append("summary()")
    if "log" in extra_verbs:
        src.append("print('--- log a habit ---')")
        first_demo = demo_inputs[0].split("  ")[0] if demo_inputs else "x"
        src.append(f"log_habit({first_demo!r})")
        src.append("print('--- after log ---')")
        src.append(f"{list_fn_name}()")
    if "total" in extra_verbs:
        src.append("print('--- total ---')")
        src.append("total()")

    return "\n".join(src) + "\n"


# ─── Public entry ────────────────────────────────────────────────


def try_compose_app(msg: str, run: bool = True) -> Optional[dict]:
    """Detect an app-build request and synthesize a complete CLI app.

    Returns a result dict matching try_write_code's contract, or None
    when no app-build intent is detected.
    """
    intent = detect_app_intent(msg)
    if intent is None:
        return None

    entity = intent["entity"] or intent["original"] or "item"
    # Sanitize entity name to be a valid Python identifier base
    entity = re.sub(r"[^a-zA-Z0-9_]", "_", entity).lower().strip("_")
    if not entity:
        return None

    # Look up schema or build generic
    schema = ENTITY_SCHEMAS.get(entity)
    if schema is None:
        # Try aliases one more time
        schema = ENTITY_SCHEMAS.get(_ENTITY_ALIASES.get(entity, ""))
    if schema is None:
        schema = _generic_schema(entity)

    code = compose_app(entity, schema)
    result = {
        "code":     code,
        "label":    f"a working CLI {entity} app — add/list/search/delete"
                       + (f" + {', '.join(schema.get('extra_verbs', []))}"
                              if schema.get("extra_verbs") else ""),
        "template": f"composer:{entity}",
        "ran":      False,
        "output":   "",
        "customized": True,
        "entity":   entity,
    }
    if run:
        # Reuse the same _safe_run as code_writer (subprocess execution)
        from mind.code_writer import _safe_run
        ok, out = _safe_run(code, timeout=10.0)
        result["ran"]    = ok
        result["output"] = out
    return result


def _self_test():
    cases = [
        "build me a note taking app",
        "make a todo list app",
        "create me a trade journal",
        "build me a bookmark manager",
        "make me an expense tracker",
        "build a contact tracker",
        "build me a reading list app",
        "build me a habit tracker",
        "build me a meditation tracker",     # unknown entity → generic
        "build me a recipe app",              # unknown entity → generic
        "hey there",                          # no intent
    ]
    for msg in cases:
        intent = detect_app_intent(msg)
        if intent is None:
            print(f"  [-] {msg!r}: no intent")
            continue
        entity = intent.get("entity") or "(generic)"
        print(f"  [{entity:<10}] {msg!r}  (raw: {intent['original']!r})")


if __name__ == "__main__":
    _self_test()
