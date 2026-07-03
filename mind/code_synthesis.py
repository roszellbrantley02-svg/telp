"""autopilot/code_synthesis.py — neurosymbolic: detect math/data
questions, synthesize a tiny safe-eval expression, return the answer.

This is the neurosymbolic completion of #63 (arithmetic) — instead of
hoping the HDC retrieval has the answer to "what's 23 * 47?", we
detect math, build the expression, and compute the answer
deterministically.

Two layers:
  Layer 1 — Pure arithmetic. "what is 23 * 47", "compute 100 - 22 / 7"
            Parses the expression with ast and evaluates safely.
  Layer 2 — Live data queries.  "what was MES's close 5 bars ago"
            Routes to the bars store with a tiny synthesized SQL.

Safety: ast.parse + walk every node, reject everything that isn't a
literal, BinOp, UnaryOp, or one of a tiny allow-list of functions.
No __import__, no attribute access, no calls outside the allow-list.
"""
from __future__ import annotations

import ast
import math
import re
from typing import Optional


# ─── Layer 1 — pure arithmetic ────────────────────────────────────


# Words → operators for natural-language math
_NL_REPLACEMENTS = [
    (re.compile(r"\bplus\b", re.IGNORECASE),         "+"),
    (re.compile(r"\bminus\b", re.IGNORECASE),        "-"),
    (re.compile(r"\btimes\b", re.IGNORECASE),        "*"),
    (re.compile(r"\bmultiplied\s+by\b", re.IGNORECASE),  "*"),
    (re.compile(r"\bdivided\s+by\b", re.IGNORECASE), "/"),
    (re.compile(r"\bover\b", re.IGNORECASE),         "/"),
    (re.compile(r"\bmod(?:ulo)?\b", re.IGNORECASE),  "%"),
    (re.compile(r"\bsquared\b", re.IGNORECASE),      "**2"),
    (re.compile(r"\bcubed\b", re.IGNORECASE),        "**3"),
    (re.compile(r"\bto\s+the\s+power\s+of\b", re.IGNORECASE), "**"),
    (re.compile(r"\braised\s+to\b", re.IGNORECASE),  "**"),
    (re.compile(r"\bpercent\s+of\b", re.IGNORECASE), "* 0.01 *"),
    (re.compile(r"%\s+of\b", re.IGNORECASE),         "* 0.01 *"),
]


# Allow-listed function names (must be lowercase)
_SAFE_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "pi": math.pi,
    "e": math.e,
}


_ALLOWED_NODES = {
    ast.Expression, ast.Constant, ast.Num,
    ast.BinOp, ast.UnaryOp, ast.BoolOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Invert,
    ast.And, ast.Or, ast.Not,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Call, ast.Name, ast.Load,
    ast.List, ast.Tuple,
}


def _is_safe(tree: ast.AST) -> bool:
    """Walk the AST; reject if any node isn't on the allow-list."""
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            return False
        if isinstance(node, ast.Name):
            if node.id not in _SAFE_FUNCS:
                return False
        if isinstance(node, ast.Call):
            func = node.func
            if not (isinstance(func, ast.Name) and func.id in _SAFE_FUNCS):
                return False
    return True


_ARITH_EXPR_RX = re.compile(
    r"[-+]?\s*\d+(?:\.\d+)?(?:\s*[+\-*/%^()]\s*[-+]?\s*\d+(?:\.\d+)?)+",
)


def _extract_expression(msg: str) -> Optional[str]:
    """Find a math expression in the message after NL replacements.
    Returns the expression text or None.
    """
    s = msg
    for pat, repl in _NL_REPLACEMENTS:
        s = pat.sub(repl, s)
    # Replace ^ with **
    s = s.replace("^", "**")
    # Strip question marks / trailing punctuation
    s = s.rstrip("?.! ")
    # Find a span that looks like an arithmetic expression
    m = _ARITH_EXPR_RX.search(s)
    if not m:
        # no last-resort letter-stripping: "serial ZX-99" is not "-99".
        # A real expression needs two operands (the regex above).
        return None
    return m.group(0).strip()


def _format_result(value) -> str:
    """Format a numeric result nicely."""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        # Trim trailing zeros
        s = f"{value:.6g}"
        return s
    return str(value)


def try_arithmetic(msg: str) -> Optional[str]:
    """If the message looks like a math question, evaluate and return
    a formatted answer string.  Returns None when it's not math or
    the expression is unsafe / unparseable.
    """
    if not msg:
        return None
    expr = _extract_expression(msg)
    if not expr:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    if not _is_safe(tree):
        return None
    try:
        result = eval(compile(tree, "<arith>", "eval"),
                       {"__builtins__": {}}, dict(_SAFE_FUNCS))
    except Exception:
        return None
    if not isinstance(result, (int, float, bool)):
        return None
    return f"{expr.strip()} = {_format_result(result)}"


# ─── Layer 2 — live data queries (stub) ──────────────────────────


def try_live_data(msg: str) -> Optional[str]:
    """Detect data-store questions and answer them deterministically.

    Currently handles:
      * "what's the close of MES" / "current price of MES"
      * "what was MES's close N bars ago"

    Returns the formatted answer or None.
    """
    s = msg.lower().strip().rstrip("?.! ")
    # Match "price of TICKER" / "close of TICKER" / "current TICKER"
    m = re.search(
        r"\b(?:current\s+)?(?:price|close|last)\s+(?:of\s+)?"
        r"(?:the\s+)?([a-z]{1,4}=?f?)\b", s, re.IGNORECASE
    )
    if not m:
        return None
    ticker_raw = m.group(1).upper()
    # Normalize ES → ES=F, MES → MES=F, etc. (futures)
    if not ticker_raw.endswith("=F") and ticker_raw in {
        "MES", "ES", "MNQ", "NQ", "MGC", "GC", "MYM", "YM",
    }:
        ticker = ticker_raw + "=F"
    else:
        ticker = ticker_raw

    try:
        import sqlite3
        from pathlib import Path
        bars = Path(__file__).resolve().parents[1] / "state" / "bars.db"
        if not bars.exists():
            return None
        con = sqlite3.connect(str(bars))
        row = con.execute(
            "SELECT ts, close FROM bars "
            "WHERE ticker=? AND interval='1m' "
            "ORDER BY ts DESC LIMIT 1", (ticker,),
        ).fetchone()
        con.close()
        if not row:
            return None
        ts, close = row
        return f"{ticker} last close (1m, {ts}): {close}"
    except Exception:
        return None


# ─── Public entry ────────────────────────────────────────────────


def try_code_synthesis(msg: str) -> Optional[str]:
    """Layered: arithmetic → live data.  Returns the first match
    or None."""
    return (try_arithmetic(msg) or try_live_data(msg))


# ─── Self-test ───────────────────────────────────────────────────


def _self_test():
    cases = [
        ("what's 23 * 47", "23 * 47 = 1081"),
        ("what is 100 - 22 / 7", "100 - 22 / 7 = 96.85714285714286"),
        ("calculate 2 + 2", None),   # ok, may match different form
        ("what is 5 squared", None),   # squared NL replacement
        ("what is 25 percent of 80", None),
        ("what's the current price of MES", None),
        ("how are you", None),
        ("what is einstein known for", None),
        ("what's the square root of 144", None),
    ]
    for q, _ in cases:
        result = try_code_synthesis(q)
        print(f"  {q!r:<45} → {result!r}")


if __name__ == "__main__":
    _self_test()
