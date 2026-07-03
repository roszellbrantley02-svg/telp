"""
lattice/arithmetic_qa.py - safe arithmetic evaluation.

Pure-python AST walker.  Supports +, -, *, /, //, %, **, parens, and
unary - on integer + float literals.  Refuses anything else.

NO use of `eval()` or `compile(..., mode='eval')`.  We parse via
ast.parse(..., mode='eval') (which produces an AST tree but does NOT
execute), then walk the tree ourselves.  Defense-in-depth: even if a
string slips past the regex, the walker rejects function calls,
attribute access, names, etc.
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass


@dataclass
class ArithResult:
    expression: str
    value:      float | int
    explain:    str


_OPS = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod:      operator.mod,
    ast.Pow:      operator.pow,
    ast.USub:     operator.neg,
    ast.UAdd:     operator.pos,
}


def _walk(node):
    if isinstance(node, ast.Expression):
        return _walk(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"non-numeric constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return op(_walk(node.left), _walk(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary: {type(node.op).__name__}")
        return op(_walk(node.operand))
    raise ValueError(f"unsupported node: {type(node).__name__}")


def safe_eval(expr: str) -> float | int:
    """Evaluate `expr` as a pure arithmetic expression.  Raises
    ValueError on anything unsupported."""
    tree = ast.parse(expr, mode="eval")
    return _walk(tree)


# Detection: pull a math expression out of a natural-language query.
# Pre-clean: turn words into ops.
_WORD_TO_OP = [
    (re.compile(r"\bplus\b",       re.I), "+"),
    (re.compile(r"\bminus\b",      re.I), "-"),
    (re.compile(r"\btimes\b",      re.I), "*"),
    (re.compile(r"\bmultiplied\s+by\b", re.I), "*"),
    (re.compile(r"\bdivided\s+by\b",    re.I), "/"),
    (re.compile(r"\bto\s+the\s+power\s+of\b",    re.I), "**"),
    (re.compile(r"\bmodulo\b",     re.I), "%"),
    (re.compile(r"\b(?:what|how\s+much)\s+is\b",  re.I), ""),
    (re.compile(r"\bcalculate\b",  re.I), ""),
    (re.compile(r"\bcompute\b",    re.I), ""),
    (re.compile(r"\bequals?\b",    re.I), ""),
    (re.compile(r"=",                  ), ""),
    (re.compile(r"\?\s*$"              ), ""),
]


# After word-replacement, what's left should look like a math expression.
_EXPR_RE = re.compile(r"^\s*[\d\s\+\-\*\/\(\)\.\%]+(?:\*\*[\d\s\+\-\*\/\(\)\.\%]+)?\s*$")
_HAS_OP_RE = re.compile(r"[\+\-\*\/\%]|\*\*")
_HAS_DIGIT_RE = re.compile(r"\d")


def detect_and_eval(query: str) -> ArithResult | None:
    """If `query` contains a clean arithmetic expression, evaluate
    and return the result.  Otherwise None."""
    s = query
    for pat, repl in _WORD_TO_OP:
        s = pat.sub(repl, s)
    s = s.strip()
    if not s or not _HAS_DIGIT_RE.search(s) or not _HAS_OP_RE.search(s):
        return None
    if not _EXPR_RE.match(s):
        return None
    try:
        value = safe_eval(s)
    except (ValueError, ZeroDivisionError, SyntaxError) as e:
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return ArithResult(
        expression=s,
        value=value,
        explain=f"{s.strip()} = {value}",
    )
