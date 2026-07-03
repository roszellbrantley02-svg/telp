"""
lattice/ingest_code.py - feed the agent its own source code.

Walks .py files, parses each with ast, and emits one record per
top-level symbol (class, function, module).  For each, captures:
  - kind:        "module" / "class" / "function" / "method"
  - name:        the symbol name
  - file:        the file it's defined in
  - docstring:   the first paragraph of its docstring (if any)
  - imports:     names this module pulls in (module-level only)

Each record becomes a sentence we can add to the agent's lattice,
plus structured claims like (X, defined_in, F), (F, imports, M),
(C, has_method, M).
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CodeRecord:
    kind:      str
    name:      str                      # function or class name
    file:      str                      # relative path
    line:      int
    docstring: str = ""
    signature: str = ""
    parent:    str | None = None        # class name if this is a method
    imports:   list[str] = field(default_factory=list)


def _first_paragraph(s: str) -> str:
    """Pull just the first paragraph of a docstring (most informative)."""
    if not s:
        return ""
    para = s.strip().split("\n\n", 1)[0]
    # Collapse internal whitespace.
    return re.sub(r"\s+", " ", para).strip()


def _format_args(args: ast.arguments) -> str:
    parts = [a.arg for a in args.args]
    if args.vararg:
        parts.append("*" + args.vararg.arg)
    parts.extend(k.arg for k in args.kwonlyargs)
    if args.kwarg:
        parts.append("**" + args.kwarg.arg)
    return ", ".join(parts)


def parse_file(path: Path, project_root: Path) -> list[CodeRecord]:
    """Parse a single .py file into a list of CodeRecords."""
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []

    rel = str(path.relative_to(project_root)).replace("\\", "/")
    out: list[CodeRecord] = []

    # Module-level record (docstring + module-level imports)
    mod_doc = ast.get_docstring(tree) or ""
    mod_imports: list[str] = []
    for n in ast.iter_child_nodes(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                mod_imports.append(a.name)
        elif isinstance(n, ast.ImportFrom) and n.module:
            mod_imports.append(n.module)

    out.append(CodeRecord(
        kind="module", name=path.stem, file=rel, line=1,
        docstring=_first_paragraph(mod_doc), imports=mod_imports,
    ))

    # Top-level functions and classes
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) or isinstance(n, ast.AsyncFunctionDef):
            out.append(CodeRecord(
                kind="function", name=n.name, file=rel, line=n.lineno,
                docstring=_first_paragraph(ast.get_docstring(n) or ""),
                signature=f"{n.name}({_format_args(n.args)})",
            ))
        elif isinstance(n, ast.ClassDef):
            out.append(CodeRecord(
                kind="class", name=n.name, file=rel, line=n.lineno,
                docstring=_first_paragraph(ast.get_docstring(n) or ""),
            ))
            # Methods inside the class
            for sub in n.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append(CodeRecord(
                        kind="method", name=sub.name, file=rel, line=sub.lineno,
                        docstring=_first_paragraph(ast.get_docstring(sub) or ""),
                        signature=f"{sub.name}({_format_args(sub.args)})",
                        parent=n.name,
                    ))
    return out


def walk_project(root: Path, subdirs: list[str] | None = None
                   ) -> list[CodeRecord]:
    """Walk all .py files in `root`/<subdirs>, parse each, return
    a flat list of CodeRecords."""
    if subdirs is None:
        subdirs = [""]
    out: list[CodeRecord] = []
    seen: set[Path] = set()
    for sub in subdirs:
        base = root / sub if sub else root
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            if p in seen:
                continue
            seen.add(p)
            out.extend(parse_file(p, root))
    return out


def record_to_sentence(r: CodeRecord) -> str:
    """Turn a CodeRecord into ONE English sentence suitable for
    storing in the lattice."""
    if r.kind == "module":
        if r.docstring:
            return f"Module {r.name} in {r.file}: {r.docstring}"
        return f"Module {r.name} is defined in {r.file}."
    if r.kind == "class":
        if r.docstring:
            return f"Class {r.name} in {r.file}: {r.docstring}"
        return f"Class {r.name} is defined in {r.file} at line {r.line}."
    if r.kind == "function":
        sig = r.signature or r.name
        if r.docstring:
            return f"Function {sig} in {r.file}: {r.docstring}"
        return f"Function {sig} is defined in {r.file} at line {r.line}."
    if r.kind == "method":
        owner = r.parent or "?"
        sig = r.signature or r.name
        if r.docstring:
            return (f"Method {owner}.{sig} in {r.file}: {r.docstring}")
        return (f"Method {owner}.{sig} is defined in {r.file}.")
    return f"{r.kind} {r.name} in {r.file}"


def record_to_claims(r: CodeRecord) -> list[tuple[str, str, str]]:
    """Turn a CodeRecord into structured (S, V, O) claims."""
    claims = []
    if r.kind == "module":
        for imp in r.imports:
            claims.append((r.name, "imports", imp))
    elif r.kind in ("function", "class"):
        claims.append((r.name, "defined_in", r.file))
    elif r.kind == "method":
        if r.parent:
            claims.append((r.parent, "has_method", r.name))
        claims.append((r.name, "defined_in", r.file))
    return claims


def ingest_into_agent(agent, project_root: Path,
                         subdirs: list[str] | None = None) -> dict:
    """Read code from `project_root`/<subdirs> and feed records into
    the agent's lattice + structured-QA store.

    Returns a stats dict.
    """
    records = walk_project(project_root, subdirs)
    n_sent = 0; n_claim = 0
    for r in records:
        sent = record_to_sentence(r)
        agent.lattice.add(sent, source=f"code:{r.file}",
                            symbol=r.name, kind=r.kind, line=r.line)
        n_sent += 1
        for s, v, o in record_to_claims(r):
            agent.structured.claim_text.append(sent)
            agent.structured.claim_source.append(f"code:{r.file}")
            agent.structured.claim_triple.append((s, v, o))
            n_claim += 1
        # Also feed the encoder so technical vocab is in scope.
        agent.encoder.add_sentence(sent)
    agent.structured._dirty = True
    return {"records": len(records), "sentences": n_sent, "claims": n_claim}


if __name__ == "__main__":
    import sys as _sys
    root = Path(_sys.argv[1]) if len(_sys.argv) > 1 else Path.cwd()
    recs = walk_project(root, ["lattice"])
    print(f"Found {len(recs)} code records under {root}/lattice")
    for r in recs[:10]:
        print(f"  [{r.kind:8s}] {r.name:25s} {r.file}:{r.line}")
        if r.docstring:
            print(f"             doc: {r.docstring[:80]}")
