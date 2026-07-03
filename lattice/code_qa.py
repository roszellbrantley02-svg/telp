"""
lattice/code_qa.py - questions about source code.

Code-specific question shapes that route directly to the agent's
lattice/structured-claim store (after ingest_self() has run):

    "What does function X do?"     -> retrieve X's docstring
    "Where is X defined?"          -> (X, defined_in, file)
    "What does X import?"          -> (X, imports, *)
    "What methods does X have?"    -> (X, has_method, *)
    "What is X?" (a code symbol)   -> module/class/function docstring
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class CodeQAResult:
    kind:    str
    target:  str
    answer:  str
    extras:  list[str]


_NAME_PAT = r"(?P<name>[A-Za-z_][A-Za-z0-9_\.]*)"


_PATTERNS = [
    ("docstring", re.compile(
        r"\bwhat\s+(?:does|is|do)\s+(?:the\s+)?(?:function\s+|method\s+|class\s+|module\s+)?"
        + _NAME_PAT + r"\s+(?:do|for)\??$", re.I)),
    ("docstring", re.compile(
        r"\bwhat\s+is\s+(?:the\s+purpose\s+of\s+)?(?:the\s+)?"
        + _NAME_PAT + r"(?:\s+module|\s+function|\s+class)?\??$", re.I)),
    ("defined_in", re.compile(
        r"\bwhere\s+is\s+(?:the\s+)?(?:function\s+|method\s+|class\s+)?"
        + _NAME_PAT + r"\s+defined\??$", re.I)),
    ("imports", re.compile(
        r"\bwhat\s+(?:modules\s+)?does\s+(?:the\s+)?"
        + _NAME_PAT + r"\s+import\??$", re.I)),
    ("methods", re.compile(
        r"\bwhat\s+methods\s+does\s+(?:the\s+)?(?:class\s+)?"
        + _NAME_PAT + r"\s+have\??$", re.I)),
]


def _normalise_name(name: str) -> str:
    """Strip leading 'lattice.' and trailing punctuation."""
    name = name.strip().rstrip(".?")
    # X.Y.Z -> Z   when looking up class methods
    return name


class CodeQA:
    def __init__(self, lattice, structured):
        self.lattice = lattice
        self.structured = structured

    def answer(self, query: str) -> CodeQAResult | None:
        for kind, pat in _PATTERNS:
            m = pat.search(query)
            if not m:
                continue
            name = _normalise_name(m.group("name"))
            res = self._handle(kind, name)
            if res:
                return res
        return None

    def _handle(self, kind: str, name: str) -> CodeQAResult | None:
        if kind == "docstring":
            return self._docstring_for(name)
        if kind == "defined_in":
            return self._defined_in(name)
        if kind == "imports":
            return self._imports(name)
        if kind == "methods":
            return self._methods(name)
        return None

    # ── Find code sentences in the lattice for `name` ──────

    def _code_sentences_for(self, name: str) -> list[tuple[str, str]]:
        """Find (sentence, source) pairs for code-tagged lattice
        entries whose text mentions `name`.  Matches on exact symbol
        appearance, not fuzzy.
        """
        out: list[tuple[str, str]] = []
        target = name.lower()
        # Strip leading dotted owner ("Lattice.analogy" -> "analogy")
        bare = target.split(".")[-1]
        for text, src in zip(self.lattice._texts, self.lattice._sources):
            if not src.startswith("code:"):
                continue
            t_lo = text.lower()
            # Match on exact-word boundary so "stats" doesn't match "statistics".
            if (re.search(rf"\b{re.escape(target)}\b", t_lo)
                    or re.search(rf"\b{re.escape(bare)}\b", t_lo)):
                out.append((text, src))
        return out

    def _docstring_for(self, name: str) -> CodeQAResult | None:
        sents = self._code_sentences_for(name)
        if not sents:
            return None
        bare = name.split(".")[-1].lower()

        def head_of(txt: str) -> str:
            return txt.split(":", 1)[0].lower()

        # Rank candidates: prefer exact symbol-name appearance early
        # in the head (where module/class/function names live), then
        # prefer MODULE > CLASS > FUNCTION > METHOD when the name
        # corresponds to a module file (e.g. "multi_hop_qa" should
        # return the module docstring, not the MultiHopPattern class).
        def kind_rank(txt: str) -> int:
            h = head_of(txt)
            if h.startswith("module "):   return 0
            if h.startswith("class "):    return 1
            if h.startswith("function "): return 2
            return 3
        def name_at_start(txt: str) -> int:
            h = head_of(txt)
            # 0 = name immediately after the kind word, 1 = anywhere, 2 = none.
            if bare in h.split()[:3]:
                return 0
            if bare in h:
                return 1
            return 2
        sents.sort(key=lambda s: (name_at_start(s[0]), kind_rank(s[0]),
                                       len(s[0])))
        best_text, best_src = sents[0]
        return CodeQAResult(
            kind="docstring", target=name, answer=best_text,
            extras=[f"...and {len(sents)-1} more code refs" if len(sents) > 1 else ""],
        )

    def _defined_in(self, name: str) -> CodeQAResult | None:
        hits = []
        for s, v, o in self.structured.claim_triple:
            if v == "defined_in" and (s == name or s.lower() == name.lower()):
                hits.append(o)
        if not hits:
            sents = self._code_sentences_for(name)
            if not sents:
                return None
            # Pull the file ref out of the source tag.
            for t, src in sents:
                file_ref = src.split(":", 1)[1]
                hits.append(file_ref)
                break
        return CodeQAResult(
            kind="defined_in", target=name,
            answer=f"{name} is defined in {hits[0]}",
            extras=hits[1:],
        )

    def _imports(self, name: str) -> CodeQAResult | None:
        bare = name.split(".")[-1]
        imports = []
        for s, v, o in self.structured.claim_triple:
            if v == "imports" and (s == name or s.lower() == bare.lower()):
                imports.append(o)
        if not imports:
            return None
        imports = sorted(set(imports))
        return CodeQAResult(
            kind="imports", target=name,
            answer=f"{name} imports {len(imports)} modules: "
                     + ", ".join(imports[:10])
                     + (f", ... ({len(imports)-10} more)" if len(imports) > 10 else ""),
            extras=imports,
        )

    def _methods(self, name: str) -> CodeQAResult | None:
        methods = []
        for s, v, o in self.structured.claim_triple:
            if v == "has_method" and (s == name or s.lower() == name.lower()):
                methods.append(o)
        if not methods:
            return None
        methods = sorted(set(methods))
        return CodeQAResult(
            kind="methods", target=name,
            answer=f"{name} has {len(methods)} methods: "
                     + ", ".join(methods[:15])
                     + (f", ... ({len(methods)-15} more)" if len(methods) > 15 else ""),
            extras=methods,
        )
