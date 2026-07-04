"""
mind/writer.py - THE WRITER: essays compiled from memory, never predicted.

An LLM writes by predicting likely words. Telp writes by COMPILING:

    gather   - every knowledge row that names the topic
    cluster  - group rows into subtopics by embedding neighborhood
             (transduction chooses the grouping; it never authors a word)
    plan     - definition cluster leads, the rest ordered by relevance
    realize  - each paragraph composed by the deterministic composer rules
    cite     - the essay ends by naming its sources

Every sentence in the output is a stored memory, cleaned by deterministic
rules. The essay can be audited line by line - that is the point.
"""
from __future__ import annotations

import re

from mind.composer import simplify
from mind.restyle import restyle

# essay-grade definition shapes: commas allowed in the subject ("The
# raccoon, sometimes called..., is a mammal" is a definition too).
# STRONG = copula + article ("is A mammal") - "is noted for" is a
# property, not a definition.
_DEF_RE = re.compile(r"^[A-Z][\w' ,-]{1,64}\s+(?:is|was|are|were)\b")
_DEF_STRONG = re.compile(
    r"^[A-Z][\w' ,-]{1,64}\s+(?:is|was|are|were)\s+(?:a|an|the|one)\b")

_SKIP_SOURCES = ("user_msg", "agent_response", "conversation_turn",
                 "image:", "video:")

_PARA_CONNECT = [
    "",                       # lead paragraph opens cold
    "There is more to it. ",
    "Beyond the basics: ",
    "It reaches further still. ",
    "And the story keeps going: ",
]

_ANCHOR_RE = re.compile(r"^([^:]{2,48}):\s+(?=[A-Z0-9\"'])")


def _clean_sentence(raw: str) -> str | None:
    """simplify + essay-grade quality gate: anchors stripped, fragments
    and unbalanced clauses rejected. Returns None for rows that cannot
    stand as prose."""
    s = simplify(raw)
    # "Raccoon: The kits are raised..." -> "The kits are raised..."
    # (simplify already substituted "Topic: It is..." pronoun forms)
    s = _ANCHOR_RE.sub("", s).strip()
    # ingest artifacts: a section header glued to the sentence start
    # ("Usage ASCII was first used...") - strip it
    s = re.sub(r"^(?:Usage|History|Etymology|Overview|Background"
               r"|Description|Terminology|Definition)\s+(?=[A-Z])", "", s)
    if len(s) < 25 or not s[0].isupper() and not s[0].isdigit():
        return None
    # rows truncated at a false sentence boundary ("...romanized: tele,
    # lit.") are broken data, not prose
    if re.search(r"\b(?:lit|cf|viz|romanized|e\.g|i\.e)\.?$", s.rstrip(".")):
        return None
    if s.count("(") != s.count(")") or s.lstrip().startswith(")"):
        return None
    # a sentence that still contains a stranded anchor colon is a fragment
    if re.match(r"^[A-Z][\w' -]{1,40}:\s*[a-z)]", s):
        return None
    # bibliography lines are references, not prose ("Livy, The History
    # of Rome, Volume III, translated by C...")
    if re.search(r"\b(?:translated by|University Press|pp\.\s*\d"
                 r"|Volume\s+[IVX0-9])", s):
        return None
    if s.count(",") >= 3 and not re.search(
            r"\b(?:is|are|was|were|has|have|had|can|will|would|became"
            r"|include[sd]?)\b", s):
        return None
    if s.rstrip().endswith((",.", ",")):
        return None
    return s


_TOPIC_STOP = frozenset(
    "history story overview facts about with from the and of in on".split())


def _topic_key(agent, topic: str) -> str:
    """The topic's most CONTENTFUL word - 'history of the telephone'
    must gather by 'telephone', never by 'history' (which matched Rome,
    Alabama and Aristotle)."""
    words = [w for w in re.findall(r"[a-z][\w'-]{3,}", topic.lower())
             if w not in _TOPIC_STOP]
    if not words:
        words = topic.lower().split() or [topic.lower()]
    dfmap = getattr(agent.encoder, "doc_freq", {}) or {}
    rarest = min(words, key=lambda w: dfmap.get(w, 0))
    return rarest[:5] if len(rarest) > 5 else rarest


def _gather(agent, topic: str) -> list:
    key = _topic_key(agent, topic)
    out, seen = [], set()
    for t, s in zip(agent.lattice._texts, agent.lattice._sources):
        if s.startswith(_SKIP_SOURCES):
            continue
        if key in t.lower() and t not in seen:
            seen.add(t)
            out.append((t, s))
    return out


def compose_essay(agent, topic: str, emb_fn,
                  max_rows: int = 40, max_paras: int = 4):
    """Compile a multi-paragraph essay about `topic` from memory.
    Returns (essay_text, used_rows, sources) or None when memory is
    too thin to write honestly."""
    import numpy as np
    rows = _gather(agent, topic)
    if len(rows) < 6:
        return None
    embs = np.asarray(emb_fn([topic] + [t for t, _ in rows]))
    q, F = embs[0], embs[1:]
    rel = F @ q
    order = np.argsort(-rel)[:max_rows]
    rows = [rows[i] for i in order]
    F = F[order]
    rel = rel[order]

    # cluster into subtopics: greedy seeds in relevance order
    clusters: list[list[int]] = []
    seeds: list[int] = []
    for i in range(len(rows)):
        placed = False
        for c, s in enumerate(seeds):
            if float(F[i] @ F[s]) >= 0.55:
                clusters[c].append(i)
                placed = True
                break
        if not placed and len(seeds) < max_paras + 2:
            seeds.append(i)
            clusters.append([i])

    # a DEFINITION means the topic itself in subject position - "The
    # original habitats of the raccoon are..." is def-shaped but defines
    # habitats, not raccoons
    tkey = _topic_key(agent, topic)

    def _def_rank(s: str) -> int:
        # a real definition: topic in subject position, and a copula +
        # article somewhere in the opening clause ("The raccoon,
        # sometimes called..., IS A mammal"). Long appositive subjects
        # defeat any subject-regex - so don't parse the subject.
        if (tkey in s[:18].lower()
                and re.search(r"\b(?:is|are|was|were)\s+(?:a|an|the|one)\b",
                              s[:140])):
            return 0
        if _DEF_RE.match(s):
            return 1
        return 2

    # the paragraph plan: definition-led cluster first, rest by relevance
    def _cluster_key(c):
        best_def = min((_def_rank(simplify(rows[i][0])) for i in c[:3]),
                       default=2)
        return (best_def, -max(float(rel[i]) for i in c))

    clusters.sort(key=_cluster_key)
    # a paragraph must be ABOUT the topic - clusters that drifted (hot-
    # Jupiter exoplanets, 1957 theater productions) don't make the essay
    clusters = [c for c in clusters
                if len(c) >= 2 and max(float(rel[i]) for i in c) >= 0.45]
    clusters = clusters[:max_paras]
    if len(clusters) < 2:
        return None

    used, paragraphs = [], []
    n_restyled = 0
    for pi, c in enumerate(clusters):
        # within a cluster: diverse picks, near-duplicates dropped
        picked: list[int] = []
        for i in c:
            if len(picked) >= 4:
                break
            if any(float(F[i] @ F[j]) > 0.85 for j in picked):
                continue
            picked.append(i)
        if len(picked) < 2:
            continue
        sents = []
        for i in picked:
            s = _clean_sentence(rows[i][0])
            if s is None:
                continue
            sents.append(s)
            used.append(rows[i])
        if len(sents) < 2:
            continue
        # the topic's own definition leads its paragraph
        sents.sort(key=_def_rank)
        # HIS voice: dictionary-licensed simplification + splits, each
        # sentence verified by embedding round-trip (reverts on drift)
        styled = []
        for s in sents:
            s2, k = restyle(s, emb_fn)
            n_restyled += k
            styled.append(s2)
        para = (_PARA_CONNECT[min(pi, len(_PARA_CONNECT) - 1)]
                + " ".join(styled))
        paragraphs.append(para)

    if len(paragraphs) < 2:
        return None
    def _label(s: str) -> str:
        t = s.split(":", 1)[-1]
        i = t.find("(")
        return (s.split(":", 1)[0] + ":" + t[:i].strip()) if i > 1 else s

    srcs = sorted({_label(s) for _, s in used})
    outro = ("Every sentence above is a memory I hold, drawn from: "
             + "; ".join(srcs[:6])
             + (" and more" if len(srcs) > 6 else "") + "."
             + ((" Rephrased only by dictionary-licensed rules, each "
                 "verified to preserve the original meaning.")
                if n_restyled else ""))
    essay = "\n\n".join(paragraphs) + "\n\n" + outro
    return essay, [t for t, _ in used], srcs
