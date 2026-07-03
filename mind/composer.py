"""
mind/composer.py - the COMPOSED VOICE: answers built from several true facts,
rewritten into plain speech and joined into one paragraph. No LLM, no word
prediction - every sentence is a stored memory, cleaned and connected:

    retrieve -> select DIVERSE facts (embedding MMR, near-duplicates dropped)
             -> simplify each (deterministic rewrite rules)
             -> order (definition first) -> join with connective tissue.

The n-gram generator stitched likely words and produced salad; this composes
known truths and produces speech. (Same philosophy as the imagination engine's
render(), applied to answering.)
"""
from __future__ import annotations

import re

# ── deterministic sentence clean-up ─────────────────────────────────

_PAREN_RE = re.compile(r"\s*\([^()]*\)")
_OFFICIAL_RE = re.compile(
    r"^([A-Z][\w' -]{1,40}?),\s+(?:officially|formally|also known as|sometimes"
    r" called)[^,]{0,80},\s+(is|was|are|were)\b")
_TOPIC_PRON_RE = re.compile(
    r"^(.{2,48}?):\s+(It|Its|He|His|She, |Her|They|Their)\b")


def simplify(fact: str, max_len: int = 230) -> str:
    """Encyclopedia sentence -> plain sentence. Deterministic rules only."""
    s = fact.strip()
    # topic-anchored pronoun rows: "Bolivia: It is multiethnic" -> "Bolivia is"
    m = _TOPIC_PRON_RE.match(s)
    if m:
        topic, pron = m.group(1), m.group(2)
        repl = {"It": topic, "Its": f"{topic}'s", "He": topic, "She": topic,
                "They": topic, "Their": f"{topic}'s", "His": f"{topic}'s",
                "Her": f"{topic}'s"}.get(pron, topic)
        s = repl + s[m.end():]
    # strip parentheticals (incl. pronunciation husks like "(; )") - twice
    # for one level of nesting
    s = _PAREN_RE.sub("", s)
    s = _PAREN_RE.sub("", s)
    # "X, officially the Republic of Y, is" -> "X is"
    s = _OFFICIAL_RE.sub(r"\1 \2", s)
    # leftover doubled spaces / stray punctuation
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = re.sub(r"^[,;:\s]+", "", s)
    # long sentence: cut at a clause boundary past the halfway point
    if len(s) > max_len:
        cut = s.rfind(",", 100, max_len)
        s = s[:cut] if cut > 100 else s[:max_len].rsplit(" ", 1)[0]
        # never end on a dangler ("...distance from Earth of.")
        s = re.sub(r"\s+(of|and|or|the|a|an|to|in|on|at|with|from|for|by|as"
                   r"|its|their|his|her)$", "", s, flags=re.I) + "."
    s = re.sub(r"\s+(of|and|or|the|a|an|to|in|on|at|with|from|for|by|as"
               r"|its|their|his|her)\s*[.]?$", "", s, flags=re.I)
    if s and s[-1] not in ".!?":
        s += "."
    return s


# ── fact selection: relevant AND diverse ────────────────────────────

def select_facts(question: str, hits: list[dict], emb_fn, k: int = 3,
                 dup_thresh: float = 0.85):
    """Greedy MMR over the retrieval cohort: each pick maximizes relevance to
    the question minus similarity to what's already picked. Near-duplicates
    are dropped entirely."""
    import numpy as np
    if not hits:
        return []
    texts = [h["text"] for h in hits]
    embs = np.asarray(emb_fn([question] + texts))
    q, F = embs[0], embs[1:]
    rel = F @ q
    picked: list[int] = []
    while len(picked) < k and len(picked) < len(hits):
        best_i, best_score = None, -1e9
        for i in range(len(hits)):
            if i in picked:
                continue
            red = max((float(F[i] @ F[j]) for j in picked), default=0.0)
            if red > dup_thresh:
                continue                    # near-duplicate of a pick
            score = float(rel[i]) - 0.5 * red
            if score > best_score:
                best_i, best_score = i, score
        if best_i is None:
            break
        picked.append(best_i)
    return [hits[i] for i in picked]


# ── composition ─────────────────────────────────────────────────────

_CONNECTORS = ["", "Beyond that, ", "Also, ", "And ", "On top of that, "]
_DEF_RE = re.compile(r"^[A-Z][\w' -]{1,48}\s+(is|was|are|were)\b")


def _lead_lower(s: str, keep_words: set) -> str:
    """Lowercase a sentence's first word after a connector, unless it looks
    like a proper noun (appears capitalized in the fact set elsewhere)."""
    toks = s.split(" ", 2)
    first = toks[0].strip(",.;:")
    nxt = toks[1].strip(",.;:") if len(toks) > 1 else ""
    if (first in keep_words or not first[:1].isupper()
            or nxt[:1].isdigit() or nxt[:1].isupper()):
        return s
    if first.lower() in ("i", "i'm", "i've"):
        return s
    return s[0].lower() + s[1:]


def compose_answer(question: str, hits: list[dict], emb_fn,
                   seed: int = 7) -> tuple[str, list[dict]] | None:
    """Compose 2-4 diverse true facts into one spoken paragraph.
    Returns (body, facts_used) or None when there's too little to compose."""
    import random
    facts = select_facts(question, hits, emb_fn, k=3)
    if len(facts) < 2:
        return None
    simple = [simplify(f["text"]) for f in facts]
    simple = [s for s in simple if len(s) > 25]
    if len(simple) < 2:
        return None
    # definition-shaped sentence leads
    simple.sort(key=lambda s: 0 if _DEF_RE.match(s) else 1)
    # proper nouns = words seen capitalized mid-sentence anywhere in the facts
    keep = set()
    lead_counts: dict = {}
    for s in simple:
        for w in re.findall(r"(?<=[a-z0-9,;] )[A-Z][\w'-]+", s):
            keep.add(w)
        lw = s.split(" ", 1)[0].strip(",.;:")
        if lw.lower() not in ("the", "a", "an", "it", "this", "these", "they"):
            lead_counts[lw] = lead_counts.get(lw, 0) + 1
    # a non-article word that BEGINS 2+ facts is a proper name (e.g. Jupiter)
    keep.update(w for w, c in lead_counts.items() if c >= 2 and w[:1].isupper())
    rng = random.Random(seed + len(question))
    parts = [simple[0]]
    used = _CONNECTORS[1:]
    rng.shuffle(used)
    for i, s in enumerate(simple[1:]):
        conn = used[i % len(used)]
        parts.append(conn + (_lead_lower(s, keep) if conn else s))
    body = " ".join(parts)
    if len(body) > 520:
        body = " ".join(parts[:2])
    return body, facts
