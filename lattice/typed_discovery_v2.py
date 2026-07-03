"""
lattice/typed_discovery_v2.py - richer type system for TypedHDC.

Extends the v1 tagger (ENTITY/NUMBER only) with:
  PERSON_NAME   - "First Last" patterns, "Sir <Name>", "Professor <Name>", etc
  PLACE         - capitalized after "in", "at", "from", "to", "between"
  DATE          - 4-digit years (1500-2100), century words, month names
  ORG           - capitalized + corporate suffix (Inc, Corp, Ltd, University, etc)
  ENTITY        - generic catch-all for capitalized mid-sentence words

All rule-based heuristics. No learned models.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from collections import defaultdict

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))


FUNCTION_WORDS = {
    "the","a","an","is","was","were","are","be","been","being","has","have","had",
    "of","in","on","at","by","to","from","with","for","into","onto","through","across",
    "and","or","but","not","nor","yet",
    "this","that","these","those","such",
    "it","its","they","their","them","he","she","his","her","him","i","we","us","our","you","your","my","mine",
    "as","than","then","also","both","either","neither",
    "which","who","whom","whose","what","where","when","why","how",
    "so","very","more","most","much","many","some","any","all","each","every",
    "if","because","while","during","since","before","after","until","when",
    "now","still","ever","never","always","often","sometimes","quite","just","only","even","too",
    "one","two","three","first","second","last","next","four","five","six","seven","eight","nine","ten",
    "between","among","over","under","above","below","near",
    "born","made","known","named","called","said","told",
    "do","did","does","done","will","would","can","could","may","might","should","shall","must",
    "about","because","into","through","without","within",
}


MONTHS = {"january","february","march","april","may","june","july","august",
           "september","october","november","december","jan","feb","mar","apr",
           "jun","jul","aug","sep","oct","nov","dec"}
CENTURY_WORDS = {"century","centuries","millennium","decade","era","period"}
ORG_SUFFIXES = {"inc","corp","corporation","ltd","limited","university","college",
                 "school","institute","foundation","association","society","union",
                 "club","party","company","co","sa","bv"}
PERSON_TITLES = {"sir","dame","lord","lady","mr","mrs","ms","dr","professor","prof",
                 "president","king","queen","emperor","empress","duke","duchess",
                 "general","admiral","captain","saint","st"}
LOCATION_PREPS = {"in","at","from","to","near","across","within","throughout",
                   "outside","inside","between","among"}


_TOKEN_RE = re.compile(r"\b[\w-]+\b")


def tokenize(sentence: str) -> list[str]:
    return _TOKEN_RE.findall(sentence)


def is_year(tok: str) -> bool:
    if not tok.isdigit() and not (tok[:-1].isdigit() and tok[-1] == "s"):
        return False
    try:
        n = int(tok.rstrip("s"))
        return 1000 <= n <= 2100
    except ValueError:
        return False


def is_number(tok: str) -> bool:
    return tok.isdigit() or (tok[:-1].isdigit() and tok[-1] == "s")


def type_tag_sentence_v2(sentence: str) -> list[tuple[str, str]]:
    """Return list of (token, type_tag) for each token.

    Tags:
      DATE          - year, century, month
      NUMBER        - digits not year-shaped
      PERSON_TITLE  - sir, dame, prof, etc (preserved for chained name detection)
      PLACE         - capitalized following location preposition
      ORG           - capitalized phrase containing corporate suffix
      PERSON_NAME   - multi-word capitalized phrase resembling a full name
      ENTITY        - other capitalized words (mid-sentence)
      <word>        - function or content words kept literal
    """
    tokens = tokenize(sentence)
    tags = ["?"] * len(tokens)

    for i, tok in enumerate(tokens):
        lo = tok.lower()
        is_first = (i == 0)
        is_cap = tok[0].isupper() if tok else False

        if is_year(tok):
            tags[i] = "DATE"
        elif is_number(tok):
            tags[i] = "NUMBER"
        elif lo in MONTHS or lo in CENTURY_WORDS:
            tags[i] = "DATE"
        elif lo in PERSON_TITLES:
            tags[i] = lo   # keep as literal — context indicator
        elif lo in FUNCTION_WORDS:
            tags[i] = lo
        elif is_cap and not is_first:
            tags[i] = "CAP"   # provisional — refined below
        elif is_cap and is_first:
            # Sentence-initial capital — could be content or entity
            if lo in FUNCTION_WORDS:
                tags[i] = lo
            else:
                tags[i] = "CAP"
        else:
            tags[i] = lo

    # Second pass — refine CAP tokens by context
    for i, tok in enumerate(tokens):
        if tags[i] != "CAP":
            continue
        # Check preceding word
        prev = tags[i - 1] if i > 0 else None
        # Look for multi-word names: prev was CAP or PERSON_TITLE
        # Detect ORG: any of next 3 tokens has org suffix
        is_org = False
        for j in range(i, min(i + 4, len(tokens))):
            if tokens[j].lower() in ORG_SUFFIXES:
                is_org = True
                break
        if is_org:
            tags[i] = "ORG"
            continue
        # Detect PLACE: preceded by location prep
        if prev in LOCATION_PREPS:
            tags[i] = "PLACE"
            continue
        # Detect PERSON_NAME: preceded by PERSON_TITLE, or by another CAP forming a name chain
        if prev in PERSON_TITLES:
            tags[i] = "PERSON_NAME"
            continue
        if prev == "PERSON_NAME":
            tags[i] = "PERSON_NAME"
            continue
        if prev == "CAP":
            # Two consecutive capitals — likely a name or place
            # Heuristic: if any of prior 4 tokens is "born", "professor", etc, lean PERSON
            person_indicator = any(
                tokens[max(0, i - 5):i][k].lower() in
                {"born","died","wrote","said","invented","discovered","founded"}
                for k in range(min(5, i))
            )
            if person_indicator:
                tags[i] = "PERSON_NAME"
                # also retroactively upgrade prev
                if tags[i - 1] == "CAP":
                    tags[i - 1] = "PERSON_NAME"
                continue
        # Default: ENTITY
        tags[i] = "ENTITY"

    # Cleanup: any remaining CAP becomes ENTITY
    for i in range(len(tags)):
        if tags[i] == "CAP":
            tags[i] = "ENTITY"

    return list(zip(tokens, tags))


TYPED_TAGS = {"ENTITY","PERSON_NAME","PLACE","ORG","DATE","NUMBER"}


def pattern_of_tagged(tagged: list[tuple[str, str]]) -> str:
    return " ".join(t[1] for t in tagged)


def entity_positions(tagged: list[tuple[str, str]]) -> list[int]:
    """Position of any typed (semantic) slot."""
    return [i for i, (_, t) in enumerate(tagged) if t in TYPED_TAGS]


def discover_typed_windows_v2(sentences: list[str],
                                  window_min: int = 4,
                                  window_max: int = 7,
                                  min_pattern_count: int = 3,
                                  min_entities_in_window: int = 2,
                                  ) -> dict[str, list[list[tuple[str, str]]]]:
    """Find recurring TYPED phrase windows with the richer tag set."""
    patterns = defaultdict(list)
    for sent in sentences:
        tagged = type_tag_sentence_v2(sent)
        for w in range(window_min, window_max + 1):
            for i in range(len(tagged) - w + 1):
                window = tagged[i:i + w]
                n_typed = sum(1 for _, t in window if t in TYPED_TAGS)
                if n_typed < min_entities_in_window:
                    continue
                pat = pattern_of_tagged(window)
                patterns[pat].append(window)
    return {p: ts for p, ts in patterns.items() if len(ts) >= min_pattern_count}


def extract_triples_v2(pattern_dict: dict, name_prefix: str = "REL") \
        -> tuple[list[tuple[str, str, str]], list[dict]]:
    """Extract subject-predicate-object triples between typed slots in patterns."""
    triples = []
    info = []
    for i, (pat, instances) in enumerate(
        sorted(pattern_dict.items(), key=lambda x: -len(x[1]))
    ):
        # Label from non-typed, non-function words in pattern
        label_parts = []
        for tag in pat.split():
            if tag not in TYPED_TAGS and tag not in FUNCTION_WORDS:
                label_parts.append(tag)
        if label_parts:
            label = name_prefix + "_" + "_".join(label_parts[:3]).upper()
        else:
            label = f"{name_prefix}_{i}"

        instance_triples = []
        sample = []
        for instance in instances:
            positions = entity_positions(instance)
            if len(positions) < 2:
                continue
            subj_tok, subj_type = instance[positions[0]]
            obj_tok, obj_type = instance[positions[-1]]
            instance_triples.append((subj_tok, label, obj_tok))
            if len(sample) < 3:
                sample.append({
                    "fragment": " ".join(t[0] for t in instance),
                    "subj": (subj_tok, subj_type),
                    "obj":  (obj_tok, obj_type),
                })
        triples.extend(instance_triples)
        info.append({
            "pattern_id": i,
            "label": label,
            "pattern_string": pat,
            "instance_count": len(instances),
            "n_triples": len(instance_triples),
            "sample": sample,
        })
    return triples, info
