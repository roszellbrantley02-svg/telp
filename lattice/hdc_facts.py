"""
lattice/hdc_facts.py - Telp's fact memory, where hypervectors do the reasoning.

Telp holds (subject, relation, object) facts - "Galileo Galilei was born in
Pisa" - in two forms that check each other:

  * one hyperdimensional RECORD per entity. Every fact becomes
    bind(RELATION, VALUE) - the XOR of two random 10,000-bit atoms - and an
    entity's facts are bundled into a single vector by bitwise majority.
    "Where was Galileo born?" is then algebra: XOR the born_in atom back out
    of Galileo's record and what is left sits measurably close to the atom
    for Pisa. "Who was born in Pisa and taught at Padua?" is a Hamming scan
    of every record against the bound question. Analogies, multi-hop chains,
    comparisons and full descriptions are the same two moves: unbind, then
    clean up against the known atoms.
  * a symbolic INDEX of the Fact rows themselves, which only VERIFIES and
    CITES. HDC proposes; a proposal is returned only when a stored Fact says
    exactly that, and the Fact (source, sentence, memory id) goes back with
    it. A proposal the index cannot confirm is dropped and counted
    (stats()["rejected"]) - never shown. If a record is damaged the answers
    degrade with it, even though the index still holds every fact: the
    index cannot answer on its own.

The numbers (D = 10,000 bits, binary atoms, majority bundling):
  * two unrelated vectors agree on half their bits: similarity 0.5 with
    standard deviation sigma = 0.5/sqrt(D) = 0.005.
  * an item bundled with k-1 others keeps similarity about 0.5 + 0.4/sqrt(k)
    with the bundle, i.e. 0.8*sqrt(D/k) sigma above chance (25 sigma at
    k = 10, 11.5 sigma at k = 48).
  * a proposal is accepted at z >= 6 (similarity >= 0.53). By chance that
    happens about once per 10^9 comparisons - and a chance hit is caught by
    verification anyway. A record holds at most as many facts as keep every
    one 5.5 sigma clear of that bar (48 at D = 10,000); an entity that knows
    more is SHARDED into several records, so recall stays perfect however
    much one entity knows.

Atoms are never stored: each is generated from a SHA-1 seed of (kind, key),
so every process derives the same vectors from nothing but the names.
Records are packed (1,250 bytes of bits, padded to whole 64-bit words) and
compared with XOR + popcount over uint64 views.

    mem = FactMemory()
    mem.add_many(facts)
    mem.lookup("Galileo", "born_in")          -> [Answer("Pisa", 0.62, [...])]
    mem.who([("born_in", "Pisa"), ("taught_at", "Padua")])  -> [Match(...)]
    mem.analogy("Galileo", "Pisa", "Newton")  -> [Answer("Woolsthorpe", ...)]
    mem.chain("Galileo", ["born_in", "country"])  -> [Answer("Italy", ...)]
    mem.check("Galileo", "born_in", "Pisa")   -> (True, [Fact(...)])
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

from lattice.facts import RELATIONS, Answer, Fact, Match, norm_key


# ─── the numbers ────────────────────────────────────────────────────

D = 10_000                 # bits per hypervector
Z_ACCEPT = 6.0             # accept an HDC proposal at >= 6 sigma above chance
RECALL_MARGIN = 5.5        # ...and keep true items >= 5.5 sigma above that bar
MAX_ALTERNATIVES = 16      # loose readings of one constraint value we probe
CANDIDATE_POOL = 256       # most entities a who() query verifies
BEAM = 16                  # most partial paths a chain() keeps per hop
CONTAINMENT_DEPTH = 3      # Arcetri -> Florence -> Tuscany -> Italy
CONTAINMENT_SCANS = 24     # most record scans one place may expand into


# ─── packed bit vectors ─────────────────────────────────────────────

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_rows_table(x: np.ndarray) -> np.ndarray:
    """Set bits per row of a uint64 matrix, by a 256-entry byte table
    (for numpy < 2, which has no bitwise_count)."""
    x = np.ascontiguousarray(x)
    return _POP8[x.view(np.uint8)].sum(axis=-1, dtype=np.int64)


def _popcount_rows_native(x: np.ndarray) -> np.ndarray:
    return np.bitwise_count(x).sum(axis=-1).astype(np.int64)


_popcount_rows = (_popcount_rows_native if hasattr(np, "bitwise_count")
                  else _popcount_rows_table)


def _make_atom(kind: str, key: str, dim: int, n_words: int) -> np.ndarray:
    """The atom for (kind, key): dim random bits from a SHA-1 seed, packed
    MSB-first like np.packbits and zero-padded to whole uint64 words. The
    pad bits are zero in every vector, so they never count as a difference."""
    digest = hashlib.sha1(f"{kind}\x1f{key}".encode("utf-8")).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest, "big")))
    n_bytes = (dim + 7) // 8
    raw = np.frombuffer(rng.bytes(n_bytes), dtype=np.uint8).copy()
    if dim % 8:
        raw[-1] &= (0xFF << (8 - dim % 8)) & 0xFF
    out = np.zeros(n_words * 8, dtype=np.uint8)
    out[:n_bytes] = raw
    return out.view(np.uint64)


class _Bank:
    """A growable matrix of packed hypervectors - records, or a codebook of
    atoms - compared against a probe in one vectorized pass. Each row has a
    label; freed rows are recycled, so replacing one record never restacks
    the rest."""

    def __init__(self, dim: int, n_words: int):
        self.dim = dim
        self.words = np.zeros((16, n_words), dtype=np.uint64)
        self.alive = np.zeros(16, dtype=bool)
        self.labels: list = []
        self.index: dict = {}          # label -> row
        self.n = 0                     # high-water mark of rows used
        self._free: list[int] = []

    def __len__(self) -> int:
        return len(self.index)

    def put(self, vec: np.ndarray, label) -> int:
        if self._free:
            row = self._free.pop()
        else:
            if self.n == len(self.words):
                grow = len(self.words)
                self.words = np.concatenate(
                    [self.words, np.zeros_like(self.words[:grow])])
                self.alive = np.concatenate(
                    [self.alive, np.zeros(grow, dtype=bool)])
            row = self.n
            self.n += 1
            self.labels.append(None)
        self.words[row] = vec
        self.alive[row] = True
        self.labels[row] = label
        self.index[label] = row
        return row

    def drop(self, label) -> None:
        row = self.index.pop(label, None)
        if row is None:
            return
        self.alive[row] = False
        self.labels[row] = None
        self.words[row] = 0
        self._free.append(row)

    def similarity(self, probe: np.ndarray) -> np.ndarray:
        """1 - Hamming/D for every row (0.5 = unrelated); dead rows get 0."""
        if self.n == 0:
            return np.zeros(0)
        dist = _popcount_rows(np.bitwise_xor(self.words[:self.n], probe))
        sims = 1.0 - dist / self.dim
        sims[~self.alive[:self.n]] = 0.0
        return sims


# ─── loose matching of the user's words to stored values ────────────
#
# "Padua" should find "the University of Padua", "Reykjavik" should find
# "Reykjavík", "1564" should find "15 February 1564". But "York" must not
# find "New York", nor "Virginia" "West Virginia": the user's words have
# to appear as a run inside the stored value, and everything else in the
# value has to be a generic head word ("university of", "city"). norm_key
# already folds case, accents and a leading article.

_GENERIC = frozenset("""
    university universities college school institute institution academy
    conservatory polytechnic seminary faculty
    city town village municipality commune borough district county
    province region state republic kingdom empire island islands
    of the at in on and
    northern southern eastern western central
    royal national federal imperial technical
    about approximately around roughly circa c ca some estimated est
""".split())
_SPECIFIERS = frozenset({"in", "of", "for"})   # "Nobel Prize" ~ "... in X"
_APPROX = frozenset({"about", "approximately", "around", "roughly", "circa",
                     "c", "ca", "some", "estimated", "est", "nearly",
                     "almost", "over", "under", "more", "less", "than"})
_MONTHS = {m: i + 1 for i, m in enumerate(
    "january february march april may june july august september "
    "october november december".split())}
_MONTH_ABBREV = {m[:3]: m for m in _MONTHS if m != "may"}
_MONTH_ABBREV["sept"] = "september"

_TOKEN = re.compile(r"[^\W_]+")
_DIGIT_GROUP = re.compile(r"(?<=\d)[,.](?=\d{3}(?!\d))")
_ORDINAL = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)$")


def _tokens(key: str) -> tuple[str, ...]:
    """Words of a norm_key'd string; "390,000" stays one number, "15th"
    reads as "15" and "Feb" as "february" (on both sides of a match)."""
    out = []
    for t in _TOKEN.findall(_DIGIT_GROUP.sub("", key)):
        m = _ORDINAL.match(t)
        if m:
            t = m.group(1)
        elif t in _MONTH_ABBREV:
            t = _MONTH_ABBREV[t]
        out.append(t)
    return tuple(out)


def _date_parts(tokens: tuple[str, ...]) -> tuple | None:
    """(day, month, year) when the tokens read as a date or a year."""
    day = month = year = None
    era = 1
    for t in tokens:                   # already normalized by _tokens
        if t in _MONTHS:
            if month is not None:
                return None
            month = _MONTHS[t]
        elif t.isdigit():
            if len(t) <= 2 and day is None and 1 <= int(t) <= 31:
                day = int(t)
            elif len(t) <= 4 and year is None:
                year = int(t)
            else:
                return None
        elif t in ("bc", "bce"):
            era = -1
        elif t not in ("c", "ca", "circa", "ad", "ce", "of", "the"):
            return None
    if year is None:
        return None
    return (day, month, era * year)


def _number(tokens: tuple[str, ...]) -> tuple[int, bool] | None:
    """(value, approximate?) for "about 390,000"-style values."""
    words = [t for t in tokens if t not in _APPROX]
    if len(words) != 1 or not words[0].isdigit():
        return None
    return int(words[0]), len(words) != len(tokens)


def _tokens_match(ut: tuple[str, ...], vt: tuple[str, ...]) -> bool:
    """Does the user's phrase (tokens ut) name the stored value (vt)?"""
    if not ut or not vt:
        return False
    if ut == vt:
        return True
    if all(t in _GENERIC for t in ut):
        return False                   # "the university" names nothing
    n = len(ut)
    for i in range(len(vt) - n + 1):
        if vt[i:i + n] != ut:
            continue
        if all(t in _GENERIC for t in vt[:i] + vt[i + n:]):
            return True                # "Padua" ~ "University of Padua"
        if i == 0 and n >= 2 and vt[n] in _SPECIFIERS:
            return True                # "Nobel Prize" ~ "Nobel Prize in X"
    pu, pv = _date_parts(ut), _date_parts(vt)
    if pu and pv:                      # "1564" ~ "15 February 1564"
        return (pu[2] == pv[2] and pu[1] in (None, pv[1])
                and pu[0] in (None, pv[0]))
    return False


def loose_match(user: str, value: str) -> bool:
    """True when the user's words name this stored value, loosely:
    "Padua" ~ "the University of Padua", "Reykjavik" ~ "Reykjavík",
    "February 1564" ~ "15 February 1564". Never "York" ~ "New York"."""
    uk, vk = norm_key(user), norm_key(value)
    if not uk or not vk:
        return False
    return uk == vk or _tokens_match(_tokens(uk), _tokens(vk))


def _agrees(relation: str, user: str, stored: str) -> bool:
    """Does the stored value confirm the user's? Loosely equal - or, for
    a year relation, the same year inside a fuller date ("born in 10
    December 1815?" is a yes for born_year 1815)."""
    if loose_match(user, stored):
        return True
    spec = RELATIONS.get(relation)
    if spec is None or spec.obj_kind != "year":
        return False
    pu = _date_parts(_tokens(norm_key(user)))
    ps = _date_parts(_tokens(norm_key(stored)))
    return bool(pu and ps) and pu[2] == ps[2]


def _contradicts(relation: str, user: str, stored: str) -> bool:
    """For a single-valued date or number relation: does the stored value
    rule out the user's? Dates conflict only on a part both give ("1564"
    doesn't rule out "15 February 1564"); approximate numbers never do."""
    spec = RELATIONS.get(relation)
    ut, st = _tokens(norm_key(user)), _tokens(norm_key(stored))
    if spec is not None and spec.obj_kind in ("date", "year"):
        pu, ps = _date_parts(ut), _date_parts(st)
        if not pu or not ps:
            return False
        return any(a is not None and b is not None and a != b
                   for a, b in zip(pu, ps))
    if spec is not None and spec.obj_kind == "number":
        nu, ns = _number(ut), _number(st)
        if not nu or not ns or ns[1] or nu[1]:
            return False
        return nu[0] != ns[0]
    return False


# ─── names ──────────────────────────────────────────────────────────

_DISAMBIGUATOR = re.compile(r"\s*\([^()]*\)\s*$")
_TITLES = frozenset("sir dame lord lady saint st dr prof professor king "
                    "queen pope mr mrs ms".split())
_PARTICLES = frozenset("de di da del della dei van von der den la le du "
                       "dos das al el bin ibn y".split())
_SUFFIXES = frozenset("jr sr ii iii iv".split())


def _name_parts(display: str) -> set[str]:
    """First and last name of a multi-word name: "Sir Isaac Newton" ->
    {"isaac", "newton"}. Used only for entities known to be people."""
    words = [norm_key(w) for w in _DISAMBIGUATOR.sub("", display).split()]
    words = [w for w in words if w and w not in _TITLES and w not in _SUFFIXES]
    if len(words) < 2:
        return set()
    return {w for w in (words[0], words[-1])
            if len(w) > 1 and w not in _PARTICLES}


# what an instance_of class says about an entity's kind
_CLASS_KIND: dict[str, str] = {}
for _kind, _words in {
    "place": "city town village capital country nation state province region "
             "county island continent river lake mountain sea ocean "
             "municipality commune district territory kingdom republic",
    "org": "university college school institute academy company corporation "
           "organization organisation society band team club party firm "
           "agency museum library hospital",
    "work": "book novel play poem painting film movie opera symphony song "
            "album sculpture essay treatise",
    "person": "person human man woman",
}.items():
    for _w in _words.split():
        _CLASS_KIND[_w] = _kind

# Containment: "X located_in Y" puts X inside Y, so someone born in X was
# born in Y. Relations whose value is a place or an organisation (and so
# has a location) may be satisfied through such a chain.
_CONTAINER_EDGES = ("located_in", "country", "capital_of", "headquarters")
_CONTAINABLE = frozenset({"born_in", "died_in", "lived_in", "educated_at",
                          "taught_at", "worked_at", "located_in", "country",
                          "headquarters"})
# single-valued, but places nest: born in Pisa doesn't rule out born in
# Italy. A different value is a "no" only for two disjoint places of the
# same kind (two different cities).
_NESTED_SINGLE = frozenset({"born_in", "died_in", "country", "headquarters"})

# The same link read from the other end: "Italy capital Rome" also answers
# "what is Rome the capital of?". lookup() scans for those facts too.
_INVERSE = {"capital": "capital_of", "capital_of": "capital",
            "founded": "founded_by", "founded_by": "founded",
            "wrote": "author", "author": "wrote",
            "parent": "child", "child": "parent", "spouse": "spouse"}

_RELATION_ORDER = {name: i for i, name in enumerate(RELATIONS)}


# ─── the memory ─────────────────────────────────────────────────────

class FactMemory:
    """Facts held as hypervector records; HDC answers, the index verifies.

    Every query resolves names through the alias index, lets the records
    propose (unbind + cleanup, or a Hamming scan), and returns only what
    stored Facts confirm - with those Facts attached for citation."""

    def __init__(self, dim: int = D):
        if dim < 512:
            raise ValueError("dim must be at least 512 bits")
        self.dim = int(dim)
        self.n_words = (self.dim + 63) // 64
        self.sigma = 0.5 / math.sqrt(self.dim)
        self.accept = 0.5 + Z_ACCEPT * self.sigma
        # an item in a k-item record sits 0.8*sqrt(D/k) sigma above chance;
        # keep that >= Z_ACCEPT + RECALL_MARGIN  ->  k <= D*(0.8/11.5)^2
        self.shard_size = max(
            3, int(self.dim * (0.8 / (Z_ACCEPT + RECALL_MARGIN)) ** 2))
        self._atom_cache: dict[tuple[str, str], np.ndarray] = {}
        self.clear()

    # ── storing ─────────────────────────────────────────────────────

    def clear(self) -> None:
        """Forget every fact (atoms are pure functions of names: kept)."""
        self._facts: dict[int, Fact] = {}
        self._fid: dict[Fact, int] = {}
        self._next_fid = 0
        # entity key -> (relation, value key) -> ids of the facts saying it
        self._items: dict[str, dict[tuple[str, str], list[int]]] = {}
        self._names: dict[str, str] = {}
        self._by_memory: dict[int, set[int]] = {}
        # relation -> codebook of the value atoms seen with it
        self._codebooks: dict[str, _Bank] = {}
        self._pair_refs: Counter = Counter()
        self._relations = _Bank(self.dim, self.n_words)
        self._val_atoms: dict[str, np.ndarray] = {}
        self._val_refs: Counter = Counter()
        self._val_tokens: dict[str, tuple[str, ...]] = {}
        self._token_index: dict[str, set[str]] = {}
        self._records = _Bank(self.dim, self.n_words)
        self._entity_rows: dict[str, list[int]] = {}
        self._dirty: set[str] = set()
        self._aliases: dict[str, Counter] = {}
        self._titles: dict[str, set[str]] = {}
        self._parts: dict[str, set[str]] = {}
        self._counts = {"queries": 0, "proposals": 0, "verified": 0,
                        "rejected": 0}

    def add(self, fact: Fact) -> bool:
        """Store one fact; False if it is empty or already stored. Records
        are rebuilt lazily, on the next query."""
        if not isinstance(fact, Fact):
            raise TypeError(f"expected a Fact, got {type(fact).__name__}")
        ekey, rel, vkey = norm_key(fact.subject), fact.relation, \
            norm_key(fact.obj)
        if not (ekey and rel and vkey) or fact in self._fid:
            return False
        fid = self._next_fid
        self._next_fid += 1
        self._facts[fid] = fact
        self._fid[fact] = fid
        if fact.memory_id is not None:
            self._by_memory.setdefault(fact.memory_id, set()).add(fid)
        items = self._items.get(ekey)
        if items is None:
            items = self._items[ekey] = {}
            self._register_entity(ekey, fact.subject)
        slot = items.get((rel, vkey))
        if slot is None:
            items[(rel, vkey)] = [fid]
            self._codebook_add(rel, vkey)
            self._dirty.add(ekey)
        else:
            slot.append(fid)             # same claim, another source
        if rel == "alias":
            self._aliases.setdefault(vkey, Counter())[ekey] += 1
        return True

    def add_many(self, facts) -> int:
        """Store several facts; returns how many were new."""
        return sum(self.add(f) for f in facts)

    def remove_memory(self, memory_id: int) -> int:
        """Forget every fact read from one stored sentence (the lattice
        row); records are rebuilt without them. Returns facts removed."""
        fids = self._by_memory.pop(memory_id, None) or set()
        for fid in sorted(fids):
            self._remove_fid(fid)
        return len(fids)

    def remove(self, fact: Fact) -> bool:
        """Forget one claim from one source; False if it wasn't stored."""
        fid = self._fid.get(fact)
        if fid is None:
            return False
        ids = self._by_memory.get(fact.memory_id)
        if ids is not None:
            ids.discard(fid)
            if not ids:
                del self._by_memory[fact.memory_id]
        self._remove_fid(fid)
        return True

    def _remove_fid(self, fid: int) -> None:
        fact = self._facts.pop(fid)
        del self._fid[fact]
        ekey, rel, vkey = norm_key(fact.subject), fact.relation, \
            norm_key(fact.obj)
        items = self._items[ekey]
        slot = items[(rel, vkey)]
        slot.remove(fid)
        if not slot:
            del items[(rel, vkey)]
            self._codebook_drop(rel, vkey)
            self._dirty.add(ekey)
        if rel == "alias":
            holders = self._aliases.get(vkey)
            if holders is not None:
                holders[ekey] -= 1
                if holders[ekey] <= 0:
                    del holders[ekey]
                if not holders:
                    del self._aliases[vkey]
        if not items:
            del self._items[ekey]
            self._unregister_entity(ekey)

    # ── atoms and codebooks ─────────────────────────────────────────

    def _atom(self, kind: str, key: str) -> np.ndarray:
        """Cached atom for the few relation and tie-breaker vectors."""
        v = self._atom_cache.get((kind, key))
        if v is None:
            v = self._atom_cache[(kind, key)] = _make_atom(
                kind, key, self.dim, self.n_words)
        return v

    def _rel_atom(self, rel: str) -> np.ndarray:
        return self._atom("relation", rel)

    def _value_vec(self, vkey: str) -> np.ndarray:
        v = self._val_atoms.get(vkey)
        return v if v is not None else _make_atom("value", vkey, self.dim,
                                                  self.n_words)

    def _codebook_add(self, rel: str, vkey: str) -> None:
        self._pair_refs[(rel, vkey)] += 1
        if self._pair_refs[(rel, vkey)] > 1:
            return
        cb = self._codebooks.get(rel)
        if cb is None:
            cb = self._codebooks[rel] = _Bank(self.dim, self.n_words)
            self._relations.put(self._rel_atom(rel), rel)
        if self._val_refs[vkey] == 0:
            self._val_atoms[vkey] = _make_atom("value", vkey, self.dim,
                                               self.n_words)
            toks = self._val_tokens[vkey] = _tokens(vkey)
            for t in set(toks):
                self._token_index.setdefault(t, set()).add(vkey)
        self._val_refs[vkey] += 1
        cb.put(self._val_atoms[vkey], vkey)

    def _codebook_drop(self, rel: str, vkey: str) -> None:
        self._pair_refs[(rel, vkey)] -= 1
        if self._pair_refs[(rel, vkey)] > 0:
            return
        del self._pair_refs[(rel, vkey)]
        cb = self._codebooks[rel]
        cb.drop(vkey)
        if not len(cb):
            del self._codebooks[rel]
            self._relations.drop(rel)
        self._val_refs[vkey] -= 1
        if self._val_refs[vkey] <= 0:
            del self._val_refs[vkey]
            del self._val_atoms[vkey]
            for t in set(self._val_tokens.pop(vkey)):
                posting = self._token_index[t]
                posting.discard(vkey)
                if not posting:
                    del self._token_index[t]

    # ── records ─────────────────────────────────────────────────────

    def _bundle(self, vecs: np.ndarray, tie: np.ndarray | None) -> np.ndarray:
        """Bitwise majority of packed vectors (k x words). With an even k
        an exact tie takes the bit of `tie`."""
        k = len(vecs)
        if k == 1:
            return vecs[0].copy()
        bits = np.unpackbits(np.ascontiguousarray(vecs).view(np.uint8),
                             axis=1, count=self.dim)
        ones = np.add.reduce(bits, axis=0,
                             dtype=np.uint8 if k < 256 else np.uint32)
        half = k // 2
        out = ones > half
        if k % 2 == 0:
            tie_bits = np.unpackbits(tie.view(np.uint8), count=self.dim)
            out |= (ones == half) & tie_bits.astype(bool)
        packed = np.zeros(self.n_words * 8, dtype=np.uint8)
        p = np.packbits(out)
        packed[:len(p)] = p
        return packed.view(np.uint64)

    def _rebuild(self, ekey: str) -> None:
        """Re-bundle an entity's record(s) from its current facts.

        Ties are broken by the entity's own atom, not one global vector: a
        shared tie vector would make every even-sized record agree with
        every other on its tied bits, lifting unrelated records above
        chance similarity to each other and to a question probe."""
        for shard in range(len(self._entity_rows.pop(ekey, ()))):
            self._records.drop((ekey, shard))
        items = self._items.get(ekey)
        if not items:
            return
        keys = sorted(items)
        n_shards = -(-len(keys) // self.shard_size)
        rows = []
        for s in range(n_shards):
            chunk = keys[s::n_shards]
            vecs = np.stack([self._rel_atom(r) ^ self._val_atoms[v]
                             for r, v in chunk])
            tie = (_make_atom("entity", ekey if s == 0 else f"{ekey}#{s}",
                              self.dim, self.n_words)
                   if len(chunk) % 2 == 0 else None)
            rows.append(self._records.put(self._bundle(vecs, tie), (ekey, s)))
        self._entity_rows[ekey] = rows

    def _sync(self) -> None:
        """Bring every record up to date with the facts (lazy, so a bulk
        add rebuilds each entity once)."""
        if self._dirty:
            for ekey in sorted(self._dirty):
                self._rebuild(ekey)
            self._dirty.clear()

    def _hits(self, vec: np.ndarray, bank: _Bank) -> list[tuple[str, float]]:
        """Cleanup: labels of the bank rows within the acceptance bar of
        `vec`, most similar first."""
        sims = bank.similarity(vec)
        idx = np.flatnonzero(sims >= self.accept)
        idx = idx[np.argsort(-sims[idx], kind="stable")]
        return [(bank.labels[i], float(sims[i])) for i in idx]

    def _tally(self, verified: bool) -> None:
        self._counts["proposals"] += 1
        self._counts["verified" if verified else "rejected"] += 1

    def _cite(self, fids: list[int]) -> list[Fact]:
        return [self._facts[f] for f in fids]

    # ── names ───────────────────────────────────────────────────────

    def _register_entity(self, ekey: str, display: str) -> None:
        display = display.strip()
        self._names[ekey] = display
        title = norm_key(_DISAMBIGUATOR.sub("", display))
        if title and title != ekey:
            self._titles.setdefault(title, set()).add(ekey)
        for part in _name_parts(display):
            self._parts.setdefault(part, set()).add(ekey)

    def _unregister_entity(self, ekey: str) -> None:
        display = self._names.pop(ekey)
        self._dirty.add(ekey)          # so its record rows are dropped
        title = norm_key(_DISAMBIGUATOR.sub("", display))
        for index, keys in ((self._titles, [title]),
                            (self._parts, _name_parts(display))):
            for k in keys:
                holders = index.get(k)
                if holders is not None:
                    holders.discard(ekey)
                    if not holders:
                        del index[k]

    def _resolve_key(self, name) -> str | None:
        key = norm_key(name or "")
        if not key:
            return None
        if key in self._items:
            return key
        # alias facts, then "Mercury (planet)" as "Mercury", then a
        # person's first or last name - each only when it is unambiguous
        for index, people_only in ((self._aliases, False),
                                   (self._titles, False),
                                   (self._parts, True)):
            holders = [e for e in index.get(key, ())
                       if e in self._items
                       and (not people_only or self._kind(e) == "person")]
            if len(holders) == 1:
                return holders[0]
            if len(holders) > 1:
                return None
        return None

    def _subject_key(self, name) -> str | None:
        """An entity to ask about - or, failing that, a name only ever seen
        as a value ("Rome" in "Italy capital Rome"), which inverse lookups
        can still answer for ("what is Rome the capital of?")."""
        ekey = self._resolve_key(name)
        if ekey is None and norm_key(name or "") in self._val_atoms:
            ekey = norm_key(name)
        return ekey

    def resolve(self, name: str) -> str | None:
        """Canonical entity name for what the user called it, or None when
        unknown or ambiguous ("Curie" with two Curies stored)."""
        ekey = self._resolve_key(name)
        return self._names[ekey] if ekey is not None else None

    def _kind(self, ekey: str) -> str | None:
        """person / place / org / work / thing, from instance_of facts or
        the kinds of relations the entity has; None if nothing says."""
        items = self._items.get(ekey, {})
        for rel, vkey in items:
            if rel == "instance_of":
                toks = self._val_tokens.get(vkey) or _tokens(vkey)
                if toks and toks[-1] in _CLASS_KIND:
                    return _CLASS_KIND[toks[-1]]
        votes = Counter(RELATIONS[rel].subj_kind for rel, _ in items
                        if rel in RELATIONS
                        and RELATIONS[rel].subj_kind != "any")
        return votes.most_common(1)[0][0] if votes else None

    # ── the symbolic side ───────────────────────────────────────────

    def facts_about(self, entity: str) -> list[Fact]:
        """Every stored fact about an entity, in the order learned."""
        ekey = self._resolve_key(entity)
        if ekey is None:
            return []
        fids = sorted(f for slot in self._items[ekey].values() for f in slot)
        return self._cite(fids)

    def __len__(self) -> int:
        return len(self._facts)

    # ── lookup ──────────────────────────────────────────────────────

    def lookup(self, subject: str, relation: str, k: int = 5) -> list[Answer]:
        """"Where was Galileo born?": unbind the relation from the
        subject's record(s) and clean up against every value ever seen
        with that relation; keep the verified ones. For a relation with an
        inverse ("capital_of" / "capital") the facts stored from the other
        end count too, found by scanning the records for them."""
        ekey = self._subject_key(subject)
        if ekey is None or k <= 0:
            return []
        self._sync()
        self._counts["queries"] += 1
        return self._lookup_key(ekey, relation)[:k]

    def _lookup_key(self, ekey: str, rel: str) -> list[Answer]:
        out = self._unbind_lookup(ekey, rel)
        inv = _INVERSE.get(rel)
        if inv in self._codebooks and ekey in self._codebooks[inv].index:
            have = {norm_key(a.value): a for a in out}
            for ans in self._inverse_lookup(ekey, inv):
                same = have.get(norm_key(ans.value))
                if same is not None:
                    same.facts.extend(ans.facts)
                else:
                    out.append(ans)
        return out

    def _unbind_lookup(self, ekey: str, rel: str) -> list[Answer]:
        cb = self._codebooks.get(rel)
        rows = self._entity_rows.get(ekey)
        if cb is None or not rows:
            return []
        ratom = self._rel_atom(rel)
        best: dict[str, float] = {}
        for row in rows:
            for vkey, s in self._hits(self._records.words[row] ^ ratom, cb):
                best[vkey] = max(s, best.get(vkey, 0.0))
        items = self._items[ekey]
        out = []
        for vkey, s in sorted(best.items(), key=lambda kv: -kv[1]):
            fids = items.get((rel, vkey))
            self._tally(bool(fids))
            if fids:
                facts = self._cite(fids)
                out.append(Answer(facts[0].obj, s, facts))
        return out

    def _inverse_lookup(self, ekey: str, inv: str) -> list[Answer]:
        """Entities whose record holds bind(inv, this entity): "Rome" via
        inverse "capital" -> Italy (from "Italy capital Rome")."""
        probe = self._rel_atom(inv) ^ self._value_vec(ekey)
        out, seen = [], {ekey}
        for (other, _), s in self._hits(probe, self._records)[
                :CANDIDATE_POOL]:
            if other in seen:
                continue
            seen.add(other)
            fids = self._items.get(other, {}).get((inv, ekey))
            self._tally(bool(fids))
            if fids:
                out.append(Answer(self._names[other], s, self._cite(fids)))
        return out

    # ── who ─────────────────────────────────────────────────────────

    def who(self, constraints, k: int = 5, answer_kind: str = "any"
            ) -> list[Match]:
        """"Who was born in Pisa and taught at Padua?"

        Each constraint value is read loosely (and through containment:
        born in Pisa counts as born in Italy when Pisa is stored as being
        in Italy); its readings are superposed into one bound item, and the
        items into one question vector. One Hamming scan ranks every record
        against the question, and each constraint item is also scanned on
        its own - a superposition of A readings carries each at only about
        1/sqrt(A) strength, so a large record could otherwise slip under
        the bar. The entities proposed are verified constraint by
        constraint; full matches come first, then partial ones with their
        missing constraints listed."""
        cons = [(str(r), str(v)) for r, v in constraints
                if r and norm_key(str(v))]
        if not cons or k <= 0:
            return []
        self._sync()
        self._counts["queries"] += 1
        return self._who(cons, k, answer_kind)

    def _who(self, cons: list[tuple[str, str]], k: int, answer_kind: str
             ) -> list[Match]:
        n = self._records.n
        if n == 0:
            return []
        readings = [self._readings(rel, val) for rel, val in cons]
        overall = self._records.similarity(self._question(cons, readings))
        hit_count = np.zeros(n, dtype=np.int32)
        for (rel, val), alts in zip(cons, readings):
            best = np.zeros(n)
            for vkey in alts or [norm_key(val)]:
                np.maximum(best, self._records.similarity(
                    self._rel_atom(rel) ^ self._value_vec(vkey)), out=best)
            hit_count += best >= self.accept
        proposed = np.flatnonzero((hit_count > 0) | (overall >= self.accept))
        by_entity: dict[str, list] = {}
        for row in proposed:
            ekey = self._records.labels[row][0]
            got = by_entity.setdefault(ekey, [0, 0.0])
            got[0] += int(hit_count[row])
            got[1] = max(got[1], float(overall[row]))
        ranked = sorted(by_entity.items(),
                        key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]))
        full, partial = [], []
        for ekey, (_, score) in ranked[:max(CANDIDATE_POOL, 16 * k)]:
            satisfied, missing, n_ok = self._verify(ekey, cons, readings)
            self._tally(n_ok > 0)
            if not n_ok:
                continue
            if answer_kind not in ("any", "", None):
                kind = self._kind(ekey)
                if kind is not None and kind != answer_kind:
                    continue
            m = Match(self._names[ekey], score, satisfied, missing)
            rank = (-n_ok, -len(self._items[ekey]), -score, ekey)
            (partial if missing else full).append((rank, m))
        full.sort(key=lambda t: t[0])
        partial.sort(key=lambda t: t[0])
        return [m for _, m in full + partial][:k]

    def _question(self, cons, readings) -> np.ndarray:
        """bundle over constraints of bind(REL, bundle(readings))."""
        items = []
        for (rel, val), alts in zip(cons, readings):
            vecs = np.stack([self._value_vec(v)
                             for v in (list(alts) or [norm_key(val)])])
            alt = self._bundle(vecs, self._atom("tie", "readings"))
            items.append(self._rel_atom(rel) ^ alt)
        return self._bundle(np.stack(items), self._atom("tie", "question"))

    def _readings(self, rel: str, user_value: str) -> dict[str, list[Fact]]:
        """Stored values of `rel` the user's words may mean -> the facts
        that justify the reading ([] for a direct match; the containment
        path for "Italy" read as "Pisa")."""
        cb = self._codebooks.get(rel)
        if cb is None:
            return {}
        out: dict[str, list[Fact]] = {
            v: [] for v in self._loose_values(user_value, cb.index)}
        if rel in _CONTAINABLE:
            for ekey, path in self._places_within(user_value).items():
                if ekey in cb.index and ekey not in out:
                    out[ekey] = path
        return dict(list(out.items())[:MAX_ALTERNATIVES])

    def _exact_values(self, user_value: str) -> list[str]:
        """Value keys that name exactly what the user named: the words
        themselves, and the entity they resolve to ("Shakespeare" ->
        "william shakespeare") when that is unambiguous."""
        uk = norm_key(user_value)
        ek = self._resolve_key(user_value)
        return [uk] if ek is None or ek == uk else [uk, ek]

    def _loose_values(self, user_value: str, within) -> list[str]:
        """Value keys in `within` that loosely match the user's words:
        exact readings first, then the closest (fewest extra words)."""
        out = [v for v in self._exact_values(user_value) if v in within]
        ut = _tokens(norm_key(user_value))
        content = [t for t in ut if t not in _GENERIC]
        if not content:
            return out
        # every match contains every content word: scan the rarest's posting
        posting = min((self._token_index.get(t, set()) for t in content),
                      key=len)
        more = [v for v in posting if v not in out and v in within
                and _tokens_match(ut, self._val_tokens[v])]
        more.sort(key=lambda v: (len(self._val_tokens[v]), v))
        return out + more

    def _verify(self, ekey: str, cons, readings
                ) -> tuple[list[Fact], list[tuple[str, str]], int]:
        """Which constraints the entity's stored facts back up."""
        items = self._items[ekey]
        satisfied: list[Fact] = []
        missing: list[tuple[str, str]] = []
        n_ok = 0
        for (rel, val), alts in zip(cons, readings):
            got = None
            for vkey, path in alts.items():
                fids = items.get((rel, vkey))
                if fids:
                    got = self._cite(fids) + path
                    break
            if got is None:            # a reading beyond the probed few
                ut = _tokens(norm_key(val))
                for (r, vkey), fids in items.items():
                    if r == rel and _tokens_match(ut, self._val_tokens[vkey]):
                        got = self._cite(fids)
                        break
            if got is None:
                missing.append((rel, val))
            else:
                satisfied.extend(got)
                n_ok += 1
        return satisfied, missing, n_ok

    # ── containment (places inside places) ──────────────────────────

    def _places_within(self, user_value: str,
                       depth: int = CONTAINMENT_DEPTH
                       ) -> dict[str, list[Fact]]:
        """Entities the memory places inside the user's place, each with
        the facts that show it: "Italy" -> {"pisa": [Pisa country Italy]}.
        Each step is an HDC scan of every record for bind(EDGE, PLACE),
        plus an unbind of the place's capital ("Italy capital Rome" puts
        Rome inside Italy). A budget of CONTAINMENT_SCANS scans keeps a
        vague place ("Europe") from turning one question into hundreds."""
        found: dict[str, list[Fact]] = {}
        start = self._loose_values(user_value, self._val_atoms)
        ek = self._resolve_key(user_value)
        if ek is not None and ek not in start:
            start.insert(0, ek)
        level = {p: [] for p in start[:MAX_ALTERNATIVES]}
        budget = CONTAINMENT_SCANS
        for _ in range(depth):
            new: dict[str, list[Fact]] = {}
            for place, path in level.items():
                for c in _CONTAINER_EDGES:
                    cb = self._codebooks.get(c)
                    if cb is None or place not in cb.index or budget <= 0:
                        continue
                    budget -= 1
                    probe = self._rel_atom(c) ^ self._value_vec(place)
                    for (ekey, _), _s in self._hits(probe, self._records)[
                            :CANDIDATE_POOL]:
                        if ekey in found or ekey in level:
                            continue
                        fids = self._items.get(ekey, {}).get((c, place))
                        self._tally(bool(fids))
                        if fids:
                            found[ekey] = new[ekey] = \
                                [self._facts[fids[0]]] + path
                if place in self._items:
                    for a in self._unbind_lookup(place, "capital"):
                        cap = norm_key(a.value)
                        if cap not in found and cap not in level:
                            found[cap] = new[cap] = a.facts[:1] + path
            if not new:
                break
            level = new
        return found

    def _container_path(self, value: str, target: str,
                        depth: int = CONTAINMENT_DEPTH) -> list[Fact] | None:
        """Facts showing the place `value` lies inside `target` ("Pisa"
        country "Italy"), following containment edges by HDC lookups
        (including inverse ones: "Iceland capital Reykjavík")."""
        start = self._subject_key(value)
        if start is None:
            return None
        frontier = [(start, [])]
        seen = {start}
        for _ in range(depth):
            nxt = []
            for ekey, path in frontier:
                for c in _CONTAINER_EDGES:
                    for a in self._lookup_key(ekey, c):
                        p = path + a.facts[:1]
                        if loose_match(target, a.value):
                            return p
                        e2 = self._subject_key(a.value)
                        if e2 is not None and e2 not in seen:
                            seen.add(e2)
                            nxt.append((e2, p))
            frontier = nxt
        return None

    def _disjoint_places(self, stored: str, user: str) -> bool:
        """Two different known places of the same class, neither inside
        the other - e.g. two cities. Only then does "born in Pisa" rule
        out "born in Florence"."""
        e1, e2 = self._resolve_key(stored), self._resolve_key(user)
        if e1 is None or e2 is None or e1 == e2:
            return False
        if self._container_path(stored, user) or \
                self._container_path(user, stored):
            return False
        c1 = {norm_key(a.value) for a in self._lookup_key(e1, "instance_of")}
        c2 = {norm_key(a.value) for a in self._lookup_key(e2, "instance_of")}
        return bool(c1 & c2)

    # ── analogy, chains, checks ─────────────────────────────────────

    def _links(self, ekey: str, value: str
               ) -> list[tuple[str, float, list[Fact]]]:
        """Relations linking the entity to a value: unbind the value's atom
        from the record(s) and clean up against the relation atoms. An
        exact reading of the value wins; loose readings only if none."""
        rows = self._entity_rows.get(ekey)
        if not rows:
            return []
        items = self._items[ekey]
        readings = self._loose_values(value, self._val_atoms)
        named = self._exact_values(value)
        exact = [v for v in readings if v in named]
        for group in (exact, [v for v in readings if v not in exact]):
            found: dict[str, tuple[float, list[Fact]]] = {}
            for vkey in group:
                vec = self._value_vec(vkey)
                for row in rows:
                    for rel, s in self._hits(self._records.words[row] ^ vec,
                                             self._relations):
                        fids = items.get((rel, vkey))
                        self._tally(bool(fids))
                        if fids and s > found.get(rel, (0.0,))[0]:
                            found[rel] = (s, self._cite(fids))
            if found:
                return sorted(((r, s, f) for r, (s, f) in found.items()),
                              key=lambda t: -t[1])
        return []

    def analogy(self, a: str, b: str, c: str, k: int = 5) -> list[Answer]:
        """"Galileo is to Pisa as Newton is to ?" - find the relation that
        links a to b (unbind b from a's record, clean up against relation
        atoms), then look that relation up for c. If b is the entity
        ("Pisa is to Galileo as Woolsthorpe is to ?") the link runs from b
        to a, and the answer is whoever has that relation to c.
        Answer.facts holds the a-b fact and the c-answer fact(s)."""
        if not all(norm_key(x or "") for x in (a, b, c)) or k <= 0:
            return []
        self._sync()
        self._counts["queries"] += 1
        out: list[Answer] = []
        ea, ec = self._resolve_key(a), self._resolve_key(c)
        if ea is not None and ec is not None and ea != ec:
            for rel, _, link in self._links(ea, b):
                for ans in self._lookup_key(ec, rel):
                    out.append(Answer(ans.value, ans.score, link + ans.facts))
        eb = self._resolve_key(b)
        if not out and eb is not None:
            for rel, _, link in self._links(eb, a):
                for m in self._who([(rel, c)], k, "any"):
                    if not m.missing and norm_key(m.entity) != eb:
                        out.append(Answer(m.entity, m.score,
                                          link + m.satisfied))
        seen: set[str] = set()
        unique = []
        for ans in out:
            if norm_key(ans.value) not in seen:
                seen.add(norm_key(ans.value))
                unique.append(ans)
        return unique[:k]

    def chain(self, subject: str, relations: list[str], k: int = 5
              ) -> list[Answer]:
        """Multi-hop: "In what country was Galileo born?" ->
        chain("Galileo", ["born_in", "country"]). Each hop is an HDC lookup
        from the entity the previous answer names; Answer.facts holds the
        whole path and the score is its weakest hop."""
        ekey = self._subject_key(subject)
        if ekey is None or not relations or k <= 0:
            return []
        self._sync()
        self._counts["queries"] += 1
        paths: list[tuple[str, float, list[Fact]]] = [(ekey, 1.0, [])]
        for hop, rel in enumerate(relations):
            last = hop == len(relations) - 1
            nxt = []
            for ent, score, facts in paths:
                for ans in self._lookup_key(ent, rel):
                    step = (min(score, ans.score), facts + ans.facts)
                    if last:
                        nxt.append((ans.value,) + step)
                        continue
                    e2 = self._subject_key(ans.value)
                    if e2 is not None and e2 != ent:
                        nxt.append((e2,) + step)
            paths = sorted(nxt, key=lambda p: -p[1])[:BEAM]
            if not paths:
                return []
        out, seen = [], set()
        for value, score, facts in paths:
            if norm_key(value) not in seen:
                seen.add(norm_key(value))
                out.append(Answer(value, score, facts))
        return out[:k]

    def check(self, subject: str, relation: str, obj: str
              ) -> tuple[bool | None, list[Fact]]:
        """"Was Galileo born in Pisa?" -> (True, facts); (False, facts)
        when a single-valued relation verifiably holds something else;
        (None, []) when the memory can't tell. Places nest, so born in
        Pisa also answers "born in Italy?" when Pisa is stored as in Italy,
        and "born in Florence?" is a "no" only if Florence is a stored city
        disjoint from Pisa. A name the memory has never seen is never a
        "no": "Roma" may just be another name for the stored Rome."""
        ekey = self._subject_key(subject)
        if ekey is None or not norm_key(obj or ""):
            return None, []
        self._sync()
        self._counts["queries"] += 1
        answers = self._lookup_key(ekey, relation)
        same = self._resolve_key(obj)
        agree = [f for a in answers
                 if _agrees(relation, obj, a.value)
                 or (same is not None and self._resolve_key(a.value) == same)
                 for f in a.facts]
        if agree:
            return True, agree
        if relation in _CONTAINABLE:
            for a in answers:
                path = self._container_path(a.value, obj)
                if path:
                    return True, a.facts + path
        spec = RELATIONS.get(relation)
        if not answers or spec is None or spec.multi or spec.internal:
            return None, []
        for a in answers:
            if relation in _NESTED_SINGLE:
                ruled_out = self._disjoint_places(a.value, obj)
            elif spec.obj_kind in ("date", "year", "number"):
                ruled_out = _contradicts(relation, obj, a.value)
            else:
                ruled_out = self._known(obj)
            if not ruled_out:
                return None, []
        return False, [f for a in answers for f in a.facts]

    def _known(self, name: str) -> bool:
        """Has the memory met this name - as an entity or a stored value?"""
        key = norm_key(name)
        return (key in self._items or key in self._val_atoms
                or self._resolve_key(name) is not None)

    # ── describe, compare ───────────────────────────────────────────

    def _decode(self, ekey: str) -> set[tuple[str, str]]:
        """Read a record back: unbind every relation atom and clean up
        against that relation's values; keep what the index confirms."""
        items = self._items.get(ekey, {})
        found: set[tuple[str, str]] = set()
        for row in self._entity_rows.get(ekey, ()):
            rec = self._records.words[row]
            for rel, cb in self._codebooks.items():
                for vkey, _ in self._hits(rec ^ self._rel_atom(rel), cb):
                    ok = (rel, vkey) in items
                    self._tally(ok)
                    if ok:
                        found.add((rel, vkey))
        return found

    def describe(self, subject: str) -> list[Fact]:
        """Everything the record says about the subject, decoded by HDC and
        verified - in the order learned (it reproduces facts_about() while
        the record is healthy)."""
        ekey = self._resolve_key(subject)
        if ekey is None:
            return []
        self._sync()
        self._counts["queries"] += 1
        found = self._decode(ekey)
        fids = sorted(f for key, slot in self._items[ekey].items()
                      if key in found for f in slot)
        return self._cite(fids)

    def compare(self, a: str, b: str) -> list[tuple[str, str, list[Fact]]]:
        """"What do Galileo and Kepler have in common?" -> (relation,
        shared value, facts from both) for every relation-value pair both
        records decode to."""
        ea, eb = self._resolve_key(a), self._resolve_key(b)
        if ea is None or eb is None or ea == eb:
            return []
        self._sync()
        self._counts["queries"] += 1
        shared = self._decode(ea) & self._decode(eb)
        out = []
        for rel, vkey in sorted(shared, key=lambda p: (
                _RELATION_ORDER.get(p[0], len(_RELATION_ORDER)), p)):
            spec = RELATIONS.get(rel)
            if spec is not None and spec.internal:
                continue
            fa = self._cite(self._items[ea][(rel, vkey)])
            fb = self._cite(self._items[eb][(rel, vkey)])
            out.append((rel, fa[0].obj, fa + fb))
        return out

    # ── bookkeeping ─────────────────────────────────────────────────

    def stats(self) -> dict:
        """Sizes, the thresholds in force, and how HDC proposals fared:
        `rejected` counts proposals the stored facts did not confirm."""
        self._sync()
        return {
            "dim": self.dim,
            "bytes_per_record": self.n_words * 8,
            "facts": len(self._facts),
            "entities": len(self._items),
            "records": len(self._records),
            "sharded_entities": sum(1 for rows in self._entity_rows.values()
                                    if len(rows) > 1),
            "relations": len(self._codebooks),
            "values": len(self._val_atoms),
            "relation_values": sum(len(cb) for cb in self._codebooks.values()),
            "shard_size": self.shard_size,
            "accept_similarity": round(self.accept, 4),
            **self._counts,
        }
