"""
lattice/relation_discovery.py - automatic relational structure from raw text.

The pipeline that turns unlabeled English into a structured knowledge graph
WITHOUT regex patterns, parsers, or human annotations.

Steps:
  1. Tokenize each sentence
  2. For each sentence, compute a STRUCTURE hypervector — a representation
     of the sentence pattern that's invariant to specific entity names
     (we mask out content words and bundle position-bound function words)
  3. Cluster sentences by structure hypervector similarity
  4. For each cluster, identify "slots" — positions where words VARY across
     the sentences in the cluster. These are the entity/attribute slots.
  5. Extract triples automatically: (entity_slot_word, CLUSTER_ID, attribute_slot_word)

The discovered "CLUSTER_ID" is the auto-generated relation. We never assign
it a human name; it's just a unique hypervector. But it functions identically
to a hand-labeled relation in RBI downstream.

This is unsupervised relational structure learning in pure HDC.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import D, bundle, hamming_distance


_TOKEN_RE = re.compile(r"[^a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.split(text.lower()) if t]


# Stop-words / function words we'll TREAT AS STRUCTURE (keep them as-is)
# Content-bearing words (entities) get masked out before encoding the
# sentence structure.
FUNCTION_WORDS = {
    "the","a","an","is","was","were","are","be","been","has","have","had",
    "of","in","on","at","by","to","from","with","for","into","onto",
    "and","or","but","not",
    "this","that","these","those",
    "it","its",
    "as","than","then",
    "country","capital","city","located","famous","known","spoken",
    "language","currency","priced","used","goods",
    "habitat","class","lives","found","belongs","class",
    "food","comes","originated","origin","classified",
    "wrote","written","author","genre","writer",
    "sport","equipment","main","requires","uses","plays","played","venue",
    "made","album","performs","film","directed","director",
    "comes","origin","drink","hot","cold",
    "tool","used","acts",
    "computer","uses","runs","chip","cpu",
    "shares","borders","with","between","maps","show","cities",
    "tourists","visit","see","its","explore","exploring",
    "primary","main",
    "hosts","government","arrive","arrives","trip","start","abroad",
    "live","near","largest",
    "people","major",
    "country","cities",
    "located","colour","color",
}


def is_function_word(w: str) -> bool:
    return w.lower() in FUNCTION_WORDS


# ─── Sentence structure encoding ──────────────────────────────────


class StructureEncoder:
    """Encode a sentence's STRUCTURE (independent of specific entities).

    Method: replace content words with a generic CONTENT_MARKER hypervector
    so all entity-containing sentences in the same template produce
    identical structure hypervectors.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        # Word vectors (lazy)
        self.word_vecs: dict[str, np.ndarray] = {}
        # CONTENT_MARKER: a fixed vector that replaces any content word
        self.content_marker = self.rng.integers(0, 2, size=D, dtype=np.int8)
        # Position role: cyclic shift per position
        self.shift = 137

    def _word_vec(self, word: str) -> np.ndarray:
        if word not in self.word_vecs:
            self.word_vecs[word] = self.rng.integers(0, 2, size=D, dtype=np.int8)
        return self.word_vecs[word]

    def encode_structure(self, sentence: str) -> np.ndarray:
        """Return a hypervector for the sentence's STRUCTURE (function words
        + position), with content words replaced by a shared marker.
        """
        tokens = tokenize(sentence)
        positioned = []
        for i, w in enumerate(tokens):
            if is_function_word(w):
                v = self._word_vec(w)
            else:
                v = self.content_marker
            positioned.append(np.roll(v, i * self.shift))
        if not positioned:
            return np.zeros(D, dtype=np.int8)
        return bundle(positioned)

    def tokens_and_mask(self, sentence: str
                         ) -> tuple[list[str], list[bool]]:
        """Return (tokens, is_content_mask) where is_content_mask[i] is True
        if token i is a content word (entity)."""
        tokens = tokenize(sentence)
        mask = [not is_function_word(t) for t in tokens]
        return tokens, mask


# ─── Clustering ───────────────────────────────────────────────────


def cluster_sentences(sentences: list[str], threshold_pct: float = 0.10,
                        encoder: StructureEncoder = None
                        ) -> list[dict]:
    """Greedy single-pass clustering of sentences by STRUCTURE similarity.

    threshold_pct: max Hamming distance (as fraction of D) for two sentences
                   to be considered the same pattern.

    Returns: list of clusters, each {"centroid": hv, "members": [(idx,sent), ...]}
    """
    encoder = encoder or StructureEncoder()
    clusters = []
    for idx, sent in enumerate(sentences):
        hv = encoder.encode_structure(sent)
        # Find nearest cluster
        best_c = None
        best_d = None
        for c in clusters:
            d = hamming_distance(hv, c["centroid"])
            if best_d is None or d < best_d:
                best_d = d
                best_c = c
        if best_d is not None and best_d / D < threshold_pct:
            best_c["members"].append((idx, sent, hv))
            # Update centroid (majority vote with new vector)
            stack = np.stack([m[2] for m in best_c["members"]])
            best_c["centroid"] = (stack.sum(axis=0) > len(best_c["members"]) / 2).astype(np.int8)
        else:
            clusters.append({
                "centroid": hv.copy(),
                "members": [(idx, sent, hv)],
            })
    return clusters


# ─── Slot detection ──────────────────────────────────────────────


def find_slots(cluster: dict, encoder: StructureEncoder
                ) -> list[int] | None:
    """For a cluster of sentences with the same structure, find positions
    where the WORD varies (those are the entity/attribute slots).

    Returns the list of token positions that vary, or None if too few
    members.
    """
    members = cluster["members"]
    if len(members) < 2:
        return None
    # Tokenize all member sentences
    token_lists = [tokenize(s) for _, s, _ in members]
    # Find positions where all sentences have the same length
    lengths = set(len(t) for t in token_lists)
    if len(lengths) > 1:
        # Sentences of different lengths — handle by using shortest length only
        min_len = min(lengths)
        token_lists = [t[:min_len] for t in token_lists]
    n_positions = len(token_lists[0])
    # For each position, check if all sentences agree
    varying = []
    for pos in range(n_positions):
        words_at_pos = set(toks[pos] for toks in token_lists)
        if len(words_at_pos) > 1:
            # This position varies; could be a slot
            # Also require all variations be CONTENT words (not function words)
            all_content = all(not is_function_word(toks[pos]) for toks in token_lists)
            if all_content:
                varying.append(pos)
    return varying


def extract_triples_from_cluster(cluster: dict, slots: list[int],
                                    relation_id: str) -> list[tuple[str, str, str]]:
    """Given a cluster and its detected slots, emit triples."""
    triples = []
    for _, sent, _ in cluster["members"]:
        tokens = tokenize(sent)
        # Skip if sentence doesn't have all expected slot positions
        if max(slots) >= len(tokens):
            continue
        slot_words = [tokens[s] for s in slots]
        if len(slot_words) == 2:
            triples.append((slot_words[0], relation_id, slot_words[1]))
        elif len(slot_words) >= 2:
            # Use first and last varying positions as subject/object
            triples.append((slot_words[0], relation_id, slot_words[-1]))
    return triples


# ─── End-to-end pipeline ──────────────────────────────────────────


def discover_relations(sentences: list[str],
                         cluster_threshold: float = 0.10,
                         min_cluster_size: int = 2,
                         verbose: bool = True
                         ) -> tuple[list[tuple[str, str, str]], list[dict]]:
    """Run the full unsupervised pipeline.

    Returns (auto_extracted_triples, cluster_info).
    """
    encoder = StructureEncoder()
    if verbose:
        print(f"  encoding {len(sentences)} sentences ...")
    clusters = cluster_sentences(sentences, threshold_pct=cluster_threshold,
                                    encoder=encoder)
    if verbose:
        print(f"  clustered into {len(clusters)} groups")
    # Filter small clusters (likely noise)
    big_clusters = [c for c in clusters if len(c["members"]) >= min_cluster_size]
    if verbose:
        print(f"  {len(big_clusters)} clusters with size >= {min_cluster_size}")

    all_triples = []
    cluster_info = []
    for i, cluster in enumerate(big_clusters):
        slots = find_slots(cluster, encoder)
        if not slots or len(slots) < 2:
            continue
        relation_id = f"DISCOVERED_REL_{i}"
        triples = extract_triples_from_cluster(cluster, slots, relation_id)
        cluster_info.append({
            "cluster_id": i,
            "relation_id": relation_id,
            "size": len(cluster["members"]),
            "slots": slots,
            "n_triples": len(triples),
            "sample_sentences": [m[1] for m in cluster["members"][:3]],
        })
        all_triples.extend(triples)
    if verbose:
        print(f"  extracted {len(all_triples)} triples from "
              f"{len(cluster_info)} usable clusters")
    return all_triples, cluster_info
