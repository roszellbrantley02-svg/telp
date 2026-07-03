"""
lattice/multi_sense_encoder.py - context-disambiguated HDC word vectors.

Solves the polysemy problem in word2vec-style flat-vocab embeddings.
Each word that appears in different contexts gets MULTIPLE sense vectors,
one per discovered context cluster.  At inference, we pick the sense
whose discovered context best matches the query's surrounding words.

This means:
  - "resistance" in "should I buy at resistance?" picks the trading sense
  - "resistance" in "the political resistance grew" picks the political sense
  - "candle" in "what is a hammer candle pattern?" picks the candlestick sense
  - "candle" in "she lit a candle on the table" picks the literal sense

Built on top of HDCNativeEncoder.  Same fast GPU pipeline.
Drop-in compatible at the encode() and encode_word() interface.

Architecture:

    For each word w with frequency >= MIN_FREQ_FOR_SENSES:
        Find every sentence containing w
        For each occurrence:
            context_hv = bundle of (other words in same sentence)
        Cluster the context_hvs via K-means (k=K_SENSES)
        Each cluster centroid = one "sense context" for w
        Each cluster has its own derived word vector

    At encode(query) time:
        query_context_hv = bundle of all query words via base encoder
        For each word w in the query:
            if w has multiple senses:
                pick the sense whose context centroid is closest to query_context_hv
            else:
                use w's base vector
        Bundle the selected vectors
"""
from __future__ import annotations

import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.hdc_native_encoder import (
    HDCNativeEncoder,
    tokenize,
    _unknown_word_hv,
)


# ─── Tunables ──────────────────────────────────────────────────────


MIN_FREQ_FOR_SENSES = 200   # higher threshold → fewer candidate words
K_SENSES            = 3      # candidate senses per polysemous word
MAX_CONTEXTS_PER_WORD = 80   # smaller cap — k-means doesn't need more
MIN_CONTEXTS_TO_SPLIT = 15   # minimum contexts to attempt clustering
CONTEXT_WINDOW      = 18     # wider window — more signal for disambiguation


# Common English stopwords — drop these from contexts before bundling
# so clusters form around content words, not "and/the/with/that".
STOPWORDS = frozenset("""
    a an the and or but if then else when while of in to on at by for with
    from into onto out off about over under up down as is are was were be
    been being have has had do does did so not no nor only also too very
    just both either neither each every all any some most more less many
    much such own same other another which who whom whose what why how here
    there now still yet again once twice ever never always sometimes often
    rarely usually i me my mine you your yours he him his she her hers it
    its we us our ours they them their theirs this that these those those
    am can could will would shall should may might must would did do does
    doing done having had had being been been over after before during between
    against among around through across along beside besides beyond inside
    outside within without upon toward towards via per than then thus hence
    therefore however moreover furthermore meanwhile otherwise nevertheless
    although though because since unless until while whereas when where why
    how whether yes okay ok please thanks sorry well good bad new old high
    low small large big little long short
""".split())


# ─── Multi-sense encoder ───────────────────────────────────────────


class MultiSenseHDCEncoder:
    """Wraps an HDCNativeEncoder with per-word sense vectors."""

    def __init__(self, base_encoder: HDCNativeEncoder,
                   min_freq_for_senses: int = MIN_FREQ_FOR_SENSES,
                   k_senses: int = K_SENSES):
        self.base = base_encoder
        self.dim = base_encoder.dim
        self.min_freq_for_senses = min_freq_for_senses
        self.k_senses = k_senses
        # word -> list[ (sense_word_hv, sense_context_centroid_hv) ]
        # both are np.ndarray int8 of size (dim,)
        self.senses: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        # Mirror the base interface
        self.vocab = base_encoder.vocab

    # ── Compatibility shims so the adapter works ─────────────

    @property
    def id2word(self):
        return self.base.id2word

    @property
    def word_counts(self):
        return self.base.word_counts

    @property
    def _bin_vectors(self):
        return self.base._bin_vectors

    @property
    def window(self):
        return self.base.window

    # ── Discovery (slow but one-time) ────────────────────────

    def discover(self, sentences: list[str],
                       verbose: bool = True) -> dict:
        t0 = time.time()
        # 1) Find candidate words (freq >= threshold)
        candidates = [
            w for w in self.base.id2word
            if self.base.word_counts.get(w, 0) >= self.min_freq_for_senses
        ]
        if verbose:
            print(f"[multi-sense] candidate polysemous words: "
                    f"{len(candidates):,} "
                    f"(min_freq={self.min_freq_for_senses})", flush=True)

        # 2) Get base vectors as a single numpy array (avoid .numpy() per
        #    token, which allocates fresh arrays).  Reference into bin_vectors
        #    rather than copying.
        bin_vectors_np = self.base._bin_vectors.numpy()    # (V, D) int8
        vocab = self.base.vocab
        dim = self.dim
        cand_set = set(candidates)

        # IDF weights: rare words count MORE in the context bundle so
        # they dominate clustering.  Computed from word_counts.
        # idf(w) = log( total_tokens / freq(w) ) — clipped to [0.3, 5.0]
        import math
        total_tokens = max(1, sum(self.base.word_counts.values()))
        idf_weight: dict[int, int] = {}
        for w, idx in vocab.items():
            if w in STOPWORDS:
                idf_weight[idx] = 0     # zero weight → effectively skipped
            else:
                f = self.base.word_counts.get(w, 1)
                idf = math.log(total_tokens / max(f, 1))
                # Scale to integer weight in [1, 8] for our int32 accumulator
                w_int = int(max(1, min(8, round(idf - 4.0))))
                idf_weight[idx] = w_int

        # Pre-allocate ONE accumulator buffer — reuse it every iteration
        # instead of np.zeros every occurrence.
        acc_buf = np.zeros(dim, dtype=np.int32)

        # Each candidate gets a fixed-size pre-allocated array.  We track
        # how many slots are filled per word (slot_count) so we don't
        # overflow.  This is far cheaper than growing lists.
        n_cand = len(candidates)
        ctx_storage = np.zeros(
            (n_cand, MAX_CONTEXTS_PER_WORD, dim), dtype=np.int8
        )
        slot_count = np.zeros(n_cand, dtype=np.int32)
        cand_id_of: dict[str, int] = {w: i for i, w in enumerate(candidates)}

        if verbose:
            mem_mb = ctx_storage.nbytes / (1024 ** 2)
            print(f"[multi-sense] pre-allocated context storage: "
                    f"{mem_mb:,.0f} MB "
                    f"({n_cand} candidates × {MAX_CONTEXTS_PER_WORD} "
                    f"× {dim}b)", flush=True)

        from random import Random
        rng = Random(42)
        for i, s in enumerate(sentences):
            toks = tokenize(s)
            if not toks:
                continue
            n = len(toks)
            # First pass: find positions of candidate words in this sentence
            cand_positions = []
            for j, t in enumerate(toks):
                if t in cand_set:
                    cand_positions.append((j, t))
            if not cand_positions:
                continue
            # For each candidate occurrence, build context on demand by
            # looking up only the window's worth of base vectors.
            for j, t in cand_positions:
                cid = cand_id_of[t]
                cur_count = slot_count[cid]
                # Reservoir replacement when full
                slot_idx: int
                if cur_count >= MAX_CONTEXTS_PER_WORD:
                    if rng.random() > 0.1:
                        continue
                    slot_idx = rng.randrange(MAX_CONTEXTS_PER_WORD)
                else:
                    slot_idx = cur_count
                lo = max(0, j - CONTEXT_WINDOW)
                hi = min(n, j + CONTEXT_WINDOW + 1)
                acc_buf.fill(0)
                weight_sum = 0
                for jj in range(lo, hi):
                    if jj == j:
                        continue
                    other = toks[jj]
                    if other not in vocab:
                        continue
                    other_idx = vocab[other]
                    w_idf = idf_weight[other_idx]
                    if w_idf == 0:
                        continue  # stopword
                    # IDF-weighted accumulation — rare content words
                    # dominate, common ones contribute less or nothing.
                    acc_buf += bin_vectors_np[other_idx] * w_idf
                    weight_sum += w_idf
                if weight_sum == 0:
                    continue
                # Threshold at half of total weight to binarize
                ctx_storage[cid, slot_idx] = (acc_buf > weight_sum / 2.0).astype(np.int8)
                if cur_count < MAX_CONTEXTS_PER_WORD:
                    slot_count[cid] = cur_count + 1
            if verbose and (i + 1) % 50_000 == 0:
                print(f"[multi-sense]   scanned {i+1:,}/{len(sentences):,} "
                        f"sentences", flush=True)

        # Build contexts dict from the storage (only what's filled)
        contexts: dict[str, np.ndarray] = {}
        for w in candidates:
            cid = cand_id_of[w]
            sc = int(slot_count[cid])
            if sc >= MIN_CONTEXTS_TO_SPLIT:
                contexts[w] = ctx_storage[cid, :sc].copy()
        # Free the giant pre-allocation
        del ctx_storage
        import gc
        gc.collect()

        # 3) Cluster each candidate's contexts
        from sklearn.cluster import MiniBatchKMeans
        n_split = 0
        for w in candidates:
            if w not in contexts:
                continue
            ctx_arr = contexts[w].astype(np.float32)
            try:
                km = MiniBatchKMeans(
                    n_clusters=self.k_senses,
                    random_state=42,
                    batch_size=64,
                    n_init=3,
                    max_iter=50,
                )
                labels = km.fit_predict(ctx_arr)
                centroids = km.cluster_centers_
                # Build per-sense word_hv as the bundle of the base
                # word_hv with each sense's context centroid (binarize).
                base_w_hv = self.base._bin_vectors[
                    self.base.vocab[w]].numpy()
                sense_pairs = []
                for ci in range(self.k_senses):
                    if (labels == ci).sum() < 3:
                        continue
                    # Sense word_hv = base word XOR-bound with context
                    # centroid (binarized).  This creates a context-
                    # conditioned vector that still anchors to the word.
                    ctx_bin = (centroids[ci] > 0.5).astype(np.int8)
                    sense_w_hv = np.bitwise_xor(base_w_hv, ctx_bin)
                    sense_pairs.append((sense_w_hv, ctx_bin))
                if len(sense_pairs) >= 2:
                    self.senses[w] = sense_pairs
                    n_split += 1
            except Exception as e:
                if verbose:
                    print(f"[multi-sense]   {w} cluster failed: {e}")
                continue

        elapsed = time.time() - t0
        if verbose:
            print(f"[multi-sense] discovered senses for {n_split:,} words "
                    f"in {elapsed:.1f}s", flush=True)
        return {
            "candidates":    len(candidates),
            "with_senses":   n_split,
            "k_senses":      self.k_senses,
            "elapsed_s":     round(elapsed, 1),
        }

    # ── Inference ────────────────────────────────────────────

    def _query_context_hv(self, toks: list[str]) -> np.ndarray:
        if not toks:
            return np.zeros(self.dim, dtype=np.int8)
        acc = np.zeros(self.dim, dtype=np.int32)
        for t in toks:
            if t in self.base.vocab:
                acc += self.base._bin_vectors[
                    self.base.vocab[t]].numpy().astype(np.int32)
            else:
                acc += _unknown_word_hv(t, self.dim).astype(np.int32)
        return (acc > len(toks) / 2.0).astype(np.int8)

    def encode_word_with_context(self, word: str,
                                          context_hv: np.ndarray) -> np.ndarray:
        """Return the word's vector selected by best context match."""
        if word not in self.senses:
            return self.base.encode_word(word)
        # Pick sense whose context centroid is most similar to query
        best_sim = -1.0
        best_hv = None
        for sense_hv, sense_ctx in self.senses[word]:
            ham = int(np.bitwise_xor(sense_ctx, context_hv).sum())
            sim = 1.0 - 2.0 * ham / self.dim
            if sim > best_sim:
                best_sim = sim
                best_hv = sense_hv
        return best_hv if best_hv is not None else self.base.encode_word(word)

    def encode_word(self, word: str) -> np.ndarray:
        """Compatibility shim — returns base vector when no context."""
        return self.base.encode_word(word)

    def encode(self, text: str) -> np.ndarray:
        """Two-pass context-aware encoding.

        Pass 1: bundle every word's base vector to get the rough context.
        Pass 2: for each word, pick the sense whose context matches the
                rough context best; bundle those.
        """
        toks = tokenize(text)
        if not toks:
            return np.zeros(self.dim, dtype=np.int8)
        # Pass 1: rough context
        rough_ctx = self._query_context_hv(toks)
        # Pass 2: re-encode each word selecting best sense
        acc = np.zeros(self.dim, dtype=np.int32)
        n = 0
        for t in toks:
            if t in self.senses:
                hv = self.encode_word_with_context(t, rough_ctx)
            elif t in self.base.vocab:
                hv = self.base._bin_vectors[
                    self.base.vocab[t]].numpy()
            else:
                hv = _unknown_word_hv(t, self.dim)
            acc += hv.astype(np.int32)
            n += 1
        return (acc > n / 2.0).astype(np.int8)

    # ── Diagnostics ──────────────────────────────────────────

    def show_senses(self, word: str, k: int = 5) -> None:
        if word not in self.senses:
            print(f"[multi-sense] {word!r}: only one sense in this corpus")
            return
        print(f"[multi-sense] {word!r}: {len(self.senses[word])} senses")
        # For each sense, find the k nearest words to its context centroid
        # via base encoder's bin_vectors
        if self.base._bin_vectors is None:
            return
        all_vectors = self.base._bin_vectors.numpy()
        for s_idx, (sense_hv, sense_ctx) in enumerate(self.senses[word]):
            xor = np.bitwise_xor(all_vectors, sense_ctx[None, :])
            dists = xor.sum(axis=1)
            order = np.argsort(dists)[:k+1]
            ctx_words = []
            for j in order:
                w_j = self.base.id2word[j]
                if w_j == word:
                    continue
                ctx_words.append(w_j)
                if len(ctx_words) >= k:
                    break
            print(f"   sense {s_idx+1}: context near {ctx_words}")

    # ── Persistence ─────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "min_freq_for_senses": self.min_freq_for_senses,
                "k_senses":            self.k_senses,
                "senses":              self.senses,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path,
                base_encoder: HDCNativeEncoder) -> "MultiSenseHDCEncoder":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        enc = cls(
            base_encoder,
            min_freq_for_senses=blob["min_freq_for_senses"],
            k_senses=blob["k_senses"],
        )
        enc.senses = blob["senses"]
        return enc
