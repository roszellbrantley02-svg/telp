"""
lattice/diff_text_encoder.py - GPU-trained differentiable text encoder.

The bridge between HDC and LLM-style learning. Instead of using random
initial vectors + co-occurrence counters (the CorpusRIEncoder approach),
this encoder LEARNS word embeddings via gradient descent on the GPU,
then binarizes them into hypervectors compatible with the rest of
Telp's HDC stack.

Training objective: skip-gram with negative sampling — for each
(center, context) pair, push their embeddings together while pushing
random "negative" word embeddings apart.  Classic word2vec, but:

  * Trained on GPU at LLM-scale dim (4096+ instead of 10000-bit sparse)
  * Loss-driven, finds optima co-occurrence counting cannot
  * At inference, sign() the float vectors to get binary hypervectors
    — drop-in compatible with CorpusRIEncoder.encode()

This implements the Hersche et al. (ETH Zurich 2023) Differentiable
HDC pattern applied to word-level language modeling.

Usage:
    enc = DifferentiableTextEncoder(dim=4096)
    enc.train_on_corpus(sentences, epochs=8, batch_size=4096)
    enc.save("state/diff_encoder.pt")
    bin_hv = enc.encode("Einstein was a physicist")   # 4096-bit np.int8
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))


# ─── Tokenizer (mirrors the rest of the stack) ─────────────────────


_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9'_-]*\b")


def tokenize(s: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(s or "")]


# ─── Encoder module ────────────────────────────────────────────────


class _SkipGramModule(nn.Module):
    """Skip-gram with negative-sampling head.

    Two embedding tables:
      * center_emb:  the "input" representation of a word
      * context_emb: the "output" representation of a word

    At inference we use center_emb only.  This is the classic word2vec
    asymmetry — keeps training cleaner.
    """

    def __init__(self, n_vocab: int, dim: int):
        super().__init__()
        self.dim = dim
        # sparse=True is crucial — skip-gram only touches ~5 rows per
        # step out of n_vocab.  With sparse gradients + SparseAdam we
        # only update those rows, not the entire 54K x 10K state.
        self.center_emb  = nn.Embedding(n_vocab, dim, sparse=True)
        self.context_emb = nn.Embedding(n_vocab, dim, sparse=True)
        nn.init.uniform_(self.center_emb.weight,  -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.context_emb.weight)

    def forward(self, centers: torch.Tensor,
                  contexts: torch.Tensor,
                  negatives: torch.Tensor) -> torch.Tensor:
        """Standard skip-gram-with-negative-sampling loss.

        centers   : (B,)  - target word ids
        contexts  : (B,)  - true context word ids
        negatives : (B, K)- random negative word ids
        """
        c    = self.center_emb(centers)            # (B, dim)
        pos  = self.context_emb(contexts)          # (B, dim)
        negs = self.context_emb(negatives)         # (B, K, dim)
        # Positive loss: dot(c, pos) should be high
        pos_score = (c * pos).sum(dim=-1)          # (B,)
        # Negative loss: dot(c, neg) should be low
        neg_score = torch.bmm(negs, c.unsqueeze(-1)).squeeze(-1)  # (B, K)
        # Binary cross-entropy with logits
        pos_loss = F.logsigmoid(pos_score).mean()
        neg_loss = F.logsigmoid(-neg_score).mean()
        return -(pos_loss + neg_loss)


# ─── Public encoder ────────────────────────────────────────────────


class DifferentiableTextEncoder:
    """GPU-trained word embeddings, exposed with a CorpusRIEncoder-
    compatible interface for drop-in use elsewhere in the stack."""

    def __init__(self, dim: int = 4096,
                   window: int = 5,
                   n_negatives: int = 10,
                   min_count: int = 2,
                   device: str | None = None):
        self.dim = dim
        self.window = window
        self.n_negatives = n_negatives
        self.min_count = min_count
        self.device = device or ("cuda" if torch.cuda.is_available()
                                       else "cpu")
        self.vocab: dict[str, int] = {}     # word -> id
        self.id2word: list[str] = []        # id   -> word
        self.word_counts: dict[str, int] = {}
        self.module: _SkipGramModule | None = None
        # Cached binary vectors after training
        self._bin_vectors: torch.Tensor | None = None   # (V, dim) int8 on CPU
        self._float_vectors: torch.Tensor | None = None # (V, dim) float CPU

    # ── Vocab building ──────────────────────────────────────────

    def _build_vocab(self, sentences: list[str]) -> None:
        counts: dict[str, int] = {}
        for s in sentences:
            for t in tokenize(s):
                counts[t] = counts.get(t, 0) + 1
        self.word_counts = counts
        # Keep words above min_count
        kept = sorted(((w, c) for w, c in counts.items()
                          if c >= self.min_count),
                        key=lambda x: -x[1])
        self.id2word = [w for w, _ in kept]
        self.vocab = {w: i for i, w in enumerate(self.id2word)}
        print(f"[diff-enc] vocab built: {len(self.vocab)} words "
                f"(min_count={self.min_count})", flush=True)

    def _build_negative_sampler(self) -> torch.Tensor:
        """Subsampled unigram distribution^0.75 as a pre-built INDEX
        TABLE (Mikolov-style).  Sampling becomes O(1) random integer
        lookup — way faster than torch.multinomial on a 54K-dim dist.

        Returns: (TABLE_SIZE,) torch.long tensor on self.device of
        word ids, each appearing proportional to freq^0.75.
        """
        TABLE_SIZE = 1_000_000
        freqs = np.array([self.word_counts[w] for w in self.id2word],
                            dtype=np.float64)
        probs = freqs ** 0.75
        probs /= probs.sum()
        counts = np.maximum(1, (probs * TABLE_SIZE).astype(np.int64))
        table = np.concatenate([
            np.full(counts[i], i, dtype=np.int64)
            for i in range(len(counts))
        ])
        if len(table) > TABLE_SIZE:
            table = table[:TABLE_SIZE]
        return torch.from_numpy(table).to(self.device)

    # ── Pair generation ────────────────────────────────────────

    def _gen_pairs(self, sentences: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Sliding-window skip-gram pair generation."""
        centers: list[int] = []
        contexts: list[int] = []
        for s in sentences:
            ids = [self.vocab[t] for t in tokenize(s) if t in self.vocab]
            n = len(ids)
            for i, c in enumerate(ids):
                lo = max(0, i - self.window)
                hi = min(n, i + self.window + 1)
                for j in range(lo, hi):
                    if i == j:
                        continue
                    centers.append(c)
                    contexts.append(ids[j])
        return (np.array(centers, dtype=np.int64),
                  np.array(contexts, dtype=np.int64))

    # ── Training ───────────────────────────────────────────────

    def train_on_corpus(self, sentences: list[str],
                              epochs: int = 8,
                              batch_size: int = 8192,
                              lr: float = 0.01,
                              verbose: bool = True) -> dict:
        """Train word embeddings on the corpus via skip-gram with
        negative sampling.  Returns a stats dict."""
        t0 = time.time()
        self._build_vocab(sentences)
        if not self.vocab:
            return {"error": "empty vocab after filter"}

        centers_np, contexts_np = self._gen_pairs(sentences)
        n_pairs = len(centers_np)
        if n_pairs == 0:
            return {"error": "no training pairs"}
        print(f"[diff-enc] {n_pairs:,} pairs, dim={self.dim}, "
                f"device={self.device}", flush=True)

        self.module = _SkipGramModule(len(self.vocab), self.dim
                                          ).to(self.device)
        neg_probs = self._build_negative_sampler()
        # SparseAdam is REQUIRED for sparse=True embeddings.  It only
        # allocates/updates Adam state for the rows that actually had
        # gradients this step — turning a 540M-param dense update into
        # an ~80K-param sparse update.  ~50-100x faster.
        opt = torch.optim.SparseAdam(self.module.parameters(), lr=lr)

        centers_t  = torch.from_numpy(centers_np ).to(self.device)
        contexts_t = torch.from_numpy(contexts_np).to(self.device)

        losses: list[float] = []
        neg_table = neg_probs   # actually the index table now
        table_size = neg_table.shape[0]
        for epoch in range(epochs):
            self.module.train()
            perm = torch.randperm(n_pairs, device=self.device)
            ep_loss = 0.0
            n_batches = 0
            t_ep0 = time.time()
            total_batches = (n_pairs + batch_size - 1) // batch_size
            for b_start in range(0, n_pairs, batch_size):
                idx = perm[b_start: b_start + batch_size]
                c = centers_t[idx]
                ctx = contexts_t[idx]
                # O(1) negative sampling via pre-built index table
                rand_idx = torch.randint(
                    0, table_size,
                    (idx.numel() * self.n_negatives,),
                    device=self.device,
                )
                negs = neg_table[rand_idx].view(-1, self.n_negatives)
                opt.zero_grad()
                loss = self.module(c, ctx, negs)
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())
                n_batches += 1
                # In-epoch progress prints so we can see it's alive
                if verbose and n_batches % 200 == 0:
                    elapsed = time.time() - t_ep0
                    pct = 100.0 * n_batches / total_batches
                    rate = n_batches / max(elapsed, 0.01)
                    eta_s = (total_batches - n_batches) / max(rate, 0.01)
                    print(f"[diff-enc]   epoch {epoch+1} "
                            f"batch {n_batches:,}/{total_batches:,} "
                            f"({pct:.1f}%) "
                            f"loss={ep_loss/n_batches:.3f} "
                            f"rate={rate:.1f}/s "
                            f"ETA {eta_s/60:.1f}min", flush=True)
            avg_loss = ep_loss / max(1, n_batches)
            losses.append(avg_loss)
            if verbose:
                ep_time = time.time() - t_ep0
                print(f"[diff-enc]   epoch {epoch+1}/{epochs}: "
                        f"loss={avg_loss:.4f}  ({ep_time:.1f}s)",
                        flush=True)

        # Snapshot the learned vectors to CPU + cache binary form
        with torch.no_grad():
            self._float_vectors = self.module.center_emb.weight.detach().cpu()
            # Binarize: sign() + map to {-1,+1}, store as int8
            # For the existing HDC stack we want {0,1} binary, so we
            # convert sign>0 to 1, else 0.
            self._bin_vectors = (self._float_vectors > 0).to(torch.int8)

        elapsed = time.time() - t0
        print(f"[diff-enc] training done in {elapsed:.1f}s. "
                f"final loss={losses[-1]:.4f}", flush=True)
        return {
            "vocab_size":  len(self.vocab),
            "dim":         self.dim,
            "pairs":       n_pairs,
            "epochs":      epochs,
            "final_loss":  losses[-1],
            "elapsed_s":   round(elapsed, 2),
        }

    # ── Inference (CorpusRIEncoder-compatible) ─────────────────

    def encode_word(self, word: str) -> np.ndarray:
        """Return D-bit binary hypervector for a word.  Unknown words
        return a deterministic-from-chars vector for graceful fallback."""
        if self._bin_vectors is None:
            raise RuntimeError("encoder not trained")
        w = word.lower()
        if w in self.vocab:
            return self._bin_vectors[self.vocab[w]].numpy()
        # Unknown — bundle char-vectors deterministically
        return _unknown_word_hv(w, self.dim)

    def encode(self, text: str) -> np.ndarray:
        """Encode a phrase/sentence to a D-bit binary hypervector by
        bundling known-word vectors (majority vote)."""
        if self._bin_vectors is None:
            raise RuntimeError("encoder not trained")
        toks = tokenize(text)
        if not toks:
            return np.zeros(self.dim, dtype=np.int8)
        # Accumulate in int32 to avoid overflow on long sentences
        acc = np.zeros(self.dim, dtype=np.int32)
        n = 0
        for t in toks:
            if t in self.vocab:
                acc += self._bin_vectors[self.vocab[t]].numpy().astype(np.int32)
                n += 1
            else:
                acc += _unknown_word_hv(t, self.dim).astype(np.int32)
                n += 1
        # Majority vote: bit is 1 if more than half the contributors voted 1
        threshold = n / 2.0
        return (acc > threshold).astype(np.int8)

    # ── Save / load ────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._bin_vectors is None:
            raise RuntimeError("nothing to save — encoder not trained")
        meta = {
            "dim":        self.dim,
            "window":     self.window,
            "vocab":      self.vocab,
            "id2word":    self.id2word,
            "word_counts": self.word_counts,
        }
        path.with_suffix(".meta.json").write_text(
            json.dumps(meta), encoding="utf-8")
        torch.save({
            "float_vectors": self._float_vectors,
            "bin_vectors":   self._bin_vectors,
        }, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "DifferentiableTextEncoder":
        path = Path(path)
        meta = json.loads(path.with_suffix(".meta.json")
                              .read_text(encoding="utf-8"))
        enc = cls(dim=meta["dim"], window=meta["window"])
        enc.vocab     = meta["vocab"]
        enc.id2word   = meta["id2word"]
        enc.word_counts = meta["word_counts"]
        payload = torch.load(str(path), map_location="cpu",
                                  weights_only=False)
        enc._float_vectors = payload["float_vectors"]
        enc._bin_vectors   = payload["bin_vectors"]
        return enc

    # ── Diagnostics ────────────────────────────────────────────

    def nearest_neighbors(self, word: str, k: int = 10
                            ) -> list[tuple[str, float]]:
        """Return the top-k most similar words (cosine, float space)."""
        if self._float_vectors is None or word.lower() not in self.vocab:
            return []
        v = self._float_vectors[self.vocab[word.lower()]]
        all_v = self._float_vectors
        # cosine
        v_n   = v / (v.norm() + 1e-9)
        all_n = all_v / (all_v.norm(dim=-1, keepdim=True) + 1e-9)
        sims = (all_n @ v_n).numpy()
        order = np.argsort(-sims)[:k+1]    # +1 because top match is self
        out = []
        for idx in order:
            w = self.id2word[idx]
            if w == word.lower():
                continue
            out.append((w, float(sims[idx])))
            if len(out) >= k:
                break
        return out


# ─── Helpers ───────────────────────────────────────────────────────


def _unknown_word_hv(word: str, dim: int) -> np.ndarray:
    """Deterministic hash-based hypervector for an unknown word.
    Used as graceful fallback when the trained vocabulary doesn't
    contain a token."""
    rng = np.random.default_rng(abs(hash(word)) % (2**32))
    return rng.integers(0, 2, size=dim, dtype=np.int8)
