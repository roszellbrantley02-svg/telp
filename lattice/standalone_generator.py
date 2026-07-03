"""
lattice/standalone_generator.py - higher-quality HDC text generation.

Wraps multiple HDCSequencePredictor instances (one per n-gram size)
with three techniques that markedly reduce drift versus greedy n=3:

  1. Back-off: try n=5, then n=4, then n=3 — only fall to a shorter
     context when no neighbour exists within a quality threshold.
  2. Beam search: keep the top-B partial continuations, expand each
     by the predictor's top-K candidates, prune to top-B by
     cumulative score.
  3. Topic anchoring: at the end, rerank the beams by encoder
     similarity to the seed hypervector — continuations that drift
     into an unrelated topic get penalised.

Everything is still pure HDC + numpy.  No LLM.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.sequence_predictor import HDCSequencePredictor, tokenize


class BackoffBeamGenerator:
    """Multi-n HDC generator with beam search and topic anchoring."""

    def __init__(self, n_gram_sizes: tuple[int, ...] = (10, 7, 5, 3),
                  encoder=None,
                  beam_width: int = 6,
                  candidate_topk: int = 8,
                  anchor_strength: float = 0.5,
                  repetition_penalty: float = 0.6,
                  style_hv: "np.ndarray | None" = None,
                  style_strength: float = 0.0):
        self.ns = tuple(sorted(n_gram_sizes, reverse=True))
        self.predictors: dict[int, HDCSequencePredictor] = {
            n: HDCSequencePredictor(n_gram=n) for n in self.ns
        }
        self.encoder = encoder    # CorpusRIEncoder (optional, for anchoring)
        self.beam_width = beam_width
        self.candidate_topk = candidate_topk
        self.anchor_strength = anchor_strength
        # NEW: repetition penalty and HDC style binding
        self.repetition_penalty = repetition_penalty
        self.style_hv = style_hv
        self.style_strength = style_strength

    def train(self, sentences: list[str]) -> None:
        for p in self.predictors.values():
            p.train(sentences)

    def stats(self) -> dict:
        return {f"n={n}": p.stats() for n, p in self.predictors.items()}

    # ─── back-off prediction with confidence threshold ─────

    def _candidates(self, prefix: list[str],
                      max_quality_distance: int = 4800
                      ) -> list[tuple[str, float]]:
        """Return list of (word, weight) candidates, trying the longest
        n-gram first and falling back if its nearest neighbour is too
        far from the query.
        """
        for n in self.ns:
            p = self.predictors[n]
            if p._stack is None:
                continue
            ctx = prefix[-n:] if len(prefix) >= n else prefix
            q = p._encode_prefix(ctx)
            xor = np.bitwise_xor(p._stack, q[None, :])
            dists = xor.sum(axis=1)
            idx = np.argsort(dists)[: self.candidate_topk]
            if int(dists[idx[0]]) > max_quality_distance:
                # No close neighbour at this n — try shorter.
                continue
            cands: dict[str, float] = {}
            for j in idx:
                w = p._next_words[j]
                weight = 1.0 / (1.0 + int(dists[j]) / 100.0)
                cands[w] = cands.get(w, 0.0) + weight
            return sorted(cands.items(), key=lambda x: -x[1])
        # Last resort: shortest n's best guess regardless of quality.
        p = self.predictors[self.ns[-1]]
        if p._stack is None:
            return []
        q = p._encode_prefix(prefix[-p.n:])
        xor = np.bitwise_xor(p._stack, q[None, :])
        dists = xor.sum(axis=1)
        idx = np.argsort(dists)[: self.candidate_topk]
        cands: dict[str, float] = {}
        for j in idx:
            w = p._next_words[j]
            weight = 1.0 / (1.0 + int(dists[j]) / 100.0)
            cands[w] = cands.get(w, 0.0) + weight
        return sorted(cands.items(), key=lambda x: -x[1])

    # ─── beam search ──────────────────────────────────────

    def generate(self, prompt: str, n_words: int = 14,
                   stop_words: set[str] | None = None,
                   topic_hv: "np.ndarray | None" = None,
                   style_hv: "np.ndarray | None" = None,
                   n_passes: int = 1) -> str:
        """Beam search with per-step HDC topic binding + repetition
        penalty + (optional) multi-pass candidate selection.

        topic_hv:  hypervector to anchor topic toward at every step
        style_hv:  hypervector to anchor style toward at every step
        n_passes:  if > 1, run generation that many times with
                   stochastic ties broken differently and pick the
                   highest composite-scoring final result.
        """
        stop_words = stop_words or set()
        seed = tokenize(prompt)
        if not seed:
            return ""
        if topic_hv is None:
            topic_hv = (self.encoder.encode(prompt)
                            if self.encoder is not None else None)
        if style_hv is None:
            style_hv = self.style_hv

        if n_passes > 1:
            best_run = None
            best_run_score = float("-inf")
            for pass_i in range(n_passes):
                run_out, run_score = self._generate_one(
                    seed, n_words, stop_words,
                    topic_hv, style_hv, jitter_seed=pass_i)
                if run_score > best_run_score:
                    best_run_score = run_score
                    best_run = run_out
            return best_run or ""
        else:
            out, _ = self._generate_one(seed, n_words, stop_words,
                                              topic_hv, style_hv,
                                              jitter_seed=0)
            return out

    def _generate_one(self, seed: list[str], n_words: int,
                          stop_words: set[str],
                          topic_hv, style_hv,
                          jitter_seed: int) -> tuple[str, float]:
        """One generation pass; returns (text, composite_score)."""
        rng = np.random.default_rng(jitter_seed + 1)

        # Beam: list of (cum_log_score, tokens_so_far)
        beams: list[tuple[float, list[str]]] = [(0.0, list(seed))]
        finished: list[tuple[float, list[str]]] = []

        for _step in range(n_words):
            new_beams: list[tuple[float, list[str]]] = []
            for score, toks in beams:
                cands = self._candidates(toks)
                if not cands:
                    finished.append((score, toks))
                    continue
                total_w = sum(w for _, w in cands) or 1.0
                # Repetition penalty: count how often each word already
                # appears in the beam's tokens; scale its weight down.
                rep_counts: dict[str, int] = {}
                for w in toks:
                    rep_counts[w] = rep_counts.get(w, 0) + 1
                for word, weight in cands[: self.candidate_topk]:
                    eff_weight = weight
                    if word in rep_counts:
                        eff_weight *= (self.repetition_penalty
                                          ** rep_counts[word])
                    # Per-step topic + style binding via HDC sim of
                    # this candidate to the topic/style anchors.
                    bonus = 0.0
                    if (self.encoder is not None
                            and (topic_hv is not None
                                 or style_hv is not None)):
                        w_hv = self.encoder.encode(word)
                        if topic_hv is not None:
                            t_sim = 1.0 - (
                                int(np.bitwise_xor(topic_hv, w_hv).sum())
                                / topic_hv.size)
                            bonus += self.anchor_strength * 0.5 * t_sim
                        if style_hv is not None and self.style_strength > 0:
                            s_sim = 1.0 - (
                                int(np.bitwise_xor(style_hv, w_hv).sum())
                                / style_hv.size)
                            bonus += self.style_strength * s_sim
                    # Add a tiny stochastic jitter so multi-pass
                    # diversifies (uses jitter_seed-derived rng).
                    jitter = float(rng.normal(0, 0.01))
                    lp = float(np.log(max(eff_weight / total_w, 1e-9)))
                    new_score = score + lp + bonus + jitter
                    new_toks = toks + [word]
                    if word in stop_words:
                        finished.append((new_score, new_toks))
                    else:
                        new_beams.append((new_score, new_toks))
            if not new_beams:
                break
            new_beams.sort(key=lambda x: -x[0] / max(len(x[1]) - len(seed), 1))
            beams = new_beams[: self.beam_width]

        candidates = beams + finished
        if not candidates:
            return ("", float("-inf"))

        # Composite end-rerank: topic similarity + length bonus + diversity
        seed_hv_local = topic_hv
        if seed_hv_local is not None and self.anchor_strength > 0.0:
            scored = []
            for score, toks in candidates:
                cont = " ".join(toks)
                cont_hv = self.encoder.encode(cont)
                ham = int(np.bitwise_xor(seed_hv_local, cont_hv).sum())
                sim = 1.0 - ham / seed_hv_local.size
                length_bonus = (len(toks) - len(seed)) * 0.02
                # Diversity: penalize candidates with low unique-word ratio
                tail = toks[len(seed):]
                if tail:
                    uniq_ratio = len(set(tail)) / len(tail)
                else:
                    uniq_ratio = 1.0
                diversity_bonus = 0.3 * uniq_ratio
                final = (score / max(len(toks) - len(seed), 1)
                            + self.anchor_strength * sim
                            + length_bonus
                            + diversity_bonus)
                scored.append((final, toks))
            scored.sort(key=lambda x: -x[0])
            best_score, best = scored[0]
        else:
            candidates.sort(key=lambda x: -x[0] / max(len(x[1]) - len(seed), 1))
            best_score, best = candidates[0]

        # Strip back-to-back duplicate triples.
        cleaned = []
        for w in best:
            if (len(cleaned) >= 2 and cleaned[-1] == cleaned[-2] == w):
                break
            cleaned.append(w)
        return (" ".join(cleaned), best_score)
