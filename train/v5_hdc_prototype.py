"""
v5_hdc_prototype.py - Hyperdimensional Computing (HDC) prototype.

The radical idea: encode every chart as a 10,000-dim BINARY vector using
brain-inspired operations (XOR, permutation, bundling). Then trading
decisions reduce to nearest-neighbor lookup in this high-dim space.

Why HDC:
  - Inference is <1ms (just Hamming distance to corpus)
  - Training is essentially "add to database"
  - Fully interpretable: you see exactly which historical charts the
    decision is based on
  - No GPU needed
  - Robust to noise (high-dim vectors have huge separation between random
    vectors)
  - Scales to billions of examples trivially

The math behind it (Pentti Kanerva / VSA):
  - Random 10k-bit vectors are nearly orthogonal (cosine ~0.5)
  - XOR binds two concepts into one (still 10k bits)
  - Majority-vote bundles many concepts (lossy compression)
  - Hamming distance preserves semantic similarity

Implementation here:
  1. Each feature has a random "atomic" vector (role)
  2. Each feature VALUE has a random vector (filler)
  3. (role, value) pair = role XOR filler
  4. Chart = bundle of all its (role, value) pairs (majority vote)
  5. Predict by finding K nearest neighbors by Hamming distance,
     vote on their outcomes

Usage:
    python -m train.v5_hdc_prototype
"""
from __future__ import annotations

import json
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]

D = 10000     # vector dimension
RNG = np.random.default_rng(42)


# ─── HDC primitives ────────────────────────────────────────────────


def random_vec() -> np.ndarray:
    """Random binary vector of dimension D (50/50 split)."""
    return RNG.integers(0, 2, size=D, dtype=np.int8)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bind two vectors (XOR) — used for role-filler pairs."""
    return np.bitwise_xor(a, b)


def bundle(vectors: list) -> np.ndarray:
    """Bundle (majority-vote) multiple vectors into one summary vector."""
    if not vectors:
        return np.zeros(D, dtype=np.int8)
    arr = np.stack(vectors)
    # Sum, then threshold at half
    s = arr.sum(axis=0)
    threshold = len(vectors) / 2
    return (s > threshold).astype(np.int8)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Number of differing bits between two binary vectors."""
    return int(np.count_nonzero(a ^ b))


def hamming_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1.0 = identical, 0.5 = random, 0.0 = opposite."""
    return 1.0 - (hamming_distance(a, b) / D)


# ─── Transition memory: Markov reasoning at the HDC primitive layer ─


class TransitionMemory:
    """A Markov-style transition store baked into the HDC base layer.

    Every encoder that uses HDVocabulary can call vocab.observe_transition()
    to record (predecessor_hv -> successor) pairs during training, and
    later call vocab.predict_successor(query_hv) to get a Markov-style
    prediction of where this state tends to go.

    Two modes:
      * kind="categorical" — successors are HVs.  Prediction bundles the
        nearest neighbours' successor HVs and (optionally) cleans up
        against a codebook to return a concrete label.
      * kind="continuous" — successors are scalars (e.g., returns).
        Prediction is the similarity-weighted mean of neighbour scalars
        with an "agreement" confidence derived from neighbour std.

    GPU acceleration mirrors the HDC v4 + Markov v2 path: stack is
    pushed to torch.cuda when device != "cpu".  Same speedup pattern.
    """

    def __init__(self, dim: int = None, kind: str = "categorical",
                  device: str = "cpu"):
        # kind:
        #   continuous  — successor is a float (e.g., future return)
        #   categorical — successor is an HV (e.g., next image patch);
        #                 predict bundles + cleanup against codebook
        #   label       — successor is a hashable tag (str/int);
        #                 predict counts weighted votes (correct mode
        #                 for discrete finite vocabularies like text)
        assert kind in ("continuous", "categorical", "label")
        self.dim = dim or D
        self.kind = kind
        self.device = device
        # Parallel append buffers (built lazily)
        self._pred_hvs: list[np.ndarray] = []
        self._succ_hvs: list[np.ndarray] = []   # categorical mode
        self._succ_vals: list[float] = []        # continuous mode
        self._succ_labels: list = []             # label mode
        # Compiled stacks (built on commit())
        self._pred_stack: np.ndarray | None = None
        self._pred_stack_t = None
        self._succ_stack: np.ndarray | None = None
        self._succ_stack_t = None
        self._succ_vals_arr: np.ndarray | None = None
        self._succ_vals_t = None
        # Codebook for categorical cleanup: label -> HV.
        self._codebook: dict[str, np.ndarray] = {}

    # ── ingestion ─────────────────────────────────────────────

    def observe(self, predecessor_hv: np.ndarray, successor) -> None:
        """Record one transition.  `successor` is an HV in categorical
        mode, a float in continuous mode, a hashable tag in label
        mode."""
        self._pred_hvs.append(predecessor_hv.astype(np.int8))
        if self.kind == "categorical":
            self._succ_hvs.append(successor.astype(np.int8))
        elif self.kind == "continuous":
            self._succ_vals.append(float(successor))
        else:  # label
            self._succ_labels.append(successor)
        # Invalidate any compiled stack
        self._pred_stack = None
        self._pred_stack_t = None

    def observe_sequence(self, hv_sequence: list[np.ndarray]) -> None:
        """Record every consecutive pair from a sequence of HVs."""
        for i in range(len(hv_sequence) - 1):
            self.observe(hv_sequence[i], hv_sequence[i + 1])

    def add_to_codebook(self, label: str, hv: np.ndarray) -> None:
        """Register a candidate output HV for categorical cleanup."""
        self._codebook[label] = hv.astype(np.int8)

    # ── compilation ──────────────────────────────────────────

    def commit(self) -> None:
        """Stack the append buffers into searchable arrays.  Called
        automatically on first predict_successor()."""
        if not self._pred_hvs:
            return
        self._pred_stack = np.stack(self._pred_hvs)
        if self.kind == "categorical":
            self._succ_stack = np.stack(self._succ_hvs)
        elif self.kind == "continuous":
            self._succ_vals_arr = np.array(self._succ_vals,
                                                 dtype=np.float64)
        # label mode: labels stay as python list (small, no compile)
        if self.device != "cpu":
            import torch
            self._pred_stack_t = torch.from_numpy(self._pred_stack).to(
                device=self.device, dtype=torch.int8, non_blocking=True
            )
            if self.kind == "categorical":
                self._succ_stack_t = torch.from_numpy(self._succ_stack).to(
                    device=self.device, dtype=torch.int8, non_blocking=True
                )
            elif self.kind == "continuous":
                self._succ_vals_t = torch.from_numpy(
                    self._succ_vals_arr
                ).to(device=self.device, dtype=torch.float64,
                       non_blocking=True)

    def __len__(self) -> int:
        return len(self._pred_hvs)

    # ── prediction ────────────────────────────────────────────

    def predict_successor(self, query_hv: np.ndarray, k: int = 12) -> dict:
        """Markov-style prediction: given a state HV, return what tends
        to come next, averaged over the k nearest stored predecessors.

        Returns a dict with mode-specific fields plus shared diagnostics
        (nearest_dist, nearest_sim, n_neighbors).
        """
        if self._pred_stack is None and self._pred_hvs:
            self.commit()
        if not self._pred_hvs:
            return {"error": "empty transition memory"}

        # Hamming distances to every stored predecessor
        if self._pred_stack_t is not None:
            import torch
            q = torch.from_numpy(query_hv.astype(np.int8)).to(
                device=self.device, dtype=torch.int8, non_blocking=True
            )
            xor = torch.bitwise_xor(self._pred_stack_t, q.unsqueeze(0))
            dists_t = xor.to(torch.int32).sum(dim=1)
            order_t = torch.argsort(dists_t, stable=True)[:k]
            order = order_t.cpu().numpy()
            dists = dists_t.cpu().numpy()
        else:
            xor = np.bitwise_xor(self._pred_stack,
                                       query_hv.astype(np.int8)[None, :])
            dists = xor.sum(axis=1)
            order = np.argsort(dists, kind="stable")[:k]

        nearest_dist = int(dists[order[0]])
        nearest_sim  = 1.0 - 2.0 * nearest_dist / self.dim
        neighbor_dists = dists[order].astype(np.float64)
        weights = 1.0 / (1.0 + neighbor_dists / 100.0)

        if self.kind == "label":
            # Discrete-vote mode: count weighted votes for each tag in
            # the k-nearest neighbours' successors.  Correct primitive
            # for finite-vocabulary tasks (text, code, categorical IDs).
            votes: dict = {}
            for j, idx in enumerate(order):
                lab = self._succ_labels[int(idx)]
                votes[lab] = votes.get(lab, 0.0) + float(weights[j])
            if not votes:
                return {"error": "no neighbours"}
            ranked = sorted(votes.items(), key=lambda kv: -kv[1])
            best_label, best_weight = ranked[0]
            total = sum(votes.values())
            top_share = best_weight / max(1e-9, total)
            runner = ranked[1][1] if len(ranked) > 1 else 0.0
            margin = (best_weight - runner) / max(1e-9, total)
            return {
                "predicted_label": best_label,
                "vote_share":      round(top_share, 4),
                "vote_margin":     round(margin, 4),
                "agreement":       round(top_share, 4),
                "candidates":      [(l, round(w / total, 4))
                                       for l, w in ranked[:5]],
                "nearest_dist":    nearest_dist,
                "nearest_sim":     round(nearest_sim, 4),
                "n_neighbors":     int(k),
            }
        elif self.kind == "categorical":
            neighbor_hvs = self._succ_stack[order].astype(np.float64)
            wsum_total = float(weights.sum())
            weighted_sum = (neighbor_hvs * weights[:, None]).sum(axis=0)
            threshold = wsum_total / 2.0
            predicted_hv = (weighted_sum > threshold).astype(np.int8)
            # Cleanup against codebook
            best_label, best_dist = None, self.dim
            second_dist = self.dim
            for name, cb_hv in self._codebook.items():
                d = int(np.count_nonzero(predicted_hv ^ cb_hv))
                if d < best_dist:
                    second_dist = best_dist
                    best_dist = d
                    best_label = name
                elif d < second_dist:
                    second_dist = d
            cleanup_sim = (1.0 - best_dist / self.dim
                              if best_label is not None else 0.0)
            # Margin = how much closer the winner is vs the runner-up.
            # Useful for abstention.
            margin = ((second_dist - best_dist) / self.dim
                          if best_label is not None else 0.0)
            return {
                "predicted_hv":   predicted_hv,
                "cleanup_label":  best_label,
                "cleanup_sim":    round(cleanup_sim, 4),
                "cleanup_margin": round(margin, 4),
                "agreement":      round(cleanup_sim, 4),
                "nearest_dist":   nearest_dist,
                "nearest_sim":    round(nearest_sim, 4),
                "n_neighbors":    int(k),
            }
        else:
            neighbor_vals = self._succ_vals_arr[order]
            wsum = float(weights.sum())
            mean_val = float((neighbor_vals * weights).sum()
                                / max(1e-9, wsum))
            std_val = float(neighbor_vals.std())
            agreement = 1.0 / (1.0 + std_val / max(1.0, abs(mean_val)))
            return {
                "predicted_value":  round(mean_val, 4),
                "std":              round(std_val, 4),
                "signal_strength":  round(abs(mean_val), 4),
                "agreement":        round(agreement, 4),
                "nearest_dist":     nearest_dist,
                "nearest_sim":      round(nearest_sim, 4),
                "n_neighbors":      int(k),
            }


# ─── Vocabulary: role vectors + filler vectors ─────────────────────


class HDVocabulary:
    """Maps feature names + values to atomic HD vectors.

    Also owns a set of named TransitionMemory channels — Markov-style
    state -> successor tables baked into the HDC primitive layer.  Any
    subsystem that holds a vocab reference can populate transitions
    during training and query them at inference time.
    """

    def __init__(self):
        self.roles: dict = {}
        self.fillers: dict = {}
        # Markov transitions, addressable by channel name so different
        # subsystems (text, trading, image patches, ...) can share or
        # isolate stores.  Built lazily.
        self.transitions: dict[str, TransitionMemory] = {}

    def role(self, name: str) -> np.ndarray:
        if name not in self.roles:
            self.roles[name] = random_vec()
        return self.roles[name]

    def filler(self, value) -> np.ndarray:
        key = str(value)
        if key not in self.fillers:
            self.fillers[key] = random_vec()
        return self.fillers[key]

    def encode_pair(self, role_name: str, value) -> np.ndarray:
        return bind(self.role(role_name), self.filler(value))

    # ── Transition memory accessors ───────────────────────────

    def get_transitions(self, channel: str = "default",
                           kind: str = "categorical",
                           device: str = "cpu") -> TransitionMemory:
        """Get or create a named transition memory.  First call sets
        kind + device; subsequent calls ignore those args."""
        if channel not in self.transitions:
            self.transitions[channel] = TransitionMemory(
                dim=D, kind=kind, device=device
            )
        return self.transitions[channel]

    def observe_transition(self, predecessor_hv: np.ndarray, successor,
                              channel: str = "default") -> None:
        """Convenience: record a transition into a named channel."""
        self.get_transitions(channel).observe(predecessor_hv, successor)

    def observe_sequence(self, hv_sequence: list[np.ndarray],
                            channel: str = "default") -> None:
        """Convenience: record every consecutive pair in a sequence."""
        self.get_transitions(channel).observe_sequence(hv_sequence)

    def predict_successor(self, query_hv: np.ndarray,
                              channel: str = "default", k: int = 12) -> dict:
        """Convenience: Markov-style 'where does this tend to go?'"""
        if channel not in self.transitions:
            return {"error": f"no transition channel '{channel}'"}
        return self.transitions[channel].predict_successor(query_hv, k=k)


# ─── Encode a chart example into HD vector ─────────────────────────


def _bucket(value, edges, labels):
    if value is None: return "none"
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def encode_chart(ex: dict, vocab: HDVocabulary) -> np.ndarray:
    """Convert a corpus example into one 10k-bit HD vector."""
    pairs = []

    # Ticker
    pairs.append(vocab.encode_pair("ticker", ex.get("ticker")))

    # Regime
    regime = (ex.get("regime_bars") or {}).get("state")
    pairs.append(vocab.encode_pair("regime", regime))

    # Session
    pairs.append(vocab.encode_pair("session", ex.get("session_phase")))

    # Hour of day
    try:
        ts = datetime.fromisoformat(ex.get("ts", "").replace("Z", "+00:00"))
        pairs.append(vocab.encode_pair("hour", ts.hour))
    except Exception:
        pass

    # Trigger candle anatomy
    cand_1m = (ex.get("candle_anatomy") or {}).get("1m") or []
    if cand_1m:
        trig = cand_1m[-1]
        pairs.append(vocab.encode_pair("trig_color", trig.get("color")))
        pairs.append(vocab.encode_pair("trig_label", trig.get("label")))
        pairs.append(vocab.encode_pair("trig_size", trig.get("body_size")))
        # Bucket the wick percentages
        up = trig.get("upper_pct"); lo = trig.get("lower_pct")
        if up is not None:
            up_b = _bucket(up, [0.15, 0.40, 0.65], ["U_LO", "U_MID", "U_HI", "U_XHI"])
            pairs.append(vocab.encode_pair("trig_upper", up_b))
        if lo is not None:
            lo_b = _bucket(lo, [0.15, 0.40, 0.65], ["L_LO", "L_MID", "L_HI", "L_XHI"])
            pairs.append(vocab.encode_pair("trig_lower", lo_b))

    # Last 3 bar color sequence
    if len(cand_1m) >= 3:
        seq = "_".join(c.get("color", "?")[0] for c in cand_1m[-3:])
        pairs.append(vocab.encode_pair("seq3", seq))

    # HTF slope direction
    slope = (ex.get("htf_context") or {}).get("slope_1h_5bar_ticks")
    if slope is not None:
        if slope > 20:    sd = "UP"
        elif slope > 5:   sd = "mild_up"
        elif slope < -20: sd = "DOWN"
        elif slope < -5:  sd = "mild_down"
        else:             sd = "FLAT"
        pairs.append(vocab.encode_pair("htf_slope", sd))

    # 4h range position
    pos = (ex.get("htf_context") or {}).get("position_in_4h_range")
    if pos is not None:
        pos_b = _bucket(pos, [0.20, 0.40, 0.60, 0.80], ["LOW", "MLOW", "MID", "MHI", "HIGH"])
        pairs.append(vocab.encode_pair("pos_4h", pos_b))

    # Volume bucket
    vol = (ex.get("volume_ofi") or {}).get("1m") or {}
    vr = vol.get("volume_ratio")
    if vr is not None:
        if vr >= 2.0:   vb = "SPIKE"
        elif vr >= 1.4: vb = "EXPANSION"
        elif vr >= 0.7: vb = "NORMAL"
        else:           vb = "LOW"
        pairs.append(vocab.encode_pair("vol", vb))

    # OFI bias
    ofi = vol.get("ofi_bias")
    if ofi:
        pairs.append(vocab.encode_pair("ofi", ofi))

    # Leading signal direction (25m)
    ld = (ex.get("leading_signal") or {}).get("lead_direction_25m")
    if ld:
        pairs.append(vocab.encode_pair("lead_dir", ld))

    # VIX regime
    vix = (ex.get("leading_signal") or {}).get("vix_regime")
    if vix:
        pairs.append(vocab.encode_pair("vix", vix))

    # Recent memory: losing streak
    streak = (ex.get("recent_memory") or {}).get("losing_streak", 0)
    streak_b = "0" if streak == 0 else "1" if streak == 1 else "2" if streak == 2 else "3+"
    pairs.append(vocab.encode_pair("streak", streak_b))

    # Detected patterns (top 3 from 1m, 5m)
    pats = ex.get("patterns_detected") or {}
    for tf in ("1m", "5m"):
        for d in (pats.get(tf) or [])[:2]:
            pairs.append(vocab.encode_pair(f"pat_{tf}", d.get("pattern")))

    return bundle(pairs)


# ─── Build database from corpus ────────────────────────────────────


def load_real_trades():
    out = []
    for p in [_TELP_ROOT / "state" / "v5_corpus" / "train_expert.jsonl",
               _TELP_ROOT / "state" / "v5_corpus" / "val_expert.jsonl"]:
        if not p.exists(): continue
        for line in open(p, encoding="utf-8"):
            ex = json.loads(line)
            if ex.get("is_expansion") or ex.get("is_synthetic_skip"): continue
            oc = ex.get("outcome") or {}
            if oc.get("side_taken") not in ("buy", "sell"): continue
            if oc.get("realized_pnl") in (None, 0.0): continue
            out.append(ex)
    return out


def load_val_set():
    """Use last 30 chronologically as TEST set; rest as KNN database."""
    all_real = load_real_trades()
    all_real.sort(key=lambda ex: ex.get("ts", ""))
    return all_real[:-30], all_real[-30:]


# ─── KNN classifier with hard_tag categories ───────────────────────


def predict_action(query_vec, database, k=5):
    """K nearest neighbors vote on action."""
    distances = [(i, hamming_distance(query_vec, db_vec))
                 for i, (db_vec, _, _) in enumerate(database)]
    distances.sort(key=lambda x: x[1])
    top_k = [database[i] for i, _ in distances[:k]]
    # Vote on the "right action" (what the trade SHOULD have been based on outcome)
    votes = Counter()
    pnl_sum = 0.0
    for _, gt_dir, pnl in top_k:
        if pnl > 0:
            # The actual side won; reuse it as a vote
            votes["take_observed"] += 1
        # Vote based on ground truth direction
        if gt_dir == "up":
            votes["buy"] += 1
        elif gt_dir == "down":
            votes["sell"] += 1
        else:
            votes["skip"] += 1
        pnl_sum += pnl
    return votes.most_common(1)[0][0], votes, pnl_sum / k


def main():
    print(f"HDC prototype (D={D} dim, binary vectors)\n")
    print("Loading corpus...")
    train, test = load_val_set()
    print(f"  Train (database): {len(train)} real trades")
    print(f"  Test (held out):  {len(test)} most recent trades")

    print(f"\nBuilding vocabulary + encoding train set...")
    vocab = HDVocabulary()
    database = []
    for ex in train:
        vec = encode_chart(ex, vocab)
        gt = (ex.get("outcome") or {}).get("ground_truth_dir")
        pnl = float(ex["outcome"]["realized_pnl"])
        database.append((vec, gt, pnl))
    print(f"  Vocabulary: {len(vocab.roles)} roles, {len(vocab.fillers)} fillers")
    print(f"  Database vectors: {len(database)}")

    print(f"\nScoring on test set...")
    # Encode test set
    test_vecs = [(encode_chart(ex, vocab), ex) for ex in test]

    # Try K=3, 5, 7, 10
    for k in [3, 5, 7, 10, 15]:
        wins = 0
        total_pnl = 0.0
        entries = 0
        disasters = 0
        for vec, ex in test_vecs:
            distances = [(i, hamming_distance(vec, db_vec))
                         for i, (db_vec, _, _) in enumerate(database)]
            distances.sort(key=lambda x: x[1])
            top_k = [database[i] for i, _ in distances[:k]]
            # Pure direction vote: how many neighbors went up vs down
            up_count = sum(1 for _, gt, _ in top_k if gt == "up")
            down_count = sum(1 for _, gt, _ in top_k if gt == "down")
            sw_count = sum(1 for _, gt, _ in top_k if gt == "sideways")
            # Decision rule: take the dominant direction if confident
            if up_count > down_count and up_count > sw_count + 1:
                action = "buy"
            elif down_count > up_count and down_count > sw_count + 1:
                action = "sell"
            else:
                action = "skip"
            # Score
            real_pnl = float(ex["outcome"]["realized_pnl"])
            real_side = ex["outcome"]["side_taken"]
            real_gt = ex["outcome"]["ground_truth_dir"]
            if action in ("buy", "sell"):
                entries += 1
                # If action matches V4's actual side → we'd get the same pnl
                # If action differs → flip the pnl (approximation)
                if action == real_side:
                    eff_pnl = real_pnl
                else:
                    eff_pnl = -real_pnl
                if eff_pnl > 0: wins += 1
                total_pnl += eff_pnl
                if (action == "buy" and real_gt == "down") or \
                   (action == "sell" and real_gt == "up"):
                    disasters += 1
        wr = wins / max(1, entries)
        print(f"  K={k:2d}: entries={entries:2d}/{len(test_vecs)}  WR={wr*100:5.1f}%  "
              f"disasters={disasters}  PnL=${total_pnl:+.2f}")


if __name__ == "__main__":
    main()
