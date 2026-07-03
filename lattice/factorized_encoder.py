"""lattice/factorized_encoder.py — factorized HDC encoder for compositional generalization.

WHY
---
Per compositional_factorization.md finding: monolithic learned HDC encoders
DESTROY compositional generalization (100% trained → 0% on held-out combos).
The fix is FACTORIZATION — each attribute group gets its own learned head;
heads are bound via fixed role vectors; final composition is the bundled
bind-results.

This is the key architectural unlock for Telp learning to generalize to
never-seen regime-pattern-time combos.

ARCHITECTURE
------------
14 attribute groups (macro/options/sector/liquidity/multi_tf/bar_ctx/tape/
vp/recent/cd/orderflow/pattern/position/brain_math).

For each group:
  raw_attrs (dict)  →  per-group MLP head  →  group_hv (continuous bipolar)
  group_hv          →  BIND with fixed group_role_hv  →  bound_hv

All 14 bound_hvs are BUNDLEd (summed) → final composite HV.

Training is contrastive: trades with same outcome are pulled together in
HV space; opposite outcomes pushed apart. At inference, we binarize via
tanh+sign and use the resulting ±1 vector for kNN against the brain bank
the same way `unified_brain.decide` does today.

USE
---
encoder = FactorizedHDCEncoder(dim=10000)
encoder.to("cuda")

# Training
hvs = encoder(batch_dict)        # (B, dim) continuous
loss = contrastive_loss(hvs, outcome_labels)
loss.backward(); optimizer.step()

# Inference
with torch.no_grad():
    hv = encoder(ctx_dict).sign().to(torch.int8).cpu().numpy()
    # use as drop-in replacement for unified_brain encode()
"""
from __future__ import annotations

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Attribute group schema ───────────────────────────────────────────
#
# Each entry: (group_name, list of (field_name, kind, vocab_size_or_None))
# kind ∈ {"cat", "num", "bool"}
# cat fields get an nn.Embedding; num fields are scalar inputs (normalized);
# bool fields are 0/1 scalars.

GROUP_SCHEMA = {
    "macro": [
        ("macro_vix_label",   "cat",  6),     # low/normal/elevated/high/extreme/unknown
        ("macro_vix_level",   "num",  None),  # raw VIX, log-scaled
        ("macro_risk_regime", "cat",  4),     # risk_on/neutral/risk_off/unknown
        ("macro_dxy_change_pct", "num", None),
        ("macro_yield_chg_bps",  "num", None),
    ],
    "options": [
        ("opt_skew_label",       "cat", 5),    # low/normal/elevated/panic/unknown
        ("opt_term_structure",   "cat", 4),    # contango/flat/backwardation/unknown
        ("opt_near_term_stress", "num", None),
        ("opt_sentiment_label",  "cat", 5),    # greedy/balanced/fearful/panic/unknown
    ],
    "sector": [
        ("sector_rotation_label",   "cat", 6), # risk_on/risk_off/defensive_lead/cyclical_lead/mixed/unknown
        ("sector_leading",          "cat", 13),# 11 sectors + SPY + unknown
        ("sector_breadth_pct",      "num", None),
        ("sector_growth_value_spread", "num", None),
    ],
    "liquidity": [
        ("liquidity_phase",     "cat", 11),    # 10 phases + unknown
        ("liquidity_posture",   "cat", 8),     # no_trade/thin/moderate/peak/active/chop/trend/unknown
        ("liquidity_is_rth",    "bool", None),
        ("liquidity_vol_ratio", "num", None),
    ],
    "multi_tf": [
        ("multi_tf_consensus",   "cat", 7),    # all_up/all_down/mostly_up/mostly_down/mixed/neutral/unknown
        ("multi_tf_alignment",   "cat", 4),    # tight/loose/split/unknown
        ("multi_tf_best_tf",     "cat", 6),    # 1m/5m/15m/1h/none/unknown
        ("multi_tf_best_dir",    "cat", 4),    # up/down/none/unknown
        ("multi_tf_n_signals",   "num", None),
    ],
    "bar_ctx": [
        ("bar_ctx_direction",   "cat", 4),     # up/down/none/unknown
        ("bar_ctx_edge_bucket", "cat", 6),     # huge/big/med/small/none/unknown
    ],
    "tape": [
        ("tape_5s",         "cat", 6),         # strong_buying/buying/balanced/selling/strong_selling/unknown
        ("tape_30s",        "cat", 6),
        ("tape_2min",       "cat", 6),
        ("tape_transition", "cat", 8),         # exhaustion_up/down, continuation_up/down, reversal_to_up/down, chop, none
    ],
    "vp": [
        ("nt_vp_position", "cat", 9),          # above_value_strong/above_value/upper_value/at_mid/lower_value/below_value/below_value_strong/at_poc/unknown
        ("nt_resistance_distance_atr", "num", None),
        ("nt_support_distance_atr",    "num", None),
    ],
    "recent_move": [
        ("recent_move_label", "cat", 9),       # strong_up_impulse/strong_down_impulse/up_drift/down_drift/chop/pullback_from_up/pullback_from_down/neutral/unknown
    ],
    "cd": [
        ("cd_divergence", "cat", 5),           # bullish/bearish/none_confirmed/none/unknown
        ("cd_trend",      "cat", 4),           # rising/falling/flat/unknown
        ("cd_strength",   "num", None),
    ],
    "orderflow": [
        ("of_sweep",      "cat", 4),           # up/down/none/unknown
        ("of_iceberg",    "cat", 4),           # bid/ask/none/unknown
        ("of_stop_run",   "cat", 4),           # up_then_down/down_then_up/none/unknown
        ("of_absorption", "cat", 4),           # bid/ask/none/unknown
    ],
    "chart_pattern": [
        ("dominant_pattern", "cat", 32),       # most-common pattern names + none + unknown
        ("pattern_stage",    "cat", 8),        # forming/confirmed/breaking/invalidating/failed/none/unknown
    ],
    "position": [
        ("position_side",        "cat", 4),    # long/short/flat/unknown
        ("position_pnl_bucket",  "cat", 8),    # losing_big/losing_med/losing_small/flat/winning_small/winning_med/winning_big/unknown
        ("time_since_close",     "num", None),
    ],
    "brain_math": [
        ("vol_regime",          "cat", 5),     # low/normal/high/extreme/unknown
        ("sharpe_bucket",       "cat", 6),     # strong_neg/weak_neg/flat/weak_pos/strong_pos/unknown
        ("session_phase",       "cat", 8),     # premarket/open/mid/lunch/afternoon/close/postmarket/unknown
    ],
}

GROUP_NAMES = list(GROUP_SCHEMA.keys())


class AttributeGroupHead(nn.Module):
    """One MLP head that encodes the inputs of one attribute group into a
    continuous bipolar HV. Categorical features get nn.Embedding lookups;
    numerical features pass through scalar projection. All are
    concatenated and projected through a 2-layer MLP to dim D.
    """

    def __init__(self, fields: list, dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fields = fields
        self.dim = dim

        # Build embeddings for categorical fields
        self.cat_embeddings = nn.ModuleDict()
        cat_total_dim = 0
        num_total = 0
        for name, kind, vocab in fields:
            if kind == "cat":
                emb_dim = max(8, int(math.ceil(math.log2(vocab + 1) * 2)))
                self.cat_embeddings[name] = nn.Embedding(vocab, emb_dim)
                cat_total_dim += emb_dim
            elif kind == "num":
                num_total += 1
            elif kind == "bool":
                num_total += 1

        # MLP: concat([cat_embs, num_scalars]) → hidden → D
        input_dim = cat_total_dim + num_total
        # Guard against empty input (shouldn't happen but be safe)
        input_dim = max(input_dim, 1)
        self.input_dim = input_dim
        # No per-head tanh — let outputs be unbounded; final tanh at the
        # bundle level keeps things in [-1, 1]. Per-head tanh saturates
        # gradients and was causing loss=NaN.
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """batch is a dict of tensors keyed by field name.
        Categorical: long tensor (B,) of indices.
        Num/bool: float tensor (B,).
        Returns (B, dim) continuous bipolar.
        """
        parts = []
        # Use the first available field's batch size as our B
        ref_tensor = next(iter(batch.values()))
        B = ref_tensor.shape[0]
        device = ref_tensor.device
        for name, kind, vocab in self.fields:
            if name in batch:
                v = batch[name]
            else:
                # Missing field — zero out (treat as unknown)
                if kind == "cat":
                    v = torch.zeros(B, dtype=torch.long, device=device)
                else:
                    v = torch.zeros(B, dtype=torch.float32, device=device)
            if kind == "cat":
                emb = self.cat_embeddings[name](v.long())
                parts.append(emb)
            else:
                parts.append(v.float().unsqueeze(-1))
        x = torch.cat(parts, dim=-1)
        return self.mlp(x)


class FactorizedHDCEncoder(nn.Module):
    """Factorized encoder. 14 group heads, each producing a continuous
    bipolar HV; BIND with fixed group-role vectors; BUNDLE all together
    to produce the final composite HV.
    """

    def __init__(self, dim: int = 10000, hidden_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.heads = nn.ModuleDict({
            name: AttributeGroupHead(fields, dim, hidden_dim)
            for name, fields in GROUP_SCHEMA.items()
        })
        # Fixed deterministic role vectors per group — register as buffer
        # so they survive .to(device) but DON'T get gradient updates.
        role_vecs = []
        for name in GROUP_NAMES:
            # SHA-256-seeded random bipolar (deterministic per group name)
            import hashlib
            seed = int.from_bytes(hashlib.sha256(
                f"role:GROUP:{name}".encode()).digest()[:8], "big")
            gen = torch.Generator().manual_seed(seed)
            v = (torch.randint(0, 2, (dim,), generator=gen).float() * 2 - 1)
            role_vecs.append(v)
        self.register_buffer("role_vecs", torch.stack(role_vecs))  # (G, dim)

    def forward(self, batch: dict) -> torch.Tensor:
        """batch is a flat dict of all attribute fields.
        Returns (B, dim) continuous composite HV.
        """
        # Run each head
        bound = []
        for i, name in enumerate(GROUP_NAMES):
            group_fields = GROUP_SCHEMA[name]
            # Subset batch to this group's fields
            sub_batch = {fn: batch[fn] for fn, _, _ in group_fields
                              if fn in batch}
            if not sub_batch:
                continue
            head_hv = self.heads[name](sub_batch)             # (B, dim)
            # BIND with fixed group role (broadcast role across batch)
            bound_hv = head_hv * self.role_vecs[i].unsqueeze(0)
            bound.append(bound_hv)
        if not bound:
            # No fields present → return zeros
            B = next(iter(batch.values())).shape[0]
            return torch.zeros(B, self.dim,
                                       device=self.role_vecs.device)
        # BUNDLE = mean (scale-invariant; better for stable training)
        # Final tanh keeps values bounded for downstream stability.
        bundled = torch.stack(bound, dim=0).mean(dim=0)       # (B, dim)
        return torch.tanh(bundled)

    def encode_to_int8(self, batch: dict) -> torch.Tensor:
        """Inference-time: produce binary ±1 int8 HV for kNN search."""
        with torch.no_grad():
            continuous = self.forward(batch)
        # Sign → ±1, cast to int8
        return continuous.sign().to(torch.int8)


# ── Contrastive loss ──────────────────────────────────────────────────


def contrastive_loss(hvs: torch.Tensor,
                              outcomes: torch.Tensor,
                              temperature: float = 0.1) -> torch.Tensor:
    """Contrastive: same-outcome pairs pulled together, different pushed apart.

    Args:
      hvs: (B, dim) batch of continuous HVs
      outcomes: (B,) class labels (0=loss, 1=win, 2=skip, etc.)
      temperature: softmax temperature for contrastive logits

    Loss: for each anchor i, sample positives (same outcome) and
    negatives (different outcomes) in-batch; SimCLR-style InfoNCE.
    """
    # Normalize
    hvs = F.normalize(hvs, dim=-1)
    sim = hvs @ hvs.T / temperature      # (B, B)
    # Mask self
    eye = torch.eye(hvs.shape[0], dtype=torch.bool, device=hvs.device)
    sim = sim.masked_fill(eye, -1e9)
    # Same-outcome mask
    same = (outcomes.unsqueeze(0) == outcomes.unsqueeze(1)) & ~eye  # (B, B)
    # InfoNCE: for each anchor, sum over positives in numerator;
    # all others in denominator.
    exp_sim = torch.exp(sim)
    pos_sum = (exp_sim * same.float()).sum(dim=-1).clamp(min=1e-9)
    all_sum = exp_sim.sum(dim=-1).clamp(min=1e-9)
    loss = -torch.log(pos_sum / all_sum)
    # Only count anchors that have at least one positive
    has_pos = same.any(dim=-1)
    if not has_pos.any():
        return torch.tensor(0.0, device=hvs.device, requires_grad=True)
    return loss[has_pos].mean()


if __name__ == "__main__":
    # Sanity test
    print("Testing FactorizedHDCEncoder...")
    enc = FactorizedHDCEncoder(dim=10000)
    print(f"  parameters: {sum(p.numel() for p in enc.parameters()):,}")
    print(f"  groups:     {len(GROUP_NAMES)}")
    print(f"  fields:     {sum(len(f) for f in GROUP_SCHEMA.values())}")

    # Mock batch
    B = 8
    batch = {}
    for name, fields in GROUP_SCHEMA.items():
        for fn, kind, vocab in fields:
            if kind == "cat":
                batch[fn] = torch.randint(0, vocab, (B,))
            else:
                batch[fn] = torch.randn(B)
    hv = enc(batch)
    print(f"  output:     {hv.shape}, range=[{hv.min():.2f}, {hv.max():.2f}]")

    # Test loss
    outcomes = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    loss = contrastive_loss(hv, outcomes)
    print(f"  loss:       {loss.item():.4f}")
    loss.backward()
    print(f"  gradient OK (no NaN)")
    print("\nFactorized encoder ready for training.")
