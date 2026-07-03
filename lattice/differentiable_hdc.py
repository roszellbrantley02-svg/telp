"""
lattice/differentiable_hdc.py - HDC operations as differentiable PyTorch ops.

The bridge between HDC and gradient descent. Replace:
  - XOR (binary)         -> elementwise product (bipolar / continuous)
  - majority bundle      -> sum or mean (then optional sign at inference)
  - Hamming distance     -> negative cosine similarity (continuous)

All operations remain composable as differentiable functions. Embeddings
are float vectors during training; we can binarize at inference for
pure-HDC deployment.

This is the technique from Hersche et al., ETH Zurich, 2023:
"Differentiable Hyperdimensional Computing".

For our task: learn entity + relation embeddings such that
    s ⊙ r ≈ o    (bind subject with relation gives object)
holds for stored (s, r, o) triples. Then for held-out subjects, the
learned embedding should let s_new ⊙ HAS_CAPITAL ≈ capital_of_s_new.

This is essentially DistMult / knowledge graph embedding, just with
the bipolar HDC convention.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableHDC(nn.Module):
    """Learn entity + relation embeddings for relational triples.

    Score function: cosine similarity of (s ⊙ r) with o.
    Training: contrastive (positive vs negative samples).
    Inference: given (s, r), return s ⊙ r and find nearest entity.
    """

    def __init__(self, n_entities: int, n_relations: int, dim: int = 256):
        super().__init__()
        self.dim = dim
        self.entity_emb   = nn.Embedding(n_entities, dim)
        self.relation_emb = nn.Embedding(n_relations, dim)
        # Initialize ~bipolar (uniform [-1, 1])
        nn.init.uniform_(self.entity_emb.weight, -1.0, 1.0)
        nn.init.uniform_(self.relation_emb.weight, -1.0, 1.0)

    def bind(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Bind = elementwise product (continuous analog of XOR on bipolar)."""
        return x * y

    def predict_object(self, subject_ids: torch.Tensor,
                         relation_ids: torch.Tensor) -> torch.Tensor:
        """Given (s, r), return predicted object embedding."""
        s = self.entity_emb(subject_ids)
        r = self.relation_emb(relation_ids)
        return self.bind(s, r)

    def score(self, subject_ids: torch.Tensor,
                relation_ids: torch.Tensor,
                object_ids: torch.Tensor) -> torch.Tensor:
        """Cosine similarity of predicted object with actual object."""
        predicted = self.predict_object(subject_ids, relation_ids)
        actual = self.entity_emb(object_ids)
        return F.cosine_similarity(predicted, actual, dim=-1)

    def score_all_objects(self, subject_ids: torch.Tensor,
                             relation_ids: torch.Tensor) -> torch.Tensor:
        """For each (s, r) pair, return scores against ALL entities.

        Returns: (batch, n_entities) of cosine similarities.
        Useful for inference (rank all candidates) and softmax loss.
        """
        predicted = self.predict_object(subject_ids, relation_ids)  # (B, dim)
        all_entities = self.entity_emb.weight   # (n_entities, dim)
        # Normalize both
        predicted = F.normalize(predicted, dim=-1)
        all_entities = F.normalize(all_entities, dim=-1)
        return predicted @ all_entities.T   # (B, n_entities)


def train_triples(model: DifferentiableHDC,
                   triples: list[tuple[int, int, int]],
                   epochs: int = 200,
                   lr: float = 0.05,
                   verbose: bool = True) -> list[float]:
    """Train the model via softmax-over-all-entities cross-entropy.

    triples: list of (subject_id, relation_id, object_id) integer triples.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    s_ids = torch.tensor([t[0] for t in triples], dtype=torch.long)
    r_ids = torch.tensor([t[1] for t in triples], dtype=torch.long)
    o_ids = torch.tensor([t[2] for t in triples], dtype=torch.long)

    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # Cross-entropy: for each (s,r), the correct o should have highest score
        scores = model.score_all_objects(s_ids, r_ids)  # (B, n_ent)
        # We want o_ids to have max score
        loss = F.cross_entropy(scores, o_ids)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            with torch.no_grad():
                # Top-1 accuracy on the training set
                preds = scores.argmax(dim=-1)
                acc = (preds == o_ids).float().mean().item()
            print(f"  epoch {epoch:3d}: loss={loss.item():.4f}  train-acc={acc*100:.0f}%")
    return losses


def predict(model: DifferentiableHDC, subject_id: int, relation_id: int,
              candidate_ids: list[int], top_k: int = 5
              ) -> list[tuple[int, float]]:
    """Predict the top-k objects for (subject, relation).

    candidate_ids: which entity IDs to rank.
    Returns list of (entity_id, score).
    """
    model.eval()
    with torch.no_grad():
        s = torch.tensor([subject_id])
        r = torch.tensor([relation_id])
        all_scores = model.score_all_objects(s, r).squeeze(0)
        cand_scores = [(int(i), float(all_scores[i])) for i in candidate_ids]
        cand_scores.sort(key=lambda x: -x[1])
        return cand_scores[:top_k]
