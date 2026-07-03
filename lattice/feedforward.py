"""
lattice/feedforward.py - HDC feed-forward layer.

The thing HDC has been missing: a way to do what feed-forward layers in
transformers do — store knowledge, transform representations, compose
features — without gradient descent and without floating-point math.

Architecture (bind + cleanup)
-----------------------------
A single layer is:
    1. A learned 'transform_key' hypervector
    2. A 'cleanup memory' of known output hypervectors

Forward pass:
    noisy_output = input XOR transform_key
    output       = cleanup_memory.nearest(noisy_output)

The XOR is a learned linear transformation. The cleanup is the source
of nonlinearity — winner-take-all on similarity returns different
outputs for inputs in different similarity basins. This is what gives
the layer expressive power beyond pure linear algebra.

Learning (Hebbian, no gradient descent)
---------------------------------------
Given (input_hv, output_hv) training pairs:
    transform_key = bundle( input_i XOR output_i  for each pair )

The bundled XORs find the 'average transformation' that maps inputs to
outputs across the training set. Outputs are stored in cleanup memory
verbatim.

The empirical question: does this generalize? If the input→output
relation has consistent geometry in HD space, the bundled key will
work on held-out pairs. If the relation is too varied, it won't.

References:
  Plate, T. (1995) - cleanup memory in HRR
  Kanerva, P. (2009) - HDC primitives
  Eliasmith & Anderson (2002) - learned transformations in vector
    symbolic systems
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import D, bundle, hamming_distance


class HDCFeedForward:
    """One HDC feed-forward layer: bind + cleanup."""

    def __init__(self):
        self.transform_key: np.ndarray | None = None
        # cleanup memory: list of (name, hypervector) pairs
        self.cleanup_items: list[tuple[str, np.ndarray]] = []
        # we keep training pairs for re-training / inspection
        self._training_pairs: list[tuple[np.ndarray, np.ndarray]] = []

    def add_cleanup_item(self, name: str, hv: np.ndarray) -> None:
        """Add an output the layer is allowed to return."""
        self.cleanup_items.append((name, hv.copy()))

    def train(self, pairs: list[tuple[str, np.ndarray, str, np.ndarray]]
              ) -> None:
        """Learn input -> output mapping from training pairs.

        pairs: list of (input_name, input_hv, output_name, output_hv)

        Hebbian learning rule:
            transform_key = bundle( input_i XOR output_i  for each pair )
        """
        if not pairs:
            return
        # Add all outputs to cleanup memory (deduplicated by name)
        seen = {n for n, _ in self.cleanup_items}
        for _in_name, _in_hv, out_name, out_hv in pairs:
            if out_name not in seen:
                self.add_cleanup_item(out_name, out_hv)
                seen.add(out_name)
        # Compute the transformation key
        xor_vectors = []
        for _, in_hv, _, out_hv in pairs:
            xor_vectors.append(np.bitwise_xor(in_hv, out_hv))
            self._training_pairs.append((in_hv.copy(), out_hv.copy()))
        self.transform_key = bundle(xor_vectors)

    def forward(self, x: np.ndarray, top_k: int = 1
                ) -> list[tuple[str, int, np.ndarray]]:
        """Apply the learned transformation + cleanup.

        Returns top_k (name, hamming_distance, hypervector) tuples
        ordered by closeness to the transformed input.
        """
        if self.transform_key is None:
            raise RuntimeError("Layer not trained")
        if not self.cleanup_items:
            raise RuntimeError("Cleanup memory is empty")
        noisy = np.bitwise_xor(x, self.transform_key)
        scored = []
        for name, hv in self.cleanup_items:
            d = hamming_distance(noisy, hv)
            scored.append((name, d, hv))
        scored.sort(key=lambda r: r[1])
        return scored[:top_k]

    def forward_raw(self, x: np.ndarray) -> np.ndarray:
        """Apply just the linear (XOR) transformation, no cleanup."""
        if self.transform_key is None:
            raise RuntimeError("Layer not trained")
        return np.bitwise_xor(x, self.transform_key)


# ─── Stack of layers ───────────────────────────────────────────────


class HDCStack:
    """A multi-layer HDC network.

    Each layer is an HDCFeedForward. Layers are applied sequentially:
    the cleaned-up output of layer N is the input to layer N+1.

    This is the HDC analog of stacking feed-forward layers in a neural
    network. If 1-layer learns 'country -> capital', 2-layer can
    learn 'country -> capital -> continent' by composing transformations.

    Two operating modes:
      forward(x):       chain layers with cleanup between each
                        (decisive at each step — uses cleanup memory output)
      forward_raw(x):   chain raw XOR transformations without cleanup
                        (defers commitment until the final layer)
    """

    def __init__(self, layers: list[HDCFeedForward] | None = None):
        self.layers: list[HDCFeedForward] = layers or []

    def add_layer(self, layer: HDCFeedForward) -> None:
        self.layers.append(layer)

    def forward(self, x: np.ndarray, top_k: int = 1, trace: bool = False
                ) -> list[tuple[str, int, np.ndarray]] | dict:
        """Run input through every layer with cleanup between layers.

        Returns the final layer's top_k results. Each intermediate
        layer commits to its top-1 cleanup output before passing on.

        If trace=True, returns a dict with intermediate predictions
        at each layer.
        """
        if not self.layers:
            raise RuntimeError("Empty stack")
        current = x
        trace_log = []
        for i, layer in enumerate(self.layers):
            results = layer.forward(current, top_k=1)
            trace_log.append({
                "layer": i,
                "top1_name": results[0][0],
                "top1_distance": results[0][1],
            })
            # Commit to top-1's cleaned-up hypervector as input to next layer
            current = results[0][2]
        # On the final layer, return top_k for inspection
        final_results = self.layers[-1].forward(x_for_final := current, top_k=top_k) \
            if False else None  # placeholder; will recompute below

        # Recompute final layer's top_k from the second-to-last cleaned output
        # so the trace is consistent.
        if len(self.layers) == 1:
            final_input = x
        else:
            # Re-run all but last with cleanup
            inp = x
            for layer in self.layers[:-1]:
                inp = layer.forward(inp, top_k=1)[0][2]
            final_input = inp
        final_results = self.layers[-1].forward(final_input, top_k=top_k)

        if trace:
            return {"trace": trace_log, "final": final_results}
        return final_results

    def forward_raw(self, x: np.ndarray, top_k: int = 1
                    ) -> list[tuple[str, int, np.ndarray]]:
        """Chain ALL layers without cleanup, then cleanup only at the end.

        This is more like 'pure linear composition' — the noisy XOR
        results from each layer accumulate, and only the final layer's
        cleanup memory is consulted. Tests whether HDC can compose
        transformations algebraically without committing to intermediate
        symbols.
        """
        if not self.layers:
            raise RuntimeError("Empty stack")
        current = x
        for layer in self.layers[:-1]:
            current = layer.forward_raw(current)
        return self.layers[-1].forward(current, top_k=top_k)
