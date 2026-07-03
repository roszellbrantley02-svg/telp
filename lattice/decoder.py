"""
lattice/decoder.py - LLM narrator for HDC retrieval results.

The decoder is the MOUTH of the Lattice. It does ONE job:
take the structured rows that HDC retrieval returns and translate
them into natural-language narration.

It does NOT:
  - Think (HDC already retrieved)
  - Reason (HDC algebra already composed)
  - Know things (the memories are the knowledge)
  - Be creative (it should be faithful to the memories)

It DOES:
  - Take a query + the top retrieved memories
  - Write a clean conversational response that presents them
  - Stay concise and faithful

Architecture
------------
A small instruction-tuned LLM (Qwen2.5-0.5B-Instruct, ~500M params).
Runs on CPU at ~5-15 tok/s. Fits in <2GB RAM. Loaded once at first use.

The prompt structure:
  SYSTEM: "You are a memory narrator. Present these recalled memories
           in a natural, conversational way. Do not add information
           that isn't in them. Be concise."
  USER:   "Query: {query}\n\nRecalled memories:\n1. ...\n2. ...\n\n
           Respond using only these memories."

Usage:
    from lattice.decoder import LatticeDecoder
    dec = LatticeDecoder()
    text = dec.narrate(query, results)
"""
from __future__ import annotations

import os
from typing import Optional


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


_SYSTEM_PROMPT = (
    "You restate stored memories. You do not invent, ask questions, or add facts. "
    "If you are given one or more memories, paraphrase the most relevant one(s) "
    "in plain second-person prose (you/your). "
    "Use only the words and ideas in the memories themselves. "
    "Never add new information. Never speculate. Never ask the user a question. "
    "Output ONE or TWO short declarative sentences and stop."
)


class LatticeDecoder:
    """Tiny LLM that narrates HDC retrieval results in natural language."""

    def __init__(self, model_name: str = DEFAULT_MODEL,
                  device: str = "cpu",
                  max_new_tokens: int = 80,
                  min_similarity: float = 0.30,
                  max_memories: int = 3):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.min_similarity = min_similarity   # filter out weak matches
        self.max_memories = max_memories        # only show top-K to the LLM
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        # Lazy import — transformers + accelerate are heavy
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Use float32 on CPU for stability; on GPU we'd use float16
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=self.device,
        )
        self._model.eval()

    def _filter_results(self, results: list[dict]) -> list[dict]:
        """Drop low-similarity matches and cap at max_memories."""
        keep = [r for r in results
                  if r.get("similarity", 0.0) >= self.min_similarity]
        return keep[:self.max_memories]

    def _format_memories(self, results: list[dict]) -> str:
        """Convert HDC result rows to a clean text block for the LLM.

        Output style is intentionally terse: numbered list, no closeness
        tags, just the memory text. Less for the small LLM to confuse.
        """
        if not results:
            return "(no memories matched the query)"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['text']}")
        return "\n".join(lines)

    def narrate(self, query: str, results: list[dict]) -> str:
        """Turn (query, results) into a natural-language response."""
        self._ensure_loaded()
        import torch

        kept = self._filter_results(results)
        if not kept:
            return ("I don't have any memories that match that closely. "
                    f"The best I found was {results[0]['distance_pct']*100:.0f}% "
                    "off, which is too far to be reliable.") if results else (
                    "I don't have any memories stored yet.")

        mem_block = self._format_memories(kept)
        user_msg = (
            f"Query: {query}\n\n"
            f"Memories:\n{mem_block}\n\n"
            f"Paraphrase the most relevant memory in one short sentence to "
            f"the user. Do not ask anything. Do not add details."
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,           # deterministic narration
                temperature=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        # Strip the prompt prefix to get only the model's response
        gen_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return text.strip()


# ─── Singleton accessor ────────────────────────────────────────────


_DECODER: Optional[LatticeDecoder] = None


def get_decoder() -> LatticeDecoder:
    global _DECODER
    if _DECODER is None:
        _DECODER = LatticeDecoder()
    return _DECODER


# ─── CLI smoke test ────────────────────────────────────────────────


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="Sarah Monday coffee")
    ap.add_argument("--db", default="state/lattice_test.db",
                      help="Path to a Lattice DB to query")
    args = ap.parse_args()

    from lattice.store import Lattice
    L = Lattice(args.db)
    print(f"Loaded {L.count()} memories from {args.db}")
    if L.count() == 0:
        print("DB is empty. Run lattice.test_store first to seed it.")
        return

    print(f"\nQuery: {args.query!r}")
    results = L.query(args.query, k=5)
    print(f"\nRaw HDC retrieval (top {len(results)}):")
    for r in results:
        print(f"  #{r['rank']} d={r['distance_pct']*100:.0f}% "
              f"'{r['text']}'")

    print(f"\nLoading LLM decoder (first run downloads ~1GB) ...")
    dec = get_decoder()

    print("Narrating ...")
    narration = dec.narrate(args.query, results)
    print(f"\nLattice says:")
    print(f"  {narration}")


if __name__ == "__main__":
    main()
