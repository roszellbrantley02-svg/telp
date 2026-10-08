"""
mind/harness_types.py - the shared shapes of Telp's LLM harness.

In harness mode Telp is the memory and the worker, and a language model
(e.g. Qwen in LM Studio) is the writer:

    question
      -> Telp: commands it handles itself (teach, forget, remember...)
      -> Telp builds a BRIEF: fixed instructions, standing memory, a
         compact conversation state, numbered evidence (memory sentences,
         facts, tool results) - only what this question needs, inside a
         token budget, fixed parts first so the model can reuse its
         cached reading of them
      -> the model writes, citing evidence as [n]; it may call Telp's
         tools to dig deeper (search memory, facts, calculate, today)
      -> Telp CHECKS every sentence against the evidence it supplied
      -> Telp updates the conversation state (constant size), so the next
         turn never resends the whole conversation

These dataclasses are the contract between mind/llm_client.py,
mind/brief.py, mind/checker.py and mind/harness.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def estimate_tokens(text: str) -> int:
    """Rough token count for English text (about 4 characters a token).
    Used for budgeting before a request; the model's own usage numbers
    replace it afterwards when the server reports them."""
    return max(1, (len(text) + 3) // 4) if text else 0


@dataclass
class Evidence:
    """One numbered item the model may cite as [n]."""
    n: int
    text: str
    source: str = ""            # "wikipedia:Iceland", "user_taught", "tool:calculate"
    created_at: str | None = None
    kind: str = "memory"        # memory | fact | tool | conversation
    score: float = 0.0          # retrieval score that selected it
    memory_id: int | None = None

    def line(self) -> str:
        """How it is shown to the model."""
        when = f", {self.created_at[:10]}" if self.created_at else ""
        src = f" ({self.source}{when})" if self.source else ""
        return f"[{self.n}] {self.text}{src}"


@dataclass
class Brief:
    """Everything the model reads for one turn.

    Order matters for speed: system and standing change rarely, so they
    come first and the model server can reuse its cached reading of them
    (LM Studio / llama.cpp reuse the longest unchanged prefix)."""
    system: str
    standing: str = ""
    state: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    question: str = ""
    naive_tokens: int = 0       # what sending everything would have cost

    def context_text(self) -> str:
        parts = []
        if self.standing:
            parts.append("What you know about the user and Telp:\n"
                         + self.standing)
        if self.state:
            parts.append("Conversation so far:\n" + self.state)
        if self.evidence:
            parts.append("Sources for this question:\n"
                         + "\n".join(e.line() for e in self.evidence))
        else:
            parts.append("Sources for this question: none found.")
        return "\n\n".join(parts)

    def messages(self) -> list[dict]:
        """OpenAI-style chat messages, stable parts first."""
        return [
            {"role": "system", "content": self.system},
            {"role": "user",
             "content": self.context_text() + "\n\nQuestion: " + self.question},
        ]

    def token_estimate(self) -> int:
        return sum(estimate_tokens(m["content"]) for m in self.messages())


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMReply:
    text: str                           # final answer, thinking removed
    thinking: str = ""                  # the model's reasoning, if any
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)   # prompt_tokens, completion_tokens, ...
    seconds: float = 0.0
    finish_reason: str = ""


@dataclass
class SentenceCheck:
    """Telp's verdict on one sentence of the model's answer."""
    sentence: str
    status: str             # supported | unsupported | no_claim
    cites: list[int] = field(default_factory=list)
    best_evidence: int | None = None
    score: float = 0.0
    note: str = ""


@dataclass
class TurnResult:
    answer: str                         # what the user sees
    evidence: list[Evidence] = field(default_factory=list)
    checks: list[SentenceCheck] = field(default_factory=list)
    brief_tokens: int = 0               # tokens Telp actually sent
    naive_tokens: int = 0               # tokens a send-everything prompt needs
    rounds: int = 0                     # model calls (1 + tool rounds)
    tools_used: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    seconds: float = 0.0
    handled_by: str = "llm"             # llm | telp (command answered directly)
