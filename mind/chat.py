"""
autopilot/chat.py - interactive REPL for talking to Telp.

Wraps FluentTelp in a simple line-based prompt loop.  Special slash
commands give you in-conversation tools:

  /help         show available commands
  /stats        lattice + claim-store + generator stats
  /teach FACT   manually add a fact ("/teach Apple was founded in 1976")
  /forget       clear current conversation context (topic words)
  /verbose      toggle internal-state debug output
  /quit         exit

No special command? Whatever you type is sent to Telp as a turn.
"""
from __future__ import annotations

import sys
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))


def _help() -> str:
    return (
        "Commands:\n"
        "  /help        show this help\n"
        "  /stats       lattice + claim-store + generator stats\n"
        "  /teach FACT  add a fact (e.g. '/teach Tesla was founded in 2003')\n"
        "  /forget      forget current topic continuity\n"
        "  /verbose     toggle internal-state debug output\n"
        "  /creative N  override creativity dial for next turn (0.0-1.0 or 'auto')\n"
        "               examples: /creative 0   (literal facts only)\n"
        "                         /creative 0.5 (synthesize a bit)\n"
        "                         /creative 1   (write me a story)\n"
        "                         /creative auto (let the router pick)\n"
        "  /quit        exit"
    )


def _print_banner() -> None:
    print()
    print("=" * 70)
    print("  TELP — fluent HDC conversational agent (no LLM)")
    print("  Type your question.  /help for commands.  /quit to exit.")
    print("=" * 70)
    print()


def _stats(telp) -> str:
    a = telp.agent
    lat = getattr(a, "lattice", None)
    sq  = getattr(a, "structured", None)
    sg  = getattr(a, "seq", None)
    parts = []
    if lat is not None:
        parts.append(f"  lattice memories : {lat.count()}")
    if sq is not None and hasattr(sq, "claim_triple"):
        parts.append(f"  claims stored    : {len(sq.claim_triple)}")
    if sg is not None and hasattr(sg, "stats"):
        try:
            s = sg.stats()
            n_gram_total = sum(v.get("memories", 0) for v in s.values()
                                  if isinstance(v, dict))
            parts.append(f"  n-gram memories  : {n_gram_total}")
        except Exception:
            pass
    parts.append(f"  turns this chat  : {len(a.turns)}")
    return "\n".join(parts) if parts else "(no stats available)"


def _main() -> None:
    import io
    # Conditional utf-8 wrap (same idiom as the ingest CLIs)
    if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                          errors="replace")

    from mind.fluency import FluentTelp
    from mind.seed_identity import seed_if_needed
    from lattice.query_router import QueryRouter
    telp = FluentTelp()
    router = QueryRouter()
    # Manual creativity override; None means "let the router pick"
    creativity_override: float | None = None
    seed_result = seed_if_needed(telp.agent)
    if seed_result.get("seeded"):
        print(f"[chat] identity seeded: +{seed_result['lattice_added']} "
              f"facts, +{seed_result['claims_added']} claims",
              flush=True)
    verbose = False

    _print_banner()

    while True:
        try:
            line = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[chat] bye.")
            return
        if not line:
            continue

        # ─── Slash commands ────────────────────────────────────────
        if line.startswith("/"):
            cmd, *rest = line.split(maxsplit=1)
            arg = rest[0] if rest else ""
            if cmd in ("/quit", "/exit"):
                print("[chat] bye.")
                return
            if cmd == "/help":
                print(_help())
                continue
            if cmd == "/stats":
                print(_stats(telp))
                continue
            if cmd == "/teach":
                fact = arg.strip()
                if not fact:
                    print("usage: /teach <a complete fact, e.g. 'X was founded in YEAR'>")
                    continue
                n = telp.agent.structured.add_sentence(fact,
                                                           source="user_taught")
                telp.agent.lattice.add(fact, source="user_taught",
                                            turn=len(telp.agent.turns))
                telp.agent.encoder.add_sentence(fact)
                print(f"[chat] taught: +{n} claim(s) added, +1 lattice memory")
                continue
            if cmd == "/forget":
                telp.last_topic_words.clear()
                print("[chat] topic continuity reset.")
                continue
            if cmd == "/verbose":
                verbose = not verbose
                print(f"[chat] verbose = {verbose}")
                continue
            if cmd == "/creative":
                a = arg.strip().lower()
                if a in ("auto", "", "off", "none"):
                    creativity_override = None
                    print("[chat] creativity = auto (router decides per turn)")
                else:
                    try:
                        v = float(a)
                        if not 0.0 <= v <= 1.0:
                            raise ValueError("out of range")
                        creativity_override = v
                        print(f"[chat] creativity locked at {v:.2f}")
                    except Exception:
                        print("[chat] /creative N where N is 0.0-1.0, "
                                f"or 'auto'.  got: {a!r}")
                continue
            print(f"[chat] unknown command: {cmd}.  Try /help.")
            continue

        # ─── Normal turn ───────────────────────────────────────────
        # Route the query: continuous creativity dial.  The dial is
        # ALSO passed to telp.respond() so the response composition
        # (not just the tag) actually varies with creativity — direct
        # gets terse, imagined gets a transition-chain extension.
        route = router.classify(line)
        from lattice.query_router import creativity_to_tag
        if creativity_override is not None:
            effective_cre = creativity_override
            effective_tag = creativity_to_tag(effective_cre)
            override_note = f" /override={effective_cre:.2f}"
        else:
            effective_cre = route.creativity
            effective_tag = route.tag
            override_note = ""
        try:
            response = telp.respond(line, creativity=effective_cre)
        except Exception as e:
            print(f"[chat] internal error: {e}")
            continue

        # Four-level tag: direct | synthesized | extrapolated | imagined
        print(f"Telp [{effective_tag} {effective_cre:.2f}"
                f"{override_note}]> {response}")
        if verbose:
            cues = []
            if route.matched_recall:
                cues.append(f"rec={route.matched_recall[:3]}")
            if route.matched_creative:
                cues.append(f"cre={route.matched_creative[:3]}")
            if cues:
                print(f"      (router: {'  '.join(cues)}  "
                        f"why: {route.rationale})")
            else:
                print(f"      (router: {route.rationale})")
        if verbose and telp.agent.turns:
            t = telp.agent.turns[-1]
            mems = t.get("retrieved_memories") or []
            if mems:
                top = mems[0]
                print(f"      (top retrieval: sim={top.get('similarity',0):.2f} "
                        f"src={top.get('source','?')})")
            if t.get("extracted_triples"):
                print(f"      (claims extracted: {t['extracted_triples']})")
            if t.get("kg_hits"):
                print(f"      (kg_hits: {len(t['kg_hits'])})")
        print()


if __name__ == "__main__":
    _main()
