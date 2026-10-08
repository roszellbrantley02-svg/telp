#!/usr/bin/env python3
"""
telp.py - THE door. One entry point to one mind.

    python telp.py                     talk to Telp (REPL)
    python telp.py ask "question"      one-shot question
    python telp.py see img.png ...     Telp looks at images (remembers them)
    python telp.py seen                what Telp has seen
    python telp.py recall "a query"    find a sight by meaning
    python telp.py stats               memory stats
    python telp.py llm "question"      ask through a local model (LM Studio):
                                       Telp finds, works out and checks;
                                       the model only writes
    python telp.py llm-chat            the same as a conversation
    python telp.py llm-status          is the model server up?

Everything routes through the same organism: perception (lattice/vision) ->
the one lattice memory -> the fluency cascade (mind/) -> voice. The retired
trading lane (autopilot/) is not loaded here.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))


DAEMON_PORT = int(__import__("os").environ.get("TELP_PORT", "7580"))


def _fluent():
    from mind.fluency import FluentTelp
    return FluentTelp()


def _daemon_request(payload: dict, timeout: float = 120.0):
    """Send one request to a running daemon. Returns dict or None (no daemon)."""
    import json
    import socket
    try:
        s = socket.create_connection(("127.0.0.1", DAEMON_PORT), timeout=0.4)
    except OSError:
        return None
    try:
        s.settimeout(timeout)
        s.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode("utf-8")) if buf.strip() else None
    except OSError:
        return None
    finally:
        s.close()


def cmd_serve(_args) -> int:
    """Run Telp as a resident mind: boot once, answer in under a second.
    `telp ask` auto-routes here when the daemon is up. Ctrl+C to stop."""
    import json
    import socket
    print("[serve] waking Telp (one-time boot) ...", flush=True)
    t = _fluent()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", DAEMON_PORT))
    srv.listen(4)
    print(f"[serve] Telp is awake on 127.0.0.1:{DAEMON_PORT} - "
          f"`python telp.py ask ...` is now instant. Ctrl+C to stop.", flush=True)
    try:
        while True:
            conn, _ = srv.accept()
            try:
                conn.settimeout(600)
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                if not buf.strip():
                    continue
                req = json.loads(buf.decode("utf-8"))
                if req.get("op") == "quit":
                    conn.sendall(b'{"reply": "Telp going to sleep."}\n')
                    break
                # another process may have taught/learned/forgotten - resync.
                # SQLite's data_version catches every outside commit (a
                # row-count check missed "forget 5, learn 5").
                lat = t.agent.lattice
                try:
                    if lat.changed_on_disk():
                        n_old = len(lat._ids)
                        lat._reload_from_disk()
                        t.agent._rebuild_structured_qa()
                        print(f"[serve] memory changed on disk "
                              f"({n_old} -> {len(lat._ids)}) - resynced",
                              flush=True)
                except Exception as e:
                    print(f"[serve] resync failed: {e}", flush=True)
                reply = t.respond(req.get("text", ""),
                                  creativity=float(req.get("creativity", 0.30)))
                conn.sendall((json.dumps({"reply": reply}) + "\n").encode("utf-8"))
            except Exception as e:
                try:
                    conn.sendall((json.dumps(
                        {"reply": f"(daemon error: {e})"}) + "\n").encode("utf-8"))
                except OSError:
                    pass
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\n[serve] Telp going to sleep.", flush=True)
    finally:
        srv.close()
    return 0


def cmd_stop(_args) -> int:
    r = _daemon_request({"op": "quit"})
    print(r["reply"] if r else "No daemon running.")
    return 0


def cmd_chat(_args) -> int:
    from mind import chat as _chat
    entry = getattr(_chat, "main", None) or getattr(_chat, "_main")
    entry()
    return 0


def cmd_ask(args) -> int:
    text = " ".join(args.question)
    # a running daemon answers in well under a second; else boot standalone
    r = _daemon_request({"text": text, "creativity": args.creative})
    if r is not None:
        print(r["reply"])
        return 0
    t = _fluent()
    print(t.respond(text, creativity=args.creative))
    return 0


def cmd_see(args) -> int:
    from lattice.vision import see, get_namer, CHAT_LATTICE
    from lattice.standalone_agent import StandaloneAgent
    agent = StandaloneAgent(lattice_path=CHAT_LATTICE)
    namer = get_namer()
    for p in args.images:
        r = see(agent, p, namer)
        lbl = ", ".join(f"{w} {s:.2f}" for w, s in r["labels"])
        print(f"[seen] {Path(p).name}: {r['caption']}   ({lbl})")
    return 0


def cmd_watch(args) -> int:
    from lattice.vision import watch, get_namer, CHAT_LATTICE
    from lattice.standalone_agent import StandaloneAgent
    agent = StandaloneAgent(lattice_path=CHAT_LATTICE)
    namer = get_namer()
    for v in args.videos:
        if "youtube.com" in v or "youtu.be" in v:
            from lattice.growth import watch_youtube
            r = watch_youtube(agent, v, namer=namer)
            if r.get("error"):
                print(f"[watch] {r['title']}: {r['error']}")
            else:
                print(f"[watched] {r['title']}: {r['scenes']} scenes, "
                      f"{r['fused']} sight+speech moments, "
                      f"{r['passages']} spoken passages")
        else:
            r = watch(agent, v, namer=namer)
            print(f"[watched] {Path(v).name}: {r['scenes']} scenes remembered")
            # local files get ears too: listen, fuse sight with speech
            try:
                from lattice.hearing import transcribe
                from lattice.growth import remember_passages
                chunks = transcribe(v)
            except Exception as e:
                print(f"[watch] hearing unavailable ({e})")
                chunks = []
            if chunks:
                import re as _re
                title = r.get("label", Path(v).stem)
                fused = 0
                for t, caption in r.get("scene_list", []):
                    near = " ".join(x for s, x in chunks if t - 4 <= s <= t + 8)
                    near = _re.sub(r"\s+", " ", near).strip()[:200]
                    if near:
                        agent.lattice.add(
                            f"In the video '{title}' at {int(t)}s, while showing "
                            f"{caption.removeprefix('an image showing ')}, the "
                            f"speaker says: \"{near}\"",
                            source=f"video:{title}")
                        fused += 1
                n_p = remember_passages(agent, title, chunks, f"video:{title}")
                agent.lattice.add(
                    f"Telp watched the video '{title}': {r['scenes']} scenes seen, "
                    f"{fused} sight+speech moments, {n_p} spoken passages heard "
                    f"with his own ears.", source=f"video:{title}")
                print(f"[heard] {fused} sight+speech moments, {n_p} passages")
    return 0


def cmd_seen(_args) -> int:
    from lattice.vision import sights, CHAT_LATTICE
    rows = sights(CHAT_LATTICE)
    if not rows:
        print("Telp hasn't seen any images yet.")
    for r in rows:
        print(f"  {r['when']}  {r['caption'].removeprefix('Image: ')}  <- {r['path']}")
    return 0


def cmd_recall(args) -> int:
    from lattice.vision import recall_semantic, CHAT_LATTICE
    for r in recall_semantic(CHAT_LATTICE, " ".join(args.query)):
        print(f"  {r['similarity']:.3f}  {r['caption'].removeprefix('Image: ')}"
              f"  <- {r['path']}")
    return 0


def cmd_learn(args) -> int:
    """Telp grows his own knowledge: fetch a topic or a URL and remember it."""
    from lattice.vision import CHAT_LATTICE
    from lattice.standalone_agent import StandaloneAgent
    from lattice.growth import learn_topic, learn_url
    agent = StandaloneAgent(lattice_path=CHAT_LATTICE, skip_ngram_retrain=True)
    for topic in args.topics:
        if "youtube.com" in topic or "youtu.be" in topic:
            from lattice.growth import learn_youtube
            r = learn_youtube(agent, topic)
        elif topic.startswith(("http://", "https://")):
            r = learn_url(agent, topic)
        else:
            r = learn_topic(agent, topic)
        if r.get("error"):
            print(f"[learn] {r['title']}: {r['error']}")
        else:
            print(f"[learn] {r['title']}: {r['added']} facts remembered")
    return 0


def cmd_teach(args) -> int:
    """Teach Telp a fact directly (claims + lattice + encoder stats)."""
    t = _fluent()
    fact = " ".join(args.fact)
    n = t.agent.structured.add_sentence(fact, source="user_taught")
    t.agent.lattice.add(fact, source="user_taught", turn=0)
    t.agent.encoder.add_sentence(fact)
    print(f"taught: +{n} claim(s), +1 memory - I'll remember that.")
    return 0


def cmd_forget(args) -> int:
    t = _fluent()
    print(t.respond("forget " + " ".join(args.what)))
    return 0


def cmd_stats(_args) -> int:
    t = _fluent()
    for k, v in t.agent.stats().items():
        print(f"  {k}: {v}")
    return 0


# ─── harness mode: Telp as memory + worker, a local model as the writer ──

def _harness(args):
    """A Harness on the one memory, talking to the model server at --url
    (default: LM Studio, http://localhost:1234/v1, or $TELP_LLM_URL)."""
    from mind.harness import Harness
    from mind.llm_client import LLMClient
    client = LLMClient(base_url=args.url, model=args.model)
    return Harness(_fluent(), client, budget_tokens=args.budget,
                   check=not args.no_check, session=args.session,
                   thinking=False if args.no_think else None)


def _llm_turn(h, text: str, stream: bool):
    """Ask once and print the answer (streamed, or whole with Telp's
    marks), any notes, and the token meter."""
    from mind.harness import check_lines, meter_line
    shown = {"answer": False, "thinking": False}

    def token(piece: str) -> None:
        shown["answer"] = True
        print(piece, end="", flush=True)

    def thinking(_piece: str) -> None:
        # a slow model can think for a minute: say so once, not every token
        if not shown["answer"] and not shown["thinking"]:
            shown["thinking"] = True
            print("(thinking...)", flush=True)

    turn = h.ask(text, stream=stream, on_token=token if stream else None,
                 on_thinking=thinking if stream else None)
    if stream:
        print()
        if turn.handled_by == "llm":
            for line in check_lines(turn):   # the check came after the text
                print(line)
    else:
        print(turn.answer)
    for note in getattr(turn, "notes", []):
        print(f"note: {note}")
    print(meter_line(turn))
    return turn


def cmd_llm(args) -> int:
    """One question through the harness: Telp finds, filters, works out;
    the model writes; Telp checks."""
    try:
        h = _harness(args)
    except ValueError as e:                   # e.g. a --budget too small
        print(f"[llm] {e}")
        return 2
    try:
        _llm_turn(h, " ".join(args.question), args.stream)
    except KeyboardInterrupt:
        print("\n[llm] stopped.")
        return 130
    return 0


def cmd_llm_chat(args) -> int:
    """Talk through the harness. The conversation is kept in the memory
    file, so it carries on where it left off."""
    from mind.harness import meter_line, session_line, sources_text
    try:
        h = _harness(args)
    except ValueError as e:
        print(f"[llm-chat] {e}")
        return 2
    url = h.client.base_url
    print(f"Telp + a local model at {url}. Telp remembers, searches and "
          "checks; the model writes.")
    print("/sources  the last turn's sources   /meter  tokens sent   "
          "/new  start a fresh conversation   /quit")
    earlier = h.state.count()
    if earlier:
        print(f"[llm-chat] carrying on our conversation ({earlier} earlier "
              "turn(s); /new starts afresh).")
    while True:
        try:
            line = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        cmd = line.lower()
        if cmd in ("/quit", "/exit", "/q"):
            break
        if cmd == "/sources":
            print(sources_text(h.last))
        elif cmd == "/meter":
            if h.last is not None:
                print(meter_line(h.last))
            print(session_line(h))
        elif cmd == "/new":
            n = h.state.clear()
            print(f"[llm-chat] fresh conversation ({n} earlier turn(s) "
                  "forgotten).")
        elif cmd.startswith("/"):
            print("[llm-chat] commands: /sources /meter /new /quit")
        else:
            try:
                _llm_turn(h, line, args.stream)
            except KeyboardInterrupt:          # a slow answer, cut short
                print("\n[llm-chat] stopped.")
        print()
    return 0


def cmd_llm_status(args) -> int:
    """Is the model server up, which model, how much context."""
    from mind.harness import status_lines
    from mind.llm_client import LLMClient
    info = LLMClient(base_url=args.url, model=args.model).status()
    for line in status_lines(info, budget_tokens=args.budget):
        print(line)
    return 0 if info["reachable"] else 1


def _llm_flags(sp, full: bool = True) -> None:
    sp.add_argument("--url", default=None,
                    help="model server (default $TELP_LLM_URL or "
                         "http://localhost:1234/v1, LM Studio)")
    sp.add_argument("--model", default=None,
                    help="model id (default: the one the server has loaded)")
    sp.add_argument("--budget", type=int, default=1800,
                    help="tokens Telp may send the model per turn")
    if not full:
        return
    sp.add_argument("--no-check", action="store_true",
                    help="don't check the answer against the sources")
    sp.add_argument("--stream", action="store_true",
                    help="show the answer as the model writes it")
    sp.add_argument("--no-think", action="store_true",
                    help="ask the model not to think first (faster)")
    sp.add_argument("--session", default="default",
                    help="which conversation to continue")


def main() -> int:
    import argparse
    # Windows consoles default to cp1252 - an essay quoting Greek
    # (telephone <- "tele") must not crash the door
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="telp", description="Telp - one door, one mind.")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("chat", help="interactive REPL").set_defaults(func=cmd_chat)
    sub.add_parser("serve", help="run Telp as a resident mind (instant asks)"
                   ).set_defaults(func=cmd_serve)
    sub.add_parser("stop", help="put the resident Telp to sleep"
                   ).set_defaults(func=cmd_stop)
    sp = sub.add_parser("ask", help="one-shot question")
    sp.add_argument("question", nargs="+")
    sp.add_argument("--creative", type=float, default=0.30,
                    help="0=terse recall .. 1=imaginative extension")
    sp.set_defaults(func=cmd_ask)
    sp = sub.add_parser("see", help="look at images and remember them")
    sp.add_argument("images", nargs="+")
    sp.set_defaults(func=cmd_see)
    sp = sub.add_parser("watch", help="watch videos scene by scene and remember them")
    sp.add_argument("videos", nargs="+")
    sp.set_defaults(func=cmd_watch)
    sub.add_parser("seen", help="list what Telp has seen").set_defaults(func=cmd_seen)
    sp = sub.add_parser("recall", help="find a sight by meaning")
    sp.add_argument("query", nargs="+")
    sp.set_defaults(func=cmd_recall)
    sp = sub.add_parser("learn", help="fetch a topic (wikipedia) and remember it")
    sp.add_argument("topics", nargs="+")
    sp.set_defaults(func=cmd_learn)
    sp = sub.add_parser("teach", help="teach Telp a fact directly")
    sp.add_argument("fact", nargs="+")
    sp.set_defaults(func=cmd_teach)
    sp = sub.add_parser("forget", help="forget specific memories on command")
    sp.add_argument("what", nargs="+")
    sp.set_defaults(func=cmd_forget)
    sub.add_parser("stats", help="memory stats").set_defaults(func=cmd_stats)
    sp = sub.add_parser("llm", help="ask through a local model (LM Studio): "
                        "Telp finds and checks, the model writes")
    sp.add_argument("question", nargs="+")
    _llm_flags(sp)
    sp.set_defaults(func=cmd_llm)
    sp = sub.add_parser("llm-chat", help="talk through a local model "
                        "(the conversation is remembered)")
    _llm_flags(sp)
    sp.set_defaults(func=cmd_llm_chat)
    sp = sub.add_parser("llm-status", help="is the local model server up? "
                        "which model, how much context")
    _llm_flags(sp, full=False)
    sp.set_defaults(func=cmd_llm_status)

    args = ap.parse_args()
    if args.cmd is None:
        return cmd_chat(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
