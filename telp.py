#!/usr/bin/env python3
"""
telp.py - THE door. One entry point to one mind.

    python telp.py                     talk to Telp (REPL)
    python telp.py ask "question"      one-shot question
    python telp.py see img.png ...     Telp looks at images (remembers them)
    python telp.py seen                what Telp has seen
    python telp.py recall "a query"    find a sight by meaning
    python telp.py stats               memory stats

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
                # another process may have taught/learned/forgotten - resync
                lat = t.agent.lattice
                try:
                    n_db = lat._con.execute(
                        "SELECT COUNT(*) FROM memories").fetchone()[0]
                    if n_db != len(lat._ids):
                        print(f"[serve] memory changed on disk "
                              f"({len(lat._ids)} -> {n_db}) - resyncing",
                              flush=True)
                        lat._reload_from_disk()
                except Exception:
                    pass
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


def main() -> int:
    import argparse
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

    args = ap.parse_args()
    if args.cmd is None:
        return cmd_chat(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
