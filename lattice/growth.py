"""
lattice/growth.py - Telp grows his own memory.

learn_topic(): fetch a topic from Wikipedia (public API, attributed source,
one request per topic - lawful retrieval with provenance), split into facts,
anchor coreference orphans, and remember into the one lattice.

extract_topics(): guess what a question is ABOUT, so learn-on-miss can go
find it. Heuristics, best-first: quoted spans, capitalized phrases from the
original casing, the question object after of/about/for, remaining content
words (rarest first).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

_PRON = ("it ", "its ", "they ", "their ", "he ", "she ", "his ", "her ",
         "this ", "these ", "the species ", "the animal ")

_STOP = frozenset(
    "what which who whom whose where when why how do does did is are was were "
    "will would can could should the a an of in on at to for from about with "
    "and or not no any some tell me you your it its they them i we us this "
    "that these those there here far away long old big biggest large largest "
    "small smallest many much capital invented discovered eat live come".split())


def learn_topic(agent, topic: str, max_facts: int = 40, force: bool = False) -> dict:
    """Fetch + remember one topic. Returns {'title', 'added', 'error'}.
    force=True re-fetches an already-known article (used when a needed fact -
    e.g. a date - is missing from the stored rows); sentence-level dedup keeps
    re-fetches from duplicating what's already remembered."""
    from lattice.fetch_wiki import fetch_full_lead, fetch_one
    r = fetch_full_lead(topic)
    if not r or r.get("error") or not r.get("extract"):
        r = fetch_one(topic)
    if not r or r.get("error") or not r.get("extract"):
        return {"title": topic, "added": 0,
                "error": (r or {}).get("error", "no data")}
    title = r["title"]
    # already known? (normalize spaces/underscores - old ingests used _)
    norm = title.replace(" ", "_").lower()
    known = any(s.split(":", 1)[-1].replace(" ", "_").lower() == norm
                for s in agent.lattice._sources if s.startswith("wikipedia:"))
    if known and not force:
        return {"title": title, "added": 0, "error": "already known"}
    existing = set(agent.lattice._texts)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", r["extract"])
             if len(s.strip()) > 30]
    n = 0
    for s in sents[:max_facts]:
        if s.lower().startswith(_PRON):
            s = f"{title}: {s}"          # anchor coreference orphans
        if s in existing:
            continue                     # sentence-level dedup
        agent.lattice.add(s, source=f"wikipedia:{title}")
        try:                             # claims update inline, not just at boot
            agent.structured.add_sentence(s, source=f"wikipedia:{title}")
        except Exception:
            pass
        n += 1
    return {"title": title, "added": n, "error": None}


# ─── Dates: find and compute (math never goes through retrieval) ────

_MONTHS = {m.lower(): i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}

_DATE_RE = re.compile(
    r"(?:(\d{1,2})\s+([A-Za-z]+)\s+(\d{3,4}))"        # 15 February 1564
    r"|(?:([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{3,4}))")    # February 15, 1564


def _dates_in(text: str) -> list[tuple[int, int, int]]:
    out = []
    for m in _DATE_RE.finditer(text):
        if m.group(1):
            d, mon, y = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        else:
            mon, d, y = m.group(4).lower(), int(m.group(5)), int(m.group(6))
        if mon in _MONTHS and 1 <= d <= 31:
            out.append((y, _MONTHS[mon], d))
    return out


def find_life_dates(agent, subject: str) -> dict:
    """Birth/death dates for a subject from remembered rows. Heuristics:
    a '(date – date)' span, or dates near 'born'/'died'."""
    sub = subject.lower()
    rows = [t for t in agent.lattice._texts if sub in t.lower()][:60]
    birth = death = None
    evidence = []
    for t in rows:
        dates = _dates_in(t)
        low = t.lower()
        if len(dates) >= 2 and (("–" in t or "—" in t or " - " in t)
                                or "born" in low):
            a, b = dates[0], dates[1]
            if a[0] < b[0] and (b[0] - a[0]) <= 120:
                birth, death = birth or a, death or b
                evidence.append(t)
                continue
        if "born" in low and dates and birth is None:
            birth = dates[0]
            evidence.append(t)
        if ("died" in low or "death" in low) and dates and death is None:
            death = dates[-1]
            evidence.append(t)
    return {"birth": birth, "death": death, "evidence": evidence[:3]}


def age_between(birth: tuple, death: tuple) -> int:
    y = death[0] - birth[0]
    if (death[1], death[2]) < (birth[1], birth[2]):
        y -= 1
    return y


def fmt_date(d: tuple) -> str:
    names = {v: k.capitalize() for k, v in _MONTHS.items()}
    return f"{names[d[1]]} {d[2]}, {d[0]}"


def search_wiki(query: str, limit: int = 3) -> list[str]:
    """Article titles best matching a free-text query (MediaWiki search API,
    one attributed request). The fallback when direct topic guesses are
    already known or unfetchable - finds 'Invention of the telephone' for
    'who invented the telephone?'."""
    import json
    import urllib.parse
    import urllib.request
    params = {"action": "query", "list": "search", "srsearch": query,
              "srlimit": str(limit), "format": "json", "formatversion": "2"}
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "telp-lattice-research/0.1 (educational use)"})
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [r["title"] for r in data.get("query", {}).get("search", [])]
    except Exception:
        return []


def learn_url(agent, url: str, max_facts: int = 60) -> dict:
    """Learn from any web page or text file (second source beyond Wikipedia).
    One attributed request; source recorded as url:<location>."""
    import urllib.parse
    import urllib.request
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "telp-lattice-research/0.1 (educational use)"})
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            raw = resp.read(1_500_000).decode("utf-8", errors="replace")
            ctype = resp.headers.get("Content-Type", "")
    except Exception as e:
        return {"title": url, "added": 0, "error": str(e)}
    if "html" in ctype or raw.lstrip()[:1] == "<":
        raw = re.sub(r"(?is)<(script|style|nav|header|footer)[^>]*>.*?</\1>", " ", raw)
        raw = re.sub(r"(?s)<[^>]+>", " ", raw)          # strip tags
        raw = re.sub(r"&[a-z#0-9]+;", " ", raw)         # entities
    raw = re.sub(r"\s+", " ", raw)
    loc = urllib.parse.urlparse(url)
    src = f"url:{loc.netloc}{loc.path}"
    existing = set(agent.lattice._texts)
    n = 0
    for s in re.split(r"(?<=[.!?])\s+", raw):
        s = s.strip()
        # keep only prose-like sentences (has letters, reasonable length)
        if not (40 < len(s) < 400) or sum(c.isalpha() for c in s) < len(s) * 0.6:
            continue
        if s in existing:
            continue
        agent.lattice.add(s, source=src)
        try:
            agent.structured.add_sentence(s, source=src)
        except Exception:
            pass
        n += 1
        if n >= max_facts:
            break
    return {"title": src, "added": n, "error": None if n else "no prose found"}


# ─── Procedures: how-to knowledge from wikiHow (steps, in order) ─────
# wikiHow's api.php is broken (HTTP 500), but its search page and article
# pages parse cleanly. One search request + one article request per learn;
# source recorded as wikihow:<slug> (content CC BY-NC-SA, attributed).

_WH_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "telp-lattice-research/0.1 (educational use)"}
_WH_SKIP = ("wikiHowTo", "Main-Page", "Special:", "Category:", "Terms-of-Use",
            "About-wikiHow", "Contact-Us", "Create-an-Account")


def _wh_get(url: str) -> str:
    import urllib.request
    req = urllib.request.Request(url, headers=_WH_UA)
    with urllib.request.urlopen(req, timeout=15.0) as resp:
        return resp.read(2_000_000).decode("utf-8", errors="replace")


def _wh_clean_step(chunk: str, cap: int = 300) -> str:
    """One '<div class=\"step\"...' chunk -> clean instruction text,
    trimmed to a sentence boundary."""
    c = chunk[:4000]
    c = re.sub(r"(?s)<script.*?</script>", " ", c)
    c = re.sub(r"\{\"smallUrl.*?\}", " ", c)            # embedded image JSON
    c = re.sub(r"(?s)<[^>]+>", " ", c)
    c = re.sub(r"&[a-z#0-9]+;", " ", c)
    c = re.sub(r"X\s+(?:Trustworthy|Research|Expert)\s+[Ss]ource\b[^.]*", " ", c)
    c = re.sub(r"\s+", " ", c).strip().lstrip('" >').strip()
    if len(c) <= cap:
        return c
    cut = c[:cap]
    dot = max(cut.rfind(". "), cut.rfind("! "))
    return cut[:dot + 1] if dot > 40 else cut


def learn_howto(agent, query: str, max_steps: int = 10) -> dict:
    """Fetch a how-to procedure from wikiHow and remember its steps IN ORDER.
    Returns {'title', 'slug', 'added', 'steps', 'error'}."""
    import urllib.parse
    try:
        html = _wh_get("https://www.wikihow.com/wikiHowTo?search="
                       + urllib.parse.quote_plus(query))
    except Exception as e:
        return {"title": query, "added": 0, "steps": [], "error": f"search: {e}"}
    slugs = []
    for m in re.finditer(r'href="https://www\.wikihow\.com/([A-Za-z0-9%\-]+)"', html):
        s = m.group(1)
        if s not in slugs and not any(x in s for x in _WH_SKIP):
            slugs.append(s)
    if not slugs:
        return {"title": query, "added": 0, "steps": [], "error": "no article found"}
    slug = slugs[0]
    title = urllib.parse.unquote(slug).replace("-", " ")
    src = f"wikihow:{slug}"
    if src in agent.lattice._sources:
        proc = procedure_steps(agent, slug=slug)
        return {"title": title, "slug": slug, "added": 0,
                "steps": [t for _, t in proc["steps"]] if proc else [],
                "error": "already known"}
    try:
        page = _wh_get(f"https://www.wikihow.com/{slug}")
    except Exception as e:
        return {"title": title, "added": 0, "steps": [], "error": f"article: {e}"}
    steps = []
    for chunk in page.split('<div class="step')[1:]:
        t = _wh_clean_step(chunk)
        if len(t) > 40 and t not in steps:
            steps.append(t)
        if len(steps) >= max_steps:
            break
    if not steps:
        return {"title": title, "added": 0, "steps": [], "error": "no steps parsed"}
    for i, t in enumerate(steps, 1):
        agent.lattice.add(f"How to {title.lower()}, step {i}: {t}", source=src)
    agent.lattice.add(f"How to {title.lower()}: a {len(steps)}-step procedure "
                      f"Telp learned from wikiHow.", source=src)
    return {"title": title, "slug": slug, "added": len(steps) + 1,
            "steps": steps, "error": None}


_WH_STEP_RE = re.compile(r"^How to (.+), step (\d+): (.+)$", re.S)


def procedure_steps(agent, query: str = None, slug: str = None) -> dict | None:
    """A known procedure's ordered steps. By slug (exact) or by query
    (best title via float-cosine, gate 0.60).
    Returns {'title', 'slug', 'steps': [(n, text), ...]} or None."""
    import urllib.parse
    by_src: dict = {}
    for t, s in zip(agent.lattice._texts, agent.lattice._sources):
        if not s.startswith("wikihow:"):
            continue
        m = _WH_STEP_RE.match(t)
        if m:
            by_src.setdefault(s, []).append((int(m.group(2)), m.group(3)))
    if not by_src:
        return None

    def _pack(src: str) -> dict:
        sg = src.split(":", 1)[1]
        return {"title": urllib.parse.unquote(sg).replace("-", " "),
                "slug": sg, "steps": sorted(by_src[src])}

    if slug is not None:
        src = f"wikihow:{slug}"
        return _pack(src) if src in by_src else None
    emb_fn = getattr(agent.encoder, "_embed", None)
    if emb_fn is None or not query:
        return None
    keys = list(by_src)
    titles = ["how to " + k.split(":", 1)[1].replace("-", " ").lower()
              for k in keys]
    import numpy as _np
    embs = _np.asarray(emb_fn([query] + titles))
    sims = embs[1:] @ embs[0]
    i = int(sims.argmax())
    if float(sims[i]) < 0.60:
        return None
    return _pack(keys[i])


_YT_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([A-Za-z0-9_\-]{11})")


def learn_youtube(agent, url_or_id: str, max_facts: int = 80) -> dict:
    """Learn from a YouTube video's transcript (free, public captions).
    Machinery inherited from the May knowledge agent, repointed at the one
    memory. Source recorded as youtube:<video_id>; chunks anchored with the
    video title so they're findable by meaning."""
    m = _YT_ID_RE.search(url_or_id)
    vid = m.group(1) if m else (url_or_id if re.fullmatch(
        r"[A-Za-z0-9_\-]{11}", url_or_id) else None)
    if not vid:
        return {"title": url_or_id, "added": 0, "error": "no video id found"}

    # transcript (handle both API generations)
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        try:
            chunks = YouTubeTranscriptApi.get_transcript(vid)
            texts = [c.get("text", "") for c in chunks]
        except AttributeError:
            fetched = YouTubeTranscriptApi().fetch(vid)
            texts = [getattr(c, "text", "") for c in fetched]
    except Exception as e:
        return {"title": vid, "added": 0, "error": f"transcript: {e}"}
    raw = " ".join(t.replace("\n", " ") for t in texts if t)
    raw = re.sub(r"\[[^\]]{0,30}\]", " ", raw)          # [Music] etc.
    raw = re.sub(r"\s+", " ", raw).strip()
    if len(raw) < 40:
        return {"title": vid, "added": 0, "error": "empty transcript"}

    # title via oEmbed (no API key)
    title = vid
    try:
        import json as _json
        import urllib.request
        u = (f"https://www.youtube.com/oembed?url="
             f"https%3A//www.youtube.com/watch%3Fv%3D{vid}&format=json")
        req = urllib.request.Request(
            u, headers={"User-Agent": "telp-lattice-research/0.1"})
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            title = _json.loads(resp.read().decode())["title"][:60]
    except Exception:
        pass

    # transcripts often lack punctuation: split on sentences if present,
    # else chunk ~30 words into pseudo-sentences
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw)
             if 40 < len(s.strip()) < 300]
    if len(parts) < 3:
        words = raw.split()
        parts = [" ".join(words[i:i + 30]) for i in range(0, len(words), 30)]
        parts = [p for p in parts if len(p) > 40]
    existing = set(agent.lattice._texts)
    src = f"youtube:{vid}"
    n = 0
    for s in parts[:max_facts]:
        s = f"In the video '{title}': {s}"
        if s in existing:
            continue
        agent.lattice.add(s, source=src)
        try:
            agent.structured.add_sentence(s, source=src)
        except Exception:
            pass
        n += 1
    if n:
        agent.lattice.add(
            f"Telp watched the YouTube video '{title}' and remembered "
            f"{n} passages from it.", source=src)
    return {"title": title, "added": n, "error": None if n else "nothing kept"}


def _yt_transcript_chunks(vid: str) -> list[tuple[float, str]]:
    """[(start_seconds, text), ...] - both API generations handled."""
    from youtube_transcript_api import YouTubeTranscriptApi
    try:
        chunks = YouTubeTranscriptApi.get_transcript(vid)
        return [(float(c.get("start", 0)), c.get("text", "")) for c in chunks]
    except AttributeError:
        fetched = YouTubeTranscriptApi().fetch(vid)
        return [(float(getattr(c, "start", 0)), getattr(c, "text", ""))
                for c in fetched]


def _yt_title(vid: str) -> str:
    import json as _json
    import urllib.request
    try:
        u = (f"https://www.youtube.com/oembed?url="
             f"https%3A//www.youtube.com/watch%3Fv%3D{vid}&format=json")
        req = urllib.request.Request(
            u, headers={"User-Agent": "telp-lattice-research/0.1"})
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return _json.loads(resp.read().decode())["title"][:60]
    except Exception:
        return vid


def remember_passages(agent, title: str, chunks: list, src: str,
                      max_facts: int = 80) -> int:
    """Store timestamped speech chunks as passage memories (shared by captions
    and heard-audio paths). Groups small chunks into 40-300 char passages."""
    existing = set(agent.lattice._texts)
    n = 0
    buf, buf_t = [], None
    def emit():
        nonlocal n, buf, buf_t
        if not buf:
            return
        text = re.sub(r"\s+", " ", " ".join(buf)).strip()
        buf, t0 = [], buf_t
        if not (30 < len(text) < 320):
            return
        s = f"In the video '{title}' (at {int(t0 or 0)}s): {text}"
        if s in existing:
            return
        agent.lattice.add(s, source=src)
        try:
            agent.structured.add_sentence(s, source=src)
        except Exception:
            pass
        n += 1
    for t, txt in chunks:
        if n >= max_facts:
            break
        if buf_t is None:
            buf_t = t
        buf.append(txt)
        if sum(len(x) for x in buf) > 180:
            emit()
            buf_t = None
    emit()
    return n


def watch_youtube(agent, url_or_id: str, namer=None, max_minutes: int = 20) -> dict:
    """Telp WATCHES a YouTube video - eyes and ears together. Downloads the
    video (yt-dlp, capped <=480p), runs scene-vision on the frames (timestamped
    CLIP-named keyframes into the one memory), fetches the timestamped
    transcript, and FUSES them: scenes near speech become single memories
    binding what was SHOWN with what was SAID. The file is deleted after;
    the keyframes and memories stay."""
    import shutil
    import tempfile
    m = _YT_ID_RE.search(url_or_id)
    vid = m.group(1) if m else (url_or_id if re.fullmatch(
        r"[A-Za-z0-9_\-]{11}", url_or_id) else None)
    if not vid:
        return {"title": url_or_id, "error": "no video id found"}
    title = _yt_title(vid)

    # download small (video only - the ears use the transcript)
    tmp = Path(tempfile.mkdtemp(prefix="telp_yt_"))
    out = tmp / f"{vid}.mp4"
    try:
        import yt_dlp
        opts = {"format": "b[height<=480][ext=mp4]/b[ext=mp4]/b",
                "outtmpl": str(out), "quiet": True, "no_warnings": True,
                "match_filter": yt_dlp.utils.match_filter_func(
                    f"duration <= {max_minutes * 60}"),
                "noplaylist": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={vid}"])
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        return {"title": title, "error": f"download: {e}"}
    if not out.exists():
        shutil.rmtree(tmp, ignore_errors=True)
        return {"title": title, "error": f"video longer than {max_minutes} min "
                                         f"or download blocked"}

    # eyes: scene vision (writes timestamped sight memories itself)
    from lattice.vision import watch
    heard = False
    try:
        w = watch(agent, out, namer=namer, label=title, verbose=True)
        # ears: captions if they exist, else LISTEN to the audio (local ASR)
        try:
            chunks = _yt_transcript_chunks(vid)
        except Exception:
            chunks = []
        if not chunks:
            try:
                from lattice.hearing import transcribe
                print("[growth] no captions - listening to the audio ...",
                      flush=True)
                chunks = transcribe(out)
                heard = True
            except Exception as e:
                print(f"[growth] hearing unavailable ({e})", flush=True)
                chunks = []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # fusion: bind what was SHOWN to what was SAID around the same moment
    fused = 0
    src = f"youtube:{vid}"
    for t, caption in w.get("scene_list", []):
        near = " ".join(txt for s, txt in chunks if t - 4 <= s <= t + 8)
        near = re.sub(r"\s+", " ", near).strip()[:200]
        if near:
            agent.lattice.add(
                f"In the video '{title}' at {int(t)}s, while showing "
                f"{caption.removeprefix('an image showing ')}, the speaker "
                f"says: \"{near}\"", source=src)
            fused += 1
    # plus the full transcript as plain knowledge (deduped)
    if heard:
        n_p = remember_passages(agent, title, chunks, src)
    else:
        n_p = learn_youtube(agent, vid).get("added", 0) if chunks else 0
    how = "heard with his own ears (no captions)" if heard else "remembered"
    agent.lattice.add(
        f"Telp watched the YouTube video '{title}': {w['scenes']} scenes seen, "
        f"{fused} moments where sight and speech were bound together, "
        f"{n_p} spoken passages {how}.",
        source=f"video:youtube:{vid}")
    return {"title": title, "scenes": w["scenes"], "fused": fused,
            "passages": n_p, "heard": heard, "error": None}


def extract_topics(question: str, max_topics: int = 3) -> list[str]:
    """Best-first guesses at what a question is about."""
    out: list[str] = []

    def _add(t: str):
        t = t.strip(" ?.!,'\"")
        if t and len(t) > 2 and t.lower() not in _STOP and t not in out:
            out.append(t)

    # quoted spans
    for m in re.findall(r"['\"]([^'\"]{3,40})['\"]", question):
        _add(m)
    # capitalized phrases (not sentence-initial word alone)
    words = question.split()
    i = 0
    while i < len(words):
        w = words[i].strip("?.!,")
        if w[:1].isupper() and (i > 0 or len(w) > 3):
            j = i
            phrase = []
            while j < len(words):
                wj = words[j].strip("?.!,")
                if wj[:1].isupper() and wj.lower() not in _STOP:
                    phrase.append(wj)
                    j += 1
                else:
                    break
            if phrase and not (i == 0 and len(phrase) == 1
                               and phrase[0].lower() in _STOP):
                if i > 0 or len(phrase) > 1 or phrase[0].lower() not in _STOP:
                    _add(" ".join(phrase))
            i = j + 1
        else:
            i += 1
    # question object: "... of X", "... about X", "... for X"
    m = re.search(r"\b(?:of|about|for)\s+(?:the\s+)?([a-zA-Z][a-zA-Z \-]{2,30})$",
                  question.rstrip(" ?.!"))
    if m:
        _add(m.group(1))
    # remaining content words (longest first as a rough rarity proxy)
    content = [w.strip("?.!,'\"") for w in question.lower().split()
               if w.strip("?.!,'\"") not in _STOP and len(w) > 3]
    for w in sorted(content, key=len, reverse=True):
        _add(w)
    return out[:max_topics]
