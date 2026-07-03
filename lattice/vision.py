"""
lattice/vision.py - see -> know -> say: the first wire of the organism.

Before this file, Telp's perception lane was test-only: remember_image existed
but nothing ever called it in production, and no captioner existed (the pixel
encoder gives similarity, not semantics). This wire closes the loop:

    see(image)  ->  CLIP zero-shot naming (encoder retrieval over a label
                    vocabulary - no LLM, no text generation)
                ->  remember_image(): image HV into the ImageBank, caption into
                    the SAME lattice the chat cascade reads
                ->  "what did you see?" answered from Telp's own memory.

Default lattice: state/concept_bridge.db - the one chat (FluentTelp) actually
reads - so sights become part of the one mind, not another island.

Usage:
    python -m lattice.vision see path/to/img.png [more images...]
    python -m lattice.vision recall "a furry animal"
    python -m lattice.vision ask "what did you see today?"
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# portable model cache: TELP_MODEL_DIR > D:\hf_home (if D: exists) > ~/.cache/telp
_cache = os.environ.get("TELP_MODEL_DIR") or (
    r"D:\hf_home" if Path(r"D:\\").exists() else str(Path.home() / ".cache" / "telp"))
os.environ.setdefault("HF_HOME", _cache)

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

CHAT_LATTICE = _TELP_ROOT / "state" / "concept_bridge.db"

# Visual vocabulary: the big curated list (~900 concepts). Falls back to the
# small inline list below if the vocab module is missing.
try:
    from lattice.vision_vocab import VOCAB as _BIG_VOCAB
except Exception:
    _BIG_VOCAB = None

# compact fallback vocabulary
_SMALL_VOCAB = [
    # living
    "person", "face", "man", "woman", "child", "crowd", "hand",
    "dog", "cat", "raccoon", "fox", "bear", "horse", "cow", "sheep", "rabbit",
    "bird", "fish", "insect", "butterfly", "spider", "snake", "monkey", "deer",
    # nature
    "tree", "forest", "grass", "flower", "leaf", "mountain", "rock", "beach",
    "ocean", "river", "lake", "sky", "cloud", "sunset", "snow", "fire", "desert",
    # built world
    "house", "building", "city", "street", "road", "bridge", "tower", "room",
    "kitchen", "bedroom", "office", "door", "window", "wall", "stairs",
    # vehicles
    "car", "truck", "bus", "train", "bicycle", "motorcycle", "airplane",
    "helicopter", "boat", "ship", "rocket",
    # objects
    "food", "fruit", "coffee", "drink", "bottle", "cup", "plate", "table",
    "chair", "sofa", "bed", "book", "phone", "computer", "screen", "keyboard",
    "television", "camera", "clock", "lamp", "toy", "ball", "tool", "knife",
    "guitar", "piano", "painting", "statue", "sign", "text", "chart", "graph",
    "map", "money", "clothing", "hat", "shoe", "bag", "umbrella", "glasses",
    # people-things
    "astronaut", "soldier", "athlete", "dancer", "chef", "doctor",
    "warrior", "knight",
    # story/fantasy things (video content)
    "dragon", "monster", "sword", "spear", "castle", "village", "cave",
    "ruins", "temple", "fight", "battle", "campfire",
    # scenes / media
    "cartoon", "animation", "video game", "landscape", "portrait", "night scene",
    "underwater scene", "aerial view", "close-up photo", "microscope image",
    "x-ray", "diagram", "screenshot", "logo",
]

VOCAB = list(dict.fromkeys(_BIG_VOCAB)) if _BIG_VOCAB else _SMALL_VOCAB


class ZeroShotNamer:
    """Names what an image shows by CLIP similarity over VOCAB.
    This is retrieval with an encoder, not generation - Telp-compatible."""

    def __init__(self, device: str = "cpu"):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(device).eval()
        self.device = device
        self._label_emb = None

    @staticmethod
    def _tensor(out):
        """transformers returns a tensor (old) or an output object (new)."""
        import torch
        if torch.is_tensor(out):
            return out
        for k in ("image_embeds", "text_embeds", "pooler_output"):
            v = getattr(out, k, None)
            if v is not None:
                return v
        raise TypeError(f"unexpected CLIP output type {type(out)}")

    def _labels(self):
        import torch
        if self._label_emb is None:
            texts = [f"a photo of a {w}" for w in VOCAB]
            with torch.no_grad():
                inp = self.proc(text=texts, return_tensors="pt", padding=True).to(self.device)
                emb = self._tensor(self.model.get_text_features(**inp))
            self._label_emb = emb / emb.norm(dim=-1, keepdim=True)
        return self._label_emb

    def name(self, image_path, k: int = 3):
        """-> (caption, [(label, score), ...])"""
        import torch
        from PIL import Image as PILImage
        img = PILImage.open(image_path).convert("RGB")
        with torch.no_grad():
            inp = self.proc(images=img, return_tensors="pt").to(self.device)
            emb = self._tensor(self.model.get_image_features(**inp))
        emb = emb / emb.norm(dim=-1, keepdim=True)
        sims = (emb @ self._labels().t())[0]
        top = torch.topk(sims, k)
        pairs = [(VOCAB[i], float(s)) for s, i in zip(top.values, top.indices)]
        # keep labels within 85% of the best match - don't pad the caption with noise
        keep = [p for p in pairs if p[1] >= 0.85 * pairs[0][1]] or pairs[:1]
        caption = "an image showing " + ", ".join(w for w, _ in keep)
        return caption, pairs


_NAMER: ZeroShotNamer | None = None


def get_namer() -> ZeroShotNamer:
    """Process-wide namer (CLIP loads once)."""
    global _NAMER
    if _NAMER is None:
        _NAMER = ZeroShotNamer()
    return _NAMER


def see(agent, image_path, namer: ZeroShotNamer | None = None):
    """Telp looks at an image: name it, READ any text in it, remember both."""
    namer = namer or get_namer()
    caption, pairs = namer.name(image_path)
    try:
        from lattice.ocr import read_text
        lines = read_text(image_path)
    except Exception:
        lines = []
    if lines:
        caption = f"{caption}, with the text: \"{' / '.join(lines)[:160]}\""
    img_id = agent.remember_image(str(image_path), description=caption)
    return {"image_id": img_id, "caption": caption, "labels": pairs,
            "text": lines}


def sights(db) -> list[dict]:
    """All image-sourced memories in a lattice - a targeted scan, so sights are
    retrievable no matter how many text memories surround them."""
    import sqlite3
    con = sqlite3.connect(str(db))
    rows = con.execute("SELECT created_at, text, source FROM memories "
                       "WHERE source LIKE 'image:%' ORDER BY id").fetchall()
    con.close()
    return [{"when": r[0], "caption": r[1],
             "path": r[2].split(":", 1)[1]} for r in rows]


def recall_semantic(db, query: str, namer: ZeroShotNamer | None = None, k: int = 3):
    """TRUE cross-modal recall: CLIP-embed the query TEXT, rank Telp's remembered
    IMAGES by direct text<->image similarity. Immune to lattice pollution and to
    out-of-vocabulary words - this is what the query means, not what it spells."""
    import torch
    from PIL import Image as PILImage
    rows = sights(db)
    if not rows:
        return []
    namer = namer or get_namer()
    with torch.no_grad():
        ti = namer.proc(text=[query], return_tensors="pt", padding=True).to(namer.device)
        q = namer._tensor(namer.model.get_text_features(**ti))
        q = q / q.norm(dim=-1, keepdim=True)
        scored = []
        for r in rows:
            try:
                img = PILImage.open(r["path"]).convert("RGB")
            except OSError:
                continue
            ii = namer.proc(images=img, return_tensors="pt").to(namer.device)
            e = namer._tensor(namer.model.get_image_features(**ii))
            e = e / e.norm(dim=-1, keepdim=True)
            scored.append({**r, "similarity": float((q @ e.t())[0, 0])})
    scored.sort(key=lambda x: -x["similarity"])
    return scored[:k]


# ─── Watching: video -> scene memories ──────────────────────────────

SIGHTS_DIR = _TELP_ROOT / "state" / "sights"


def _ffmpeg():
    import shutil
    p = shutil.which("ffmpeg")
    if p:
        return p
    import glob as _g
    for c in _g.glob(r"C:\Users\*\upscaler\bin\**\ffmpeg.exe", recursive=True):
        return str(c)
    raise RuntimeError("ffmpeg not found")


def _probe_wh(video):
    import subprocess, json
    ff = _ffmpeg()
    fp = ff[:-len("ffmpeg.exe")] + "ffprobe.exe" if ff.endswith("ffmpeg.exe") else "ffprobe"
    r = subprocess.run([fp, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "json", str(video)],
                       capture_output=True, text=True)
    s = json.loads(r.stdout)["streams"][0]
    return int(s["width"]), int(s["height"])


def watch(agent, video_path, sample_fps=4.0, cut_thresh=22.0, every_s=10.0,
          max_scenes=40, namer: ZeroShotNamer | None = None, verbose=True,
          label: str | None = None):
    """Telp watches a video: detect scene changes, name each scene's keyframe,
    remember every scene (with its timestamp) in the one lattice, and store a
    watch-summary so 'what did you watch?' has an answer. `label` overrides the
    filename as the video's human name in memories (e.g. a YouTube title)."""
    import subprocess
    import numpy as np_
    video_path = Path(video_path)
    W, H = _probe_wh(video_path)
    fb = W * H * 3
    namer = namer or get_namer()
    label = label or video_path.stem
    safe = re.sub(r"[^A-Za-z0-9_\- ]", "", label)[:40].strip() or video_path.stem
    out_dir = SIGHTS_DIR / safe.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    dec = subprocess.Popen([_ffmpeg(), "-loglevel", "error", "-i", str(video_path),
                            "-vf", f"fps={sample_fps}", "-f", "rawvideo",
                            "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE, bufsize=fb * 2)
    scenes = []
    _seen_text: set = set()
    n_text = 0
    prev_small = None
    last_key_t = -1e9
    i = 0
    while True:
        raw = dec.stdout.read(fb)
        if len(raw) < fb:
            break
        t = i / sample_fps
        i += 1
        fr = np_.frombuffer(raw, np_.uint8).reshape(H, W, 3)
        # keep COLOR: a dark-blue -> dark-red cut is invisible in luma alone
        small = fr[::8, ::8].astype(np_.float32)
        is_cut = prev_small is not None and np_.abs(small - prev_small).mean() > cut_thresh
        prev_small = small
        # min 1.5s between keyframes: grainy/handheld footage fires the cut
        # detector on every wobble (30 keyframes for an 18s clip otherwise)
        if (is_cut and t - last_key_t < 1.5) and t - last_key_t < every_s:
            continue
        if (is_cut or t - last_key_t >= every_s) and len(scenes) < max_scenes:
            # skip near-black frames (fades/cuts to black)
            if small.mean() < 12:
                continue
            last_key_t = t
            from PIL import Image as PILImage
            kp = out_dir / f"t{int(t):04d}s.png"
            PILImage.fromarray(fr).save(kp)
            caption, pairs = namer.name(kp)
            if scenes and caption == scenes[-1][1]:
                kp.unlink(missing_ok=True)      # same sight as last keyframe
                continue
            desc = f"At {int(t)}s in video '{safe}': {caption}"
            agent.remember_image(str(kp), description=desc)
            scenes.append((t, caption, pairs))
            if verbose:
                print(f"  [scene @{int(t):>4}s] {caption}", flush=True)
            # reading eyes: text burned into the frame (titles, signs, slides)
            try:
                from lattice.ocr import read_text
                lines = read_text(kp)
            except Exception:
                lines = []
            joined = " / ".join(lines)[:200]
            if joined and joined not in _seen_text:
                _seen_text.add(joined)
                agent.lattice.add(
                    f"At {int(t)}s in video '{safe}', the screen shows the "
                    f"text: \"{joined}\"", source=f"video:{safe}")
                n_text += 1
                if verbose:
                    print(f"    [text @{int(t):>4}s] {joined[:70]}", flush=True)
    dec.stdout.close()
    dec.wait()

    timeline = "; ".join(f"{int(t)}s: {c.removeprefix('an image showing ')}"
                         for t, c, _ in scenes)
    summary = (f"Telp watched the video '{safe}' and remembers "
               f"{len(scenes)} scenes - {timeline}.")
    agent.lattice.add(summary, source=f"video:{video_path}")
    return {"video": str(video_path), "scenes": len(scenes), "summary": summary,
            "scene_list": [(t, c) for t, c, _ in scenes], "label": safe,
            "text_memories": n_text}


def screen_texts(db) -> list[dict]:
    """All on-screen text memories (OCR rows) - targeted scan, like sights()."""
    import sqlite3
    con = sqlite3.connect(str(db))
    rows = con.execute("SELECT created_at, text FROM memories WHERE text LIKE "
                       "'%the screen shows the text%' ORDER BY id").fetchall()
    con.close()
    return [{"when": r[0], "text": r[1]} for r in rows]


def watched(db) -> list[dict]:
    """All watch-summary memories."""
    import sqlite3
    con = sqlite3.connect(str(db))
    rows = con.execute("SELECT created_at, text, source FROM memories "
                       "WHERE source LIKE 'video:%' ORDER BY id").fetchall()
    con.close()
    return [{"when": r[0], "summary": r[1], "path": r[2].split(":", 1)[1]}
            for r in rows]


# ─── Forgetting: selective, on command ──────────────────────────────


def forget(db, *, video: str | None = None, query: str | None = None,
           min_sim: float = 0.20):
    """Selective forgetting - remove SPECIFIC memories, leave the rest intact.

    video=<name/stem> : forget a watched video - its watch-summary, every scene
                        memory, and their image-bank entries.
    query=<text>      : forget the sight that best matches the query by meaning
                        (CLIP cross-modal), if it matches at all.

    Deletes lattice rows + image-bank rows. Keyframe PNGs stay on disk (they're
    artifacts, not memory - Telp can no longer recall them). Returns the list of
    forgotten memory texts. NOTE: a long-lived agent process should be restarted
    (or its stack reloaded) after forgetting - the CLI reloads per call anyway.
    """
    import sqlite3
    img_db = str(db).replace(".db", "_images.db")
    con = sqlite3.connect(str(db))
    icon = sqlite3.connect(img_db) if Path(img_db).exists() else None
    rows = []
    if video:
        stem = Path(video).stem
        rows = con.execute(
            "SELECT id, text, source FROM memories WHERE "
            "(source LIKE 'video:%' AND source LIKE ?) OR "
            "(source LIKE 'image:%' AND text LIKE ?)",
            (f"%{stem}%", f"%in video '{stem}'%")).fetchall()
    elif query:
        for h in recall_semantic(db, query, k=1):
            if h["similarity"] < min_sim:
                continue
            rows += con.execute(
                "SELECT id, text, source FROM memories WHERE source=?",
                (f"image:{h['path']}",)).fetchall()
    gone = []
    for rid, text, src in rows:
        con.execute("DELETE FROM memories WHERE id=?", (rid,))
        if icon is not None and src.startswith("image:"):
            icon.execute("DELETE FROM images WHERE image_path=?",
                         (src.split(":", 1)[1],))
        gone.append(text)
    con.commit()
    con.close()
    if icon is not None:
        icon.commit()
        icon.close()
    return gone


def _agent(db):
    from lattice.standalone_agent import StandaloneAgent
    return StandaloneAgent(lattice_path=Path(db))


def main():
    import argparse
    ap = argparse.ArgumentParser(prog="lattice.vision")
    ap.add_argument("cmd", choices=["see", "recall", "ask"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--db", default=str(CHAT_LATTICE),
                    help="lattice DB (default: the chat's concept_bridge.db)")
    a = ap.parse_args()

    if a.cmd == "see":
        agent = _agent(a.db)
        namer = ZeroShotNamer()
        for p in a.args:
            r = see(agent, p, namer)
            lbl = ", ".join(f"{w} {s:.2f}" for w, s in r["labels"])
            print(f"[seen] {Path(p).name}: {r['caption']}   ({lbl})")
    elif a.cmd == "recall":
        for r in recall_semantic(a.db, " ".join(a.args)):
            print(f"[recall] {r['similarity']:.3f}  {r['caption']}  <- {r['path']}")
    else:
        q = " ".join(a.args)
        # vision route: questions about seeing answer from sight memories directly
        if any(w in q.lower() for w in ("see", "saw", "seen", "look", "watch", "image")):
            rows = sights(a.db)
            if rows:
                latest = rows[-3:]
                caps = "; ".join(r["caption"] for r in latest)
                print(f"[telp] I have seen {len(rows)} image(s). Most recently: {caps}.")
            else:
                print("[telp] I haven't seen any images yet.")
        else:
            print(f"[telp] {_agent(a.db).respond(q)}")


if __name__ == "__main__":
    main()
