#!/usr/bin/env python3
"""Rebuild runs/listen-hack-vs-gold from Cloud-gold CTC F1 ranking."""

from __future__ import annotations

import html
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LISTEN = ROOT / "runs" / "listen-hack-vs-gold"
RANK = ROOT / "runs" / "misaki-iter" / "cloud-rank.json"
MANIFEST = ROOT / "data" / "gold_gemini_ipa" / "manifest.jsonl"
PINS = ROOT / "runs" / "misaki-iter" / "user_pins.json"
CHOSEN = ROOT / "runs" / "misaki-iter" / "chosen.json"
GOLD_MP3 = LISTEN / "mp3" / "gold"
MISAKI_MP3 = LISTEN / "mp3" / "misaki"
GOLD_WAV = ROOT / "data" / "gold_gemini_ipa" / "wavs"
SPAN = ROOT / "runs" / "misaki-iter" / "cloud-span"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_full_items() -> list[dict]:
    items = []
    for line in (ROOT / "data" / "dose_v1.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ingredients = row.get("ingredients") or [row["name"]]
        spans = row.get("spans") or []
        sentence = row["sentence"]
        for i, ing in enumerate(ingredients):
            if i < len(spans) and isinstance(spans[i], (list, tuple)) and len(spans[i]) == 2:
                a, b = spans[i]
                spoken = sentence[a:b]
            else:
                idx = sentence.lower().find(ing.lower())
                if idx < 0:
                    continue
                spoken = sentence[idx : idx + len(ing)]
            items.append({"drug": ing, "spoken": spoken, "sentence": sentence, "slug": slug(ing)})
    seen: dict[str, dict] = {}
    for it in items:
        seen.setdefault(it["slug"], it)
    return list(seen.values())


def teacher_label(t: str) -> str:
    if t == "en-US-Standard-C":
        return "Cloud Standard-C"
    if t == "drugs.com":
        return "drugs.com human"
    return t


def dur(path: Path) -> str:
    if not path.exists():
        return "?"
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        text=True,
    ).strip()
    return f"{float(out):.2f}s"


def wav_to_mp3(wav: Path, mp3: Path) -> None:
    mp3.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame", "-qscale:a", "5", str(mp3)]
    )


def main() -> int:
    rank = json.loads(RANK.read_text())
    rows = rank["rows"]
    manifest = {}
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        manifest[Path(r["audio"]).stem] = r
        # also index hyphenated dose form of underscore stems
        manifest.setdefault(Path(r["audio"]).stem.replace("_", "-"), r)

    pins = json.loads(PINS.read_text()) if PINS.exists() else {}
    chosen = json.loads(CHOSEN.read_text()) if CHOSEN.exists() else {}
    phones = dict(chosen.get("phones") or {})
    for s, pin in pins.items():
        phones[s] = pin["misaki"]
    items = {it["slug"]: it for it in load_full_items()}
    GOLD_MP3.mkdir(parents=True, exist_ok=True)
    MISAKI_MP3.mkdir(parents=True, exist_ok=True)

    cards = []
    for r in rows:
        s = r["slug"]
        it = items.get(s)
        man = manifest.get(s) or manifest.get(s.replace("-", "_"))
        if it is None or man is None:
            continue
        misaki = phones.get(s, "?")
        ipa = man["ipa"]
        teacher = teacher_label(man.get("teacher", "en-US-Standard-C"))
        spoken = it["spoken"]
        word = pins.get(s, {}).get("word") or spoken
        inject = f"[{word}](/{misaki}/)"
        gold_wav = GOLD_WAV / Path(man["audio"]).name
        gold_mp3 = GOLD_MP3 / f"{s}.mp3"
        if gold_wav.exists():
            wav_to_mp3(gold_wav, gold_mp3)
        misaki_mp3 = MISAKI_MP3 / f"{s}.mp3"
        span_wav = SPAN / f"{s}.wav"
        if span_wav.exists():
            wav_to_mp3(span_wav, misaki_mp3)
        f1 = r["ctc_f1"]
        kind = r.get("kind", "")
        pin_note = " · pin" if (s in pins or kind in {"pin", "v0-rec", "lowest75", "pin-rescored-newgold"}) else ""
        cards.append(
            f"""<article class="card">
<div class="name">{html.escape(spoken)} <span class="f1">CTC F1 {f1:.3f}</span></div>
<p class="sent">{html.escape(it["sentence"])}</p>
<label>{html.escape(teacher)} /{html.escape(ipa)}/ · {dur(gold_mp3)}</label>
<audio controls preload="none" src="mp3/gold/{html.escape(s)}.mp3?v=3"></audio>
<label>Misaki scored crop {html.escape(misaki)} · {dur(misaki_mp3)} · sentence CTC span{html.escape(pin_note)}</label>
<audio controls preload="none" src="mp3/misaki/{html.escape(s)}.mp3?v=3"></audio>
<div class="row"><code>{html.escape(inject)}</code><button type="button" data-copy="{html.escape(inject)}">Copy inject</button></div>
</article>"""
        )

    n = len(cards)
    mean = sum(r["ctc_f1"] for r in rows) / len(rows) if rows else 0
    missing = len(rank.get("missing") or [])
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Misaki vs Cloud gold</title>
<style>
body{{font:16px/1.45 -apple-system,BlinkMacSystemFont,sans-serif;margin:32px auto;max-width:820px;color:#1a1a1a;background:#fafafa;padding:0 16px 64px}}
h1{{font-size:22px;font-weight:650}}
.note{{color:#444}}
.card{{background:#fff;border:1px solid #e4e4e4;border-radius:8px;padding:14px 16px;margin:12px 0}}
.name{{font-weight:650}}
.f1{{color:#9a3412;font-weight:500}}
.sent{{margin:6px 0 10px;color:#333}}
audio{{width:100%;margin:4px 0 12px}}
label{{display:block;font-size:14px;color:#333}}
.row{{display:flex;gap:8px;align-items:center}}
.row code{{flex:1;font-family:ui-monospace,Menlo,monospace;font-size:12px;background:#f3f3f3;padding:6px 8px;border-radius:4px;overflow-x:auto}}
button{{padding:6px 10px;border:1px solid #ccc;border-radius:4px;background:#fff;cursor:pointer}}
button.ok{{background:#dcfce7}}
</style></head><body>
<h1>Kokoro Misaki vs Cloud gold</h1>
<p class="note">Ranked lowest CTC F1 first. n={n} at speed 1.0 vs locked Cloud gold (mean {mean:.3f}). You hear the same forced-align drug crop from the DoSE sentence that the F1 score uses (not an isolated word). Gold is the isolated teacher word.</p>
{''.join(cards)}
<script>
document.querySelectorAll('button[data-copy]').forEach(btn => {{
  btn.addEventListener('click', async () => {{
    await navigator.clipboard.writeText(btn.dataset.copy);
    btn.textContent = 'Copied';
    btn.classList.add('ok');
    setTimeout(() => {{ btn.textContent = 'Copy inject'; btn.classList.remove('ok'); }}, 1200);
  }});
}});
</script>
</body></html>
"""
    LISTEN.mkdir(parents=True, exist_ok=True)
    (LISTEN / "index.html").write_text(page)
    print(f"WROTE {LISTEN / 'index.html'} cards={n} mean={mean:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
