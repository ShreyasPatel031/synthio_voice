#!/usr/bin/env python3
"""Listen page using exact scored WAV paths and full provenance metadata."""

from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LISTEN = ROOT / "runs" / "listen-hack-vs-gold"
RANK = ROOT / "runs" / "misaki-iter" / "cloud-rank.json"
PROVENANCE = ROOT / "runs" / "misaki-iter" / "scored-provenance.jsonl"
CONTEXT = ROOT / "runs" / "misaki-iter" / "context-controlled" / "manifest.json"
MANIFEST = ROOT / "data" / "gold_gemini_ipa" / "manifest.jsonl"
PINS = ROOT / "runs" / "misaki-iter" / "user_pins.json"
STORE = ROOT / "runs" / "misaki-iter"
AUDIO = LISTEN / "audio"
MP3 = LISTEN / "mp3"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def load_items() -> dict[str, dict]:
    items = {}
    for line in (ROOT / "data/dose_v1.jsonl").read_text().splitlines():
        row = json.loads(line)
        for ing in row.get("ingredients") or [row["name"]]:
            s = slug(ing)
            items[s] = {"drug": ing, "sentence": row["sentence"]}
    return items


def load_provenance() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if PROVENANCE.exists():
        for line in PROVENANCE.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["slug"]] = row
    if CONTEXT.exists():
        for row in json.loads(CONTEXT.read_text()).get("rows", []):
            out[row["slug"]] = row
    return out


def pick_clip_paths(s: str, prov: dict | None) -> dict[str, Path]:
    """Exact scored audio paths. Prefer context-controlled manifest clips."""
    if prov:
        by = {c["label"]: ROOT / c["wav"] for c in prov["clips"]}
        return {
            "cloud_iso": by.get("cloud_iso") or by.get("cloud_iso_ref"),
            "kokoro_iso": by["kokoro_iso"],
            "kokoro_sent": by["kokoro_sent"],
            "kokoro_span": by["kokoro_span"],
            "cloud_span": by.get("cloud_span"),
        }
    # Fallback: user rescore artifacts (exact pin synthesis).
    return {
        "cloud_iso": ROOT / "data/gold_gemini_ipa/wavs" / f"{s}.wav",
        "kokoro_iso": STORE / "user-iso" / f"{s}.wav",
        "kokoro_sent": STORE / "user-sent" / f"{s}.wav",
        "kokoro_span": STORE / "user-span" / f"{s}.wav",
        "cloud_span": None,
    }


def publish_audio(s: str, paths: dict[str, Path]) -> dict[str, str]:
    """Copy exact scored wavs into listen tree; return mp3 relative urls."""
    urls: dict[str, str] = {}
    for label, src in paths.items():
        if src is None or not src.exists():
            continue
        dest_wav = AUDIO / s / f"{label}.wav"
        dest_wav.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dest_wav.resolve():
            shutil.copy2(src, dest_wav)
        dest_mp3 = MP3 / s / f"{label}.mp3"
        wav_to_mp3(dest_wav, dest_mp3)
        urls[label] = f"mp3/{s}/{label}.mp3"
    return urls


def ipa_for(s: str) -> str:
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if slug(rec["ingredient"]) == s:
            return rec["ipa"]
    return "?"


def main() -> int:
    rank = json.loads(RANK.read_text())
    rows = rank["rows"]
    items = load_items()
    pins = json.loads(PINS.read_text()) if PINS.exists() else {}
    provenance = load_provenance()

    cards = []
    for r in rows:
        s = r["slug"]
        it = items.get(s)
        if it is None:
            continue
        prov = provenance.get(s)
        pin = (prov or {}).get("pin") or (pins.get(s) or {}).get("misaki") or "?"
        paths = pick_clip_paths(s, prov)
        urls = publish_audio(s, paths)

        f1 = r["ctc_f1"]
        ref_sha = sha256_file(paths["cloud_iso"]) if paths["cloud_iso"].exists() else "?"
        span_sha = sha256_file(paths["kokoro_span"]) if paths["kokoro_span"].exists() else "?"
        iso_sha = sha256_file(paths["kokoro_iso"]) if paths["kokoro_iso"].exists() else "?"

        bounds = ""
        if prov and prov.get("boundary", {}).get("kokoro", {}).get("bounds_s"):
            b = prov["boundary"]["kokoro"]["bounds_s"]
            bounds = f"crop {b['start']:.3f}–{b['end']:.3f}s · "

        audios = []
        if "cloud_iso" in urls:
            audios.append(
                f"<label>Cloud iso (reference) · {dur(paths['cloud_iso'])} · sha {ref_sha[:12]}…</label>"
                f"<audio controls preload='none' src='{html.escape(urls['cloud_iso'])}'></audio>"
            )
        if "kokoro_span" in urls:
            audios.append(
                f"<label>Scored Kokoro sentence-span · {dur(paths['kokoro_span'])} · "
                f"{bounds}sha {span_sha[:12]}…</label>"
                f"<audio controls preload='none' src='{html.escape(urls['kokoro_span'])}'></audio>"
            )
        if "kokoro_iso" in urls:
            audios.append(
                f"<label>Kokoro isolated (same pin) · {dur(paths['kokoro_iso'])} · sha {iso_sha[:12]}…</label>"
                f"<audio controls preload='none' src='{html.escape(urls['kokoro_iso'])}'></audio>"
            )

        prov_note = " · context-controlled provenance" if prov else ""
        cards.append(
            f"""<article class="card" id="{html.escape(s)}">
<div class="name">{html.escape(it['drug'])} <span class="f1">sentence-span WavLM F1 {f1:.3f}</span></div>
<p class="meta">pin <code>{html.escape(pin)}</code> · gold /{html.escape(ipa_for(s))}/ · scorer microsoft/wavlm-large SpeechBERTScore F1{prov_note}</p>
<p class="sent">{html.escape(it['sentence'])}</p>
{''.join(audios)}
<div class="row"><code>[{html.escape(it['drug'])}](/{html.escape(pin)}/)</code></div>
</article>"""
        )

    n = len(cards)
    mean = sum(r["ctc_f1"] for r in rows) / len(rows) if rows else 0
    controlled = sorted(provenance.keys())
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Misaki vs Cloud gold (provenance)</title>
<style>
body{{font:16px/1.45 -apple-system,sans-serif;margin:32px auto;max-width:860px;background:#fafafa;padding:0 16px 64px}}
h1{{font-size:22px}} .note,.meta{{color:#444;font-size:14px}}
.card{{background:#fff;border:1px solid #e4e4e4;border-radius:8px;padding:14px 16px;margin:12px 0}}
.name{{font-weight:650}} .f1{{color:#9a3412;font-weight:500}}
.sent{{margin:6px 0 10px;color:#333}} audio{{width:100%;margin:4px 0 10px}}
label{{display:block;font-size:13px;color:#333}}
.row code{{font-family:ui-monospace,Menlo,monospace;font-size:12px;background:#f3f3f3;padding:6px 8px;border-radius:4px}}
</style></head><body>
<h1>Kokoro Misaki vs Cloud gold</h1>
<p class="note">Plays the <strong>exact scored WAVs</strong> (sentence-span crop + reference iso). Score = sentence-span WavLM F1 vs locked Cloud iso. CTC phones reported separately in context-controlled run. No pass/fail threshold. n={n} mean {mean:.3f}.</p>
<p class="note">Context-controlled slugs with full provenance: {html.escape(", ".join(controlled) or "none yet")}.
<a href="../misaki-iter/context-controlled/index.html">Open 4-way comparison page</a></p>
{''.join(cards)}
</body></html>"""
    LISTEN.mkdir(parents=True, exist_ok=True)
    (LISTEN / "index.html").write_text(page)
    print(f"WROTE {LISTEN / 'index.html'} cards={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
