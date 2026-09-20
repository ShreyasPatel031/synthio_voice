#!/usr/bin/env python
"""Write IPA-worse-than-plain names into listen-google-ipa/index.html.

Lowest raw IPA F1 hid names that still lose to plain spelling (Biktarvy
0.721 IPA vs 0.819 plain never made the old top-20-worst-F1 list).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "runs" / "verify-google-ipa" / "results.json"
SRC = ROOT / "runs" / "verify-google-ipa"
TTS = SRC / "tts"
DST = ROOT / "runs" / "listen-google-ipa"
WORSE = -0.005


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _copy_audio(rec: dict) -> None:
    s = slug(rec["ingredient"])
    ext = rec.get("human_ext") or ".wav"
    pairs = [
        (DST / f"{s}__human{ext}", SRC / f"{s}__human{ext}"),
        (DST / f"{s}__plain.wav", SRC / f"{s}__plain.wav"),
        (DST / f"{s}__ipa.wav", SRC / f"{s}__ipa.wav"),
    ]
    # TTS dir uses dotted names from the original eval.
    tts_plain = TTS / f"{s}.plain.wav"
    tts_ipa = TTS / f"{s}.ipa.wav"
    if not pairs[1][1].exists() and tts_plain.exists():
        pairs[1] = (DST / f"{s}__plain.wav", tts_plain)
    if not pairs[2][1].exists() and tts_ipa.exists():
        pairs[2] = (DST / f"{s}__ipa.wav", tts_ipa)
    for dest, src in pairs:
        if src.exists():
            shutil.copy2(src, dest)


def write_lowest20(rows: list[dict] | None = None) -> Path:
    """Keep this name: apply/iter_fix import it. Page is IPA-loses-to-plain."""
    if rows is None:
        rows = json.loads(RESULTS.read_text())["rows"]
    lose = [
        r
        for r in rows
        if r.get("ipa_f1") is not None
        and r.get("plain") is not None
        and r["ipa_f1"] < r["plain"] + WORSE
    ]
    lose.sort(key=lambda r: r["ipa_f1"] - r["plain"])
    DST.mkdir(parents=True, exist_ok=True)
    for rec in lose:
        _copy_audio(rec)
    n = len(lose)
    mean_d = (
        sum(r["ipa_f1"] - r["plain"] for r in lose) / n if n else 0
    )
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<meta http-equiv='refresh' content='12'>",
        "<title>IPA worse than plain TTS</title>",
        "<style>body{font:16px/1.4 system-ui;max-width:760px;margin:2rem auto;padding:0 1rem}",
        "section{border:1px solid #ccc;border-radius:8px;padding:1rem 1.2rem;margin:1rem 0}",
        "h1{font-size:1.25rem} h2{font-size:1.05rem;margin:0 0 .35rem}",
        ".ipa{font-family:ui-monospace,monospace} .meta{color:#444}",
        ".rank{color:#888;font-size:.9rem} audio{width:100%}",
        "label{display:block;font-weight:600;margin:.45rem 0 .1rem}</style>",
        "<h1>IPA sidecar loses to plain spelling</h1>",
        f"<p class='meta'>{n} names, worst Δ first (mean Δ {mean_d:+.3f}). "
        "These were missing from the old lowest-F1-20 list because IPA F1 "
        "can still be ~0.72 while plain is 0.82. Reload every 12s.</p>",
    ]
    for i, rec in enumerate(lose, 1):
        s = slug(rec["ingredient"])
        ext = rec.get("human_ext") or ".wav"
        d = rec["ipa_f1"] - rec["plain"]
        flags = ", ".join(rec.get("flags") or []) or "—"
        parts.append(f"<section id='{s}'>")
        parts.append(f"<p class='rank'>#{i} of {n} &nbsp; IPA worse than plain</p>")
        parts.append(f"<h2>{rec['ingredient']}</h2>")
        parts.append(
            f"<p>IPA <span class='ipa'>{rec.get('ipa','')}</span><br>"
            f"used <span class='ipa'>{rec.get('ipa_used') or rec.get('ipa')}</span></p>"
        )
        parts.append(
            f"<p class='meta'>IPA F1 <b>{rec['ipa_f1']:.3f}</b> &nbsp; "
            f"plain {rec['plain']:.3f} &nbsp; Δ <b>{d:+.3f}</b><br>{flags}</p>"
        )
        parts.append(f"<label>Human ({rec.get('clip_source') or ''})</label>")
        parts.append(f"<audio controls src='{s}__human{ext}'></audio>")
        parts.append("<label>Plain spelling</label>")
        parts.append(f"<audio controls src='{s}__plain.wav'></audio>")
        parts.append("<label>Google IPA sidecar</label>")
        if (DST / f"{s}__ipa.wav").exists():
            parts.append(f"<audio controls src='{s}__ipa.wav'></audio>")
        else:
            parts.append("<p class='meta'>no IPA wav</p>")
        parts.append("</section>")
    (DST / "index.html").write_text("\n".join(parts) + "\n")
    return DST / "index.html"


if __name__ == "__main__":
    path = write_lowest20()
    print("wrote", path)
