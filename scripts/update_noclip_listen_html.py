#!/usr/bin/env python
"""Listen page for names with no human clip: plain vs IPA vs compact.

Compare Google IPA against the original dictionary respelling (not a G2P
of that respelling). Dupixent is not on this page — it has a clip.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VERIFY = ROOT / "runs" / "verify-google-ipa" / "results.json"
AGREE = ROOT / "runs" / "no-clip-tts-agree" / "results.json"
TTS = ROOT / "runs" / "no-clip-tts-agree" / "tts"
PRON = ROOT / "dose_r" / "references" / "pronunciations.jsonl"
DST = ROOT / "runs" / "listen-no-clip"

from dose_r.references.reference_clips import available_clips


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def main() -> Path:
    clips = available_clips()
    verify = {r["ingredient"]: r for r in json.loads(VERIFY.read_text())["rows"]}
    agree_rows = {r["ingredient"]: r for r in json.loads(AGREE.read_text())["rows"]}
    pron = {}
    for line in PRON.read_text().splitlines():
        rec = json.loads(line)
        pron[rec["ingredient"]] = rec

    names = [n for n in verify if n not in clips]
    rows = []
    for name in names:
        a = agree_rows.get(name) or {}
        p = pron.get(name) or {}
        v = verify[name]
        rows.append(
            {
                "ingredient": name,
                "ipa": a.get("ipa") or v.get("ipa") or "",
                "ipa_used": a.get("ipa_used") or "",
                "url": v.get("url") or "",
                "respelling": p.get("respelling") or "",
                "respelling_source": p.get("source") or "",
                "compact": a.get("compact") or p.get("compact") or "",
                "agree_plain_ipa": a.get("agree_plain_ipa"),
                "agree_ipa_compact": a.get("agree_ipa_compact"),
            }
        )

    def sort_key(r: dict):
        a = r["agree_plain_ipa"]
        return (0 if a is not None and a < 0.80 else 1, a if a is not None else 9)

    rows.sort(key=sort_key)
    DST.mkdir(parents=True, exist_ok=True)
    for rec in rows:
        s = slug(rec["ingredient"])
        for kind in ("plain", "ipa", "compact", "xsampa"):
            src = TTS / f"{s}.{kind}.wav"
            if src.exists():
                shutil.copy2(src, DST / f"{s}__{kind}.wav")

    n_dis = sum(
        1 for r in rows if r["agree_plain_ipa"] is not None and r["agree_plain_ipa"] < 0.80
    )
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>No human clip — plain vs IPA</title>",
        "<style>body{font:16px/1.4 system-ui;max-width:760px;margin:2rem auto;padding:0 1rem}",
        "section{border:1px solid #ccc;border-radius:8px;padding:1rem 1.2rem;margin:1rem 0}",
        "section.dis{border-color:#a44}",
        "h1{font-size:1.25rem} h2{font-size:1.05rem;margin:0 0 .35rem}",
        ".ipa{font-family:ui-monospace,monospace} .meta{color:#444}",
        ".rank{color:#888;font-size:.9rem} audio{width:100%}",
        "label{display:block;font-weight:600;margin:.45rem 0 .1rem}</style>",
        "<h1>No human clip — compare plain vs IPA</h1>",
        "<p class='meta'>These names have <b>no human recording</b>. "
        "Dupixent / <span class='ipa'>DU-pix-ent</span> is not here; that one has a clip. "
        f"{len(rows)} names. First {n_dis} are where Cloud TTS plain and IPA "
        "already sound different (F1 &lt; 0.80) — those are the ones to ear-check. "
        "Hold the original dictionary respelling (CAPS = stress). "
        "Do not treat compact-as-text as gold.</p>",
        "<p><a href='#agree'>Skip to names that already agree</a></p>",
    ]
    marked_agree = False
    for i, rec in enumerate(rows, 1):
        s = slug(rec["ingredient"])
        agree = rec["agree_plain_ipa"]
        disagree = agree is not None and agree < 0.80
        if not disagree and not marked_agree:
            parts.append("<h2 id='agree'>Already agree (plain ≈ IPA)</h2>")
            marked_agree = True
        cls = "dis" if disagree else ""
        tag = "DISAGREE — ear-check" if disagree else "agree"
        parts.append(f"<section class='{cls}' id='{s}'>")
        parts.append(f"<p class='rank'>#{i} of {len(rows)} &nbsp; {tag}</p>")
        parts.append(f"<h2>{rec['ingredient']}</h2>")
        src = rec["respelling_source"] or "—"
        parts.append(
            f"<p>Source respelling <span class='ipa'>{rec['respelling'] or '—'}</span> "
            f"<span class='meta'>({src})</span><br>"
            f"Google IPA <span class='ipa'>{rec['ipa'] or '—'}</span></p>"
        )
        if agree is not None:
            parts.append(
                f"<p class='meta'>plain↔IPA F1 <b>{agree:.3f}</b> &nbsp; "
                f"IPA↔compact {rec.get('agree_ipa_compact') if rec.get('agree_ipa_compact') is not None else 'n/a'}</p>"
            )
        else:
            parts.append("<p class='meta'>no IPA synth (Cloud rejected or not IPA)</p>")
        parts.append("<label>Plain spelling</label>")
        if (DST / f"{s}__plain.wav").exists():
            parts.append(f"<audio controls src='{s}__plain.wav'></audio>")
        else:
            parts.append("<p class='meta'>no plain wav</p>")
        parts.append("<label>Google IPA sidecar</label>")
        if (DST / f"{s}__ipa.wav").exists():
            parts.append(f"<audio controls src='{s}__ipa.wav'></audio>")
        else:
            parts.append("<p class='meta'>no IPA wav</p>")
        if (DST / f"{s}__compact.wav").exists():
            parts.append(
                "<label>Compact respelling as text (not gold — other format)</label>"
            )
            parts.append(f"<audio controls src='{s}__compact.wav'></audio>")
        parts.append("</section>")
    path = DST / "index.html"
    path.write_text("\n".join(parts) + "\n")
    print("wrote", path, "n=", len(rows), "disagree=", n_dis)
    return path


if __name__ == "__main__":
    main()
