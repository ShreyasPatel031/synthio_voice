#!/usr/bin/env python
"""No-clip names: Gemini 3.1 hyphen vs Cloud Standard-C + Google IPA.

Hyphen is the third voice. Do not G2P respelling to IPA.

Buckets:
  ipa_suspect   hyphen and compact agree with each other, both disagree with IPA
  split         hyphen disagrees with IPA, compact does too, but hyphen≠compact
  hyphen_artifact  compact agrees with IPA; hyphen is the odd one (often CAPS spelling)
  agree         hyphen matches IPA
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
_hf = ROOT / ".cache" / "huggingface"
_hf.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_hf))
os.environ.setdefault("HF_HUB_CACHE", str(_hf / "hub"))

import librosa  # noqa: E402

from eval_gemini31_holdout import cloud_ipa_path, mean, slug  # noqa: E402
from dose_r.references.reference_clips import available_clips  # noqa: E402
from dose_r.scoring.speech_similarity import (  # noqa: E402
    extract_frame_embeddings,
    speech_bertscore,
)

IN = ROOT / "runs" / "gemini31-respell-all" / "results.json"
TTS = ROOT / "runs" / "gemini31-respell-all" / "tts"
VERIFY = ROOT / "runs" / "verify-google-ipa" / "results.json"
OUT = ROOT / "runs" / "gemini31-respell-all" / "noclip_ipa_diverge.json"
LISTEN = ROOT / "runs" / "listen-hyphen-ipa-noclip"
AGREE = 0.75
DIVERGE = 0.70
GAP = 0.03
_EMB: dict = {}


def emb(path: Path):
    key = (str(path), path.stat().st_size)
    if key not in _EMB:
        _EMB[key] = extract_frame_embeddings(path)
    return _EMB[key]


def prf(a: Path, b: Path) -> dict[str, float]:
    return speech_bertscore(emb(a), emb(b))


def dur(path: Path) -> float:
    audio, sr = librosa.load(str(path), sr=None, mono=True)
    return len(audio) / sr


def classify(h_ipa: float, c_ipa: float | None, h_c: float | None, h_ratio: float) -> str:
    compact_ok = c_ipa is not None and c_ipa >= AGREE
    gemini_agree = h_c is not None and h_c >= AGREE
    hyphen_bad = h_ipa < DIVERGE
    if not hyphen_bad:
        return "agree"
    if compact_ok and h_ratio >= 1.8:
        return "hyphen_artifact"
    if compact_ok and (h_c is None or h_c < AGREE):
        return "hyphen_artifact"
    if gemini_agree:
        return "ipa_suspect"
    if c_ipa is not None and c_ipa < DIVERGE:
        return "split"
    return "ipa_suspect"


def write_listen(rows: list[dict]) -> Path:
    LISTEN.mkdir(parents=True, exist_ok=True)
    order = ["ipa_suspect", "split", "hyphen_artifact", "agree", "no_ipa_wav"]
    ranked = sorted(
        rows,
        key=lambda r: (
            order.index(r["bucket"]) if r["bucket"] in order else 9,
            r.get("hyphen_vs_ipa", {}).get("f1") or 9,
        ),
    )
    counts = Counter(r["bucket"] for r in ranked)
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>No-clip: hyphen vs Cloud IPA</title>",
        "<style>body{font:16px/1.4 system-ui;max-width:820px;margin:2rem auto;padding:0 1rem}",
        "section{border:1px solid #ccc;border-radius:8px;padding:1rem 1.2rem;margin:1rem 0}",
        "section.ipa_suspect{border-color:#a44} section.split{border-color:#c80}",
        "section.hyphen_artifact{border-color:#888} section.agree{border-color:#6a6}",
        "h1{font-size:1.25rem} h2{font-size:1.05rem;margin:0 0 .35rem}",
        "p{margin:.2rem 0 .5rem} .ipa{font-family:ui-monospace,monospace}",
        "label{display:block;font-weight:600;margin:.45rem 0 .1rem} audio{width:100%}",
        ".meta{font-size:.9rem;color:#555} .tag{display:inline-block;background:#eee;",
        "padding:.1rem .45rem;border-radius:4px;margin-right:.3rem;font-size:.85rem}</style>",
        "<h1>No-clip names: Gemini 3.1 hyphen vs Cloud IPA</h1>",
        "<p>No human clip. Hyphen = published respelling as-is. "
        "<b>ipa_suspect</b> = hyphen and compact agree, both disagree with IPA. "
        f"{dict(counts)}</p>",
    ]
    for rec in ranked:
        s = slug(rec["ingredient"])
        parts.append(f"<section class='{rec['bucket']}' id='{s}'>")
        parts.append(f"<h2>{rec['ingredient']}</h2>")
        parts.append(f"<span class='tag'>{rec['bucket']}</span>")
        parts.append(
            f"<p>hyphen <span class='ipa'>{rec.get('respelling') or '—'}</span><br>"
            f"Cloud IPA <span class='ipa'>{rec.get('cloud_ipa') or '—'}</span></p>"
        )
        hi = rec.get("hyphen_vs_ipa") or {}
        ci = rec.get("compact_vs_ipa") or {}
        parts.append(
            "<p class='meta'>"
            f"hyphen vs IPA F1 {hi.get('f1')} (P {hi.get('precision')} R {hi.get('recall')}) · "
            f"compact vs IPA F1 {ci.get('f1')} · "
            f"hyphen vs compact {rec.get('hyphen_vs_compact')}</p>"
        )
        ipa_src = cloud_ipa_path(rec["ingredient"])
        if ipa_src is not None:
            dest = LISTEN / f"{s}__cloud_ipa.wav"
            if not dest.exists():
                shutil.copy2(ipa_src, dest)
            parts.append(
                "<label>Cloud Standard-C + Google IPA</label>"
                f"<audio controls src='{dest.name}'></audio>"
            )
        for kind, label in (("hyphen", "Gemini 3.1 hyphen"), ("compact", "Gemini 3.1 compact")):
            src = TTS / f"{s}.{kind}.wav"
            dest = LISTEN / f"{s}__{kind}.wav"
            if src.exists():
                if not dest.exists():
                    shutil.copy2(src, dest)
                extra = rec.get("respelling") if kind == "hyphen" else rec.get("compact")
                parts.append(
                    f"<label>{label} (<span class='ipa'>{extra or ''}</span>)</label>"
                    f"<audio controls src='{dest.name}'></audio>"
                )
        parts.append("</section>")
    dest = LISTEN / "index.html"
    dest.write_text("\n".join(parts) + "\n")
    return dest


def main() -> int:
    prev = json.loads(IN.read_text())
    verify = {}
    if VERIFY.exists():
        verify = {r["ingredient"]: r for r in json.loads(VERIFY.read_text())["rows"]}
    clips = available_clips()
    rows = []
    for rec in prev["rows"]:
        name = rec["ingredient"]
        if name in clips:
            continue
        hp = TTS / f"{slug(name)}.hyphen.wav"
        cp = TTS / f"{slug(name)}.compact.wav"
        ipa = cloud_ipa_path(name)
        v = verify.get(name) or {}
        compact = rec.get("texts", {}).get("compact")
        base = {
            "ingredient": name,
            "respelling": rec.get("respelling"),
            "compact": compact,
            "cloud_ipa": v.get("ipa_used") or v.get("ipa") or "",
            "cloud_ipa_url": v.get("url") or "",
        }
        if ipa is None or not hp.exists():
            rows.append({**base, "bucket": "no_ipa_wav"})
            print(f"  {name} no_ipa_wav", flush=True)
            continue
        h_ipa = prf(hp, ipa)
        c_ipa = prf(cp, ipa) if cp.exists() else None
        h_c = rec.get("hyphen_vs_compact")
        if h_c is None and cp.exists():
            h_c = prf(hp, cp)["f1"]
        hd, idur = dur(hp), dur(ipa)
        ratio = hd / idur if idur else 9
        bucket = classify(
            h_ipa["f1"],
            c_ipa["f1"] if c_ipa else None,
            h_c,
            ratio,
        )
        row = {
            **base,
            "hyphen_vs_ipa": h_ipa,
            "compact_vs_ipa": c_ipa,
            "hyphen_vs_compact": h_c,
            "hyphen_dur": round(hd, 3),
            "ipa_dur": round(idur, 3),
            "hyphen_ipa_dur_ratio": round(ratio, 3),
            "bucket": bucket,
        }
        rows.append(row)
        print(
            f"  {name} {bucket} H-IPA={h_ipa['f1']} C-IPA="
            f"{None if c_ipa is None else c_ipa['f1']} H-C={h_c} ratio={ratio:.2f}",
            flush=True,
        )

    buckets = Counter(r["bucket"] for r in rows)
    scored = [r for r in rows if r.get("hyphen_vs_ipa")]
    summary = {
        "n_no_clip": len(rows),
        "n_compared": len(scored),
        "agree_threshold": AGREE,
        "diverge_threshold": DIVERGE,
        "buckets": dict(buckets),
        "mean_hyphen_vs_ipa_f1": mean([r["hyphen_vs_ipa"]["f1"] for r in scored]),
        "mean_hyphen_vs_ipa_recall": mean([r["hyphen_vs_ipa"]["recall"] for r in scored]),
        "ipa_suspect": [
            r["ingredient"] for r in scored if r["bucket"] == "ipa_suspect"
        ],
        "split": [r["ingredient"] for r in scored if r["bucket"] == "split"],
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    html = write_listen(rows)
    print(json.dumps(summary, indent=2), flush=True)
    print(html, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
