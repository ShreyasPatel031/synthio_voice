#!/usr/bin/env python
"""Validate pronunciations on names with no human clip.

Path 2 (TTS vs human) only covers names with audio. For the rest, compare
independent Cloud TTS encodings to each other:

  plain     ingredient spelling (engine G2P)
  ipa       name + source-published Google IPA sidecar
  xsampa    same IPA rewritten to X-SAMPA (encoding check, not a new source)
  compact   dictionary respelling with hyphens stripped, spoken as text
            (original source string; NOT converted to IPA)

Calibrate on the 60% that have clips: does F1(plain, ipa) predict which arm
is closer to the human?
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
_hf = ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_hf))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_hf))
os.environ.setdefault("HF_HUB_CACHE", str(_hf / "hub"))

from dose_r.references.reference_clips import available_clips
from dose_r.references.tts_pronunciation import (
    compact_ascii,
    custom_pronunciation,
    ipa_to_xsampa,
)
from verify_google_ipa_tts import (
    TTS as CLIP_TTS,
    cloud_ipa,
    f1,
    looks_like_ipa,
    synth,
    token,
    try_ipa_synth,
)

OUT = ROOT / "runs" / "no-clip-tts-agree"
TTS = OUT / "tts"
LISTEN = ROOT / "runs" / "listen-no-clip-agree"
IPA_JSON = ROOT / "runs" / "gemini3-ipa-query" / "ipa.json"
PRON = ROOT / "dose_r" / "references" / "pronunciations.jsonl"
VERIFY = ROOT / "runs" / "verify-google-ipa" / "results.json"
WORKERS = int(os.environ.get("TTS_WORKERS", "8"))


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def mean(xs: list[float]) -> float | None:
    return round(statistics.mean(xs), 4) if xs else None


def load_pron() -> dict[str, dict]:
    out = {}
    for line in PRON.read_text().splitlines():
        rec = json.loads(line)
        out[rec["ingredient"]] = rec
    return out


def load_ipa() -> dict[str, dict]:
    return {r["ingredient"]: r for r in json.loads(IPA_JSON.read_text())}


def try_xsampa_synth(tok: str, name: str, ipa: str) -> tuple[bytes, str]:
    raw = ipa_to_xsampa(cloud_ipa(ipa))
    nodot = ipa_to_xsampa(cloud_ipa(ipa.replace(".", "")))
    last = None
    for cand in dict.fromkeys([raw, nodot]):
        if not cand:
            continue
        try:
            wav = synth(
                tok,
                text=name,
                pronunciations=custom_pronunciation(
                    name, cand, encoding="PHONETIC_ENCODING_X_SAMPA"
                ),
            )
            return wav, cand
        except RuntimeError as exc:
            last = exc
    raise last or RuntimeError("no X-SAMPA candidate")


def cached_synth(path: Path, fn) -> bytes:
    if path.exists() and path.stat().st_size > 500:
        return path.read_bytes()
    wav = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav)
    return wav


def calibrate(verify_rows: list[dict]) -> dict:
    """TTS-vs-TTS on names that already have human scores."""
    rows = []
    for rec in verify_rows:
        name = rec["ingredient"]
        if rec.get("plain") is None or rec.get("ipa_f1") is None:
            continue
        s = slug(name)
        p_plain = CLIP_TTS / f"{s}.plain.wav"
        p_ipa = CLIP_TTS / f"{s}.ipa.wav"
        if not (p_plain.exists() and p_ipa.exists()):
            continue
        agree = round(f1(p_plain.read_bytes(), p_ipa.read_bytes()), 4)
        plain_h = rec["plain"]
        ipa_h = rec["ipa_f1"]
        if abs(ipa_h - plain_h) < 0.005:
            winner = "tie"
        elif ipa_h > plain_h:
            winner = "ipa"
        else:
            winner = "plain"
        rows.append(
            {
                "ingredient": name,
                "agree_plain_ipa": agree,
                "plain_vs_human": plain_h,
                "ipa_vs_human": ipa_h,
                "winner": winner,
            }
        )
        print(
            f"cal {name}: agree={agree:.3f} plain={plain_h:.3f} "
            f"ipa={ipa_h:.3f} {winner}",
            flush=True,
        )
    disagrees = [r for r in rows if r["agree_plain_ipa"] < 0.80]
    agrees = [r for r in rows if r["agree_plain_ipa"] >= 0.80]
    def win_rate(subset, arm):
        n = len(subset)
        return round(sum(1 for r in subset if r["winner"] == arm) / n, 3) if n else None
    summary = {
        "n": len(rows),
        "mean_agree": mean([r["agree_plain_ipa"] for r in rows]),
        "n_agree_ge_0.80": len(agrees),
        "n_disagree_lt_0.80": len(disagrees),
        "when_agree_winner_plain": win_rate(agrees, "plain"),
        "when_agree_winner_ipa": win_rate(agrees, "ipa"),
        "when_agree_tie": win_rate(agrees, "tie"),
        "when_disagree_winner_plain": win_rate(disagrees, "plain"),
        "when_disagree_winner_ipa": win_rate(disagrees, "ipa"),
        "when_disagree_tie": win_rate(disagrees, "tie"),
        "mean_human_plain": mean([r["plain_vs_human"] for r in rows]),
        "mean_human_ipa": mean([r["ipa_vs_human"] for r in rows]),
    }
    return {"summary": summary, "rows": rows}


def write_listen(rows: list[dict]) -> None:
    scored = [r for r in rows if r.get("agree_plain_ipa") is not None]
    scored.sort(key=lambda r: r["agree_plain_ipa"])
    LISTEN.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>No-clip TTS agreement</title>",
        "<style>body{font:16px/1.4 system-ui;max-width:760px;margin:2rem auto;padding:0 1rem}",
        "section{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0}",
        ".ipa{font-family:ui-monospace,monospace} audio{width:100%}",
        "label{display:block;font-weight:600;margin:.4rem 0 .1rem}</style>",
        "<h1>No human clip — Cloud TTS encodings vs each other</h1>",
        "<p>Lowest F1(plain, IPA) first. Compact is the original dictionary "
        "respelling with hyphens stripped, spoken as text — not G2P'd to IPA.</p>",
    ]
    for rec in scored:
        s = slug(rec["ingredient"])
        for kind in ("plain", "ipa", "xsampa", "compact"):
            src = TTS / f"{s}.{kind}.wav"
            if src.exists():
                dest = LISTEN / f"{s}__{kind}.wav"
                dest.write_bytes(src.read_bytes())
        parts.append(f"<section><h2>{rec['ingredient']}</h2>")
        parts.append(
            f"<p class='ipa'>IPA {rec.get('ipa','')}<br>"
            f"compact {rec.get('compact','')}</p>"
        )
        parts.append(
            f"<p>plain↔IPA <b>{rec.get('agree_plain_ipa')}</b> &nbsp; "
            f"IPA↔X-SAMPA {rec.get('agree_ipa_xsampa')} &nbsp; "
            f"IPA↔compact {rec.get('agree_ipa_compact')}</p>"
        )
        for label, kind in (
            ("Plain spelling", "plain"),
            ("Google IPA sidecar", "ipa"),
            ("Same IPA as X-SAMPA", "xsampa"),
            ("Compact respelling as text", "compact"),
        ):
            if (LISTEN / f"{s}__{kind}.wav").exists():
                parts.append(f"<label>{label}</label>")
                parts.append(f"<audio controls src='{s}__{kind}.wav'></audio>")
        parts.append("</section>")
    (LISTEN / "index.html").write_text("\n".join(parts) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    TTS.mkdir(parents=True, exist_ok=True)
    clips = available_clips()
    ipa_map = load_ipa()
    pron = load_pron()
    verify = json.loads(VERIFY.read_text())["rows"]
    names = [r["ingredient"] for r in verify]
    no_clip = [n for n in names if n not in clips]
    print(
        f"names={len(names)} clips={len(clips)} no_clip={len(no_clip)}",
        flush=True,
    )

    tok = token()
    rows: list[dict] = []

    def one(name: str) -> dict:
        rec = {
            "ingredient": name,
            "name_type": pron.get(name, {}).get("name_type", ""),
            "ipa": (ipa_map.get(name) or {}).get("ipa") or "",
            "compact": pron.get(name, {}).get("compact") or "",
            "respelling": pron.get(name, {}).get("respelling") or "",
        }
        if not rec["compact"] and rec["respelling"]:
            rec["compact"] = compact_ascii(rec["respelling"])
        s = slug(name)
        try:
            rec["_plain"] = cached_synth(
                TTS / f"{s}.plain.wav", lambda: synth(tok, text=name)
            )
        except Exception as exc:
            rec["plain_error"] = str(exc)[:200]
        ipa = rec["ipa"]
        if ipa and looks_like_ipa(ipa):
            try:
                def _ipa():
                    wav, used = try_ipa_synth(tok, name, ipa)
                    rec["ipa_used"] = used
                    return wav

                rec["_ipa"] = cached_synth(TTS / f"{s}.ipa.wav", _ipa)
            except Exception as exc:
                rec["ipa_error"] = str(exc)[:200]
            try:
                def _xs():
                    wav, used = try_xsampa_synth(tok, name, ipa)
                    rec["xsampa_used"] = used
                    return wav

                rec["_xsampa"] = cached_synth(TTS / f"{s}.xsampa.wav", _xs)
            except Exception as exc:
                rec["xsampa_error"] = str(exc)[:200]
        compact = rec["compact"]
        if compact and compact.lower() != name.lower():
            try:
                rec["_compact"] = cached_synth(
                    TTS / f"{s}.compact.wav", lambda: synth(tok, text=compact)
                )
            except Exception as exc:
                rec["compact_error"] = str(exc)[:200]
        return rec

    print(f"=== synth {len(no_clip)} no-clip names ===", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(one, n): n for n in no_clip}
        for fut in as_completed(futs):
            rec = fut.result()
            done += 1
            print(f"synth [{done}/{len(no_clip)}] {rec['ingredient']}", flush=True)
            rows.append(rec)

    print("=== calibrate on names with a human clip ===", flush=True)
    cal = calibrate(verify)
    (OUT / "calibration.json").write_text(
        json.dumps(cal, indent=2, ensure_ascii=False) + "\n"
    )
    print("calibration", json.dumps(cal["summary"], indent=2), flush=True)

    print("=== score pairwise ===", flush=True)
    for rec in rows:
        plain, ipa_w, xs, comp = (
            rec.pop("_plain", None),
            rec.pop("_ipa", None),
            rec.pop("_xsampa", None),
            rec.pop("_compact", None),
        )
        if plain is not None and ipa_w is not None:
            rec["agree_plain_ipa"] = round(f1(plain, ipa_w), 4)
        if ipa_w is not None and xs is not None:
            rec["agree_ipa_xsampa"] = round(f1(ipa_w, xs), 4)
        if ipa_w is not None and comp is not None:
            rec["agree_ipa_compact"] = round(f1(ipa_w, comp), 4)
        if plain is not None and comp is not None:
            rec["agree_plain_compact"] = round(f1(plain, comp), 4)
        print(
            f"score {rec['ingredient']}: plain↔ipa={rec.get('agree_plain_ipa')} "
            f"ipa↔xsampa={rec.get('agree_ipa_xsampa')} "
            f"ipa↔compact={rec.get('agree_ipa_compact')}",
            flush=True,
        )

    scored = [r for r in rows if r.get("agree_plain_ipa") is not None]
    disagree = [r for r in scored if r["agree_plain_ipa"] < 0.80]
    summary = {
        "n_no_clip": len(no_clip),
        "n_scored_plain_ipa": len(scored),
        "mean_plain_ipa": mean([r["agree_plain_ipa"] for r in scored]),
        "n_disagree_lt_0.80": len(disagree),
        "mean_ipa_xsampa": mean(
            [r["agree_ipa_xsampa"] for r in rows if r.get("agree_ipa_xsampa") is not None]
        ),
        "mean_ipa_compact": mean(
            [r["agree_ipa_compact"] for r in rows if r.get("agree_ipa_compact") is not None]
        ),
        "mean_plain_compact": mean(
            [r["agree_plain_compact"] for r in rows if r.get("agree_plain_compact") is not None]
        ),
        "lowest_plain_ipa": [
            {
                "ingredient": r["ingredient"],
                "agree_plain_ipa": r["agree_plain_ipa"],
                "ipa": r.get("ipa"),
                "compact": r.get("compact"),
            }
            for r in sorted(scored, key=lambda x: x["agree_plain_ipa"])[:20]
        ],
        "calibration": cal["summary"],
    }
    (OUT / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False)
        + "\n"
    )
    write_listen(rows)
    print("summary", json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("wrote", OUT / "results.json", LISTEN / "index.html", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
