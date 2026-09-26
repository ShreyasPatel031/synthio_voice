#!/usr/bin/env python
"""Gemini 3.1 Flash TTS: published respelling vs compact, all names.

Hyphen: original source string (`ak-oh-RAM-id-is`, `lin-e-RIX-i-bat`).
Compact: hyphens stripped (`akohramidis`). Never G2P respelling to IPA.

Score wavlm F1 vs the human clip and vs Cloud Standard-C + Google IPA.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
_hf = ROOT / ".cache" / "huggingface"
_hf.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_hf))
os.environ.setdefault("HF_HUB_CACHE", str(_hf / "hub"))

from eval_gemini31_holdout import (  # noqa: E402
    MODEL,
    VOICE,
    cached_synth,
    cloud_ipa_path,
    f1,
    mean,
    slug,
)
from dose_r.references.reference_clips import available_clips  # noqa: E402
from dose_r.references.tts_pronunciation import compact_ascii  # noqa: E402

PRON = ROOT / "dose_r" / "references" / "pronunciations.jsonl"
HOLD_TTS = ROOT / "runs" / "gemini31-holdout" / "tts"
OUT = ROOT / "runs" / "gemini31-respell-all"
TTS = OUT / "tts"
LISTEN = ROOT / "runs" / "listen-gemini31-full"
WORKERS = int(os.environ.get("TTS_WORKERS", "4"))
VARIANTS = ("hyphen", "compact")
GAP = 0.03


def load_pron() -> list[dict]:
    rows = []
    for line in PRON.read_text().splitlines():
        rec = json.loads(line)
        if (rec.get("respelling") or "").strip():
            rows.append(rec)
    return rows


def texts(name: str, canonical: str) -> dict[str, str]:
    return {
        "hyphen": canonical.strip(),
        "compact": compact_ascii(canonical) or name,
    }


def reuse_or_synth(path: Path, text: str) -> bytes:
    if path.exists() and path.stat().st_size > 500:
        return path.read_bytes()
    alt = HOLD_TTS / path.name
    if alt.exists() and alt.stat().st_size > 500:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(alt, path)
        return path.read_bytes()
    return cached_synth(path, text)


def wav_path(name: str, kind: str) -> Path:
    return TTS / f"{slug(name)}.{kind}.wav"


def wins(a: float, b: float) -> str:
    if a >= b + GAP:
        return "hyphen"
    if b >= a + GAP:
        return "compact"
    return "tie"


def write_listen(rows: list[dict], clips: dict, summary: dict) -> Path:
    LISTEN.mkdir(parents=True, exist_ok=True)
    human = [r for r in rows if r.get("vs_human", {}).get("hyphen") is not None]
    ranked = sorted(
        human,
        key=lambda r: (
            (r["vs_human"].get("compact") or 0) - (r["vs_human"].get("hyphen") or 0)
        ),
    )
    rest = [r for r in rows if r not in ranked]
    rest.sort(
        key=lambda r: -abs(
            (r.get("vs_ipa", {}).get("compact") or 0)
            - (r.get("vs_ipa", {}).get("hyphen") or 0)
        )
    )
    bv = summary.get("by_variant", {})
    pair = summary.get("paired_vs_human", {})
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Gemini 3.1 hyphen vs compact (full)</title>",
        "<style>body{font:16px/1.4 system-ui;max-width:820px;margin:2rem auto;padding:0 1rem}",
        "section{border:1px solid #ccc;border-radius:8px;padding:1rem 1.2rem;margin:1rem 0}",
        "section.hyphen{border-color:#26a} section.compact{border-color:#a62}",
        "h1{font-size:1.25rem} h2{font-size:1.05rem;margin:0 0 .35rem}",
        "p{margin:.2rem 0 .5rem} .ipa{font-family:ui-monospace,monospace}",
        "label{display:block;font-weight:600;margin:.45rem 0 .1rem} audio{width:100%}",
        ".meta{font-size:.9rem;color:#555}</style>",
        "<h1>Gemini 3.1 Flash TTS — hyphen vs compact (full set)</h1>",
        "<p>Hyphen = published respelling as-is. Compact = hyphens stripped. "
        "Not G2P to IPA. Names with a human clip are first, hyphen-better at the top.</p>",
        "<p class='meta'>"
        f"vs human: hyphen {bv.get('hyphen', {}).get('mean_vs_human')} "
        f"(n={bv.get('hyphen', {}).get('n_human')}) · "
        f"compact {bv.get('compact', {}).get('mean_vs_human')} "
        f"(n={bv.get('compact', {}).get('n_human')}). "
        f"Paired wins hyphen {pair.get('hyphen_wins')} / "
        f"compact {pair.get('compact_wins')} / tie {pair.get('ties')}. "
        f"Cloud IPA vs human {summary.get('cloud_ipa_vs_human')}.</p>",
    ]
    for rec in ranked + rest:
        s = slug(rec["ingredient"])
        vh = rec.get("vs_human") or {}
        klass = rec.get("winner_human") or ""
        parts.append(f"<section class='{klass}' id='{s}'>")
        parts.append(f"<h2>{rec['ingredient']}</h2>")
        parts.append(
            f"<p>respelling <span class='ipa'>{rec.get('respelling') or '—'}</span></p>"
        )
        parts.append(
            "<p class='meta'>"
            f"hyphen vs human {vh.get('hyphen')} / vs IPA {rec.get('vs_ipa', {}).get('hyphen')} · "
            f"compact vs human {vh.get('compact')} / vs IPA {rec.get('vs_ipa', {}).get('compact')} · "
            f"Cloud IPA vs human {rec.get('cloud_ipa_vs_human')} · "
            f"hyphen vs compact {rec.get('hyphen_vs_compact')}</p>"
        )
        clip = clips.get(rec["ingredient"])
        if clip is not None:
            dest = LISTEN / f"{s}__human{clip.path.suffix}"
            if not dest.exists():
                dest.write_bytes(clip.path.read_bytes())
            parts.append(
                f"<label>human ({clip.source})</label>"
                f"<audio controls src='{dest.name}'></audio>"
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
        fed = rec.get("texts") or {}
        for kind in VARIANTS:
            src = wav_path(rec["ingredient"], kind)
            dest = LISTEN / f"{s}__{kind}.wav"
            if src.exists():
                if not dest.exists():
                    shutil.copy2(src, dest)
                parts.append(
                    f"<label>Gemini 3.1 {kind} "
                    f"(<span class='ipa'>{fed.get(kind, '')}</span>)</label>"
                    f"<audio controls src='{dest.name}'></audio>"
                )
        parts.append("</section>")
    dest = LISTEN / "index.html"
    dest.write_text("\n".join(parts) + "\n")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    items = load_pron()
    if args.only:
        want = {n.lower() for n in args.only}
        items = [r for r in items if r["ingredient"].lower() in want]
    clips = available_clips()
    TTS.mkdir(parents=True, exist_ok=True)

    jobs = []
    for rec in items:
        name = rec["ingredient"]
        fed = texts(name, rec["respelling"])
        for kind, text in fed.items():
            jobs.append((name, kind, text, wav_path(name, kind)))

    print(f"{len(items)} names, {len(jobs)} synths", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {
            ex.submit(reuse_or_synth, path, text): (name, kind, text)
            for name, kind, text, path in jobs
        }
        done = 0
        for fut in as_completed(futs):
            name, kind, text = futs[fut]
            done += 1
            try:
                fut.result()
                print(f"  ok {done}/{len(jobs)} {name} {kind} {text!r}", flush=True)
            except Exception as exc:
                print(f"  FAIL {done}/{len(jobs)} {name} {kind}: {exc}", flush=True)

    rows = []
    for rec in items:
        name = rec["ingredient"]
        fed = texts(name, rec["respelling"])
        clip = clips.get(name)
        ipa_path = cloud_ipa_path(name)
        vs_h: dict[str, float | None] = {}
        vs_i: dict[str, float | None] = {}
        paths = {kind: wav_path(name, kind) for kind in VARIANTS}
        for kind in VARIANTS:
            p = paths[kind]
            if not p.exists() or p.stat().st_size < 500:
                vs_h[kind] = None
                vs_i[kind] = None
                continue
            vs_h[kind] = round(f1(p, clip.path), 4) if clip is not None else None
            vs_i[kind] = round(f1(p, ipa_path), 4) if ipa_path else None
        cloud_ipa_human = None
        if clip is not None and ipa_path is not None:
            cloud_ipa_human = round(f1(ipa_path, clip.path), 4)
        h_vs_c = None
        hp, cp = paths["hyphen"], paths["compact"]
        if hp.exists() and cp.exists() and hp.stat().st_size > 500 and cp.stat().st_size > 500:
            h_vs_c = round(f1(hp, cp), 4)
        winner = None
        if vs_h["hyphen"] is not None and vs_h["compact"] is not None:
            winner = wins(vs_h["hyphen"], vs_h["compact"])
        rows.append(
            {
                "ingredient": name,
                "respelling": rec.get("respelling"),
                "respelling_source": rec.get("source"),
                "texts": fed,
                "clip_source": clip.source if clip is not None else None,
                "cloud_ipa_vs_human": cloud_ipa_human,
                "vs_human": vs_h,
                "vs_ipa": vs_i,
                "hyphen_vs_compact": h_vs_c,
                "winner_human": winner,
            }
        )
        print(
            f"  {name} H={vs_h['hyphen']} C={vs_h['compact']} "
            f"ipaH={vs_i['hyphen']} ipaC={vs_i['compact']} "
            f"cloud={cloud_ipa_human} HvsC={h_vs_c} win={winner}",
            flush=True,
        )

    summary: dict = {
        "n": len(rows),
        "model": MODEL,
        "voice": VOICE,
        "gap": GAP,
        "cloud_ipa_vs_human": mean(
            [r["cloud_ipa_vs_human"] for r in rows if r.get("cloud_ipa_vs_human") is not None]
        ),
        "mean_hyphen_vs_compact": mean(
            [r["hyphen_vs_compact"] for r in rows if r.get("hyphen_vs_compact") is not None]
        ),
        "by_variant": {},
    }
    for kind in VARIANTS:
        h = [r["vs_human"][kind] for r in rows if r["vs_human"].get(kind) is not None]
        i = [r["vs_ipa"][kind] for r in rows if r["vs_ipa"].get(kind) is not None]
        summary["by_variant"][kind] = {
            "n_human": len(h),
            "mean_vs_human": mean(h),
            "n_ipa": len(i),
            "mean_vs_cloud_ipa": mean(i),
        }
    paired = [
        r
        for r in rows
        if r["vs_human"].get("hyphen") is not None
        and r["vs_human"].get("compact") is not None
    ]
    wc = Counter(r["winner_human"] for r in paired)
    summary["paired_vs_human"] = {
        "n": len(paired),
        "mean_hyphen": mean([r["vs_human"]["hyphen"] for r in paired]),
        "mean_compact": mean([r["vs_human"]["compact"] for r in paired]),
        "hyphen_wins": wc.get("hyphen", 0),
        "compact_wins": wc.get("compact", 0),
        "ties": wc.get("tie", 0),
    }
    paired_ipa = [
        r
        for r in rows
        if r["vs_ipa"].get("hyphen") is not None and r["vs_ipa"].get("compact") is not None
    ]
    summary["paired_vs_cloud_ipa"] = {
        "n": len(paired_ipa),
        "mean_hyphen": mean([r["vs_ipa"]["hyphen"] for r in paired_ipa]),
        "mean_compact": mean([r["vs_ipa"]["compact"] for r in paired_ipa]),
    }
    both_cloud = [
        r
        for r in paired
        if r.get("cloud_ipa_vs_human") is not None
    ]
    summary["vs_cloud_ipa_baseline"] = {
        "n": len(both_cloud),
        "hyphen_beats_cloud": sum(
            1
            for r in both_cloud
            if r["vs_human"]["hyphen"] > r["cloud_ipa_vs_human"] + GAP
        ),
        "compact_beats_cloud": sum(
            1
            for r in both_cloud
            if r["vs_human"]["compact"] > r["cloud_ipa_vs_human"] + GAP
        ),
        "cloud_beats_hyphen": sum(
            1
            for r in both_cloud
            if r["cloud_ipa_vs_human"] > r["vs_human"]["hyphen"] + GAP
        ),
        "cloud_beats_compact": sum(
            1
            for r in both_cloud
            if r["cloud_ipa_vs_human"] > r["vs_human"]["compact"] + GAP
        ),
    }
    ranked = sorted(
        summary["by_variant"].items(),
        key=lambda kv: (kv[1]["mean_vs_human"] or 0, kv[1]["mean_vs_cloud_ipa"] or 0),
        reverse=True,
    )
    summary["best_vs_human"] = ranked[0][0] if ranked else None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False)
        + "\n"
    )
    html = write_listen(rows, clips, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(html, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
