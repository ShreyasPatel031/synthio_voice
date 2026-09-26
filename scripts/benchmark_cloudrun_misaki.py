#!/usr/bin/env python3
"""Score the kokoro-misaki Cloud Run endpoint the same way as the dashboard.

POST each DOSE carrier sentence as plain text. The service is stock
Kokoro-82M with the 284-name Misaki lexicon already loaded. Do not wrap
names in [Word](/phones/). Do not set voice or speed.

Crops the drug with forced alignment and reports WavLM SpeechBERTScore F1
against the locked Cloud Standard-C gold wav. One request at a time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dose_r.forced_align import extract_drug_span_forced_align
from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore

DEFAULT_URL = "https://kokoro-misaki-347838016394.us-east4.run.app"
LEXICON_PATH = ROOT / "data" / "kokoro_misaki_lexicon.json"
DOSE_PATH = ROOT / "data" / "dose_v1.jsonl"
GOLD_DIR = ROOT / "data" / "gold_gemini_ipa" / "wavs"
OUT_DIR = ROOT / "runs" / "cloudrun-misaki-bench"


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_sentences() -> dict[str, str]:
    sentences: dict[str, str] = {}
    for line in DOSE_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sentence = row["sentence"]
        for ing in row.get("ingredients") or [row["name"]]:
            sentences.setdefault(slugify(ing), sentence)
    return sentences


def gold_wav(slug: str, word: str) -> Path | None:
    names = [
        slug,
        slug.replace("-", "_"),
        re.sub(r"[^a-z0-9]+", "_", word.lower()).strip("_"),
    ]
    seen: set[str] = set()
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        path = GOLD_DIR / f"{name}.wav"
        if path.exists() and path.stat().st_size > 500:
            return path
    return None


def synth(session: requests.Session, base_url: str, text: str, timeout: float) -> bytes:
    resp = session.post(
        f"{base_url.rstrip('/')}/v1/audio/speech",
        json={"input": text},
        timeout=timeout,
    )
    if not resp.ok:
        raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
    if len(resp.content) < 44 or resp.content[:4] != b"RIFF":
        raise RuntimeError(f"response is not a wav ({len(resp.content)} bytes)")
    return resp.content


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=DEFAULT_URL)
    ap.add_argument("--lexicon", type=Path, default=LEXICON_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--only", nargs="*", default=())
    ap.add_argument("--skip-synth", action="store_true")
    args = ap.parse_args(argv)

    lexicon = json.loads(args.lexicon.read_text())
    entries = lexicon["entries"]
    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries if e["slug"] in wanted]
    sentences = load_sentences()
    sent_dir = args.out / "sent"
    sent_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    gold_emb: dict[str, object] = {}
    rows = []
    missing = []

    for n, entry in enumerate(entries, 1):
        slug = entry["slug"]
        word = entry["word"]
        sentence = sentences.get(slug)
        gold = gold_wav(slug, word)
        if sentence is None:
            missing.append((slug, "no-sentence"))
            continue
        if gold is None:
            missing.append((slug, "no-gold"))
            continue
        wav = sent_dir / f"{slug}.wav"
        if not args.skip_synth and (not wav.exists() or wav.stat().st_size < 500):
            t0 = time.perf_counter()
            try:
                wav.write_bytes(synth(session, args.base_url, sentence, args.timeout))
            except Exception as exc:
                missing.append((slug, f"synth:{exc}"))
                print(f"[{n}/{len(entries)}] ERR {slug} {exc}", flush=True)
                continue
            print(f"[{n}/{len(entries)}] synth {slug} {time.perf_counter() - t0:.1f}s", flush=True)
        if not wav.exists():
            missing.append((slug, "no-wav"))
            continue
        span = extract_drug_span_forced_align(wav.read_bytes(), sentence, word)
        if span is None:
            missing.append((slug, "no-span"))
            print(f"[{n}/{len(entries)}] NOSPAN {slug}", flush=True)
            continue
        key = str(gold)
        if key not in gold_emb:
            gold_emb[key] = extract_frame_embeddings(gold.read_bytes())
        f1 = float(speech_bertscore(extract_frame_embeddings(span), gold_emb[key])["f1"])
        rows.append({
            "slug": slug,
            "drug": word,
            "sentence": sentence,
            "ctc_f1": round(f1, 4),
        })
        print(f"[{n}/{len(entries)}] score {slug} {f1:.4f}", flush=True)

    rows.sort(key=lambda r: r["ctc_f1"])
    mean = sum(r["ctc_f1"] for r in rows) / len(rows) if rows else 0.0
    report = {
        "endpoint": args.base_url,
        "request": {"input": "<full carrier sentence, plain text>"},
        "lexicon_on_server": 284,
        "lexicon_mean_f1": lexicon.get("mean_f1"),
        "n": len(rows),
        "mean_ctc_f1": round(mean, 4),
        "missing": missing,
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cloud-rank.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"WROTE {args.out}/cloud-rank.json n={len(rows)} mean={mean:.4f} missing={len(missing)}",
        flush=True,
    )
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
