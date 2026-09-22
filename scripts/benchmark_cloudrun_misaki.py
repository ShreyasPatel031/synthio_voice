#!/usr/bin/env python3
"""Full Misaki vs Cloud-gold benchmark through the Cloud Run endpoint.

1. Synthesize each scored drug sentence via POST /v1/audio/speech (speed 1.0).
2. Force-align drug span; WavLM F1 vs locked Cloud gold iso (same as cloud-rank).
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


def gold_wav(drug: str) -> Path | None:
    p = ROOT / "data" / "gold_gemini_ipa" / "wavs" / f"{slug(drug)}.wav"
    if p.exists() and p.stat().st_size > 500:
        return p
    return None


from dose_r.forced_align import extract_drug_span_forced_align
from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore

STORE = ROOT / "runs" / "oss-selfhosted" / "cloudrun-misaki-bench"
_GOLD_EMB: dict[str, object] = {}


def score_sentence(wav: Path, item: dict, gold: Path) -> float | None:
    raw = wav.read_bytes()
    span = extract_drug_span_forced_align(raw, item["sentence"], item["spoken"])
    if span is None:
        span = extract_drug_span_forced_align(raw, item["sentence"], item["drug"])
    if span is None:
        return None
    emb_c = extract_frame_embeddings(span)
    key = str(gold)
    emb_g = _GOLD_EMB.get(key)
    if emb_g is None:
        emb_g = extract_frame_embeddings(gold.read_bytes())
        _GOLD_EMB[key] = emb_g
    return float(speech_bertscore(emb_c, emb_g)["f1"])

STORE = ROOT / "runs" / "oss-selfhosted" / "cloudrun-misaki-bench"
DEFAULT_URL = "https://kokoro-misaki-347838016394.us-east4.run.app"


def inject(sentence: str, spoken: str, phones: str, word: str | None = None) -> str:
    w = word or spoken
    if spoken not in sentence:
        idx = sentence.lower().find(spoken.lower())
        if idx < 0:
            raise ValueError(f"cannot find {spoken!r} in sentence")
        spoken = sentence[idx : idx + len(spoken)]
    return sentence.replace(spoken, f"[{w}](/{phones}/)", 1)


def synth(session: requests.Session, base_url: str, text: str, timeout: float) -> bytes:
    url = f"{base_url.rstrip('/')}/v1/audio/speech"
    resp = session.post(
        url,
        json={"input": text, "voice": "af_heart", "speed": 1.0, "response_format": "wav"},
        timeout=timeout,
    )
    if not resp.ok:
        raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
    if len(resp.content) < 44:
        raise RuntimeError("empty audio")
    return resp.content


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_URL)
    ap.add_argument("--out", type=Path, default=STORE)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--only", nargs="*", default=())
    ap.add_argument("--skip-synth", action="store_true", help="rescore existing wavs only")
    args = ap.parse_args(argv)

    out = args.out
    sent_dir = out / "sent"
    sent_dir.mkdir(parents=True, exist_ok=True)

    pins = json.loads((ROOT / "runs/misaki-iter/user_pins.json").read_text())
    chosen = json.loads((ROOT / "runs/misaki-iter/chosen.json").read_text())
    phones = dict(chosen.get("phones") or {})
    for slug, pin in pins.items():
        phones[slug] = pin["misaki"]

    items = {it["slug"]: it for it in load_full_items()}
    session = requests.Session()
    rows = []
    missing = []
    keys = sorted(items)
    if args.only:
        keys = [k for k in args.only if k in items]

    for n, slug in enumerate(keys, 1):
        it = items[slug]
        gold = gold_wav(it["drug"])
        if gold is None:
            missing.append((slug, "no-locked-gold"))
            continue
        ph = phones.get(slug)
        if not ph:
            missing.append((slug, "no-misaki"))
            continue
        wav = sent_dir / f"{slug}.wav"
        if not args.skip_synth:
            if not wav.exists() or wav.stat().st_size < 500:
                word = pins.get(slug, {}).get("word")
                text = inject(it["sentence"], it["spoken"], ph, word=word)
                t0 = time.perf_counter()
                try:
                    wav.write_bytes(synth(session, args.base_url, text, args.timeout))
                except Exception as exc:
                    missing.append((slug, f"synth:{exc}"))
                    print(f"[{n}/{len(keys)}] ERR {slug} {exc}", flush=True)
                    continue
                dt = time.perf_counter() - t0
                print(f"[{n}/{len(keys)}] synth {slug} {dt:.1f}s", flush=True)
        if not wav.exists():
            missing.append((slug, "no-wav"))
            continue
        f1 = score_sentence(wav, it, gold)
        if f1 is None:
            missing.append((slug, "no-span"))
            continue
        kind = "pin" if slug in pins else "chosen"
        rows.append({
            "slug": slug,
            "drug": it["drug"],
            "kind": kind,
            "misaki": ph,
            "ctc_f1": round(float(f1), 4),
        })
        print(f"[{n}/{len(keys)}] score {slug} {rows[-1]['ctc_f1']:.4f}", flush=True)

    rows.sort(key=lambda r: r["ctc_f1"])
    mean = sum(r["ctc_f1"] for r in rows) / len(rows) if rows else 0.0
    report = {
        "endpoint": args.base_url,
        "n": len(rows),
        "mean_ctc_f1": round(mean, 4),
        "missing": missing,
        "rows": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "cloud-rank.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "summary.json").write_text(
        json.dumps({"mean_ctc_f1": report["mean_ctc_f1"], "n": len(rows), "missing": len(missing)}, indent=2)
        + "\n"
    )
    print(f"WROTE {out}/cloud-rank.json n={len(rows)} mean={mean:.4f} missing={len(missing)}", flush=True)
    print("LOWEST 10", flush=True)
    for r in rows[:10]:
        print(f"  {r['ctc_f1']:.4f} {r['slug']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
