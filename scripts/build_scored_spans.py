#!/usr/bin/env python3
"""Build forced-align drug spans from current Misaki (what CTC F1 scores)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_iter = _load("misaki_iter", ROOT / "scripts" / "misaki_bottom_iterate.py")
from dose_r.forced_align import extract_drug_span_forced_align

STORE = ROOT / "runs" / "misaki-iter"
OUT = STORE / "cloud-span"
OUT.mkdir(parents=True, exist_ok=True)


def inject(sentence: str, spoken: str, phones: str) -> str:
    if spoken not in sentence:
        idx = sentence.lower().find(spoken.lower())
        if idx < 0:
            raise ValueError(spoken)
        spoken = sentence[idx : idx + len(spoken)]
    return sentence.replace(spoken, f"[{spoken}](/{phones}/)", 1)


def main() -> int:
    rank = json.loads((STORE / "cloud-rank.json").read_text())["rows"]
    pins = json.loads((STORE / "user_pins.json").read_text()) if (STORE / "user_pins.json").exists() else {}
    chosen = json.loads((STORE / "chosen.json").read_text()) if (STORE / "chosen.json").exists() else {}
    phones = dict(chosen.get("phones") or {})
    for s, pin in pins.items():
        phones[s] = pin["misaki"]
    items = {it["slug"]: it for it in _iter.load_full_items()}

    import torch
    from kokoro import KPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
    print(f"device={device} n={len(rank)}", flush=True)

    sent_dir = STORE / "listen-sent"
    sent_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for n, row in enumerate(rank, 1):
        slug = row["slug"]
        it = items.get(slug)
        ph = phones.get(slug)
        if it is None or not ph:
            print("skip", slug, flush=True)
            continue
        user_sent = STORE / "user-sent" / f"{slug}.wav"
        if slug in pins and user_sent.exists() and user_sent.stat().st_size > 500:
            raw = user_sent.read_bytes()
        else:
            text = inject(it["sentence"], it["spoken"], ph)
            dest = sent_dir / f"{slug}.wav"
            result = next(pipeline(text, voice=_iter.VOICE, speed=1.0))
            if result.audio is None:
                print("noaudio", slug, flush=True)
                continue
            sf.write(dest, result.audio.detach().cpu().numpy(), 24000)
            raw = dest.read_bytes()
        span = extract_drug_span_forced_align(raw, it["sentence"], it["spoken"])
        if span is None:
            span = extract_drug_span_forced_align(raw, it["sentence"], it["drug"])
        if span is None:
            print("nospan", slug, flush=True)
            continue
        (OUT / f"{slug}.wav").write_bytes(span)
        ok += 1
        if n % 40 == 0:
            print(n, slug, flush=True)
    print(f"WROTE {ok} spans -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
