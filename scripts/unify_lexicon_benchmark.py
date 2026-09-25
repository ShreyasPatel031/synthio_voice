#!/usr/bin/env python3
"""Resynthesize the full-sentence Kokoro benchmark through the Misaki lexicon.

Every name uses the phoneme string already chosen (user_pins override
chosen.json). The string is written to pipeline.g2p.lexicon.golds under the
exact token text, then the plain sentence is synthesized with pipeline(text).
No markdown inject and no token splice.

Scores the forced-align drug crop against the Cloud gold wav. Looks in
runs/misaki-iter/cloud-gold first, then data/gold_gemini_ipa/wavs. Tries
hyphen slugs and underscore names, because multi-word golds use
datopotamab_deruxtecan.wav, not datopotamab-deruxtecan.wav.
Does not write data/gold_gemini_ipa or chosen.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dose_r.forced_align import extract_drug_span_forced_align
from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore

STORE = ROOT / "runs" / "misaki-iter"
CLOUD = STORE / "cloud-gold"
LOCKED = ROOT / "data" / "gold_gemini_ipa" / "wavs"
SENT = STORE / "lexicon-sent"
SPAN = STORE / "lexicon-span"
VOICE = "af_heart"
SPEED = 1.0
_MISSING = object()


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def gold_wav(item: dict) -> Path | None:
    """Resolve the teacher wav. Hyphen slugs miss multi-word golds."""
    names = [
        item["slug"],
        item["spoken"].lower().replace(" ", "_"),
        item["drug"].lower().replace(" ", "_"),
        item["spoken"].lower().replace(" ", "-"),
        item["drug"].lower().replace(" ", "-"),
        item["slug"].replace("-", "_"),
    ]
    seen: set[str] = set()
    for name in names:
        stem = re.sub(r"[^a-z0-9_-]+", "", name).strip("_-")
        if not stem or stem in seen:
            continue
        seen.add(stem)
        for folder in (CLOUD, LOCKED):
            path = folder / f"{stem}.wav"
            if path.exists() and path.stat().st_size > 500:
                return path
    return None


def load_items() -> list[dict]:
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
    seen = {}
    for it in items:
        seen.setdefault(it["slug"], it)
    return list(seen.values())


def load_phones() -> dict[str, str]:
    chosen = json.loads((STORE / "chosen.json").read_text())
    phones = dict(chosen.get("phones") or {})
    pins = json.loads((STORE / "user_pins.json").read_text())
    for key, pin in pins.items():
        phones[key] = pin["misaki"]
    return phones


def lexicon_assignments(spoken: str, phones: str) -> dict[str, str]:
    words = spoken.split()
    parts = phones.split()
    if len(words) > 1 and len(parts) == len(words):
        return dict(zip(words, parts))
    return {spoken: phones}


def apply_lexicon(golds: dict, spoken: str, phones: str) -> dict:
    saved = {}
    for key, value in lexicon_assignments(spoken, phones).items():
        saved[key] = golds.get(key, _MISSING)
        golds[key] = value
    return saved


def restore_lexicon(golds: dict, saved: dict) -> None:
    for key, old in saved.items():
        if old is _MISSING:
            golds.pop(key, None)
        else:
            golds[key] = old


def drug_token_phones(tokens, spoken: str) -> str:
    words = spoken.split()
    texts = [t.text for t in tokens]
    lowered = [t.lower() for t in texts]
    want = [w.lower() for w in words]
    for i in range(len(tokens) - len(words) + 1):
        if lowered[i : i + len(words)] == want:
            return " ".join((tokens[i + j].phonemes or "") for j in range(len(words))).strip()
    return ""


def score_span(span: bytes, gold: Path, cache: dict) -> float:
    key = str(gold)
    emb_g = cache.get(key)
    if emb_g is None:
        emb_g = extract_frame_embeddings(gold.read_bytes())
        cache[key] = emb_g
    emb_c = extract_frame_embeddings(span)
    return float(speech_bertscore(emb_c, emb_g)["f1"])


def main() -> int:
    import torch
    from kokoro import KPipeline

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=())
    args = ap.parse_args()

    phones = load_phones()
    items = [it for it in load_items() if it["slug"] in phones]
    if args.only:
        only = set(args.only)
        items = [it for it in items if it["slug"] in only]
    old_rank = json.loads((STORE / "cloud-rank.before-lexicon.json").read_text()) if (STORE / "cloud-rank.before-lexicon.json").exists() else json.loads((STORE / "cloud-rank.json").read_text())
    old_f1 = {r["slug"]: r["ctc_f1"] for r in old_rank["rows"]}
    SENT.mkdir(parents=True, exist_ok=True)
    SPAN.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
    golds = pipeline.g2p.lexicon.golds
    print(f"device={device} speed={SPEED} n={len(items)} injection=lexicon", flush=True)

    rows = []
    cache: dict = {}
    missed = []
    for n, it in enumerate(items, 1):
        ph = phones[it["slug"]]
        gold = gold_wav(it)
        if gold is None:
            missed.append((it["slug"], "no-cloud-gold"))
            print("missing gold", it["slug"], flush=True)
            continue
        saved = apply_lexicon(golds, it["spoken"], ph)
        try:
            _, tokens = pipeline.g2p(it["sentence"])
            applied = drug_token_phones(tokens, it["spoken"])
            dest = SENT / f"{it['slug']}.wav"
            result = next(pipeline(it["sentence"], voice=VOICE, speed=SPEED))
            if result.audio is None:
                raise RuntimeError("no audio")
            sf.write(dest, result.audio.detach().cpu().numpy(), 24000)
            raw = dest.read_bytes()
            span = extract_drug_span_forced_align(raw, it["sentence"], it["spoken"])
            if span is None:
                span = extract_drug_span_forced_align(raw, it["sentence"], it["drug"])
            if span is None:
                missed.append((it["slug"], "no-span"))
                print("nospan", it["slug"], flush=True)
                continue
            (SPAN / f"{it['slug']}.wav").write_bytes(span)
            f1 = score_span(span, gold, cache)
        finally:
            restore_lexicon(golds, saved)
        prev = old_f1.get(it["slug"])
        row = {
            "slug": it["slug"],
            "drug": it["drug"],
            "kind": "lexicon",
            "misaki": ph,
            "applied": applied,
            "ctc_f1": round(f1, 4),
            "previous_f1": prev,
        }
        rows.append(row)
        delta = "" if prev is None else f" {f1 - prev:+.3f}"
        print(f"{n}/{len(items)} {it['slug']} {f1:.4f}{delta} applied={applied!r}", flush=True)

    existing_path = STORE / "lexicon-rank.json"
    if args.only and existing_path.exists():
        existing = json.loads(existing_path.read_text())
        by_slug = {r["slug"]: r for r in existing.get("rows") or []}
        by_slug.update({r["slug"]: r for r in rows})
        rows = list(by_slug.values())
        missed = [m for m in (existing.get("missing") or []) if m[0] not in by_slug] + missed
    rows.sort(key=lambda r: r["ctc_f1"])
    scored = [r["ctc_f1"] for r in rows]
    prevs = [r["previous_f1"] for r in rows if r["previous_f1"] is not None]
    paired = [(r["previous_f1"], r["ctc_f1"]) for r in rows if r["previous_f1"] is not None]
    mean = sum(scored) / len(scored) if scored else 0.0
    old_mean = sum(prevs) / len(prevs) if prevs else 0.0
    report = {
        "injection": "lexicon",
        "speed": SPEED,
        "n": len(rows),
        "mean_f1": round(mean, 4),
        "previous_mean_f1": round(old_mean, 4),
        "delta_mean": round(mean - old_mean, 4),
        "n_up": sum(1 for a, b in paired if b > a + 0.005),
        "n_down": sum(1 for a, b in paired if b < a - 0.005),
        "missing": missed,
        "rows": rows,
    }
    out = STORE / "lexicon-rank.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    rank_rows = [
        {"slug": r["slug"], "drug": r["drug"], "kind": "lexicon", "ctc_f1": r["ctc_f1"]}
        for r in rows
    ]
    (STORE / "cloud-rank.json").write_text(
        json.dumps({"n": len(rank_rows), "missing": missed, "injection": "lexicon", "rows": rank_rows}, indent=2) + "\n"
    )
    print(
        f"WROTE {out} n={len(rows)} mean={mean:.4f} prev={old_mean:.4f} "
        f"up={report['n_up']} down={report['n_down']} missing={len(missed)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
