#!/usr/bin/env python3
"""Main DOSE benchmark vs the current Cloud TTS gold wav.

Protocol (same as the lexicon / misaki-iter board):
  synthesize the carrier sentence, forced-align the drug span, score
  WavLM-large SpeechBERTScore F1 against the newest Cloud Standard-C wav.

Systems:
  kokoro-plain     Kokoro-82M, Misaki G2P, no lexicon
  kokoro-pronounce Kokoro-82M + chosen/user_pins Misaki lexicon
  qwen-plain       Qwen3-TTS 1.7B-Base, public clone.wav speaker
  qwen-wav         Qwen3-TTS 1.7B-Base, clone the current Cloud gold wav
  qwen-ft          best full-set fine-tune (existing sentence wavs rescored)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dose_r.forced_align import extract_drug_span_forced_align
from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore

STORE = ROOT / "runs" / "misaki-iter"
CLOUD = STORE / "cloud-gold"
LOCKED = ROOT / "data" / "gold_gemini_ipa" / "wavs"
OUT_ROOT = ROOT / "runs" / "main-bench-cloud"
VOICE = "af_heart"
PUBLIC_CLONE = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
PUBLIC_CLONE_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it! And thanks to you."
)
QWEN_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def stems(item: dict) -> list[str]:
    names = [
        item["slug"],
        item["spoken"].lower().replace(" ", "_"),
        item["drug"].lower().replace(" ", "_"),
        item["spoken"].lower().replace(" ", "-"),
        item["drug"].lower().replace(" ", "-"),
        item["slug"].replace("-", "_"),
    ]
    seen: set[str] = set()
    out = []
    for name in names:
        stem = re.sub(r"[^a-z0-9_-]+", "", name).strip("_-")
        if stem and stem not in seen:
            seen.add(stem)
            out.append(stem)
    return out


def load_products() -> list[dict]:
    """274 original DOSE rows. Combo products stay one row."""
    products = []
    for line in (ROOT / "data" / "dose_v1.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ingredients = row.get("ingredients") or [row["name"]]
        products.append(
            {
                "id": row["id"],
                "name": row["name"],
                "slug": slug(row["name"]),
                "ingredients": ingredients,
                "ing_slugs": [slug(ing) for ing in ingredients],
                "is_combination": bool(row.get("is_combination") or len(ingredients) > 1),
                "sentence": row["sentence"],
            }
        )
    return products


def load_items() -> list[dict]:
    """Ingredient spans for synth/score. Gold wavs are per-ingredient.

    Reporting must go through collapse_to_products so a 2–3 ingredient
    row is one vote, not two or three.
    """
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
            items.append(
                {
                    "product_id": row["id"],
                    "product": row["name"],
                    "drug": ing,
                    "spoken": spoken,
                    "sentence": sentence,
                    "slug": slug(ing),
                }
            )
    seen: dict[str, dict] = {}
    for it in items:
        seen.setdefault(it["slug"], it)
    return list(seen.values())


def collapse_to_products(slug_f1: dict[str, float]) -> list[dict]:
    """One F1 per original DOSE row: mean of that row's ingredient scores."""
    rows = []
    for product in load_products():
        vals = [slug_f1[s] for s in product["ing_slugs"] if s in slug_f1]
        if not vals:
            continue
        rows.append(
            {
                "id": product["id"],
                "name": product["name"],
                "slug": product["slug"],
                "n_ingredients": len(product["ing_slugs"]),
                "n_scored": len(vals),
                "complete": len(vals) == len(product["ing_slugs"]),
                "cloud_f1": round(sum(vals) / len(vals), 4),
            }
        )
    return rows


def product_summary(rows: list[dict]) -> dict:
    xs = [r["cloud_f1"] for r in rows]
    xs_sorted = sorted(xs)
    mean = sum(xs) / len(xs) if xs else 0.0
    lo, hi = bootstrap_ci(xs)
    return {
        "unit": "dose-product",
        "n_products": 274,
        "n": len(rows),
        "n_missing": 274 - len(rows),
        "n_complete": sum(1 for r in rows if r["complete"]),
        "mean_f1": round(mean, 4),
        "median_f1": round(xs_sorted[len(xs_sorted) // 2], 4) if xs_sorted else None,
        "p10": round(xs_sorted[max(0, int(0.10 * (len(xs_sorted) - 1)))], 4) if xs_sorted else None,
        "ci95": [lo, hi],
        "pass_ge_0_70": sum(1 for x in xs if x >= 0.70),
        "pass_rate_ge_0_70": round(sum(1 for x in xs if x >= 0.70) / len(xs), 4) if xs else None,
    }


def resolve_path(folders: list[Path], item: dict) -> Path | None:
    """Newest existing wav among name variants. Ignores AppleDouble files."""
    best: Path | None = None
    best_mtime = -1.0
    for folder in folders:
        if not folder.exists():
            continue
        for stem in stems(item):
            path = folder / f"{stem}.wav"
            if path.name.startswith("._"):
                continue
            if path.exists() and path.stat().st_size > 500:
                mtime = path.stat().st_mtime
                if mtime > best_mtime:
                    best = path
                    best_mtime = mtime
    return best


def gold_wav(item: dict, *, prefer: str = "cloud") -> Path | None:
    """Resolve the Cloud Standard-C teacher wav.

    prefer=cloud  — main-benchmark order (cloud-gold, then gold_gemini).
    prefer=locked — most current curated take (gold_gemini, then cloud-gold).
    """
    folders = [LOCKED, CLOUD] if prefer == "locked" else [CLOUD, LOCKED]
    for folder in folders:
        found = resolve_path([folder], item)
        if found is not None:
            return found
    return None


def cand_wav(folder: Path, item: dict) -> Path | None:
    return resolve_path([folder], item)


def load_phones() -> dict[str, str]:
    phones = {}
    chosen = STORE / "chosen.json"
    if chosen.exists():
        phones.update(json.loads(chosen.read_text()).get("phones") or {})
    pins = STORE / "user_pins.json"
    if pins.exists():
        for key, pin in json.loads(pins.read_text()).items():
            phones[key] = pin["misaki"]
    return phones


def lexicon_assignments(spoken: str, phones: str) -> dict[str, str]:
    words = spoken.split()
    parts = phones.split()
    if len(words) > 1 and len(parts) == len(words):
        return dict(zip(words, parts))
    return {spoken: phones}


def extract_span(raw: bytes, sentence: str, spoken: str, drug: str) -> bytes | None:
    span = extract_drug_span_forced_align(raw, sentence, spoken)
    if span is None and drug != spoken:
        span = extract_drug_span_forced_align(raw, sentence, drug)
    return span


def score_span(span: bytes, gold: Path, cache: dict) -> float:
    key = str(gold.resolve())
    emb_g = cache.get(key)
    if emb_g is None:
        emb_g = extract_frame_embeddings(gold.read_bytes())
        cache[key] = emb_g
    return float(speech_bertscore(extract_frame_embeddings(span), emb_g)["f1"])


def bootstrap_ci(xs: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not xs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(xs)
    means = []
    for _ in range(n_boot):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return (round(lo, 4), round(hi, 4))


def score_dir(wav_dir: Path, condition: str, span_dir: Path | None = None) -> dict:
    items = load_items()
    out_dir = OUT_ROOT / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    if span_dir is None:
        span_dir = out_dir / "span"
    span_dir.mkdir(parents=True, exist_ok=True)

    cache: dict = {}
    rows = []
    missed = []
    for n, it in enumerate(items, 1):
        gold = gold_wav(it, prefer="cloud")
        cand = cand_wav(wav_dir, it)
        if gold is None:
            missed.append((it["slug"], "no-cloud-gold"))
            print(f"{n}/{len(items)} {it['slug']} missing gold", flush=True)
            continue
        if cand is None:
            missed.append((it["slug"], "no-candidate"))
            print(f"{n}/{len(items)} {it['slug']} missing cand", flush=True)
            continue
        raw = cand.read_bytes()
        span = extract_span(raw, it["sentence"], it["spoken"], it["drug"])
        if span is None:
            missed.append((it["slug"], "no-span"))
            print(f"{n}/{len(items)} {it['slug']} nospan", flush=True)
            continue
        (span_dir / f"{it['slug']}.wav").write_bytes(span)
        f1 = score_span(span, gold, cache)
        row = {
            "slug": it["slug"],
            "drug": it["drug"],
            "spoken": it["spoken"],
            "condition": condition,
            "cloud_f1": round(f1, 4),
            "gold": _rel(gold),
            "cand": _rel(cand),
        }
        rows.append(row)
        print(f"{n}/{len(items)} {it['slug']} {f1:.4f} gold={gold.name}", flush=True)

    rows.sort(key=lambda r: r["cloud_f1"])
    xs = [r["cloud_f1"] for r in rows]
    mean = sum(xs) / len(xs) if xs else 0.0
    lo, hi = bootstrap_ci(xs)
    report = {
        "condition": condition,
        "metric": "wavlm-large SpeechBERTScore F1 vs current Cloud Standard-C gold",
        "gold_folders": [str(LOCKED), str(CLOUD)],
        "wav_dir": str(wav_dir),
        "n_items": len(items),
        "n": len(rows),
        "n_missing": len(missed),
        "mean_f1": round(mean, 4),
        "median_f1": round(xs[len(xs) // 2], 4) if xs else None,
        "p10": round(xs[max(0, int(0.10 * (len(xs) - 1)))], 4) if xs else None,
        "ci95": [lo, hi],
        "pass_ge_0_70": sum(1 for x in xs if x >= 0.70),
        "pass_rate_ge_0_70": round(sum(1 for x in xs if x >= 0.70) / len(xs), 4) if xs else None,
        "missing": missed,
        "rows": rows,
    }
    (out_dir / "scores.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "summary.json").write_text(
        json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2) + "\n"
    )
    prod_rows = collapse_to_products({r["slug"]: r["cloud_f1"] for r in rows})
    prod = {"condition": condition, **product_summary(prod_rows), "rows": prod_rows}
    (out_dir / "products.json").write_text(json.dumps(prod, indent=2) + "\n")
    print(
        f"WROTE {out_dir} slugs={len(rows)} mean={mean:.4f} | "
        f"products n={prod['n']} mean={prod['mean_f1']} "
        f"ci95={prod['ci95']} pass>=0.70={prod['pass_ge_0_70']}",
        flush=True,
    )
    return report


def synth_kokoro(mode: str, out_dir: Path, only: set[str] | None = None) -> None:
    import soundfile as sf
    import torch
    from kokoro import KPipeline

    _MISSING = object()
    items = load_items()
    if only:
        items = [it for it in items if it["slug"] in only]
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
    phones = load_phones() if mode == "lexicon" else {}
    golds = pipeline.g2p.lexicon.golds
    print(f"kokoro device={device} mode={mode} n={len(items)}", flush=True)
    for n, it in enumerate(items, 1):
        dest = out_dir / f"{it['slug']}.wav"
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"{n}/{len(items)} skip {it['slug']}", flush=True)
            continue
        saved = {}
        try:
            if mode == "lexicon":
                ph = phones.get(it["slug"])
                if not ph:
                    print(f"{n}/{len(items)} no-phones {it['slug']}", flush=True)
                    continue
                for key, value in lexicon_assignments(it["spoken"], ph).items():
                    saved[key] = golds.get(key, _MISSING)
                    golds[key] = value
            result = next(pipeline(it["sentence"], voice=VOICE, speed=1.0))
            if result.audio is None:
                raise RuntimeError("no audio")
            sf.write(dest, result.audio.detach().cpu().numpy(), 24000)
            print(f"{n}/{len(items)} ok {it['slug']}", flush=True)
        except Exception as exc:
            print(f"{n}/{len(items)} FAIL {it['slug']} {exc}", flush=True)
        finally:
            for key, old in saved.items():
                if old is _MISSING:
                    golds.pop(key, None)
                else:
                    golds[key] = old


def _load_qwen(model_path: str):
    import torch
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def synth_qwen(
    mode: str,
    out_dir: Path,
    model_path: str,
    speaker: str | None,
    *,
    batch_size: int = 8,
    only: set[str] | None = None,
) -> None:
    import soundfile as sf

    items = load_items()
    if only:
        items = [it for it in items if it["slug"] in only]
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for it in items:
        dest = out_dir / f"{it['slug']}.wav"
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        pending.append(it)
    model = _load_qwen(model_path)
    public_prompt = None
    if mode == "plain":
        public_prompt = model.create_voice_clone_prompt(
            ref_audio=PUBLIC_CLONE, ref_text=PUBLIC_CLONE_TEXT
        )
    print(
        f"qwen mode={mode} model={model_path} pending={len(pending)} "
        f"batch={batch_size}",
        flush=True,
    )
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        texts = [it["sentence"] for it in chunk]
        langs = ["English"] * len(chunk)
        try:
            if mode == "ft":
                wavs, sr = model.generate_custom_voice(
                    text=texts,
                    language=langs,
                    speaker=[speaker or "dose_cloud_ipa"] * len(chunk),
                )
            elif mode == "wav":
                refs = []
                keep = []
                for it in chunk:
                    gold = gold_wav(it, prefer="locked")
                    if gold is None:
                        print(f"no-gold {it['slug']}", flush=True)
                        continue
                    refs.append(str(gold))
                    keep.append(it)
                if not keep:
                    continue
                chunk = keep
                wavs, sr = model.generate_voice_clone(
                    text=[it["sentence"] for it in chunk],
                    language=["English"] * len(chunk),
                    ref_audio=refs,
                    ref_text=[it["spoken"] for it in chunk],
                )
            else:
                wavs, sr = model.generate_voice_clone(
                    text=texts,
                    language=langs,
                    voice_clone_prompt=[public_prompt] * len(chunk),
                )
            if len(wavs) != len(chunk):
                raise RuntimeError(f"batch size mismatch got {len(wavs)} want {len(chunk)}")
            for it, audio in zip(chunk, wavs):
                dest = out_dir / f"{it['slug']}.wav"
                sf.write(dest, audio, sr)
                print(f"ok {it['slug']}", flush=True)
            print(f"batch {start // batch_size + 1} n={len(chunk)}", flush=True)
        except Exception as exc:
            print(f"BATCH FAIL { [it['slug'] for it in chunk] } {exc}", flush=True)
            for it in chunk:
                dest = out_dir / f"{it['slug']}.wav"
                if dest.exists() and dest.stat().st_size > 1000:
                    continue
                try:
                    if mode == "wav":
                        gold = gold_wav(it, prefer="locked")
                        if gold is None:
                            continue
                        prompt = model.create_voice_clone_prompt(
                            ref_audio=str(gold), ref_text=it["spoken"]
                        )
                        wavs, sr = model.generate_voice_clone(
                            text=it["sentence"],
                            language="English",
                            voice_clone_prompt=prompt,
                        )
                    elif mode == "ft":
                        wavs, sr = model.generate_custom_voice(
                            text=it["sentence"],
                            language="English",
                            speaker=speaker or "dose_cloud_ipa",
                        )
                    else:
                        wavs, sr = model.generate_voice_clone(
                            text=it["sentence"],
                            language="English",
                            voice_clone_prompt=public_prompt,
                        )
                    sf.write(dest, wavs[0], sr)
                    print(f"ok-single {it['slug']}", flush=True)
                except Exception as exc2:
                    print(f"FAIL {it['slug']} {exc2}", flush=True)


def summarize_all(bench: Path) -> dict:
    rows = []
    for p in sorted(bench.glob("*/summary.json")):
        d = json.loads(p.read_text())
        rows.append(d)
    rows.sort(key=lambda r: r.get("mean_f1") or 0, reverse=True)
    out = {"n_systems": len(rows), "systems": rows}
    (bench / "leaderboard.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({r["condition"]: r.get("mean_f1") for r in rows}, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_score = sub.add_parser("score")
    p_score.add_argument("--wav-dir", required=True)
    p_score.add_argument("--condition", required=True)
    p_score.add_argument("--span-dir")

    p_k = sub.add_parser("synth-kokoro")
    p_k.add_argument("--mode", choices=("plain", "lexicon"), required=True)
    p_k.add_argument("--out", required=True)
    p_k.add_argument("--only", nargs="*")

    p_q = sub.add_parser("synth-qwen")
    p_q.add_argument("--mode", choices=("plain", "wav", "ft"), required=True)
    p_q.add_argument("--out", required=True)
    p_q.add_argument("--model", default=QWEN_BASE)
    p_q.add_argument("--speaker")
    p_q.add_argument("--batch-size", type=int, default=8)
    p_q.add_argument("--only", nargs="*")

    sub.add_parser("summarize")
    p_col = sub.add_parser("collapse")
    p_col.add_argument("--scores", required=True, help="ingredient scores.json")
    p_gold = sub.add_parser("list-gold")
    p_gold.add_argument("--limit", type=int, default=0)

    args = ap.parse_args()
    if args.cmd == "score":
        score_dir(Path(args.wav_dir), args.condition, Path(args.span_dir) if args.span_dir else None)
    elif args.cmd == "synth-kokoro":
        synth_kokoro(args.mode, Path(args.out), set(args.only) if args.only else None)
    elif args.cmd == "synth-qwen":
        synth_qwen(
            args.mode,
            Path(args.out),
            args.model,
            args.speaker,
            batch_size=args.batch_size,
            only=set(args.only) if args.only else None,
        )
    elif args.cmd == "summarize":
        summarize_all(OUT_ROOT)
    elif args.cmd == "collapse":
        d = json.loads(Path(args.scores).read_text())
        slug_f1 = {r["slug"]: r["cloud_f1"] for r in d.get("rows") or []}
        prod_rows = collapse_to_products(slug_f1)
        prod = {"condition": d.get("condition"), **product_summary(prod_rows)}
        print(json.dumps(prod, indent=2))
    elif args.cmd == "list-gold":
        items = load_items()
        missing = []
        for it in items:
            g = gold_wav(it)
            if g is None:
                missing.append(it["slug"])
            elif args.limit:
                print(it["slug"], g)
        print(f"items={len(items)} gold={len(items)-len(missing)} missing={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
