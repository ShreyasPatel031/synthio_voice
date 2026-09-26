#!/usr/bin/env python3
"""Hard-subset Base vs word-audio inject, batched.

Gold audio is read from data/gold_gemini_ipa only. This script does not
write that folder. Score later with score_dose_ctc_vs_gemini_ipa.py --scope hard.

One process = one arm. Run four arms on four GPUs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
GOLD = ROOT / "data" / "gold_gemini_ipa"
GOLD_MANIFEST = GOLD / "manifest.jsonl"
DOSE = ROOT / "data" / "dose_v1.jsonl"
HARD = ROOT / "runs" / "hard-subset-v1.json"
OUT_ROOT = ROOT / "runs" / "oss-eval"
PUBLIC_REF = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
PUBLIC_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it! And thanks to you."
)
MAX_NEW_TOKENS = 220
MAX_WAV_BYTES = 2_000_000


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_gold() -> dict[str, dict]:
    out = {}
    for line in GOLD_MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        wav = ROOT / rec["audio"]
        if wav.exists() and wav.stat().st_size > 500:
            out[rec["ingredient"].lower()] = {
                "wav": str(wav.resolve()),
                "ipa": rec.get("ipa") or "",
                "spoken": rec.get("spoken_text") or rec["ingredient"],
            }
    return out


def load_items(scope: str) -> list[dict]:
    hard_drugs = None
    if scope == "hard":
        hard = json.loads(HARD.read_text())["items"]
        hard_drugs = {v["drug"].lower() for v in hard.values()}
    items = []
    for line in DOSE.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ingredients = row.get("ingredients") or [row["name"]]
        spans = row.get("spans") or []
        sentence = row["sentence"]
        for i, ing in enumerate(ingredients):
            if hard_drugs is not None and ing.lower() not in hard_drugs:
                continue
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
                    "drug": ing,
                    "spoken": spoken,
                    "sentence": sentence,
                    "dose_id": row["id"],
                    "slug": slug(ing),
                }
            )
    # one wav per drug slug
    seen = {}
    for it in items:
        seen.setdefault(it["slug"], it)
    return list(seen.values())


def _gen_batch(model, items, *, arm: str, gold: dict, public_prompt):
    texts = [it["sentence"] for it in items]
    n = len(texts)
    if arm == "base":
        return model.generate_voice_clone(
            text=texts,
            language=["English"] * n,
            voice_clone_prompt=public_prompt,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    greedy = arm.endswith("greedy")
    xvec = "xvec" in arm
    refs = []
    ref_texts = []
    for it in items:
        g = gold[it["drug"].lower()]
        refs.append(g["wav"])
        # Transcript of the gold clip is the published IPA, not a new G2P.
        ref_texts.append(g["ipa"] or g["spoken"])
    kw = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=0.2, top_p=0.8)
    return model.generate_voice_clone(
        text=texts,
        language=["English"] * n,
        ref_audio=refs,
        ref_text=ref_texts,
        x_vector_only_mode=xvec,
        max_new_tokens=MAX_NEW_TOKENS,
        **kw,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=(
        "base",
        "inject_word_greedy",
        "inject_word_xvec_greedy",
        "inject_word_sample",
    ))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--scope", choices=("hard", "full"), default="hard")
    args = ap.parse_args()

    gold = load_gold()
    loaded = load_items(args.scope)
    items = [it for it in loaded if it["drug"].lower() in gold]
    missing = [it["drug"] for it in loaded if it["drug"].lower() not in gold]
    out = OUT_ROOT / f"pron-gemini-{args.arm}"
    out.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if not (out / f"{it['slug']}.wav").exists()]
    print(
        f"arm={args.arm} scope={args.scope} items={len(items)} todo={len(todo)} "
        f"batch={args.batch_size} missing_gold={missing}",
        flush=True,
    )
    if not todo:
        print("nothing to synth", flush=True)
        return

    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    public_prompt = None
    if args.arm == "base":
        public_prompt = model.create_voice_clone_prompt(
            ref_audio=PUBLIC_REF, ref_text=PUBLIC_TEXT
        )

    bs = args.batch_size
    for start in range(0, len(todo), bs):
        batch = todo[start : start + bs]
        try:
            wavs, sr = _gen_batch(
                model, batch, arm=args.arm, gold=gold, public_prompt=public_prompt
            )
        except Exception as exc:
            print(f"BATCH_FAIL {start} {exc} — falling back to 1", flush=True)
            wavs = []
            sr = 24000
            for it in batch:
                w, sr = _gen_batch(
                    model, [it], arm=args.arm, gold=gold, public_prompt=public_prompt
                )
                wavs.append(w[0])
        for it, wav in zip(batch, wavs):
            dest = out / f"{it['slug']}.wav"
            sf.write(dest, wav, sr)
            if dest.stat().st_size > MAX_WAV_BYTES:
                dest.unlink()
                print("RUNAWAY", it["slug"], flush=True)
            else:
                print("ok", it["slug"], flush=True)
    print(f"done {args.arm}", flush=True)


if __name__ == "__main__":
    main()
