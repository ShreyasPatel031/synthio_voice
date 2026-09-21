#!/usr/bin/env python3
"""Hard-subset Base Qwen, dictionary respelling spelled out in the sentence.

No gold audio. The drug span is replaced with the original dictionary
respelling (VIV-gart, ak-oh-RAM-id-is). Hyphens and stress caps stay.
This does not write data/gold_gemini_ipa or convert respelling to IPA.

Toujeo and Avlayah have no dictionary source. The two strings below were
supplied for this test only and are not stored as sourced respellings.

Score with score_dose_ctc_vs_gemini_ipa.py --scope hard. The scorer still
cuts the original drug spelling out of the wav.
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
DOSE = ROOT / "data" / "dose_v1.jsonl"
HARD = ROOT / "runs" / "hard-subset-v1.json"
RESPELL = ROOT / "dose_r" / "references" / "respellings.jsonl"
OUT = ROOT / "runs" / "oss-eval" / "pron-gemini-respell-greedy"
PUBLIC_REF = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
PUBLIC_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it! And thanks to you."
)
# User-supplied for this test. Not a published dictionary source.
USER_RESPELL = {
    "toujeo": "tow-ZHAY-oh",
    "avlayah": "av-LAY-uh",
}
MAX_NEW_TOKENS = 220
MAX_WAV_BYTES = 2_000_000


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_respellings() -> dict[str, str]:
    out = dict(USER_RESPELL)
    for line in RESPELL.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        text = (rec.get("respelling") or "").strip()
        if text:
            out[rec["ingredient"].lower()] = text
    return out


def load_items(respellings: dict[str, str]) -> tuple[list[dict], list[str]]:
    hard = json.loads(HARD.read_text())["items"]
    hard_drugs = {v["drug"].lower() for v in hard.values()}
    items = []
    missing = []
    for line in DOSE.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ingredients = row.get("ingredients") or [row["name"]]
        spans = row.get("spans") or []
        sentence = row["sentence"]
        for i, ing in enumerate(ingredients):
            if ing.lower() not in hard_drugs:
                continue
            if i < len(spans) and isinstance(spans[i], (list, tuple)) and len(spans[i]) == 2:
                a, b = spans[i]
                spoken = sentence[a:b]
                start = a
            else:
                start = sentence.lower().find(ing.lower())
                if start < 0:
                    continue
                spoken = sentence[start : start + len(ing)]
            spell = respellings.get(ing.lower())
            if not spell:
                missing.append(ing)
                continue
            spelled = sentence[:start] + spell + sentence[start + len(spoken) :]
            items.append(
                {
                    "drug": ing,
                    "spoken": spoken,
                    "sentence": sentence,
                    "spelled": spelled,
                    "respelling": spell,
                    "slug": slug(ing),
                }
            )
    seen = {}
    for it in items:
        seen.setdefault(it["slug"], it)
    return list(seen.values()), sorted(set(missing))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    items, missing = load_items(load_respellings())
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if not (OUT / f"{it['slug']}.wav").exists()]
    print(
        f"respell-greedy items={len(items)} todo={len(todo)} "
        f"batch={args.batch_size} missing={missing}",
        flush=True,
    )
    if missing:
        raise SystemExit("missing respelling")
    if not todo:
        print("nothing to synth", flush=True)
        return

    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    prompt = model.create_voice_clone_prompt(ref_audio=PUBLIC_REF, ref_text=PUBLIC_TEXT)

    bs = args.batch_size
    for start in range(0, len(todo), bs):
        batch = todo[start : start + bs]
        texts = [it["spelled"] for it in batch]
        try:
            wavs, sr = model.generate_voice_clone(
                text=texts,
                language=["English"] * len(texts),
                voice_clone_prompt=prompt,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        except Exception as exc:
            print(f"BATCH_FAIL {start} {exc} — falling back to 1", flush=True)
            wavs = []
            sr = 24000
            for it in batch:
                w, sr = model.generate_voice_clone(
                    text=[it["spelled"]],
                    language=["English"],
                    voice_clone_prompt=prompt,
                    do_sample=False,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                wavs.append(w[0])
        for it, wav in zip(batch, wavs):
            dest = OUT / f"{it['slug']}.wav"
            sf.write(dest, wav, sr)
            if dest.stat().st_size > MAX_WAV_BYTES:
                dest.unlink()
                print("RUNAWAY", it["slug"], flush=True)
            else:
                print("ok", it["slug"], it["respelling"], flush=True)
    print("done respell-greedy", flush=True)


if __name__ == "__main__":
    main()
