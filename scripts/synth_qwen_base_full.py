#!/usr/bin/env python3
"""Untouched Qwen3-TTS 1.7B Base on every DoSE carrier sentence.

Voice is the published English clone prompt, not a fine-tune and not an
IPA injection. Greedy decode so the benchmark is one draw, not a sample.
Wavs are named by product slug, same as kokoro-full-plain, so
score_dose_ctc_vs_gemini_ipa.py can find them.

Does not read or write data/gold_gemini_ipa.
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
PUBLIC_REF = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
PUBLIC_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it! And thanks to you."
)
MAX_NEW_TOKENS = 384
MAX_WAV_BYTES = 2_000_000


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_sentences() -> list[dict]:
    items = []
    seen = set()
    for line in DOSE.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = slug(row["name"])
        if key in seen:
            raise SystemExit(f"duplicate product slug {key}")
        seen.add(key)
        items.append({"slug": key, "product": row["name"], "sentence": row["sentence"], "dose_id": row["id"]})
    return items


def _generate(model, prompt, sentences: list[str]):
    n = len(sentences)
    return model.generate_voice_clone(
        text=sentences,
        language=["English"] * n,
        voice_clone_prompt=prompt,
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "oss-eval" / "qwen3-base-full")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    items = load_sentences()
    todo = [it for it in items if not (out / f"{it['slug']}.wav").exists()]
    print(f"sentences={len(items)} todo={len(todo)} start_batch={args.batch_size}", flush=True)
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

    bs = max(1, args.batch_size)
    failures = []
    i = 0
    while i < len(todo):
        batch = todo[i : i + bs]
        texts = [it["sentence"] for it in batch]
        try:
            wavs, sr = _generate(model, prompt, texts)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            if bs == 1:
                print(f"OOM {batch[0]['slug']} {exc}", flush=True)
                failures.append({"slug": batch[0]["slug"], "error": "oom"})
                i += 1
                continue
            bs = max(1, bs // 2)
            print(f"OOM — batch now {bs}", flush=True)
            continue
        except Exception as exc:
            print(f"BATCH_FAIL n={len(batch)} {exc} — falling back to 1", flush=True)
            if len(batch) == 1:
                failures.append({"slug": batch[0]["slug"], "error": str(exc)})
                i += 1
                continue
            bs = 1
            continue

        for it, wav in zip(batch, wavs):
            dest = out / f"{it['slug']}.wav"
            sf.write(dest, wav, sr)
            if dest.stat().st_size > MAX_WAV_BYTES:
                dest.unlink()
                failures.append({"slug": it["slug"], "error": "runaway"})
                print("RUNAWAY", it["slug"], flush=True)
            else:
                print("ok", it["slug"], flush=True)
        i += len(batch)
        free, total = torch.cuda.mem_get_info()
        used_frac = 1.0 - (free / total)
        print(f"gpu_used={used_frac:.2f} batch={bs} done={i}/{len(todo)}", flush=True)
        # Fill the card. Back off is handled by OOM above.
        if used_frac < 0.72 and bs < 32:
            bs = min(32, bs + 4)
        torch.cuda.empty_cache()

    (out / "failures.json").write_text(json.dumps(failures, indent=2))
    print(f"done failures={len(failures)} wavs={len(list(out.glob('*.wav')))}", flush=True)


if __name__ == "__main__":
    main()
