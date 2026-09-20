"""Qwen3-TTS-12Hz-1.7B-Base, plain spelling, on the hard subset.

Base has no built-in speaker. One voice-clone prompt is built once from
Qwen's published English reference clip and reused for every item, so
the speaker is constant and this run is the baseline a later fine-tune
of this checkpoint has to beat.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oss_eval_items import load_items

REF_AUDIO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it! And thanks to you."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    prompt = model.create_voice_clone_prompt(ref_audio=REF_AUDIO, ref_text=REF_TEXT)
    failures = []
    for item in load_items():
        dest = out / f"{item['item_id']}.wav"
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        try:
            wavs, sr = model.generate_voice_clone(
                text=item["sentence"],
                language="English",
                voice_clone_prompt=prompt,
            )
            sf.write(dest, wavs[0], sr)
            print("ok", item["item_id"], flush=True)
        except Exception as exc:
            failures.append({"item_id": item["item_id"], "error": str(exc)})
            print("FAIL", item["item_id"], exc, flush=True)
    (out / "failures.json").write_text(__import__("json").dumps(failures, indent=2))
    print(f"done failures={len(failures)}")


if __name__ == "__main__":
    main()
