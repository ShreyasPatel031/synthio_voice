"""VoxCPM2 plain spelling on the hard subset.

No IPA injection. One fixed reference clip sets the speaker for every
item, matching the Qwen and CosyVoice plain runs. The clip is not a
drug-name recording, so Path 2 is not scored against the prompt itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oss_eval_items import load_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference-wav", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from voxcpm import VoxCPM

    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    sr = model.tts_model.sample_rate
    ref = args.reference_wav or None
    failures = []
    for item in load_items():
        dest = out / f"{item['item_id']}.wav"
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        try:
            wav = model.generate(
                text=item["sentence"],
                reference_wav_path=ref,
                cfg_value=2.0,
                inference_timesteps=10,
            )
            sf.write(dest, wav, sr)
            print("ok", item["item_id"], flush=True)
        except Exception as exc:
            failures.append({"item_id": item["item_id"], "error": str(exc)})
            print("FAIL", item["item_id"], exc, flush=True)
    (out / "failures.json").write_text(json.dumps(failures, indent=2))
    print(f"done failures={len(failures)}")


if __name__ == "__main__":
    main()
