"""Kokoro-82M on the hard subset.

plain: misaki G2P, voice af_heart.
phoneme: same pipeline, but the drug-name token's phonemes are replaced
with the gold IPA (r rewritten to ɹ). Symbols outside Kokoro's vocab abort
that item instead of being dropped, so a "win" is not a silently edited string.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from kokoro import KPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oss_eval_items import load_items

VOICE = "af_heart"


def _vocab() -> set[str]:
    path = hf_hub_download("hexgrad/Kokoro-82M", "config.json")
    vocab = json.loads(Path(path).read_text())["vocab"]
    return set(vocab)


def _ipa_for_kokoro(ipa: str, vocab: set[str]) -> str:
    text = ipa.replace("r", "ɹ").replace("g", "ɡ").replace(" ", "")
    missing = sorted({ch for ch in text if ch not in vocab})
    if missing:
        raise ValueError(f"IPA symbols not in Kokoro vocab: {missing} in {ipa!r}")
    return text


def _replace_drug_phonemes(tokens, drug: str, phones: str) -> None:
    words = drug.split()
    texts = [t.text for t in tokens]
    for i in range(len(tokens) - len(words) + 1):
        if [t.lower() for t in texts[i:i + len(words)]] == [w.lower() for w in words]:
            tokens[i].phonemes = phones
            for extra in tokens[i + 1:i + len(words)]:
                extra.phonemes = ""
            return
    raise ValueError(f"could not locate {drug!r} in tokens {texts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("plain", "phoneme"), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
    vocab = _vocab() if args.mode == "phoneme" else set()
    failures = []
    for item in load_items():
        dest = out / f"{item['item_id']}.wav"
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        try:
            if args.mode == "plain":
                result = next(pipeline(item["sentence"], voice=VOICE, speed=1))
                audio = result.audio
            else:
                ipa = (item["ipa_variants"] or [None])[0]
                if not ipa:
                    raise ValueError("no gold IPA")
                phones = _ipa_for_kokoro(ipa, vocab)
                _, tokens = pipeline.g2p(item["sentence"])
                _replace_drug_phonemes(tokens, item["spoken"], phones)
                result = next(pipeline.generate_from_tokens(tokens, voice=VOICE, speed=1))
                audio = result.audio
            if audio is None:
                raise RuntimeError("no audio")
            sf.write(dest, audio.detach().cpu().numpy(), 24000)
            print("ok", item["item_id"], flush=True)
        except Exception as exc:
            failures.append({"item_id": item["item_id"], "error": str(exc)})
            print("FAIL", item["item_id"], exc, flush=True)
    (out / "failures.json").write_text(json.dumps(failures, indent=2))
    print(f"done failures={len(failures)}")


if __name__ == "__main__":
    main()
