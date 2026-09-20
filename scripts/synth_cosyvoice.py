"""Fun-CosyVoice3-0.5B-2512 on the hard subset.

plain: inference_zero_shot on the carrier sentence.
phoneme: same call, drug name replaced with CMU tokens [AH0] [B] ...,
text_frontend off so the brackets survive.

Prompt audio is CosyVoice's own English-capable zero-shot asset. The
prompt transcript is the one the CosyVoice 3 examples pair with that file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oss_eval_items import inject_arpabet, load_items

# Official CosyVoice 3 English example: Chinese prompt transcript, English text.
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"


def _first_speech(generator):
    chunks = []
    for piece in generator:
        chunks.append(piece["tts_speech"])
    if not chunks:
        raise RuntimeError("model returned no audio")
    return torch.cat(chunks, dim=1) if chunks[0].ndim == 2 else torch.cat(chunks, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("plain", "phoneme"), required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--cosyvoice-root", required=True)
    ap.add_argument("--prompt-wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--llm-pt",
        default="",
        help="Load this LLM file instead of model-dir/llm.pt (CosyVoice RL).",
    )
    args = ap.parse_args()

    sys.path.insert(0, args.cosyvoice_root)
    sys.path.insert(0, str(Path(args.cosyvoice_root) / "third_party" / "Matcha-TTS"))
    # diffusers 0.29 still imports cached_download, removed from current huggingface_hub.
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download"):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    from cosyvoice.cli.cosyvoice import AutoModel

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    if args.llm_pt:
        # AutoModel always reads llm.pt. Stage a sibling dir that points at
        # the RL file without touching the base checkpoint.
        staged = model_dir.parent / (model_dir.name + "-rl-load")
        if staged.exists():
            import shutil
            shutil.rmtree(staged)
        staged.mkdir()
        for child in model_dir.iterdir():
            if child.name == "llm.pt":
                continue
            (staged / child.name).symlink_to(child)
        (staged / "llm.pt").symlink_to(Path(args.llm_pt).resolve())
        model_dir = staged
        print("LLM", args.llm_pt, "via", model_dir / "llm.pt", flush=True)
    model = AutoModel(model_dir=str(model_dir))
    if args.mode == "phoneme":
        probe = "[AE1] [D] [K] [IY0]"
        ids = model.frontend.tokenizer.encode(probe)
        pieces = model.frontend.tokenizer.tokenizer.convert_ids_to_tokens(ids)
        print("TOKEN_PROBE", pieces, flush=True)
        missing = [t for t in ("[AE1]", "[D]", "[K]", "[IY0]") if t not in pieces]
        if missing:
            raise SystemExit(f"phoneme tokens were split, missing {missing}: {pieces}")
    failures = []
    for item in load_items():
        dest = out / f"{item['item_id']}.wav"
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        try:
            if args.mode == "plain":
                text = item["sentence"]
                frontend = True
            else:
                arpabet = (item["arpabet_variants"] or [None])[0]
                if not arpabet:
                    raise ValueError("no gold ARPABET")
                text = inject_arpabet(item["sentence"], item["spoken"], arpabet)
                frontend = False
            try:
                gen = model.inference_zero_shot(
                    text, PROMPT_TEXT, args.prompt_wav, stream=False, text_frontend=frontend,
                )
            except TypeError:
                gen = model.inference_zero_shot(
                    text, PROMPT_TEXT, args.prompt_wav, stream=False,
                )
            speech = _first_speech(gen)
            wav = speech.detach().cpu().float().numpy()
            if wav.ndim == 2:
                wav = wav.T
            sf.write(str(dest), wav, model.sample_rate)
            print("ok", item["item_id"], flush=True)
        except Exception as exc:
            failures.append({"item_id": item["item_id"], "error": str(exc)})
            print("FAIL", item["item_id"], exc, flush=True)
    (out / "failures.json").write_text(json.dumps(failures, indent=2))
    print(f"done failures={len(failures)}")


if __name__ == "__main__":
    main()
