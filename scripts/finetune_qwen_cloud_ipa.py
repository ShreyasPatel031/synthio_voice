#!/usr/bin/env python3
"""Fine-tune Qwen3-TTS on cloud IPA teacher clips; compare plain vs FT F1.

Train: spoken_text -> en-US-Standard-C teacher wav (data/finetune_cloud_ipa).
Eval (167 scored names): synthesize spoken_text only, score SpeechBERTScore
F1 against (1) human reference clips and (2) the teacher wav itself.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "data" / "finetune_cloud_ipa"
REF_AUDIO = str(ROOT / "data" / "finetune_cloud_ipa" / "qwen_ref_clone.wav")
REF_AUDIO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it! And thanks to you."
)


def _abs(p: str | Path) -> str:
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve())


def build_qwen_jsonl(out: Path) -> int:
    rows = [json.loads(l) for l in (PACK / "manifest.jsonl").open() if l.strip()]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        "audio": _abs(r["audio"]),
                        "text": r["spoken_text"],
                        "ref_audio": REF_AUDIO if Path(REF_AUDIO).exists() else REF_AUDIO_URL,
                    }
                )
                + "\n"
            )
    return len(rows)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def find_finetune_scripts() -> Path:
    # qwen_tts package or cloned Qwen3-TTS repo
    candidates = [
        Path.home() / "Qwen3-TTS" / "finetuning",
        ROOT / "third_party" / "Qwen3-TTS" / "finetuning",
        Path("/home/shreyaspatel/Qwen3-TTS/finetuning"),
    ]
    try:
        import qwen_tts

        pkg = Path(qwen_tts.__file__).resolve().parent
        for rel in ("finetuning", "../finetuning"):
            p = (pkg / rel).resolve()
            if (p / "sft_12hz.py").exists():
                return p
    except Exception:
        pass
    for c in candidates:
        if (c / "sft_12hz.py").exists():
            return c
    raise SystemExit(
        "Qwen3-TTS finetuning scripts not found. Clone "
        "https://github.com/QwenLM/Qwen3-TTS next to the repo or pip show qwen-tts."
    )


def train(model_id: str, work: Path, epochs: int, batch_size: int, lr: float) -> Path:
    ft = find_finetune_scripts()
    raw = work / "train_raw.jsonl"
    coded = work / "train_with_codes.jsonl"
    out_model = work / "checkpoint"
    n = build_qwen_jsonl(raw)
    print(f"train rows={n} model={model_id}", flush=True)
    if not coded.exists() or coded.stat().st_size < 1000:
        run(
            [
                sys.executable,
                str(ft / "prepare_data.py"),
                "--device",
                "cuda:0",
                "--tokenizer_model_path",
                "Qwen/Qwen3-TTS-Tokenizer-12Hz",
                "--input_jsonl",
                str(raw),
                "--output_jsonl",
                str(coded),
            ]
        )
    run(
        [
            sys.executable,
            str(ft / "sft_12hz.py"),
            "--init_model_path",
            model_id,
            "--output_model_path",
            str(out_model),
            "--train_jsonl",
            str(coded),
            "--batch_size",
            str(batch_size),
            "--lr",
            str(lr),
            "--num_epochs",
            str(epochs),
            "--speaker_name",
            "dose_cloud_ipa",
        ]
    )
    return out_model


def _wav_bytes(path: Path) -> bytes:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="int16", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype("int16")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    return buf.getvalue()


def _load_model(model_path: str):
    import torch
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def synthesize(
    model_path: str,
    eval_items: list[dict],
    out_dir: Path,
    *,
    custom_speaker: str | None = None,
) -> None:
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(model_path)
    prompt = None
    if not custom_speaker:
        prompt = model.create_voice_clone_prompt(ref_audio=REF_AUDIO, ref_text=REF_TEXT)
    for item in eval_items:
        dest = out_dir / f"{_slug(item['ingredient'])}.wav"
        if dest.exists() and dest.stat().st_size > 500:
            continue
        if custom_speaker:
            wavs, sr = model.generate_custom_voice(
                text=item["spoken_text"],
                language="English",
                speaker=custom_speaker,
            )
        else:
            wavs, sr = model.generate_voice_clone(
                text=item["spoken_text"],
                language="English",
                voice_clone_prompt=prompt,
            )
        sf.write(dest, wavs[0], sr)
        print("ok", item["ingredient"], flush=True)


def _slug(name: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def score_pair(cand_dir: Path, eval_items: list[dict], label: str) -> list[dict]:
    import sys as _sys

    _sys.path.insert(0, str(ROOT))
    from dose_r.references.reference_clips import available_clips_all
    from dose_r.scoring.candidate_eval import score_against_best_reference
    from dose_r.scoring.speech_similarity import (
        extract_frame_embeddings,
        speech_bertscore,
    )

    clips = {}
    for ing, clist in available_clips_all().items():
        clips.setdefault(ing.lower(), []).extend(clist)

    rows = []
    for item in eval_items:
        slug = _slug(item["ingredient"])
        cand = cand_dir / f"{slug}.wav"
        row = {
            "ingredient": item["ingredient"],
            "spoken_text": item["spoken_text"],
            "arm": item["arm"],
            "condition": label,
        }
        if not cand.exists():
            row["error"] = "missing cand"
            rows.append(row)
            continue
        cand_bytes = _wav_bytes(cand)

        # vs teacher IPA/plain cloud clip (whole utterance)
        teacher = Path(_abs(item["audio"]))
        if teacher.exists():
            try:
                emb_c = extract_frame_embeddings(cand_bytes)
                emb_t = extract_frame_embeddings(_wav_bytes(teacher))
                f1_t = float(speech_bertscore(emb_c, emb_t)["f1"])
                row["teacher_f1"] = round(f1_t, 4)
            except Exception as exc:
                row["teacher_error"] = str(exc)

        # vs human (best-of available)
        human = clips.get(item["ingredient"].lower(), [])
        if human:
            try:
                best = score_against_best_reference(cand_bytes, human)
                row["human_f1"] = round(best.best_f1, 4)
                row["human_source"] = best.best_source
            except Exception as exc:
                row["human_error"] = str(exc)
        else:
            row["human_error"] = "no human clip"
        rows.append(row)
        print(
            f"{label} {item['ingredient']}: human={row.get('human_f1')} "
            f"teacher={row.get('teacher_f1')}",
            flush=True,
        )
    return rows


def summarize(plain: list[dict], ft: list[dict]) -> dict:
    by_p = {r["ingredient"].lower(): r for r in plain}
    by_f = {r["ingredient"].lower(): r for r in ft}
    keys = sorted(set(by_p) & set(by_f))

    def mean(rows, field, keys):
        vals = [
            by[k][field]
            for k in keys
            for by in (rows,)
            if by[k].get(field) is not None
        ]
        # fix: use the right dict
        return None

    def paired(field: str):
        pairs = []
        for k in keys:
            a, b = by_p[k].get(field), by_f[k].get(field)
            if a is not None and b is not None:
                pairs.append((a, b, k))
        if not pairs:
            return {"n": 0}
        mp = sum(a for a, _, _ in pairs) / len(pairs)
        mf = sum(b for _, b, _ in pairs) / len(pairs)
        up = sum(1 for a, b, _ in pairs if b > a + 1e-6)
        down = sum(1 for a, b, _ in pairs if b < a - 1e-6)
        return {
            "n": len(pairs),
            "plain_mean": round(mp, 4),
            "ft_mean": round(mf, 4),
            "delta": round(mf - mp, 4),
            "n_up": up,
            "n_down": down,
        }

    return {
        "vs_human": paired("human_f1"),
        "vs_teacher": paired("teacher_f1"),
        "n_eval": len(keys),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    ap.add_argument("--work", default=str(ROOT / "runs" / "finetune-qwen-cloud-ipa"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--eval-limit", type=int, default=0)
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    eval_items = json.loads((PACK / "eval_scored.json").read_text())
    if args.eval_limit:
        eval_items = eval_items[: args.eval_limit]

    tag = "qwen06" if "0.6B" in args.model_id else "qwen17"
    ckpt = work / f"{tag}-checkpoint"
    if not args.skip_train:
        trained = train(args.model_id, work / tag, args.epochs, args.batch_size, args.lr)
        # sft writes into output_model_path; keep a stable pointer
        if trained != ckpt and trained.exists():
            ckpt = trained

    plain_dir = work / f"{tag}-plain-synth"
    ft_dir = work / f"{tag}-ft-synth"
    print("=== synth plain base ===", flush=True)
    synthesize(args.model_id, eval_items, plain_dir)
    print("=== synth fine-tuned ===", flush=True)
    # Fine-tuned checkpoint: last epoch dir (custom_voice + speaker slot).
    ft_root = work / tag / "checkpoint"
    epochs = sorted(ft_root.glob("checkpoint-epoch-*"), key=lambda p: p.stat().st_mtime)
    if not epochs:
        raise SystemExit(f"no FT epochs under {ft_root}")
    ft_path = epochs[-1]
    print("FT path", ft_path, flush=True)
    synthesize(str(ft_path), eval_items, ft_dir, custom_speaker="dose_cloud_ipa")

    print("=== score ===", flush=True)
    plain_rows = score_pair(plain_dir, eval_items, f"{tag}-plain")
    ft_rows = score_pair(ft_dir, eval_items, f"{tag}-ft")
    (work / f"{tag}-plain-scores.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in plain_rows)
    )
    (work / f"{tag}-ft-scores.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in ft_rows)
    )
    summary = summarize(plain_rows, ft_rows)
    summary["model_id"] = args.model_id
    summary["ft_path"] = str(ft_path)
    summary["epochs"] = args.epochs
    (work / f"{tag}-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
