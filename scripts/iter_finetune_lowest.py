#!/usr/bin/env python3
"""Iterative Qwen 1.7B SFT on lowest Path-2 names only (from Base).

Curriculum: train only hard/low names; skip names that already score well.
Default first run: vorasidenib alone, many epochs, score every checkpoint.

Train audio = cloud IPA teacher wav. ref_audio = same teacher wav so the
learned custom-voice speaker matches the target (avoids clone.wav mismatch).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "data" / "gold_gemini_ipa"
PLAIN_SCORES = ROOT / "runs" / "finetune-qwen-cloud-ipa" / "qwen17-plain-scores.jsonl"
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
FT_SCRIPTS = Path.home() / "Qwen3-TTS" / "finetuning"
if not (FT_SCRIPTS / "sft_12hz.py").exists():
    FT_SCRIPTS = Path("/home/shreyaspatel/Qwen3-TTS/finetuning")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _abs(p: str | Path) -> str:
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve())


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def load_manifest() -> dict[str, dict]:
    out = {}
    for line in (PACK / "manifest.jsonl").open():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["ingredient"].lower()] = r
    return out


def load_plain_human() -> dict[str, float]:
    scores = {}
    if not PLAIN_SCORES.exists():
        return scores
    for line in PLAIN_SCORES.open():
        r = json.loads(line)
        if r.get("human_f1") is not None:
            scores[r["ingredient"].lower()] = float(r["human_f1"])
    return scores


def pick_names(args: argparse.Namespace, manifest: dict[str, dict], plain: dict[str, float]) -> list[str]:
    if args.names:
        return [n.strip() for n in args.names.split(",") if n.strip()]
    # lowest-first: bottom K by plain human F1, among names with teacher audio
    ranked = sorted(
        ((plain[k], k) for k in plain if k in manifest),
        key=lambda t: t[0],
    )
    if args.max_plain_f1 is not None:
        ranked = [(s, k) for s, k in ranked if s <= args.max_plain_f1]
    picked = [k for _, k in ranked[: args.bottom]]
    if not picked:
        raise SystemExit("no names matched lowest-first filters")
    return picked


def build_train_raw(names: list[str], manifest: dict[str, dict], out: Path, repeats: int) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for name in names:
            row = manifest[name.lower()]
            audio = _abs(row["audio"])
            rec = {
                "audio": audio,
                "text": row["spoken_text"],
                # speaker emb from the teacher itself — not public clone.wav
                "ref_audio": audio,
            }
            for _ in range(repeats):
                f.write(json.dumps(rec) + "\n")
                n += 1
    return n


def prepare_codes(raw: Path, coded: Path) -> None:
    if coded.exists() and coded.stat().st_size > 100:
        # rebuild if source names changed
        coded.unlink()
    _run(
        [
            sys.executable,
            str(FT_SCRIPTS / "prepare_data.py"),
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


def train(coded: Path, out_model: Path, epochs: int, batch_size: int, lr: float, speaker: str) -> None:
    out_model.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(FT_SCRIPTS / "sft_12hz.py"),
            "--init_model_path",
            MODEL_ID,
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
            speaker,
        ]
    )


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


def synth_custom(ckpt: Path, names: list[str], manifest: dict[str, dict], out_dir: Path, speaker: str) -> None:
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    out_dir.mkdir(parents=True, exist_ok=True)
    model = Qwen3TTSModel.from_pretrained(
        str(ckpt),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    for name in names:
        dest = out_dir / f"{_slug(name)}.wav"
        if dest.exists() and dest.stat().st_size > 500:
            continue
        text = manifest[name.lower()]["spoken_text"]
        wavs, sr = model.generate_custom_voice(
            text=text, language="English", speaker=speaker
        )
        sf.write(dest, wavs[0], sr)
        print("ok", name, dest.name, flush=True)
    del model
    import torch as _t

    _t.cuda.empty_cache()


def synth_plain_base(names: list[str], manifest: dict[str, dict], out_dir: Path) -> None:
    """Baseline: voice_clone with teacher wav as ref (fairer than public clone)."""
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    out_dir.mkdir(parents=True, exist_ok=True)
    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    for name in names:
        dest = out_dir / f"{_slug(name)}.wav"
        if dest.exists() and dest.stat().st_size > 500:
            continue
        row = manifest[name.lower()]
        audio = _abs(row["audio"])
        prompt = model.create_voice_clone_prompt(
            ref_audio=audio,
            ref_text=row["spoken_text"],
        )
        wavs, sr = model.generate_voice_clone(
            text=row["spoken_text"],
            language="English",
            voice_clone_prompt=prompt,
        )
        sf.write(dest, wavs[0], sr)
        print("ok-plain", name, flush=True)
    del model
    import torch as _t

    _t.cuda.empty_cache()


def score_wavs(cand_dir: Path, names: list[str], manifest: dict[str, dict], label: str) -> list[dict]:
    sys.path.insert(0, str(ROOT))
    from dose_r.references.reference_clips import available_clips_all
    from dose_r.scoring.candidate_eval import score_against_best_reference
    from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore

    clips = {}
    for ing, clist in available_clips_all().items():
        clips.setdefault(ing.lower(), []).extend(clist)

    rows = []
    for name in names:
        key = name.lower()
        row = {"ingredient": manifest[key]["ingredient"], "condition": label}
        cand = cand_dir / f"{_slug(name)}.wav"
        if not cand.exists():
            row["error"] = "missing"
            rows.append(row)
            continue
        cand_b = _wav_bytes(cand)
        teacher = Path(_abs(manifest[key]["audio"]))
        try:
            emb_c = extract_frame_embeddings(cand_b)
            emb_t = extract_frame_embeddings(_wav_bytes(teacher))
            row["teacher_f1"] = round(float(speech_bertscore(emb_c, emb_t)["f1"]), 4)
        except Exception as exc:
            row["teacher_error"] = str(exc)
        human = clips.get(key, [])
        if human:
            try:
                best = score_against_best_reference(cand_b, human)
                row["human_f1"] = round(best.best_f1, 4)
                row["human_source"] = best.best_source
            except Exception as exc:
                row["human_error"] = str(exc)
        else:
            row["human_error"] = "no human clip"
        rows.append(row)
        print(
            f"{label} {row['ingredient']}: human={row.get('human_f1')} teacher={row.get('teacher_f1')}",
            flush=True,
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="vorasidenib", help="comma list, or empty to use --bottom")
    ap.add_argument("--bottom", type=int, default=0, help="if >0 and --names empty, take N lowest")
    ap.add_argument("--max-plain-f1", type=float, default=None, help="skip names above this plain F1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=5, help="score checkpoint every N epochs")
    ap.add_argument("--repeats", type=int, default=16, help="duplicate each clip per epoch")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--speaker", default="dose_hard")
    ap.add_argument("--work", default=str(ROOT / "runs" / "iter-ft-lowest"))
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-plain", action="store_true")
    args = ap.parse_args()

    if not args.names and not args.bottom:
        args.bottom = 1  # default lowest-first single name

    manifest = load_manifest()
    plain = load_plain_human()
    # argparse default names=vorasidenib; allow --names '' with --bottom
    if args.names == "" and args.bottom:
        names = pick_names(args, manifest, plain)
    elif args.bottom and not args.names:
        names = pick_names(args, manifest, plain)
    else:
        names = [n.strip() for n in args.names.split(",") if n.strip()]

    for n in names:
        if n.lower() not in manifest:
            raise SystemExit(f"not in finetune pack: {n}")

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    tag = "+".join(_slug(n) for n in names)
    run_dir = work / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "names": names,
        "plain_f1": {n: plain.get(n.lower()) for n in names},
        "epochs": args.epochs,
        "repeats": args.repeats,
        "lr": args.lr,
        "note": "train only listed (low) names; high-F1 names excluded",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)

    raw = run_dir / "train_raw.jsonl"
    coded = run_dir / "train_with_codes.jsonl"
    ckpt_root = run_dir / "checkpoint"
    n_rows = build_train_raw(names, manifest, raw, args.repeats)
    print(f"train rows={n_rows}", flush=True)
    prepare_codes(raw, coded)

    if not args.skip_train:
        train(coded, ckpt_root, args.epochs, args.batch_size, args.lr, args.speaker)

    # plain baseline with teacher-as-ref (fairer)
    curve = []
    if not args.skip_plain:
        plain_dir = run_dir / "synth-plain-teacher-ref"
        synth_plain_base(names, manifest, plain_dir)
        plain_rows = score_wavs(plain_dir, names, manifest, "plain-teacher-ref")
        (run_dir / "scores-plain.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in plain_rows)
        )
        curve.append({"epoch": -1, "label": "plain-teacher-ref", "rows": plain_rows})

    epochs_dir = sorted(ckpt_root.glob("checkpoint-epoch-*"), key=lambda p: p.name)
    for ckpt in epochs_dir:
        m = re.search(r"checkpoint-epoch-(\d+)$", ckpt.name)
        if not m:
            continue
        ep = int(m.group(1))
        # sft saves after each epoch with 0-index; eval epoch 0, every eval_every, and last
        if ep != 0 and (ep + 1) % args.eval_every != 0 and ep != args.epochs - 1:
            continue
        out_dir = run_dir / f"synth-epoch-{ep}"
        print(f"=== synth epoch {ep} ===", flush=True)
        synth_custom(ckpt, names, manifest, out_dir, args.speaker)
        rows = score_wavs(out_dir, names, manifest, f"epoch-{ep}")
        (run_dir / f"scores-epoch-{ep}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        curve.append({"epoch": ep, "label": f"epoch-{ep}", "rows": rows})

    # compact curve summary
    summary_rows = []
    for block in curve:
        for r in block["rows"]:
            summary_rows.append(
                {
                    "epoch": block["epoch"],
                    "label": block["label"],
                    "ingredient": r.get("ingredient"),
                    "human_f1": r.get("human_f1"),
                    "teacher_f1": r.get("teacher_f1"),
                    "error": r.get("error") or r.get("human_error"),
                }
            )
    (run_dir / "curve.json").write_text(json.dumps(summary_rows, indent=2))
    print("=== curve ===", flush=True)
    print(json.dumps(summary_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
