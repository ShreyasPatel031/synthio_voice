#!/usr/bin/env python3
"""Few-word Qwen recipes to match cloud-IPA teacher pronunciation.

Goal: beat/match the ceiling
  Base + voice_clone(ref=teacher_wav) + plain spoken_text
  (vorasidenib was human 0.763 / vs-teacher 0.867).

Recipes (same small name set):
  A  base_clone_teacher_plain   — ceiling: Base, clone teacher, text=spoken
  B  base_clone_teacher_ipa     — Base, clone teacher, text=ipa_used
  C  base_clone_public_plain    — Base, public clone.wav, text=spoken (old bad)
  D  sft_text_custom            — SFT text=spoken → teacher; infer custom_voice
  E  sft_ipa_custom             — SFT text=ipa_used → teacher; infer custom_voice + IPA
  F  sft_text_then_clone_teacher— SFT like D, but infer voice_clone(ref=teacher)
                                  on the FT weights if the API still works

Only a few hard names. Sparse checkpoint keep. Primary metric: vs-teacher F1.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "data" / "finetune_cloud_ipa"
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
PUBLIC_CLONE = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
PUBLIC_CLONE_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it! And thanks to you."
)
FT_SCRIPTS = Path("/home/shreyaspatel/Qwen3-TTS/finetuning")
if not (FT_SCRIPTS / "sft_12hz.py").exists():
    FT_SCRIPTS = Path.home() / "Qwen3-TTS" / "finetuning"

DEFAULT_NAMES = ["vorasidenib", "Advair", "Benadryl", "Vraylar"]


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
        if line.strip():
            r = json.loads(line)
            out[r["ingredient"].lower()] = r
    return out


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


def _load_model(path: str):
    import torch
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(
        path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def _free(model) -> None:
    del model
    import torch

    torch.cuda.empty_cache()


def build_train(
    names: list[str],
    manifest: dict[str, dict],
    out_raw: Path,
    *,
    text_mode: str,
    repeats: int,
) -> int:
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_raw.open("w") as f:
        for name in names:
            row = manifest[name.lower()]
            audio = _abs(row["audio"])
            if text_mode == "spoken":
                text = row["spoken_text"]
            elif text_mode == "ipa":
                text = (row.get("ipa_used") or "").strip()
                if not text:
                    raise SystemExit(f"no ipa_used for {name}")
            else:
                raise SystemExit(text_mode)
            rec = {"audio": audio, "text": text, "ref_audio": audio}
            for _ in range(repeats):
                f.write(json.dumps(rec) + "\n")
                n += 1
    return n


def prepare_and_train(
    raw: Path,
    coded: Path,
    ckpt_root: Path,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    speaker: str,
) -> Path:
    if coded.exists():
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
    if ckpt_root.exists():
        shutil.rmtree(ckpt_root)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(FT_SCRIPTS / "sft_12hz.py"),
            "--init_model_path",
            MODEL_ID,
            "--output_model_path",
            str(ckpt_root),
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
    # keep only last + mid; drop the rest to save disk
    epochs_dirs = sorted(ckpt_root.glob("checkpoint-epoch-*"))
    keep = set()
    if epochs_dirs:
        keep.add(epochs_dirs[-1].name)
        keep.add(epochs_dirs[len(epochs_dirs) // 2].name)
        keep.add(epochs_dirs[0].name)
    for d in epochs_dirs:
        if d.name not in keep:
            shutil.rmtree(d)
    return sorted(ckpt_root.glob("checkpoint-epoch-*"))[-1]


def synth_clone(
    model_path: str,
    names: list[str],
    manifest: dict[str, dict],
    out_dir: Path,
    *,
    text_mode: str,
    ref: str,
) -> None:
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(model_path)
    public_prompt = None
    if ref == "public":
        public_prompt = model.create_voice_clone_prompt(
            ref_audio=PUBLIC_CLONE, ref_text=PUBLIC_CLONE_TEXT
        )
    for name in names:
        dest = out_dir / f"{_slug(name)}.wav"
        if dest.exists() and dest.stat().st_size > 500:
            continue
        row = manifest[name.lower()]
        if text_mode == "spoken":
            text = row["spoken_text"]
        else:
            text = row["ipa_used"]
        if ref == "teacher":
            prompt = model.create_voice_clone_prompt(
                ref_audio=_abs(row["audio"]),
                ref_text=row["spoken_text"],
            )
        else:
            prompt = public_prompt
        wavs, sr = model.generate_voice_clone(
            text=text, language="English", voice_clone_prompt=prompt
        )
        sf.write(dest, wavs[0], sr)
        print("ok-clone", name, text_mode, ref, flush=True)
    _free(model)


def synth_custom(
    ckpt: Path,
    names: list[str],
    manifest: dict[str, dict],
    out_dir: Path,
    *,
    text_mode: str,
    speaker: str,
) -> None:
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(str(ckpt))
    for name in names:
        dest = out_dir / f"{_slug(name)}.wav"
        if dest.exists() and dest.stat().st_size > 500:
            continue
        row = manifest[name.lower()]
        text = row["spoken_text"] if text_mode == "spoken" else row["ipa_used"]
        wavs, sr = model.generate_custom_voice(
            text=text, language="English", speaker=speaker
        )
        sf.write(dest, wavs[0], sr)
        print("ok-custom", name, text_mode, flush=True)
    _free(model)


def score_dir(
    cand_dir: Path,
    names: list[str],
    manifest: dict[str, dict],
    label: str,
) -> list[dict]:
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
        row = {
            "ingredient": manifest[key]["ingredient"],
            "recipe": label,
            "ipa_used": manifest[key].get("ipa_used"),
        }
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
            except Exception as exc:
                row["human_error"] = str(exc)
        rows.append(row)
        print(
            f"{label} {row['ingredient']}: human={row.get('human_f1')} "
            f"teacher={row.get('teacher_f1')}",
            flush=True,
        )
    return rows


def summarize(all_rows: list[dict], names: list[str]) -> dict:
    by_recipe: dict[str, list] = {}
    for r in all_rows:
        by_recipe.setdefault(r["recipe"], []).append(r)

    def mean(vals):
        return round(sum(vals) / len(vals), 4) if vals else None

    out = {"names": names, "by_recipe": {}}
    for recipe, rows in by_recipe.items():
        th = [r["teacher_f1"] for r in rows if r.get("teacher_f1") is not None]
        hu = [r["human_f1"] for r in rows if r.get("human_f1") is not None]
        out["by_recipe"][recipe] = {
            "n": len(rows),
            "mean_teacher_f1": mean(th),
            "mean_human_f1": mean(hu),
            "per_name": {
                r["ingredient"]: {
                    "teacher_f1": r.get("teacher_f1"),
                    "human_f1": r.get("human_f1"),
                }
                for r in rows
            },
        }
    # rank by mean teacher
    ranked = sorted(
        out["by_recipe"].items(),
        key=lambda kv: kv[1]["mean_teacher_f1"] or -1,
        reverse=True,
    )
    out["rank_by_teacher"] = [
        {"recipe": k, "mean_teacher_f1": v["mean_teacher_f1"], "mean_human_f1": v["mean_human_f1"]}
        for k, v in ranked
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=",".join(DEFAULT_NAMES))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=12)
    # batch_size must be 1 when mixing short drug names: collate fails on
    # unequal codec lengths (Expected size 77 but got size 83).
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--work", default=str(ROOT / "runs" / "teacher-match-recipes"))
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument(
        "--only",
        default="",
        help="comma recipes to run; empty = all",
    )
    args = ap.parse_args()

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    manifest = load_manifest()
    for n in names:
        if n.lower() not in manifest:
            raise SystemExit(f"missing from pack: {n}")
        if not Path(_abs(manifest[n.lower()]["audio"])).exists():
            raise SystemExit(f"missing wav: {manifest[n.lower()]['audio']}")

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    only = {x.strip() for x in args.only.split(",") if x.strip()} or None

    def want(tag: str) -> bool:
        return only is None or tag in only

    all_rows: list[dict] = []

    # --- A ceiling ---
    if want("A"):
        d = work / "A_base_clone_teacher_plain"
        synth_clone(MODEL_ID, names, manifest, d, text_mode="spoken", ref="teacher")
        all_rows += score_dir(d, names, manifest, "A_base_clone_teacher_plain")

    # --- B IPA as text at inference ---
    if want("B"):
        d = work / "B_base_clone_teacher_ipa"
        synth_clone(MODEL_ID, names, manifest, d, text_mode="ipa", ref="teacher")
        all_rows += score_dir(d, names, manifest, "B_base_clone_teacher_ipa")

    # --- C old bad baseline ---
    if want("C"):
        d = work / "C_base_clone_public_plain"
        synth_clone(MODEL_ID, names, manifest, d, text_mode="spoken", ref="public")
        all_rows += score_dir(d, names, manifest, "C_base_clone_public_plain")

    # --- D SFT spoken → teacher ---
    if want("D") or want("F"):
        d_dir = work / "D_sft_text"
        raw, coded, ckpt = d_dir / "train_raw.jsonl", d_dir / "coded.jsonl", d_dir / "checkpoint"
        speaker = "dose_teacher"
        if not args.skip_train and want("D"):
            n = build_train(names, manifest, raw, text_mode="spoken", repeats=args.repeats)
            print(f"D train rows={n}", flush=True)
            last = prepare_and_train(
                raw, coded, ckpt, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, speaker=speaker
            )
        else:
            last = sorted(ckpt.glob("checkpoint-epoch-*"))[-1]
        if want("D"):
            out = work / "D_sft_text_custom"
            synth_custom(last, names, manifest, out, text_mode="spoken", speaker=speaker)
            all_rows += score_dir(out, names, manifest, "D_sft_text_custom")
        if want("F"):
            out = work / "F_sft_text_clone_teacher"
            try:
                synth_clone(str(last), names, manifest, out, text_mode="spoken", ref="teacher")
                all_rows += score_dir(out, names, manifest, "F_sft_text_clone_teacher")
            except Exception as exc:
                print("F failed (expected if custom_voice-only):", exc, flush=True)
                all_rows.append(
                    {
                        "ingredient": names[0],
                        "recipe": "F_sft_text_clone_teacher",
                        "error": str(exc),
                    }
                )

    # --- E SFT IPA → teacher ---
    if want("E"):
        d_dir = work / "E_sft_ipa"
        raw, coded, ckpt = d_dir / "train_raw.jsonl", d_dir / "coded.jsonl", d_dir / "checkpoint"
        speaker = "dose_ipa"
        if not args.skip_train:
            n = build_train(names, manifest, raw, text_mode="ipa", repeats=args.repeats)
            print(f"E train rows={n}", flush=True)
            last = prepare_and_train(
                raw, coded, ckpt, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, speaker=speaker
            )
        else:
            last = sorted(ckpt.glob("checkpoint-epoch-*"))[-1]
        out = work / "E_sft_ipa_custom"
        synth_custom(last, names, manifest, out, text_mode="ipa", speaker=speaker)
        all_rows += score_dir(out, names, manifest, "E_sft_ipa_custom")

    (work / "all_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in all_rows))
    summary = summarize([r for r in all_rows if "error" not in r or r.get("teacher_f1")], names)
    # clean summarize for rows with scores
    summary = summarize([r for r in all_rows if r.get("teacher_f1") is not None], names)
    (work / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
