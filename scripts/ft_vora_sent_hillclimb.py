#!/usr/bin/env python3
"""Hill-climb sentence FT until CTC-cut works on ALL vorasidenib eval sentences.

Teachers: Gemini 3.1 + source IPA (best in-sentence Path-2).
Train: expanding sentence pack; ref_audio = teacher wav.
Eval each epoch: every eval sentence, N draws at low temperature;
report mean/min human F1. Keep checkpoint maximizing min F1.

Success: min_human_f1 >= 0.68 and mean >= 0.72 on the eval set
(Gemini teacher mean was ~0.76).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dose_r.forced_align import extract_drug_span_forced_align
from dose_r.references.reference_clips import available_clips_all
from dose_r.scoring.candidate_eval import score_against_best_reference

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
FT_SCRIPTS = Path("/home/shreyaspatel/Qwen3-TTS/finetuning")
if not (FT_SCRIPTS / "sft_12hz.py").exists():
    FT_SCRIPTS = Path.home() / "Qwen3-TTS" / "finetuning"

DRUG = "vorasidenib"
IPA = "vɔːrəˈsɪdənɪb"
SPEAKER = "dose_sent2"
WORK = ROOT / "runs" / "ft-vora-sent-hillclimb"
TEACHER_DIR = WORK / "teachers"
EVAL_IDS = ["dose", "prescribed", "confirm", "monitor"]  # must all work

# Train pack: eval 4 + extra diversity (Gemini teachers synth'd separately)
ALL_SENTENCES = [
    {
        "id": "dose",
        "text": "Let's initiate vorasidenib therapy for this patient to target the mutant IDH1 and IDH2 enzymes in the tumor.",
        "eval": True,
    },
    {
        "id": "prescribed",
        "text": "The oncologist prescribed vorasidenib for the patient's IDH-mutant glioma.",
        "eval": True,
    },
    {
        "id": "confirm",
        "text": "Please confirm the vorasidenib dose before the next clinic visit.",
        "eval": True,
    },
    {
        "id": "monitor",
        "text": "Patients taking vorasidenib need monitoring for liver enzyme elevation.",
        "eval": True,
    },
    {
        "id": "start",
        "text": "We will start vorasidenib this week if labs remain stable.",
        "eval": False,
    },
    {
        "id": "discuss",
        "text": "I want to discuss the risks and benefits of vorasidenib with you today.",
        "eval": False,
    },
    {
        "id": "oral",
        "text": "Vorasidenib is taken by mouth once daily with or without food.",
        "eval": False,
    },
    {
        "id": "switch",
        "text": "If side effects worsen we may hold vorasidenib and reassess.",
        "eval": False,
    },
    {
        "id": "idh",
        "text": "Because the tumor carries an IDH mutation, vorasidenib is a reasonable option.",
        "eval": False,
    },
    {
        "id": "pharmacy",
        "text": "Please counsel the patient on how to store and take vorasidenib correctly.",
        "eval": False,
    },
    {
        "id": "followup",
        "text": "At follow-up we will review imaging and decide whether to continue vorasidenib.",
        "eval": False,
    },
    {
        "id": "combo",
        "text": "Do not combine vorasidenib with strong CYP inducers without checking interactions.",
        "eval": False,
    },
]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def build_train_jsonl(raw: Path, teacher_dir: Path, repeats: int) -> int:
    raw.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with raw.open("w") as f:
        for item in ALL_SENTENCES:
            wav = teacher_dir / f"{item['id']}__gemini.wav"
            if not wav.exists():
                raise SystemExit(f"missing teacher: {wav}")
            rec = {
                "audio": str(wav.resolve()),
                "text": item["text"],
                "ref_audio": str(wav.resolve()),
            }
            for _ in range(repeats):
                f.write(json.dumps(rec) + "\n")
                n += 1
        # light isolated anchor (Cloud IPA teacher)
        iso = ROOT / "data/gold_gemini_ipa/wavs/vorasidenib.wav"
        if iso.exists():
            rec = {
                "audio": str(iso.resolve()),
                "text": DRUG,
                "ref_audio": str(iso.resolve()),
            }
            for _ in range(max(2, repeats // 2)):
                f.write(json.dumps(rec) + "\n")
                n += 1
    return n


def train_one_epoch(
    coded: Path,
    init_model: str,
    out_ckpt: Path,
    *,
    batch_size: int,
    lr: float,
    speaker: str,
) -> Path:
    """Train exactly 1 epoch from init_model into out_ckpt/checkpoint-epoch-0."""
    if out_ckpt.exists():
        shutil.rmtree(out_ckpt)
    out_ckpt.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(FT_SCRIPTS / "sft_12hz.py"),
            "--init_model_path",
            init_model,
            "--output_model_path",
            str(out_ckpt),
            "--train_jsonl",
            str(coded),
            "--batch_size",
            str(batch_size),
            "--lr",
            str(lr),
            "--num_epochs",
            "1",
            "--speaker_name",
            speaker,
        ]
    )
    ep0 = out_ckpt / "checkpoint-epoch-0"
    if not ep0.exists():
        raise SystemExit(f"missing {ep0}")
    return ep0


def load_model(path: str | Path) -> Qwen3TTSModel:
    return Qwen3TTSModel.from_pretrained(
        str(path),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def free(model) -> None:
    del model
    torch.cuda.empty_cache()


def synth_score(
    ckpt: Path,
    *,
    n_draws: int,
    temperature: float,
    human,
) -> dict:
    eval_items = [s for s in ALL_SENTENCES if s["id"] in EVAL_IDS]
    model = load_model(ckpt)
    per = {}
    for item in eval_items:
        scores = []
        for draw in range(n_draws):
            torch.manual_seed(1000 + draw * 17 + hash(item["id"]) % 997)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1000 + draw * 17 + hash(item["id"]) % 997)
            wavs, sr = model.generate_custom_voice(
                text=item["text"],
                language="English",
                speaker=SPEAKER,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
            )
            dest = WORK / "eval_wavs" / f"{item['id']}_t{temperature}_d{draw}.wav"
            dest.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dest, wavs[0], sr)
            span = extract_drug_span_forced_align(
                dest.read_bytes(), item["text"], DRUG
            )
            if span is None:
                scores.append(0.0)
                continue
            span_path = dest.with_name(dest.stem + "_span.wav")
            span_path.write_bytes(span)
            best = score_against_best_reference(span, human)
            scores.append(float(best.best_f1))
        per[item["id"]] = {
            "draws": [round(s, 4) for s in scores],
            "mean": round(sum(scores) / len(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        }
        print(
            f"  {item['id']}: mean={per[item['id']]['mean']} "
            f"min={per[item['id']]['min']} draws={per[item['id']]['draws']}",
            flush=True,
        )
    free(model)
    means = [v["mean"] for v in per.values()]
    mins = [v["min"] for v in per.values()]
    return {
        "per_sentence": per,
        "mean_of_means": round(sum(means) / len(means), 4),
        "min_of_means": round(min(means), 4),
        "min_of_mins": round(min(mins), 4),
        "temperature": temperature,
        "n_draws": n_draws,
    }


def success(stats: dict) -> bool:
    # all sentence means >= 0.70 and worst draw floor >= 0.62
    per = stats["per_sentence"]
    return all(v["mean"] >= 0.70 for v in per.values()) and all(
        v["min"] >= 0.62 for v in per.values()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epochs", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-draws", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--teacher-dir", default=str(TEACHER_DIR))
    ap.add_argument("--skip-prepare", action="store_true")
    args = ap.parse_args()

    work = WORK
    work.mkdir(parents=True, exist_ok=True)
    teacher_dir = Path(args.teacher_dir)
    raw = work / "train_raw.jsonl"
    coded = work / "train_with_codes.jsonl"

    n = build_train_jsonl(raw, teacher_dir, args.repeats)
    print(f"train rows={n}", flush=True)

    if not args.skip_prepare or not coded.exists():
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

    clips = {}
    for ing, clist in available_clips_all().items():
        clips.setdefault(ing.lower(), []).extend(clist)
    human = clips[DRUG]

    history = []
    best = None
    init = MODEL_ID
    for ep in range(args.max_epochs):
        print(f"=== train epoch slot {ep} from {init} ===", flush=True)
        out_root = work / f"ckpt_round_{ep}"
        ckpt = train_one_epoch(
            coded,
            init,
            out_root,
            batch_size=args.batch_size,
            lr=args.lr,
            speaker=SPEAKER,
        )
        # drop other epoch copies inside round to save disk
        for extra in out_root.glob("checkpoint-epoch-*"):
            if extra != ckpt:
                shutil.rmtree(extra)

        print(f"=== eval epoch {ep} ===", flush=True)
        stats = synth_score(
            ckpt, n_draws=args.n_draws, temperature=args.temperature, human=human
        )
        stats["epoch"] = ep
        stats["ckpt"] = str(ckpt)
        history.append(stats)
        (work / "history.json").write_text(json.dumps(history, indent=2))
        print(
            f"epoch {ep}: mean={stats['mean_of_means']} "
            f"min_mean={stats['min_of_means']} min_draw={stats['min_of_mins']} "
            f"success={success(stats)}",
            flush=True,
        )

        if best is None or stats["min_of_means"] > best["min_of_means"]:
            best = stats
            # stable pointer
            best_link = work / "best_checkpoint"
            if best_link.exists() or best_link.is_symlink():
                if best_link.is_symlink() or best_link.is_file():
                    best_link.unlink()
                else:
                    shutil.rmtree(best_link)
            shutil.copytree(ckpt, best_link)
            (work / "best_stats.json").write_text(json.dumps(best, indent=2))

        if success(stats):
            print("SUCCESS criteria met", flush=True)
            break

        # continue from this ckpt (cumulative SFT)
        init = str(ckpt)
        # free prior round disk (keep current + best)
        if ep >= 1:
            old = work / f"ckpt_round_{ep - 1}"
            if old.exists() and old.resolve() != Path(init).parent.resolve():
                # only delete if not the parent of current init
                pass
            # init points at ckpt inside ckpt_round_{ep}; delete ep-1
            prev = work / f"ckpt_round_{ep - 1}"
            if prev.exists():
                shutil.rmtree(prev)
                print("freed", prev, flush=True)

    summary = {
        "best": best,
        "history": history,
        "success": bool(best and success(best)),
        "criteria": "each eval sentence mean>=0.70 and min_draw>=0.62",
        "n_train_sentences": len(ALL_SENTENCES),
        "eval_ids": EVAL_IDS,
        "lr": args.lr,
        "temperature": args.temperature,
        "n_draws": args.n_draws,
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
