#!/usr/bin/env python3
"""Hard-subset: Base Qwen vs pronunciation-list word-audio inject (top 3).

Arms:
  base                  — public clone.wav (no drug audio)
  inject_word_greedy    — list wav, full clone, greedy
  inject_word_xvec_greedy — list wav, x-vector only, greedy
  inject_word_sample    — list wav, full clone, sample T=0.2

When a drug is missing from the list, inject arms fall back to the public
clone prompt (same as base). Score with scripts/score_oss_run.py after.
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
sys.path.insert(0, str(ROOT / "scripts"))

from oss_eval_items import load_items

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
PACK = ROOT / "data" / "gold_gemini_ipa"
LIST_PATH = ROOT / "data" / "pronunciation_list.json"
OUT_ROOT = ROOT / "runs" / "oss-eval"
PUBLIC_REF = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
PUBLIC_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it! And thanks to you."
)
MAX_WAV_BYTES = 2_000_000

ARMS = [
    "base",
    "inject_word_greedy",
    "inject_word_xvec_greedy",
    "inject_word_sample",
]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def build_list(items: list[dict]) -> dict:
    manifest = {}
    for line in (PACK / "manifest.jsonl").open():
        if line.strip():
            r = json.loads(line)
            manifest[r["ingredient"].lower()] = r
    entries = {}
    missing = []
    for item in items:
        drug = item["drug"]
        key = drug.lower()
        row = manifest.get(key)
        wav = None
        ipa = ""
        if row:
            p = ROOT / row["audio"]
            if p.exists():
                wav = p
            ipa = (row.get("ipa") or "").strip("/") or row.get("ipa_used") or ""
        if wav is None:
            cand = PACK / "wavs" / f"{_slug(drug)}.wav"
            if cand.exists():
                wav = cand
        if wav is None:
            missing.append(drug)
            continue
        entry = {
            "spoken": row.get("spoken_text", drug) if row else drug,
            "ipa": ipa,  # source-published / Cloud pack only — never G2P
            "audio": str(wav.relative_to(ROOT)),
            "audio_abs": str(wav.resolve()),
        }
        entries[drug] = entry
        entries[drug.lower()] = entry
        if item.get("spoken"):
            entries[item["spoken"]] = entry
    data = {
        "version": 1,
        "n_entries": len({e["audio"] for e in entries.values()}),
        "n_missing_hard_subset": len(missing),
        "missing": missing,
        "entries": entries,
    }
    LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIST_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return data


def lookup(plist: dict, drug: str, spoken: str) -> dict | None:
    e = plist["entries"]
    return e.get(spoken) or e.get(drug) or e.get(drug.lower())


def generate(model, *, text: str, ref_audio, ref_text: str, arm: str, public_prompt):
    if arm == "base" or ref_audio is None:
        return model.generate_voice_clone(
            text=text, language="English", voice_clone_prompt=public_prompt
        )
    greedy = arm.endswith("_greedy")
    xvec = "_xvec_" in arm
    kw = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=0.2, top_p=0.8)
    return model.generate_voice_clone(
        text=text,
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=xvec,
        **kw,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = load_items()
    if args.limit:
        items = items[: args.limit]
    plist = build_list(load_items())  # full hard subset for list coverage
    print(
        f"items={len(items)} list_entries={plist['n_entries']} "
        f"missing={plist['missing']}",
        flush=True,
    )

    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    public_prompt = model.create_voice_clone_prompt(
        ref_audio=PUBLIC_REF, ref_text=PUBLIC_TEXT
    )

    for arm in args.arms:
        out = OUT_ROOT / f"pron-list-{arm}"
        out.mkdir(parents=True, exist_ok=True)
        print(f"=== {arm} → {out} ===", flush=True)
        n_inject = n_base = n_fail = 0
        for i, item in enumerate(items):
            dest = out / f"{item['item_id']}.wav"
            if dest.exists() and 1000 < dest.stat().st_size < MAX_WAV_BYTES:
                continue
            entry = lookup(plist, item["drug"], item["spoken"])
            use_inject = arm != "base" and entry is not None
            try:
                torch.manual_seed(10000 + i)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(10000 + i)
                wavs, sr = generate(
                    model,
                    text=item["sentence"],
                    ref_audio=entry["audio_abs"] if use_inject else None,
                    ref_text=entry["spoken"] if use_inject else PUBLIC_TEXT,
                    arm=arm,
                    public_prompt=public_prompt,
                )
                sf.write(dest, wavs[0], sr)
                if dest.stat().st_size > MAX_WAV_BYTES:
                    dest.unlink()
                    n_fail += 1
                    print("RUNAWAY", item["item_id"], flush=True)
                    continue
                if use_inject:
                    n_inject += 1
                else:
                    n_base += 1
                print(
                    "ok",
                    item["item_id"],
                    "inject" if use_inject else "fallback_base",
                    flush=True,
                )
            except Exception as exc:
                n_fail += 1
                print("FAIL", item["item_id"], exc, flush=True)
        (out / "synth_meta.json").write_text(
            json.dumps(
                {
                    "arm": arm,
                    "n_inject": n_inject,
                    "n_fallback_base": n_base,
                    "n_fail": n_fail,
                    "list": str(LIST_PATH),
                },
                indent=2,
            )
        )
        print(f"done {arm} inject={n_inject} fallback={n_base} fail={n_fail}", flush=True)

    print("SYNTH_DONE — score with score_oss_run.py", flush=True)


if __name__ == "__main__":
    main()
