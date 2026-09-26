#!/usr/bin/env python
"""SpeechBERTScore precision / recall / F1 for Gemini 3.1 hyphen vs compact.

Candidate = TTS, reference = human clip. Precision is each TTS frame's
best match in the human (paper default; extra length can still drag it).
Recall is each human frame's best match in the TTS (covers the whole name;
extra TTS length is mostly ignored).

Does not synthesize. Reads wavs from runs/gemini31-respell-all/tts.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
_hf = ROOT / ".cache" / "huggingface"
_hf.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_hf))
os.environ.setdefault("HF_HUB_CACHE", str(_hf / "hub"))

from eval_gemini31_holdout import cloud_ipa_path, mean, slug  # noqa: E402
from dose_r.references.reference_clips import available_clips  # noqa: E402
from dose_r.scoring.speech_similarity import (  # noqa: E402
    extract_frame_embeddings,
    speech_bertscore,
)

IN = ROOT / "runs" / "gemini31-respell-all" / "results.json"
TTS = ROOT / "runs" / "gemini31-respell-all" / "tts"
OUT = ROOT / "runs" / "gemini31-respell-all" / "prf.json"
GAP = 0.03
_EMB: dict = {}


def emb(path: Path):
    key = (str(path), path.stat().st_size)
    if key not in _EMB:
        _EMB[key] = extract_frame_embeddings(path)
    return _EMB[key]


def prf(cand: Path, ref: Path) -> dict[str, float]:
    return speech_bertscore(emb(cand), emb(ref))


def wins(a: float, b: float) -> str:
    if a >= b + GAP:
        return "hyphen"
    if b >= a + GAP:
        return "compact"
    return "tie"


def main() -> int:
    prev = json.loads(IN.read_text())
    clips = available_clips()
    rows = []
    for rec in prev["rows"]:
        name = rec["ingredient"]
        clip = clips.get(name)
        if clip is None:
            continue
        hp = TTS / f"{slug(name)}.hyphen.wav"
        cp = TTS / f"{slug(name)}.compact.wav"
        if not hp.exists() or not cp.exists():
            continue
        h = prf(hp, clip.path)
        c = prf(cp, clip.path)
        ipa = cloud_ipa_path(name)
        ipa_h = prf(ipa, clip.path) if ipa is not None else None
        row = {
            "ingredient": name,
            "respelling": rec.get("respelling"),
            "hyphen": h,
            "compact": c,
            "cloud_ipa": ipa_h,
            "winner": {
                "precision": wins(h["precision"], c["precision"]),
                "recall": wins(h["recall"], c["recall"]),
                "f1": wins(h["f1"], c["f1"]),
            },
        }
        rows.append(row)
        print(
            f"  {name} P h={h['precision']} c={c['precision']} "
            f"R h={h['recall']} c={c['recall']} "
            f"F1 h={h['f1']} c={c['f1']}",
            flush=True,
        )

    summary: dict = {"n": len(rows), "gap": GAP, "metrics": {}}
    for metric in ("precision", "recall", "f1"):
        hv = [r["hyphen"][metric] for r in rows]
        cv = [r["compact"][metric] for r in rows]
        iv = [r["cloud_ipa"][metric] for r in rows if r.get("cloud_ipa")]
        wc = Counter(r["winner"][metric] for r in rows)
        summary["metrics"][metric] = {
            "mean_hyphen": mean(hv),
            "mean_compact": mean(cv),
            "mean_cloud_ipa": mean(iv) if iv else None,
            "n_cloud_ipa": len(iv),
            "hyphen_wins": wc.get("hyphen", 0),
            "compact_wins": wc.get("compact", 0),
            "ties": wc.get("tie", 0),
            "delta_compact_minus_hyphen": mean([b - a for a, b in zip(hv, cv)]),
        }
        print(metric, json.dumps(summary["metrics"][metric]), flush=True)

    # Does recall (or precision) flip the F1 compact wins that were length penalties?
    f1_compact = [r for r in rows if r["winner"]["f1"] == "compact"]
    summary["among_f1_compact_wins"] = {
        "n": len(f1_compact),
        "recall_prefers_hyphen": sum(1 for r in f1_compact if r["winner"]["recall"] == "hyphen"),
        "precision_prefers_hyphen": sum(1 for r in f1_compact if r["winner"]["precision"] == "hyphen"),
        "recall_still_compact": sum(1 for r in f1_compact if r["winner"]["recall"] == "compact"),
        "precision_still_compact": sum(1 for r in f1_compact if r["winner"]["precision"] == "compact"),
    }
    f1_hyphen = [r for r in rows if r["winner"]["f1"] == "hyphen"]
    summary["among_f1_hyphen_wins"] = {
        "n": len(f1_hyphen),
        "recall_still_hyphen": sum(1 for r in f1_hyphen if r["winner"]["recall"] == "hyphen"),
        "precision_still_hyphen": sum(1 for r in f1_hyphen if r["winner"]["precision"] == "hyphen"),
    }
    summary["best_vs_human"] = {
        m: (
            "compact"
            if summary["metrics"][m]["mean_compact"] > summary["metrics"][m]["mean_hyphen"]
            else "hyphen"
        )
        for m in ("precision", "recall", "f1")
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
