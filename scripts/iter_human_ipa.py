"""4–5 pass human → IPA → name+sidecar injection on the Path 2 gaps.

Keeps a proposal only when wavlm F1 vs the human clip rises. CTC is a
proposal generator, not gold: tails are trimmed to the sidecar length.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_hf = ROOT / ".cache" / "huggingface"
_hf.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_hf))
os.environ.setdefault("HF_HUB_CACHE", str(_hf / "hub"))

import importlib.util  # noqa: E402

from dose_r.references.human_ipa_loop import propose_round  # noqa: E402
from dose_r.references.tts_pronunciation import (  # noqa: E402
    custom_pronunciation,
    to_cloud_en_us_ipa,
)
from dose_r.scoring.phoneme_model import transcribe_phonemes  # noqa: E402

_eval_spec = importlib.util.spec_from_file_location(
    "eval_uniform_pron_tts", ROOT / "scripts" / "eval_uniform_pron_tts.py"
)
_eval = importlib.util.module_from_spec(_eval_spec)
assert _eval_spec.loader is not None
_eval_spec.loader.exec_module(_eval)
HOLDOUT = _eval.HOLDOUT
cached = _eval.cached
f1 = _eval.f1
token_and_project = _eval.token_and_project
wav_duration_s = _eval.wav_duration_s

PRIOR = ROOT / "runs" / "uniform-pron-tts"
OUT = ROOT / "runs" / "human-ipa-loop"
ROUNDS = 5


def load_prior() -> list[dict]:
    return json.loads((PRIOR / "results.json").read_text())["rows"]


def gap_names(rows: list[dict], extra_worst: int) -> list[str]:
    hold = set()
    if HOLDOUT.exists():
        hold = {
            r["ingredient"]
            for r in json.loads(HOLDOUT.read_text()).get("core_holdout", [])
        }
    worst = [
        r["ingredient"]
        for r in sorted(
            [x for x in rows if x.get("delta_ipa") is not None],
            key=lambda x: x["delta_ipa"],
        )[:extra_worst]
    ]
    ordered: list[str] = []
    for name in list(hold) + worst:
        if name not in ordered:
            ordered.append(name)
    return ordered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--extra-worst", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = {r["ingredient"]: r for r in load_prior()}
    names = [n for n in gap_names(list(rows.values()), args.extra_worst) if n in rows]
    if args.limit:
        names = names[: args.limit]
    print(f"gap names n={len(names)} rounds={args.rounds}", flush=True)

    tok, project = token_and_project()
    tts_dir = OUT / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)

    out_rows = []
    for i, name in enumerate(names, 1):
        rec = rows[name]
        slug = rec["key"]
        human = Path(rec["human_path"]).read_bytes()
        human_ctc = transcribe_phonemes(rec["human_path"])
        sidecar = rec["ipa"]
        best_ipa = sidecar
        best_f1 = float(rec["ipa_f1"])
        best_round = 0
        synth_ctc = None
        history = [
            {
                "round": 0,
                "ipa": sidecar,
                "f1": rec["ipa_f1"],
                "kept": True,
                "note": "current sidecar",
            }
        ]
        for rnd in range(1, args.rounds + 1):
            proposal = to_cloud_en_us_ipa(
                propose_round(best_ipa, human_ctc, synth_ctc, rnd)
            )
            if proposal == history[-1]["ipa"]:
                history.append(
                    {
                        "round": rnd,
                        "ipa": proposal,
                        "f1": history[-1]["f1"],
                        "kept": False,
                        "note": "unchanged",
                    }
                )
                continue
            wav_path = tts_dir / f"{slug}.r{rnd}.wav"
            err = None
            try:
                wav = cached(
                    wav_path,
                    tok,
                    project,
                    text=name,
                    pronunciations=custom_pronunciation(name, proposal),
                )
            except RuntimeError as exc:
                err = str(exc)[:200]
                history.append(
                    {
                        "round": rnd,
                        "ipa": proposal,
                        "f1": None,
                        "kept": False,
                        "note": err,
                    }
                )
                print(f"{i}/{len(names)} {name} r{rnd} FAIL {err}", flush=True)
                continue
            score = f1(wav, human)
            kept = score > best_f1 + 0.005
            if kept:
                best_ipa = proposal
                best_f1 = score
                best_round = rnd
            try:
                synth_ctc = transcribe_phonemes(wav_path)
            except Exception:
                synth_ctc = None
            history.append(
                {
                    "round": rnd,
                    "ipa": proposal,
                    "f1": round(score, 4),
                    "kept": kept,
                    "dur": round(wav_duration_s(wav), 3),
                    "synth_ctc": synth_ctc,
                }
            )
            print(
                f"{i}/{len(names)} {name:28s} r{rnd} "
                f"{score:.3f} {'KEEP' if kept else 'drop'} {proposal}",
                flush=True,
            )

        out_rows.append(
            {
                "ingredient": name,
                "plain": rec["plain"],
                "ipa0": rec["ipa_f1"],
                "ipa_best": round(best_f1, 4),
                "delta_vs_plain0": rec["delta_ipa"],
                "delta_vs_plain_best": round(best_f1 - rec["plain"], 4),
                "delta_vs_ipa0": round(best_f1 - rec["ipa_f1"], 4),
                "ipa0_str": sidecar,
                "ipa_best_str": best_ipa,
                "best_round": best_round,
                "in_holdout": rec.get("in_holdout", False),
                "human_ctc": human_ctc,
                "history": history,
            }
        )

    def mean(key, subset=None):
        xs = subset if subset is not None else out_rows
        vals = [r[key] for r in xs if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    hold = [r for r in out_rows if r["in_holdout"]]
    summary = {
        "n": len(out_rows),
        "rounds": args.rounds,
        "mean_plain": mean("plain"),
        "mean_ipa0": mean("ipa0"),
        "mean_ipa_best": mean("ipa_best"),
        "mean_delta_vs_ipa0": mean("delta_vs_ipa0"),
        "n_improved": sum(1 for r in out_rows if r["delta_vs_ipa0"] > 0.005),
        "n_best_beats_plain": sum(
            1 for r in out_rows if r["delta_vs_plain_best"] > 0.01
        ),
        "holdout_n": len(hold),
        "holdout_plain": mean("plain", hold),
        "holdout_ipa0": mean("ipa0", hold),
        "holdout_ipa_best": mean("ipa_best", hold),
        "best_gains": sorted(out_rows, key=lambda r: r["delta_vs_ipa0"], reverse=True)[
            :8
        ],
        "still_below_plain": sorted(
            [r for r in out_rows if r["delta_vs_plain_best"] < -0.05],
            key=lambda r: r["delta_vs_plain_best"],
        )[:8],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps({"summary": summary, "rows": out_rows}, indent=2)
    )
    skip = {"best_gains", "still_below_plain"}
    print("MEANS", json.dumps({k: summary[k] for k in summary if k not in skip}))
    print("wrote", OUT / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
