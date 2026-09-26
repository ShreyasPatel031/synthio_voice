"""4–5 pass human → IPA → name+sidecar injection on the Path 2 gaps.

DEAD as a way to invent IPA. Do not G2P respelling or paste CTC as gold.
Source-published IPA only. See dose_r/references/README.md.
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

from dose_r.references.human_ipa_loop import (  # noqa: E402
    propose_round,
    sanitize_cloud_ipa,
    schwa_to_wedge,
)
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
LOOP0 = ROOT / "runs" / "human-ipa-loop" / "results.json"
OUT = ROOT / "runs" / "human-ipa-loop2"
REFS = ROOT / "dose_r" / "references" / "references.jsonl"
ROUNDS = 5


def load_prior() -> list[dict]:
    return json.loads((PRIOR / "results.json").read_text())["rows"]


def official_for(name: str) -> list[str]:
    for line in REFS.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("ingredient", "").lower() == name.lower():
            return [to_cloud_en_us_ipa(v) for v in rec.get("ipa_variants") or [] if v]
    return []


def previous_best() -> dict[str, dict]:
    if not LOOP0.exists():
        return {}
    return {
        r["ingredient"]: r
        for r in json.loads(LOOP0.read_text()).get("rows", [])
    }


def open_gaps(rows: list[dict], prev: dict[str, dict]) -> list[str]:
    """Names still below 0.70, or still losing to plain, or hard holdout."""
    out: list[str] = []
    for r in rows:
        p = prev.get(r["ingredient"], {})
        best = p.get("ipa_best", r.get("ipa_f1"))
        dplain = (best - r["plain"]) if best is not None else r.get("delta_ipa") or 0
        hold = r.get("in_holdout")
        if best is None:
            continue
        if best < 0.70 or dplain < -0.03 or (hold and best < 0.72):
            out.append(r["ingredient"])
    return out


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
    ap.add_argument("--all-holdout", action="store_true")
    args = ap.parse_args()

    prior_rows = load_prior()
    rows = {r["ingredient"]: r for r in prior_rows}
    prev = previous_best()
    if args.all_holdout:
        names = [n for n in gap_names(prior_rows, args.extra_worst) if n in rows]
    else:
        names = [n for n in open_gaps(prior_rows, prev) if n in rows]
    if args.limit:
        names = names[: args.limit]
    print(f"open gaps n={len(names)} rounds={args.rounds}", flush=True)

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
        prev_rec = prev.get(name, {})
        best_ipa = prev_rec.get("ipa_best_str") or sidecar
        best_f1 = float(prev_rec.get("ipa_best") or rec["ipa_f1"])
        best_round = 0
        synth_ctc = None
        official = [
            v
            for v in official_for(name)
            if sanitize_cloud_ipa(v, sidecar) not in {sidecar, best_ipa}
        ]
        history = [
            {
                "round": 0,
                "ipa": best_ipa,
                "f1": best_f1,
                "kept": True,
                "note": "resume" if prev_rec else "current sidecar",
            }
        ]
        for rnd in range(1, args.rounds + 1):
            raw = propose_round(
                best_ipa, human_ctc, synth_ctc, rnd, official=official
            )
            proposal = sanitize_cloud_ipa(raw, sidecar)
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
                retry = schwa_to_wedge(proposal)
                if retry != proposal:
                    try:
                        wav_path = tts_dir / f"{slug}.r{rnd}b.wav"
                        wav = cached(
                            wav_path,
                            tok,
                            project,
                            text=name,
                            pronunciations=custom_pronunciation(name, retry),
                        )
                        proposal = retry
                    except RuntimeError as exc2:
                        err = str(exc2)[:200]
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
                else:
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
                "ipa_resume": float(prev_rec.get("ipa_best") or rec["ipa_f1"]),
                "ipa_best": round(best_f1, 4),
                "delta_vs_plain0": rec["delta_ipa"],
                "delta_vs_plain_best": round(best_f1 - rec["plain"], 4),
                "delta_vs_ipa0": round(best_f1 - rec["ipa_f1"], 4),
                "delta_vs_resume": round(
                    best_f1 - float(prev_rec.get("ipa_best") or rec["ipa_f1"]), 4
                ),
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
