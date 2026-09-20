"""Score one hard-subset synthesis run.

Two internal metrics, neither of which is Synthio's unpublished 0-5 judge:

* phoneme_score: Path 3. The extracted drug span is decoded with
  wav2vec2-lv-60-espeak and compared to every gold IPA variant
  (panphon weighted feature edit distance). 0-5 via rate_to_score.
  Stress is not in this number; the recognizer does not mark it.
  Pass is >= 4.0.
* f1_score: Path 2. SpeechBERTScore F1 (WavLM-Large) against the human
  clip on disk, mapped with `f1_to_score`: 1 at F1 0, 5 at the
  human-vs-human floor (F1 0.747), clamped. Pass is score >= 4.0.
  Items with no clip on disk are not scoreable on this metric.

Writes runs/oss-eval/<condition>/scores.jsonl and summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dose_r.forced_align import extract_drug_span_forced_align
from dose_r.references.reference_clips import available_clips_all
from dose_r.scoring.candidate_eval import score_against_best_reference
from dose_r.scoring.phoneme_distance import best_phoneme_distance, rate_to_score
from dose_r.scoring.speech_similarity import f1_to_score
from dose_r.scoring.phoneme_model import transcribe_phonemes
from oss_eval_items import load_items

PASS = 4.0


def _wav_bytes(path: Path) -> bytes:
    audio, sr = sf.read(path, dtype="int16", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype("int16")
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    return buf.getvalue()


def _clips_by_drug() -> dict[str, list]:
    raw = available_clips_all()
    out: dict[str, list] = {}
    for ingredient, clips in raw.items():
        out.setdefault(ingredient.lower(), []).extend(clips)
    return out


def score_condition(condition: str, wav_dir: Path) -> None:
    items = load_items()
    clips = _clips_by_drug()
    out_dir = ROOT / "runs" / "oss-eval" / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in items:
        wav = wav_dir / f"{item['item_id']}.wav"
        row = {
            "item_id": item["item_id"],
            "drug": item["drug"],
            "tier": item["tier"],
            "condition": condition,
        }
        if not wav.exists():
            row.update(phoneme_scoreable=False, f1_scoreable=False, error="missing wav")
            rows.append(row)
            continue
        try:
            span = extract_drug_span_forced_align(_wav_bytes(wav), item["sentence"], item["drug"])
        except Exception as exc:
            row.update(phoneme_scoreable=False, f1_scoreable=False, error=f"align: {exc}")
            rows.append(row)
            continue
        if span is None:
            row.update(phoneme_scoreable=False, f1_scoreable=False, error="align returned none")
            rows.append(row)
            continue

        ipa = item["ipa_variants"]
        if not ipa:
            row["phoneme_scoreable"] = False
            row["phoneme_error"] = "no gold IPA"
        else:
            decoded = transcribe_phonemes(span)
            dist = best_phoneme_distance(decoded, ipa)
            score = rate_to_score(dist["rate"])
            row.update(
                phoneme_scoreable=True,
                decoded_phonemes=decoded,
                phoneme_rate=dist["rate"],
                phoneme_score=score,
                phoneme_pass=score >= PASS,
                best_ipa=dist["best_variant"],
            )

        drug_clips = clips.get(item["drug"].lower(), [])
        if not drug_clips:
            row["f1_scoreable"] = False
            row["f1_error"] = "no human clip on disk"
        else:
            best = score_against_best_reference(span, drug_clips)
            f1_score = f1_to_score(best.best_f1)
            row.update(
                f1_scoreable=True,
                best_f1=round(best.best_f1, 4),
                best_f1_source=best.best_source,
                f1_by_source={k: round(v, 4) for k, v in best.all_scores.items()},
                f1_score=f1_score,
                f1_pass=f1_score >= PASS,
                n_refs=len(best.all_scores),
            )
        rows.append(row)
        print(f"{item['item_id']}: phoneme={row.get('phoneme_score')} f1={row.get('f1_score')}", flush=True)

    (out_dir / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = _summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _rate(rows, key_scoreable, key_pass):
    scored = [r for r in rows if r.get(key_scoreable)]
    if not scored:
        return None
    return round(sum(1 for r in scored if r.get(key_pass)) / len(scored), 3)


def _summarize(rows: list[dict]) -> dict:
    by_tier = defaultdict(list)
    for r in rows:
        by_tier[r["tier"]].append(r)

    def block(group):
        return {
            "n": len(group),
            "phoneme_scored": sum(1 for r in group if r.get("phoneme_scoreable")),
            "phoneme_pass_rate": _rate(group, "phoneme_scoreable", "phoneme_pass"),
            "f1_scored": sum(1 for r in group if r.get("f1_scoreable")),
            "f1_pass_rate": _rate(group, "f1_scoreable", "f1_pass"),
            "align_fail": sum(1 for r in group if r.get("error")),
        }

    ceiling = by_tier.get("F_ceiling_control", [])
    ear = by_tier.get("A_confirmed_bad_by_ear", [])
    return {
        "note": "Hard subset only. Not a DOSE leaderboard pass rate.",
        "overall": block(rows),
        "ceiling_control": block(ceiling),
        "confirmed_bad_by_ear": [
            {k: r.get(k) for k in ("drug", "phoneme_score", "phoneme_pass", "f1_score", "f1_pass", "error")}
            for r in ear
        ],
        "by_tier": {tier: block(group) for tier, group in sorted(by_tier.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--wav-dir", required=True)
    args = ap.parse_args()
    score_condition(args.condition, Path(args.wav_dir))


if __name__ == "__main__":
    main()
