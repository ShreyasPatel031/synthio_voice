"""Probe Kokoro plain vs user-corrected IPA on five DOSE names.

Source-published IPA only. Do not G2P DailyMed/USAN respelling.
See dose_r/references/README.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from kokoro import KPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VOICE = "af_heart"

# User-supplied corrected IPA (slashes/brackets stripped below).
TARGETS = {
    "Attruby": "æˈtruːbi",
    "Dupixent": "ˈduːpɪksɛnt",
    "Humira": "hjuːˈmɛr.ə",
    "Nexium": "ˈnɛksiəm",
    "aripiprazole": "ˌæɹ.ɪˈpɪp.ɹəˌzoʊl",
}


def _vocab() -> set[str]:
    path = hf_hub_download("hexgrad/Kokoro-82M", "config.json")
    return set(json.loads(Path(path).read_text())["vocab"])


def ipa_to_misaki(raw: str, vocab: set[str]) -> str:
    """Map corrected IPA into Kokoro's American Misaki alphabet."""
    s = raw.strip().strip("/[]")
    s = s.replace("'", "ˈ").replace(".", "")
    # Longest diphthong / affricate first (Misaki single symbols).
    for old, new in (
        ("tʃ", "ʧ"),
        ("dʒ", "ʤ"),
        ("eɪ", "A"),
        ("aɪ", "I"),
        ("aʊ", "W"),
        ("ɔɪ", "Y"),
        ("oʊ", "O"),
        ("əʊ", "O"),
        ("ɝ", "ɜɹ"),
        ("ɚ", "əɹ"),
    ):
        s = s.replace(old, new)
    s = s.replace("ː", "").replace("r", "ɹ").replace("g", "ɡ").replace(" ", "")
    missing = sorted({ch for ch in s if ch not in vocab})
    if missing:
        raise ValueError(f"not in Misaki vocab: {missing} from {raw!r} -> {s!r}")
    return s


def _replace_drug_phonemes(tokens, drug: str, phones: str) -> None:
    words = drug.split()
    texts = [t.text for t in tokens]
    for i in range(len(tokens) - len(words) + 1):
        if [t.lower() for t in texts[i : i + len(words)]] == [w.lower() for w in words]:
            tokens[i].phonemes = phones
            for extra in tokens[i + 1 : i + len(words)]:
                extra.phonemes = ""
            return
    raise ValueError(f"could not locate {drug!r} in tokens {texts}")


def _load_rows() -> list[dict]:
    by_name = {}
    for line in (ROOT / "data" / "dose_v1.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            by_name[row["name"].lower()] = row
    items = []
    for drug, ipa in TARGETS.items():
        row = by_name[drug.lower()]
        sentence = row["sentence"]
        idx = sentence.lower().find(drug.lower())
        spoken = sentence[idx : idx + len(drug)]
        items.append(
            {
                "item_id": re.sub(r"[^a-z0-9]+", "-", drug.lower()).strip("-"),
                "drug": drug,
                "spoken": spoken,
                "sentence": sentence,
                "ipa": ipa,
            }
        )
    return items


def synthesize(out: Path) -> dict[str, dict]:
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
    vocab = _vocab()
    meta = {}
    for item in _load_rows():
        misaki = ipa_to_misaki(item["ipa"], vocab)
        print("CONVERT", item["drug"], item["ipa"], "->", misaki, flush=True)
        plain_path = out / f"{item['item_id']}__plain.wav"
        ipa_path = out / f"{item['item_id']}__corrected.wav"
        if not (plain_path.exists() and plain_path.stat().st_size > 1000):
            result = next(pipeline(item["sentence"], voice=VOICE, speed=1))
            sf.write(plain_path, result.audio.detach().cpu().numpy(), 24000)
            print("ok plain", item["item_id"], flush=True)
        if not (ipa_path.exists() and ipa_path.stat().st_size > 1000):
            _, tokens = pipeline.g2p(item["sentence"])
            _replace_drug_phonemes(tokens, item["spoken"], misaki)
            result = next(pipeline.generate_from_tokens(tokens, voice=VOICE, speed=1))
            sf.write(ipa_path, result.audio.detach().cpu().numpy(), 24000)
            print("ok corrected", item["item_id"], flush=True)
        meta[item["item_id"]] = {
            "drug": item["drug"],
            "ipa": item["ipa"],
            "misaki": misaki,
            "sentence": item["sentence"],
            "spoken": item["spoken"],
            "plain_wav": str(plain_path),
            "corrected_wav": str(ipa_path),
        }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def score(meta: dict[str, dict]) -> list[dict]:
    import io
    import wave

    from dose_r.forced_align import extract_drug_span_forced_align
    from dose_r.references.reference_clips import available_clips_all
    from dose_r.scoring.candidate_eval import score_against_best_reference

    def wav_bytes(path: Path) -> bytes:
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

    clips = {}
    for ingredient, clist in available_clips_all().items():
        clips.setdefault(ingredient.lower(), []).extend(clist)

    rows = []
    for item_id, info in meta.items():
        drug_clips = clips.get(info["drug"].lower(), [])
        row = {
            "item_id": item_id,
            "drug": info["drug"],
            "ipa": info["ipa"],
            "misaki": info["misaki"],
            "n_refs": len(drug_clips),
        }
        if not drug_clips:
            row["error"] = "no human clip"
            rows.append(row)
            continue
        for label, key in (("plain", "plain_wav"), ("corrected", "corrected_wav")):
            span = extract_drug_span_forced_align(
                wav_bytes(Path(info[key])), info["sentence"], info["spoken"]
            )
            if span is None:
                row[f"{label}_f1"] = None
                row[f"{label}_error"] = "align failed"
                continue
            best = score_against_best_reference(span, drug_clips)
            row[f"{label}_f1"] = round(best.best_f1, 4)
            row[f"{label}_source"] = best.best_source
        if row.get("plain_f1") is not None and row.get("corrected_f1") is not None:
            row["delta"] = round(row["corrected_f1"] - row["plain_f1"], 4)
        rows.append(row)
        print(
            f"{info['drug']}: plain={row.get('plain_f1')} "
            f"corrected={row.get('corrected_f1')} delta={row.get('delta')}",
            flush=True,
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / "oss-eval" / "kokoro-corrected-ipa-probe"))
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    meta = synthesize(out)
    if args.score:
        rows = score(meta)
        (out / "scores.json").write_text(json.dumps(rows, indent=2))
        scored = [r for r in rows if r.get("delta") is not None]
        if scored:
            mean_p = sum(r["plain_f1"] for r in scored) / len(scored)
            mean_c = sum(r["corrected_f1"] for r in scored) / len(scored)
            print(
                f"n={len(scored)} plain_mean={mean_p:.3f} "
                f"corrected_mean={mean_c:.3f} delta={mean_c - mean_p:+.3f}"
            )


if __name__ == "__main__":
    main()
