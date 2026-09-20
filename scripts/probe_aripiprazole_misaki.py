"""Inject one Misaki string for aripiprazole; score Path 2 vs plain."""
from __future__ import annotations

import io
import json
import sys
import wave
from pathlib import Path

import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from kokoro import KPipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PHONE = "ˌæɹɪpˈɪpɹəzOɫ"
VOICE = "af_heart"
OUT = ROOT / "runs" / "oss-eval" / "kokoro-aripiprazole-misaki-probe"


def vocab() -> set[str]:
    path = hf_hub_download("hexgrad/Kokoro-82M", "config.json")
    return set(json.loads(Path(path).read_text())["vocab"])


def replace(tokens, drug: str, phones: str) -> None:
    texts = [t.text for t in tokens]
    for i, text in enumerate(texts):
        if text.lower() == drug.lower():
            tokens[i].phonemes = phones
            return
    raise ValueError(texts)


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


def main() -> None:
    v = vocab()
    missing = sorted({c for c in PHONE if c not in v})
    spoken = PHONE
    note = None
    if missing == ["ɫ"]:
        spoken = PHONE.replace("ɫ", "l")
        note = "ɫ not in Kokoro vocab; spoke with l"
    elif missing:
        raise SystemExit(f"missing from vocab: {missing}")
    print("INPUT", PHONE, flush=True)
    print("SPOKEN", spoken, note or "exact", flush=True)

    row = None
    for line in (ROOT / "data" / "dose_v1.jsonl").open():
        cand = json.loads(line)
        if cand["name"].lower() == "aripiprazole":
            row = cand
            break
    assert row is not None
    sentence = row["sentence"]
    drug = "aripiprazole"
    idx = sentence.lower().find(drug)
    spoken_word = sentence[idx : idx + len(drug)]

    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)

    plain_p = OUT / "plain.wav"
    inj_p = OUT / "injected.wav"
    if not (plain_p.exists() and plain_p.stat().st_size > 1000):
        result = next(pipe(sentence, voice=VOICE, speed=1))
        sf.write(plain_p, result.audio.detach().cpu().numpy(), 24000)
        print("ok plain", flush=True)
    _, tokens = pipe.g2p(sentence)
    replace(tokens, spoken_word, spoken)
    result = next(pipe.generate_from_tokens(tokens, voice=VOICE, speed=1))
    sf.write(inj_p, result.audio.detach().cpu().numpy(), 24000)
    print("ok injected", flush=True)

    from dose_r.forced_align import extract_drug_span_forced_align
    from dose_r.references.reference_clips import available_clips_all
    from dose_r.scoring.candidate_eval import score_against_best_reference

    clips: dict[str, list] = {}
    for key, clist in available_clips_all().items():
        clips.setdefault(key.lower(), []).extend(clist)
    drug_clips = clips.get("aripiprazole", [])
    print("n_refs", len(drug_clips), flush=True)
    out = {"input": PHONE, "spoken": spoken, "note": note, "n_refs": len(drug_clips)}
    for label, path in (("plain", plain_p), ("injected", inj_p)):
        span = extract_drug_span_forced_align(wav_bytes(path), sentence, spoken_word)
        best = score_against_best_reference(span, drug_clips)
        out[f"{label}_f1"] = round(best.best_f1, 4)
        out[f"{label}_source"] = best.best_source
    out["delta"] = round(out["injected_f1"] - out["plain_f1"], 4)
    print(json.dumps(out, indent=2), flush=True)
    (OUT / "score.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
