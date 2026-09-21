#!/usr/bin/env python3
"""Rounds 4-6: bottom half again, slower Kokoro, sentence F1 for every variant.

Does not write data/gold_gemini_ipa. Resumes from runs/misaki-iter/chosen.json.
Gemini is given the Misaki American phoneme list and must reply in that
alphabet. IPA replies are dropped. Nothing is rewritten into Misaki.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dose_r.forced_align import extract_drug_span_forced_align
from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_iter = _load("misaki_iter", ROOT / "scripts" / "misaki_bottom_iterate.py")

STORE = _iter.STORE
WAV_ROOT = _iter.WAV_ROOT
VOWELS = _iter.VOWELS
START_ROUND = 5
ROUNDS = 2
SPEED = 0.5
ISO_SPEED = 0.45
SEED = {
    "latuda": ["ləˈtudə", "lətˈudə", "ləˈtudɑ", "ləˈtʊdə", "ləˈtudʌ"],
    "veozah": ["vˈiOzə", "viˈɑzə", "ˈviOzə", "viˈɔzə", "vˈiəzə"],
    "resmetirom": ["ɹɛsˈmɛtɪɹəm", "ɹɛsˈmɛtɪɹɔm", "ɹɛsmˈɛtɪɹəm", "ɹɛsˈmɛtəɹəm"],
    "adquey": ["ˈædki", "ædkˈi", "ˈædki", "ædˈki"],
}
# American Misaki only. Not Kokoro's full vocab, not IPA.
US_MISAKI = frozenset("AIWYbdfhijklmnpstuvwzðŋɑɔəɛɜɡɪɹʃʊʌʒʤʧˈˌθᵊOæɾᵻT")
IPA_MARKERS = ("oʊ", "aɪ", "eɪ", "aʊ", "ɔɪ", "əʊ", "ː", "ɒ")
MISAKI_PHONES = """
Misaki American English phonemes. This is the alphabet. Write only these.

Stress: ˈ primary, ˌ secondary. Put the tick immediately BEFORE the vowel.

Consonants: b d f h j k l m n p s t v w z ɡ ŋ ɹ ʃ ʒ ð θ ʤ ʧ T
  j is "y" as in yes => jˈɛs
  ɡ is hard g as in get
  ɹ is r as in red => ɹˈɛd
  ʤ is j/dg as in jump => ʤˈʌmp
  ʧ is ch as in chump => ʧˈʌmp
  T is the flap in butter => bˈʌTəɹ  (do not write ɾ)

Vowels: ə i u ɑ ɔ ɛ ɜ ɪ ʊ ʌ æ O ᵻ ᵊ
  O is American "oh". Never write oʊ.
  A is "ay" in hey => hˈA. Never write eɪ.
  I is "eye" in high => hˈI. Never write aɪ.
  W is "ow" in how => hˈW. Never write aʊ.
  Y is "oy" in soy => sˈY. Never write ɔɪ.
  æ is ash => ˈæʃ
  Do not write a, ɒ, Q, ː, o, e as vowels.

Gold-dictionary examples (word => Misaki, not IPA):
  hey => hˈA
  high => hˈI
  how => hˈW
  soy => sˈY
  flu => flˈu
  spa => spˈɑ
  sun => sˈʌn
  brick => bɹˈɪk
  wood => wˈʊd
  her => hɜɹ
  red => ɹˈɛd
  Kokoro => kˈOkəɹO
  Misaki => misˈɑki
"""


def legal(ps: str, symbols: set[str]) -> bool:
    return bool(ps) and all(ch in symbols or ch.isspace() for ch in ps)


def ending_variants(ps: str, symbols: set[str]) -> list[str]:
    out: list[str] = []

    def add(s: str) -> None:
        s = s.strip()
        if s and s != ps and s not in out and legal(s, symbols):
            out.append(s)

    for piece in ps.split() or [ps]:
        bare = piece.replace("ˈ", "").replace("ˌ", "")
        for i, ch in enumerate(bare):
            if ch in VOWELS:
                add(bare[:i] + "ˈ" + bare[i:])
        last = None
        for i in range(len(piece) - 1, -1, -1):
            if piece[i] in VOWELS:
                last = i
                break
        if last is not None:
            for rep in "əɑɔOʌæʊuɪi":
                add(piece[:last] + rep + piece[last + 1 :])
        if piece.endswith("ɑm"):
            for tail in ("əm", "ɔm", "ʌm", "əm"):
                add(piece[:-2] + tail)
        if piece.endswith("Ozə"):
            for tail in ("ɑzə", "ɔzə", "əzə", "Ozɑ"):
                add(piece[:-3] + tail)
        if piece.endswith("udə"):
            for tail in ("udɑ", "ʊdə", "udʌ", "udə"):
                add(piece[:-3] + tail)
    return out


def wav_to_mp3(wav: Path, mp3: Path) -> Path:
    mp3.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame", "-qscale:a", "5", str(mp3)]
    )
    return mp3


def span_mp3(sent_wav: Path, item: dict, mp3: Path) -> Path | None:
    from dose_r.forced_align import extract_drug_span_forced_align

    span = extract_drug_span_forced_align(sent_wav.read_bytes(), item["sentence"], item["spoken"])
    if span is None:
        span = extract_drug_span_forced_align(sent_wav.read_bytes(), item["sentence"], item["drug"])
    if span is None:
        return None
    mp3.parent.mkdir(parents=True, exist_ok=True)
    tmp = mp3.with_suffix(".wav")
    tmp.write_bytes(span)
    return wav_to_mp3(tmp, mp3)


def gemini_proposal(
    gold_mp3: Path,
    current: str,
    current_f1: float,
    current_mp3: Path,
    history: list[dict],
) -> str | None:
    import base64

    try:
        tok = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

    def mp3_part(path: Path) -> dict:
        return {"inlineData": {"mimeType": "audio/mpeg", "data": base64.b64encode(path.read_bytes()).decode()}}

    lines = [
        "Listen in order. Do not output IPA. We already have IPA.",
        "Output one new Misaki string for the TARGET, same alphabet as Misaki US gold.",
        "Each iteration below is Kokoro. Use the F1 scores and the mp3s together:",
        "a higher F1 is closer to the TARGET. Do not repeat a string that already scored worse.",
        MISAKI_PHONES,
        f"Current best Misaki: {current}  F1={current_f1:.3f}",
    ]
    for h in history:
        lines.append(
            f"Iteration {h['round']}: Misaki {h['phonemes']}  F1={h['f1']:.3f}"
        )
    lines.append('JSON only: {"phonemes":"..."}')
    parts = [{"text": "\n".join(lines)}, {"text": "TARGET mp3:"}, mp3_part(gold_mp3)]
    for h in history:
        parts.append({"text": f"Iteration {h['round']} mp3. Misaki {h['phonemes']}. F1 {h['f1']:.3f}:"})
        parts.append(mp3_part(Path(h["mp3"])))
    parts.append({"text": f"CURRENT best mp3. Misaki {current}. F1 {current_f1:.3f}:"})
    parts.append(mp3_part(current_mp3))
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.2},
    }
    url = (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/"
        "project-amer-scs-sandbox/locations/us-central1/publishers/google/"
        "models/gemini-2.5-flash:generateContent"
    )
    try:
        import requests

        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
            json=body,
            timeout=120,
        )
        if resp.status_code != 200:
            print("gemini", resp.status_code, resp.text[:240], flush=True)
            return None
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            return None
        raw = (json.loads(text[start : end + 1]).get("phonemes") or "").strip().strip("/")
        if any(m in raw for m in IPA_MARKERS):
            print("gemini ipa dropped", raw, flush=True)
            return None
        if not legal(raw, US_MISAKI):
            bad = sorted({ch for ch in raw if ch not in US_MISAKI and not ch.isspace()})
            print("gemini not misaki", raw, "bad", bad, flush=True)
            return None
        return raw
    except Exception as exc:
        print("gemini skip", exc, flush=True)
    return None


def main() -> None:
    chosen = json.loads((STORE / "chosen.json").read_text())
    best_phones = dict(chosen["phones"])
    best_f1 = {k: float(v) for k, v in chosen["f1"].items()}
    items = _iter.load_full_items()
    by_slug = {it["slug"]: it for it in items}
    symbols = _iter.vocab() | set("T ")
    best_dir = WAV_ROOT / "kokoro-full-best"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
    hist_path = STORE / "gemini-history.json"
    history: dict[str, list[dict]] = json.loads(hist_path.read_text()) if hist_path.exists() else {}
    gold_mp3_dir = STORE / "gemini-mp3" / "gold"
    print(
        f"resume n={len(best_f1)} mean={sum(best_f1.values())/len(best_f1):.3f} speed={SPEED}",
        flush=True,
    )

    for offset in range(ROUNDS):
        rnd = START_ROUND + offset
        ranked = sorted(best_f1, key=lambda k: best_f1[k])
        half = ranked[: max(1, len(ranked) // 2)]
        print(f"round {rnd} bottom {len(half)}", flush=True)
        trial = STORE / f"round{rnd}"
        improved = 0
        for n, key in enumerate(half, 1):
            it = by_slug[key]
            gold = _iter.gold_wav(it["drug"])
            if gold is None:
                continue
            current = best_phones[key]
            cands: list[str] = [current]
            for extra in SEED.get(key, []):
                if extra not in cands:
                    cands.append(extra)
            for v in ending_variants(current, symbols):
                if v not in cands:
                    cands.append(v)
            cands = cands[:8]
            gold_mp3 = gold_mp3_dir / f"{key}.mp3"
            if not gold_mp3.exists():
                wav_to_mp3(gold, gold_mp3)
            cur_wav = trial / "sent" / f"{key}-cur.wav"
            cur_mp3 = trial / "mp3" / f"{key}-cur.mp3"
            try:
                _iter.write_sentence(pipeline, it, current, cur_wav, speed=SPEED)
                if span_mp3(cur_wav, it, cur_mp3) is None:
                    wav_to_mp3(cur_wav, cur_mp3)
                proposal = gemini_proposal(
                    gold_mp3,
                    current,
                    best_f1[key],
                    cur_mp3,
                    history.get(key, []),
                )
            except Exception as exc:
                print("gemini context fail", key, exc, flush=True)
                proposal = None
            if proposal and proposal not in cands:
                cands.append(proposal)
                _iter.log_row({
                    "round": rnd,
                    "slug": key,
                    "drug": it["drug"],
                    "phonemes": proposal,
                    "source": "gemini",
                    "kept": False,
                })
            best_local = None
            for j, phones in enumerate(cands):
                sent = trial / "sent" / f"{key}-{j}.wav"
                try:
                    _iter.write_sentence(pipeline, it, phones, sent, speed=SPEED)
                    f1 = _iter.sentence_f1(sent, it, gold)
                except Exception as exc:
                    print("sent fail", key, phones, exc, flush=True)
                    continue
                if f1 is None:
                    continue
                _iter.log_row({
                    "round": rnd,
                    "slug": key,
                    "drug": it["drug"],
                    "phonemes": phones,
                    "sentence_f1": round(f1, 4),
                    "previous_f1": round(best_f1[key], 4),
                    "speed": SPEED,
                    "source": "gemini" if phones == proposal else "variant",
                    "kept": False,
                })
                if best_local is None or f1 > best_local[0]:
                    best_local = (f1, phones, sent)
            if best_local is None:
                continue
            f1, phones, sent = best_local
            kept = f1 > best_f1[key] + 0.005
            _iter.log_row({
                "round": rnd,
                "slug": key,
                "drug": it["drug"],
                "phonemes": phones,
                "sentence_f1": round(f1, 4),
                "previous_f1": round(best_f1[key], 4),
                "speed": SPEED,
                "kept": kept,
            })
            if kept:
                best_f1[key] = f1
                best_phones[key] = phones
                (best_dir / f"{key}.wav").write_bytes(sent.read_bytes())
                improved += 1
            rec_mp3 = trial / "mp3" / f"{key}-iter.mp3"
            if span_mp3(sent, it, rec_mp3) is None:
                wav_to_mp3(sent, rec_mp3)
            history.setdefault(key, []).append({
                "round": rnd,
                "phonemes": phones,
                "f1": round(f1, 4),
                "mp3": str(rec_mp3),
                "kept": kept,
            })
            hist_path.write_text(json.dumps(history))
            if n % 10 == 0:
                print(f"  {n}/{len(half)} improved={improved} last={key} {f1:.3f}", flush=True)
        mean = sum(best_f1.values()) / len(best_f1)
        print(f"round {rnd} done improved={improved} mean={mean:.3f}", flush=True)
        (STORE / "chosen.json").write_text(json.dumps(
            {
                "round": rnd,
                "speed": SPEED,
                "phones": best_phones,
                "f1": {k: round(v, 4) for k, v in best_f1.items()},
            },
            indent=2,
        ))

    vals = list(best_f1.values())
    summary = {
        "n": len(vals),
        "mean_sentence_f1": round(sum(vals) / len(vals), 4),
        "median_sentence_f1": round(sorted(vals)[len(vals) // 2], 4),
        "pass_f1_ge_0_70": sum(v >= 0.70 for v in vals),
        "rounds": f"{START_ROUND}-{START_ROUND + ROUNDS - 1}",
        "speed": SPEED,
    }
    (STORE / "summary-r4.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
