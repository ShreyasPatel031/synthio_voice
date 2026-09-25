#!/usr/bin/env python3
"""Compare forced-align auto spans vs manually verified boundaries on fixed sentence WAVs.

Keeps gold IPA, reference wav, scorer, and sentence synthesis fixed. Reports
sentence-span WavLM F1 (SpeechBERTScore) separately from CTC-decoded phones.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import re
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dose_r.forced_align import extract_drug_span_forced_align
from dose_r.scoring.phoneme_distance import normalize_phonemes
from dose_r.scoring.phoneme_model import locate_drug_phonemes, transcribe_phonemes
from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore

STORE = ROOT / "runs" / "misaki-iter"
OUT_ROOT = STORE / "span-investigation"
GOLD_DIR = ROOT / "data" / "gold_gemini_ipa" / "wavs"
DOSE = ROOT / "data" / "dose_v1.jsonl"
PAD_SWEEP = (-0.05, 0.0, 0.05, 0.10, 0.15)
_TOKEN_RE = re.compile(r"\S+")
_TARGET_SR = 16_000
_MAX_GAP_RECOVERY_S = 0.15


@dataclass
class Case:
    slug: str
    label: str
    pin: str
    sent_wav: Path
    note: str = ""


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_item(slug_name: str) -> dict:
    for line in DOSE.read_text().splitlines():
        row = json.loads(line)
        ingredients = row.get("ingredients") or [row["name"]]
        spans = row.get("spans") or []
        sentence = row["sentence"]
        for i, ing in enumerate(ingredients):
            s = slug(ing)
            if s != slug_name:
                continue
            if i < len(spans) and isinstance(spans[i], (list, tuple)) and len(spans[i]) == 2:
                a, b = spans[i]
                spoken = sentence[a:b]
            else:
                idx = sentence.lower().find(ing.lower())
                if idx < 0:
                    continue
                spoken = sentence[idx : idx + len(ing)]
            return {"drug": ing, "spoken": spoken, "sentence": sentence, "slug": s}
    raise KeyError(slug_name)


def gold_wav(slug_name: str) -> Path:
    p = GOLD_DIR / f"{slug_name}.wav"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def wav_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def crop_wav(audio_bytes: bytes, start_s: float, end_s: float) -> bytes:
    with wave.open(io.BytesIO(audio_bytes), "rb") as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        nframes = w.getnframes()
        start_f = max(0, int(start_s * rate))
        end_f = min(nframes, int(end_s * rate))
        w.setpos(start_f)
        frames = w.readframes(end_f - start_f)
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(frames)
    return out.getvalue()


def wavlm_f1(span: bytes, gold: Path) -> float:
    return float(
        speech_bertscore(
            extract_frame_embeddings(span),
            extract_frame_embeddings(gold.read_bytes()),
        )["f1"]
    )


def decode_span(span: bytes) -> dict[str, str]:
    raw = transcribe_phonemes(span)
    return {"raw": raw, "norm": normalize_phonemes(raw)}


def align_debug(audio_bytes: bytes, sentence: str, drug: str) -> dict | None:
    """Return forced-align frame/time boundaries for auto, raw, and full-gap modes."""
    import torch
    import torchaudio

    from dose_r.forced_align import (
        _build_target_sequence,
        _decode_segments,
        _locate_drug_word_indices,
    )
    from dose_r.scoring import phoneme_model

    torch_m, processor, model = phoneme_model._get_model()
    vocab = processor.tokenizer.get_vocab()
    blank_id = processor.tokenizer.pad_token_id

    word_idx = _locate_drug_word_indices(sentence, drug)
    if word_idx is None:
        return None
    target_ids, word_of_token = _build_target_sequence(sentence, vocab)
    drug_token_positions = [i for i, w in enumerate(word_of_token) if w in word_idx]
    if not drug_token_positions:
        return None

    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=_TARGET_SR, mono=True)
    duration_s = len(audio) / _TARGET_SR
    inputs = processor(audio, sampling_rate=_TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        log_probs = torch.log_softmax(model(inputs.input_values).logits, dim=-1)
    targets = torch.tensor([target_ids], dtype=torch.int32)
    input_lengths = torch.tensor([log_probs.shape[1]])
    target_lengths = torch.tensor([len(target_ids)])
    aligned_tokens, _ = torchaudio.functional.forced_align(
        log_probs, targets, input_lengths, target_lengths, blank=blank_id,
    )
    frame_labels = aligned_tokens[0].tolist()
    segments = _decode_segments(frame_labels, blank_id)
    if len(segments) != len(target_ids):
        return None

    frame_stride = duration_s / log_probs.shape[1]
    first_pos, last_pos = min(drug_token_positions), max(drug_token_positions)
    drug_start_frame = segments[first_pos][1]
    drug_end_frame = segments[last_pos][2]

    raw_start_s = drug_start_frame * frame_stride
    raw_end_s = drug_end_frame * frame_stride

    if first_pos > 0:
        prev_end_frame = segments[first_pos - 1][2]
        half_gap_s = (drug_start_frame - prev_end_frame) / 2 * frame_stride
        auto_start_s = drug_start_frame * frame_stride - min(half_gap_s, _MAX_GAP_RECOVERY_S)
        full_start_s = drug_start_frame * frame_stride - half_gap_s
    else:
        auto_start_s = max(0.0, raw_start_s - 0.03)
        full_start_s = auto_start_s
    if last_pos < len(segments) - 1:
        next_start_frame = segments[last_pos + 1][1]
        half_gap_s = (next_start_frame - drug_end_frame) / 2 * frame_stride
        auto_end_s = drug_end_frame * frame_stride + min(half_gap_s, _MAX_GAP_RECOVERY_S)
        full_end_s = drug_end_frame * frame_stride + half_gap_s
    else:
        auto_end_s = min(duration_s, raw_end_s + 0.03)
        full_end_s = auto_end_s

    return {
        "duration_s": round(duration_s, 4),
        "drug_token_count": len(drug_token_positions),
        "raw": (max(0.0, raw_start_s), min(duration_s, raw_end_s)),
        "auto": (max(0.0, auto_start_s), min(duration_s, auto_end_s)),
        "full_gap": (max(0.0, full_start_s), min(duration_s, full_end_s)),
    }


def char_proportional_bounds(sentence: str, spoken: str, duration_s: float) -> tuple[float, float]:
    start = sentence.lower().find(spoken.lower())
    if start < 0:
        start = sentence.lower().find(spoken.split()[0].lower())
    end = start + len(spoken)
    return start / len(sentence) * duration_s, end / len(sentence) * duration_s


def energy_verified_bounds(
    audio_bytes: bytes, hint_start: float, hint_end: float, expand: float = 0.35
) -> tuple[float, float]:
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    dur = len(y) / sr
    lo = max(0.0, hint_start - expand)
    hi = min(dur, hint_end + expand)
    seg = y[int(lo * sr) : int(hi * sr)]
    if len(seg) < 10:
        return hint_start, hint_end
    yt, idx = librosa.effects.trim(seg, top_db=22)
    if len(yt) < 10:
        return hint_start, hint_end
    return lo + idx[0] / sr, lo + idx[1] / sr


def recognizer_hint_bounds(
    audio_bytes: bytes, sentence: str, spoken: str, duration_s: float
) -> tuple[float, float] | None:
    """Map CTC-located drug phoneme span to coarse time bounds via token proportion."""
    full_raw = transcribe_phonemes(audio_bytes)
    drug_ph = locate_drug_phonemes(sentence, spoken, full_raw)
    if not drug_ph:
        drug_ph = locate_drug_phonemes(sentence, spoken.split()[0], full_raw)
    if not drug_ph:
        return None
    full_tokens = full_raw.split()
    drug_tokens = drug_ph.split()
    if not drug_tokens:
        return None
    # find first matching subsequence
    start_idx = None
    for i in range(len(full_tokens) - len(drug_tokens) + 1):
        if full_tokens[i : i + len(drug_tokens)] == drug_tokens:
            start_idx = i
            break
    if start_idx is None:
        from difflib import SequenceMatcher

        sm = SequenceMatcher(None, full_tokens, drug_tokens)
        match = sm.find_longest_match(0, 0, len(full_tokens), len(drug_tokens))
        if match.size < max(2, len(drug_tokens) // 2):
            return None
        start_idx = match.a
        end_idx = match.a + match.size
    else:
        end_idx = start_idx + len(drug_tokens)
    n = max(len(full_tokens), 1)
    return (start_idx / n) * duration_s, (end_idx / n) * duration_s


def evaluate_crop(
    audio_bytes: bytes,
    start_s: float,
    end_s: float,
    gold: Path,
    label: str,
    out_dir: Path,
) -> dict:
    span = crop_wav(audio_bytes, start_s, end_s)
    dur = (end_s - start_s)
    wav_path = out_dir / f"{label}.wav"
    mp3_path = out_dir / f"{label}.mp3"
    wav_path.write_bytes(span)
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
            "-codec:a", "libmp3lame", "-qscale:a", "5", str(mp3_path),
        ]
    )
    phones = decode_span(span)
    return {
        "label": label,
        "start_s": round(start_s, 4),
        "end_s": round(end_s, 4),
        "dur_s": round(dur, 4),
        "sentence_span_wavlm_f1": round(wavlm_f1(span, gold), 4),
        "ctc_phones_raw": phones["raw"],
        "ctc_phones_norm": phones["norm"],
        "wav": str(wav_path.relative_to(ROOT)),
        "mp3": str(mp3_path.relative_to(ROOT)),
    }


def run_case(case: Case) -> dict:
    item = load_item(case.slug)
    gold = gold_wav(case.slug)
    audio_bytes = case.sent_wav.read_bytes()
    sent_dur = wav_duration(case.sent_wav)
    out_dir = OUT_ROOT / case.slug / case.label
    out_dir.mkdir(parents=True, exist_ok=True)

    dbg = align_debug(audio_bytes, item["sentence"], item["spoken"])
    if dbg is None:
        dbg = align_debug(audio_bytes, item["sentence"], item["drug"])
    if dbg is None:
        raise RuntimeError(f"alignment debug failed for {case.slug}/{case.label}")

    auto_span = extract_drug_span_forced_align(audio_bytes, item["sentence"], item["spoken"])
    if auto_span is None:
        auto_span = extract_drug_span_forced_align(audio_bytes, item["sentence"], item["drug"])
    auto_repro = None
    if auto_span is not None:
        auto_repro = round(wavlm_f1(auto_span, gold), 4)

    crops: list[dict] = []
    bounds: dict[str, tuple[float, float]] = {
        "auto": dbg["auto"],
        "raw_spike": dbg["raw"],
        "full_gap": dbg["full_gap"],
    }
    cp = char_proportional_bounds(item["sentence"], item["spoken"], sent_dur)
    bounds["char_proportional"] = cp
    bounds["energy_verified"] = energy_verified_bounds(audio_bytes, *dbg["auto"])
    rec = recognizer_hint_bounds(audio_bytes, item["sentence"], item["spoken"], sent_dur)
    if rec is not None:
        bounds["recognizer_proportion"] = rec
        bounds["energy_on_recognizer"] = energy_verified_bounds(audio_bytes, *rec)

    for name, (st, en) in bounds.items():
        crops.append(evaluate_crop(audio_bytes, st, en, gold, name, out_dir))

    verified = bounds["energy_verified"]
    for pad in PAD_SWEEP:
        label = f"verified_pad_{pad:+.2f}".replace("+", "p").replace("-", "m")
        crops.append(
            evaluate_crop(
                audio_bytes,
                max(0.0, verified[0] + pad),
                min(sent_dur, verified[1] + pad),
                gold,
                label,
                out_dir,
            )
        )

    full_decode = transcribe_phonemes(audio_bytes)
    drug_decode = locate_drug_phonemes(item["sentence"], item["spoken"], full_decode)

    return {
        "slug": case.slug,
        "drug": item["drug"],
        "label": case.label,
        "pin": case.pin,
        "note": case.note,
        "sentence": item["sentence"],
        "spoken": item["spoken"],
        "sent_wav": str(case.sent_wav.relative_to(ROOT)),
        "sent_dur_s": round(sent_dur, 4),
        "gold_wav": str(gold.relative_to(ROOT)),
        "align_debug": dbg,
        "auto_span_repro_wavlm_f1": auto_repro,
        "sentence_ctc_full": full_decode,
        "sentence_ctc_drug_located": drug_decode,
        "bounds": {k: {"start_s": round(v[0], 4), "end_s": round(v[1], 4)} for k, v in bounds.items()},
        "crops": crops,
    }


def synth_sentence(slug_name: str, pin: str, dest: Path, speed: float = 1.0) -> None:
    from kokoro import KPipeline

    item = load_item(slug_name)
    spoken = item["spoken"]
    sentence = item["sentence"]
    if spoken not in sentence:
        raise ValueError(f"{spoken!r} not in sentence")
    text = sentence.replace(spoken, f"[{spoken}](/{pin}/)", 1)
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cpu")
    result = next(pipeline(text, voice="af_heart", speed=speed))
    if result.audio is None:
        raise RuntimeError("synthesis failed")
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, result.audio.detach().cpu().numpy(), 24000)


def default_cases() -> list[Case]:
    synth_dir = OUT_ROOT / "_synth"
    cases: list[Case] = []

    def add(slug_name: str, label: str, pin: str, wav: Path, note: str = "") -> None:
        cases.append(Case(slug_name, label, pin, wav, note))

    add(
        "icotyde",
        "pin_space_user_sent",
        "ˌI kˈOtId",
        STORE / "user-sent" / "icotyde.wav",
        "misaki_user_rescore pin; user-span source",
    )
    add(
        "icotyde",
        "stored_round3_sent",
        "IkˈOtId",
        STORE / "round3" / "sent" / "icotyde.wav",
        "chosen.json stored phones; likely cloud-span source",
    )
    icotyde_pref = synth_dir / "icotyde_pref_tide_break.wav"
    if not icotyde_pref.exists():
        synth_sentence("icotyde", "ˌIkˈO tId", icotyde_pref)
    add(
        "icotyde",
        "pref_tide_break_synth",
        "ˌIkˈO tId",
        icotyde_pref,
        "user preferred candidate; freshly synthesized sentence",
    )

    add(
        "idvynso",
        "pin_space_user_sent",
        "ɪdvˈɪn sˌO",
        STORE / "user-sent" / "idvynso.wav",
        "misaki_user_rescore pin",
    )

    add(
        "vorasidenib",
        "pin_new_space_user_sent",
        "vˌɔɹə sˈɪdənɪb",
        STORE / "user-sent" / "vorasidenib.wav",
        "current user pin with syllable break",
    )
    vora_old = synth_dir / "vorasidenib_old.wav"
    if not vora_old.exists():
        synth_sentence("vorasidenib", "vɔɹəsˈɪdənɪb", vora_old)
    add(
        "vorasidenib",
        "pin_old_no_space_synth",
        "vɔɹəsˈɪdənɪb",
        vora_old,
        "previous stored pin before user pin",
    )
    vora_sec = synth_dir / "vorasidenib_sec_no_space.wav"
    if not vora_sec.exists():
        synth_sentence("vorasidenib", "ˌvɔɹəsˈɪdənɪb", vora_sec)
    add(
        "vorasidenib",
        "pin_sec_no_space_synth",
        "ˌvɔɹəsˈɪdənɪb",
        vora_sec,
        "secondary stress on VOR, no syllable break",
    )

    advair_sent = STORE / "user-sent" / "advair.wav"
    if not advair_sent.exists():
        synth_sentence("advair", "ˈædvɛɹ", advair_sent)
    add(
        "advair",
        "stored_pin_sent",
        "ˈædvɛɹ",
        advair_sent,
        "chosen.json stored phones; sentence wav synthesized if missing",
    )
    return cases


def _crop_audio_src(mp3: str) -> str:
    p = Path(mp3)
    if not p.is_absolute():
        p = ROOT / p
    return p.relative_to(OUT_ROOT.resolve()).as_posix()


def write_html(report: list[dict], dest: Path) -> None:
    sections = []
    for row in report:
        crop_rows = []
        for c in sorted(row["crops"], key=lambda x: x["label"]):
            crop_rows.append(
                f"""<tr>
<td><code>{html.escape(c['label'])}</code></td>
<td>{c['dur_s']:.3f}s</td>
<td>{c['start_s']:.3f}–{c['end_s']:.3f}</td>
<td><strong>{c['sentence_span_wavlm_f1']:.3f}</strong></td>
<td><code>{html.escape(c['ctc_phones_raw'])}</code></td>
<td><audio controls preload="none" src="{html.escape(_crop_audio_src(c['mp3']))}"></audio></td>
</tr>"""
            )
        sections.append(
            f"""<article class="card">
<h2>{html.escape(row['drug'])} — {html.escape(row['label'])}</h2>
<p class="meta">pin <code>{html.escape(row['pin'])}</code> · sent <code>{html.escape(row['sent_wav'])}</code> ({row['sent_dur_s']:.2f}s)</p>
<p class="meta">{html.escape(row.get('note') or '')}</p>
<p class="meta">auto-span reproduce WavLM F1: <strong>{row.get('auto_span_repro_wavlm_f1')}</strong></p>
<p class="meta">full-sentence CTC drug locate: <code>{html.escape(row.get('sentence_ctc_drug_located') or '')}</code></p>
<table>
<thead><tr><th>crop</th><th>dur</th><th>bounds</th><th>sentence-span WavLM F1</th><th>CTC phones</th><th>listen</th></tr></thead>
<tbody>{''.join(crop_rows)}</tbody>
</table>
</article>"""
        )
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Span extraction investigation</title>
<style>
body{{font:15px/1.45 -apple-system,sans-serif;margin:24px auto;max-width:1100px;background:#fafafa;color:#111}}
h1{{font-size:22px}} .card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;margin:16px 0}}
.meta{{color:#444;font-size:14px}} table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{border-bottom:1px solid #eee;padding:6px 8px;vertical-align:top}}
code{{font-family:ui-monospace,Menlo,monospace;font-size:12px}}
audio{{width:220px;height:28px}}
</style></head><body>
<h1>Span extraction investigation</h1>
<p class="meta">Gold IPA unchanged. Same sentence WAV + Cloud gold reference per case.
<strong>sentence-span WavLM F1</strong> = SpeechBERTScore on the crop vs gold iso wav.
CTC phones are decoded independently and do not define the score.</p>
{''.join(sections)}
</body></html>"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(page)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path, default=OUT_ROOT / "report.json")
    ap.add_argument("--html-out", type=Path, default=OUT_ROOT / "index.html")
    args = ap.parse_args()

    report = []
    for case in default_cases():
        if not case.sent_wav.exists():
            print(f"SKIP missing {case.sent_wav}", flush=True)
            continue
        print(f"=== {case.slug} / {case.label} ===", flush=True)
        row = run_case(case)
        report.append(row)
        auto = next(c for c in row["crops"] if c["label"] == "auto")
        best = max(row["crops"], key=lambda c: c["sentence_span_wavlm_f1"])
        print(
            f"  auto WavLM={auto['sentence_span_wavlm_f1']:.3f} dur={auto['dur_s']:.3f}s "
            f"CTC={auto['ctc_phones_raw']!r}",
            flush=True,
        )
        print(
            f"  best={best['label']} WavLM={best['sentence_span_wavlm_f1']:.3f} "
            f"dur={best['dur_s']:.3f}s CTC={best['ctc_phones_raw']!r}",
            flush=True,
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    write_html(report, args.html_out)
    print(f"WROTE {args.json_out}", flush=True)
    print(f"WROTE {args.html_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
