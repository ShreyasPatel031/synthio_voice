#!/usr/bin/env python3
"""Controlled isolated vs sentence-context comparison with full audio provenance.

Gold IPA unchanged. Uses forced-align text-grounded crop boundaries (documented,
listen-verify pending). Reports symmetric WavLM F1 between clip pairs and CTC
phones separately. No pass/fail threshold.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

import librosa
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dose_r.forced_align import extract_drug_span_forced_align
from dose_r.scoring.phoneme_distance import normalize_phonemes
from dose_r.scoring.phoneme_model import transcribe_phonemes
from dose_r.scoring.speech_similarity import (
    MODEL_ID,
    extract_frame_embeddings,
    speech_bertscore,
)

OUT = ROOT / "runs" / "misaki-iter" / "context-controlled"
MANIFEST = ROOT / "data" / "gold_gemini_ipa" / "manifest.jsonl"
DOSE = ROOT / "data" / "dose_v1.jsonl"
GOLD_WAV = ROOT / "data" / "gold_gemini_ipa" / "wavs"

KOKORO_VOICE = "af_heart"
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_SPEED = 1.0
CLOUD_VOICE = "en-US-Standard-C"
CLOUD_RATE = 24000
CLOUD_ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
CLOUD_PROJECT = __import__("os").environ.get("GOOGLE_CLOUD_PROJECT", "project-amer-scs-sandbox")
_TARGET_SR = 16_000
_MAX_GAP_RECOVERY_S = 0.15

SCORER = {
    "metric": "speech_bertscore_f1",
    "model": MODEL_ID,
    "layer": "final",
    "symmetric": True,
}

# Controlled pins for this run (gold IPA in manifest unchanged).
PINS = {
    "icotyde": "ˌIkˈO tId",
    "idvynso": "ɪdvˈɪn sˌO",  # final vowel explicitly unresolved
    "vorasidenib": "vˌɔɹəsˈɪdənɪb",  # space removed; ˌ before consonant cluster
    "advair": "ˈædvɛɹ",
}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_item(slug_name: str) -> dict:
    for line in DOSE.read_text().splitlines():
        row = json.loads(line)
        ingredients = row.get("ingredients") or [row["name"]]
        spans = row.get("spans") or []
        sentence = row["sentence"]
        for i, ing in enumerate(ingredients):
            if slug(ing) != slug_name:
                continue
            if i < len(spans) and isinstance(spans[i], (list, tuple)) and len(spans[i]) == 2:
                a, b = spans[i]
                spoken = sentence[a:b]
            else:
                idx = sentence.lower().find(ing.lower())
                if idx < 0:
                    continue
                spoken = sentence[idx : idx + len(ing)]
            return {"drug": ing, "spoken": spoken, "sentence": sentence, "slug": slug_name}
    raise KeyError(slug_name)


def gold_manifest_row(slug_name: str) -> dict:
    for line in MANIFEST.read_text().splitlines():
        rec = json.loads(line)
        if slug(rec["ingredient"]) == slug_name:
            return rec
    raise KeyError(slug_name)


def wav_duration(path: Path) -> float:
    info = sf.info(str(path))
    return round(info.frames / info.samplerate, 4)


def wav_to_mp3(wav: Path, mp3: Path) -> None:
    mp3.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
            "-codec:a", "libmp3lame", "-qscale:a", "5", str(mp3),
        ]
    )


def pair_wavlm_f1(a: bytes, b: bytes) -> float:
    return float(
        speech_bertscore(
            extract_frame_embeddings(a),
            extract_frame_embeddings(b),
        )["f1"]
    )


def decode_ctc(data: bytes) -> dict[str, str]:
    raw = transcribe_phonemes(data)
    return {"raw": raw, "norm": normalize_phonemes(raw)}


def align_bounds(audio_bytes: bytes, sentence: str, spoken: str) -> tuple[float, float] | None:
    """Text-grounded forced align; returns (start_s, end_s) on the source wav."""
    import torch
    import torchaudio

    from dose_r.forced_align import (
        _build_target_sequence,
        _decode_segments,
        _locate_drug_word_indices,
    )
    from dose_r.scoring import phoneme_model

    _torch, processor, model = phoneme_model._get_model()
    vocab = processor.tokenizer.get_vocab()
    blank_id = processor.tokenizer.pad_token_id
    word_idx = _locate_drug_word_indices(sentence, spoken)
    if word_idx is None:
        return None
    target_ids, word_of_token = _build_target_sequence(sentence, vocab)
    drug_token_positions = [i for i, w in enumerate(word_of_token) if w in word_idx]
    if not drug_token_positions:
        return None
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=_TARGET_SR, mono=True)
    duration_s = len(audio) / _TARGET_SR
    inputs = processor(audio, sampling_rate=_TARGET_SR, return_tensors="pt")
    with _torch.no_grad():
        log_probs = _torch.log_softmax(model(inputs.input_values).logits, dim=-1)
    aligned_tokens, _ = torchaudio.functional.forced_align(
        log_probs,
        _torch.tensor([target_ids], dtype=torch.int32),
        _torch.tensor([log_probs.shape[1]]),
        _torch.tensor([len(target_ids)]),
        blank=blank_id,
    )
    segments = _decode_segments(aligned_tokens[0].tolist(), blank_id)
    if len(segments) != len(target_ids):
        return None
    frame_stride = duration_s / log_probs.shape[1]
    first_pos, last_pos = min(drug_token_positions), max(drug_token_positions)
    drug_start_frame = segments[first_pos][1]
    drug_end_frame = segments[last_pos][2]
    if first_pos > 0:
        prev_end_frame = segments[first_pos - 1][2]
        half_gap_s = (drug_start_frame - prev_end_frame) / 2 * frame_stride
        auto_start_s = drug_start_frame * frame_stride - min(half_gap_s, _MAX_GAP_RECOVERY_S)
    else:
        auto_start_s = max(0.0, drug_start_frame * frame_stride - 0.03)
    if last_pos < len(segments) - 1:
        next_start_frame = segments[last_pos + 1][1]
        half_gap_s = (next_start_frame - drug_end_frame) / 2 * frame_stride
        auto_end_s = drug_end_frame * frame_stride + min(half_gap_s, _MAX_GAP_RECOVERY_S)
    else:
        auto_end_s = min(duration_s, drug_end_frame * frame_stride + 0.03)
    dbg = {"auto": (max(0.0, auto_start_s), min(duration_s, auto_end_s))}
    if dbg is None:
        return None
    return dbg["auto"]


def crop_span(audio_bytes: bytes, bounds: tuple[float, float]) -> bytes:
    start_s, end_s = bounds
    with wave.open(io.BytesIO(audio_bytes), "rb") as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        start_f = max(0, int(start_s * rate))
        end_f = min(w.getnframes(), int(end_s * rate))
        w.setpos(start_f)
        frames = w.readframes(end_f - start_f)
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(frames)
    return out.getvalue()


def extract_span(audio_bytes: bytes, sentence: str, spoken: str, drug: str) -> tuple[bytes | None, dict]:
    span = extract_drug_span_forced_align(audio_bytes, sentence, spoken)
    if span is None:
        span = extract_drug_span_forced_align(audio_bytes, sentence, drug)
    bounds = align_bounds(audio_bytes, sentence, spoken)
    meta = {
        "boundary_method": "forced_align_text_grounded_v4",
        "listen_verify_pending": True,
        "bounds_s": None if bounds is None else {"start": bounds[0], "end": bounds[1]},
    }
    return span, meta


def custom_pronunciation(phrase: str, pronunciation: str) -> dict:
    return {
        "pronunciations": [
            {
                "phrase": phrase,
                "phoneticEncoding": "PHONETIC_ENCODING_IPA",
                "pronunciation": pronunciation.strip().strip("/"),
            }
        ]
    }


def cloud_token() -> str:
    try:
        import google.auth
        from google.auth.transport.requests import Request

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(Request())
        return creds.token
    except Exception:
        return subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()


def cloud_synth_sentence(sentence: str, spoken: str, gold_ipa: str) -> bytes:
    tok = cloud_token()
    pron = custom_pronunciation(spoken, gold_ipa.strip("/"))
    body = {
        "input": {"text": sentence, "customPronunciations": pron},
        "voice": {"languageCode": "en-US", "name": CLOUD_VOICE},
        "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": CLOUD_RATE},
    }
    resp = requests.post(
        CLOUD_ENDPOINT,
        headers={
            "Authorization": f"Bearer {tok}",
            "Content-Type": "application/json",
            "x-goog-user-project": CLOUD_PROJECT,
        },
        json=body,
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Cloud TTS {resp.status_code}: {resp.text[:400]}")
    return base64.b64decode(resp.json()["audioContent"])


def kokoro_inject(sentence: str, spoken: str, pin: str) -> str:
    return sentence.replace(spoken, f"[{spoken}](/{pin}/)", 1)


def synth_kokoro(text: str, dest: Path) -> None:
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="a", repo_id=KOKORO_REPO, device="cpu")
    result = next(pipeline(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED))
    if result.audio is None:
        raise RuntimeError(f"kokoro failed: {text[:80]!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, result.audio.detach().cpu().numpy(), 24000)


def write_wav_bytes(data: bytes, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def clip_record(
    label: str,
    data: bytes,
    dest: Path,
    mp3_dir: Path,
    *,
    engine: str,
    context: str,
    bounds: dict | None = None,
    parent_sent: str | None = None,
) -> dict:
    write_wav_bytes(data, dest)
    mp3 = mp3_dir / f"{label}.mp3"
    wav_to_mp3(dest, mp3)
    ctc = decode_ctc(data)
    return {
        "label": label,
        "engine": engine,
        "context": context,
        "wav": str(dest.relative_to(ROOT)),
        "mp3": str(mp3.relative_to(OUT)),
        "sha256": sha256_bytes(data),
        "duration_s": wav_duration(dest),
        "bounds_s": bounds,
        "parent_sentence_wav": parent_sent,
        "ctc_phones_raw": ctc["raw"],
        "ctc_phones_norm": ctc["norm"],
    }


def run_slug(slug_name: str, pin: str, *, resynth: bool) -> dict:
    item = load_item(slug_name)
    man = gold_manifest_row(slug_name)
    gold_ipa = man["ipa"]
    gold_path = GOLD_WAV / f"{slug_name}.wav"
    if not gold_path.exists():
        raise FileNotFoundError(gold_path)

    slug_dir = OUT / slug_name
    mp3_dir = slug_dir / "mp3"
    slug_dir.mkdir(parents=True, exist_ok=True)

    kokoro_iso = slug_dir / "kokoro_iso.wav"
    kokoro_sent = slug_dir / "kokoro_sent.wav"
    cloud_sent = slug_dir / "cloud_sent.wav"

    if resynth or not kokoro_iso.exists():
        synth_kokoro(f"[{item['spoken']}](/{pin}/)", kokoro_iso)
    if resynth or not kokoro_sent.exists():
        synth_kokoro(kokoro_inject(item["sentence"], item["spoken"], pin), kokoro_sent)
    if resynth or not cloud_sent.exists():
        write_wav_bytes(
            cloud_synth_sentence(item["sentence"], item["spoken"], gold_ipa),
            cloud_sent,
        )

    kokoro_sent_bytes = kokoro_sent.read_bytes()
    cloud_sent_bytes = cloud_sent.read_bytes()

    kokoro_span_bytes, kokoro_span_meta = extract_span(
        kokoro_sent_bytes, item["sentence"], item["spoken"], item["drug"]
    )
    cloud_span_bytes, cloud_span_meta = extract_span(
        cloud_sent_bytes, item["sentence"], item["spoken"], item["drug"]
    )
    if kokoro_span_bytes is None or cloud_span_bytes is None:
        raise RuntimeError(f"span extraction failed for {slug_name}")

    kokoro_span = slug_dir / "kokoro_span.wav"
    cloud_span = slug_dir / "cloud_span.wav"
    write_wav_bytes(kokoro_span_bytes, kokoro_span)
    write_wav_bytes(cloud_span_bytes, cloud_span)

    cloud_iso_bytes = gold_path.read_bytes()
    cloud_iso_copy = slug_dir / "cloud_iso_ref.wav"
    write_wav_bytes(cloud_iso_bytes, cloud_iso_copy)

    clips = [
        clip_record(
            "cloud_iso", cloud_iso_bytes, cloud_iso_copy, mp3_dir,
            engine="cloud", context="isolated",
            bounds=None,
        ),
        clip_record(
            "kokoro_iso", kokoro_iso.read_bytes(), kokoro_iso, mp3_dir,
            engine="kokoro", context="isolated",
            bounds=None,
        ),
        clip_record(
            "kokoro_sent", kokoro_sent_bytes, kokoro_sent, mp3_dir,
            engine="kokoro", context="sentence_full",
            bounds=None,
        ),
        clip_record(
            "cloud_sent", cloud_sent_bytes, cloud_sent, mp3_dir,
            engine="cloud", context="sentence_full",
            bounds=None,
        ),
        clip_record(
            "kokoro_span", kokoro_span_bytes, kokoro_span, mp3_dir,
            engine="kokoro", context="sentence_crop",
            bounds=kokoro_span_meta.get("bounds_s"),
            parent_sent=str(kokoro_sent.relative_to(ROOT)),
        ),
        clip_record(
            "cloud_span", cloud_span_bytes, cloud_span, mp3_dir,
            engine="cloud", context="sentence_crop",
            bounds=cloud_span_meta.get("bounds_s"),
            parent_sent=str(cloud_sent.relative_to(ROOT)),
        ),
    ]
    by_label = {c["label"]: c for c in clips}

    def pair(name: str, left: str, right: str, kind: str) -> dict:
        lb, rb = by_label[left], by_label[right]
        lv = Path(ROOT / lb["wav"]).read_bytes() if left != "cloud_iso" else cloud_iso_bytes
        rv = Path(ROOT / rb["wav"]).read_bytes() if right != "cloud_iso" else cloud_iso_bytes
        f1 = round(pair_wavlm_f1(lv, rv), 4)
        return {
            "comparison": name,
            "kind": kind,
            "left": left,
            "right": right,
            "left_sha256": lb["sha256"],
            "right_sha256": rb["sha256"],
            "wavlm_f1": f1,
        }

    comparisons = [
        pair("cloud_iso_vs_cloud_span", "cloud_iso", "cloud_span", "within_cloud"),
        pair("kokoro_iso_vs_kokoro_span", "kokoro_iso", "kokoro_span", "within_kokoro"),
        pair("kokoro_iso_vs_cloud_iso", "kokoro_iso", "cloud_iso", "cross_engine_isolated"),
        pair("kokoro_span_vs_cloud_span", "kokoro_span", "cloud_span", "cross_engine_sentence_crop"),
    ]

    # Official sentence-span score vs locked gold (what cloud-rank measures).
    official_span_f1 = round(pair_wavlm_f1(kokoro_span_bytes, cloud_iso_bytes), 4)

    row = {
        "slug": slug_name,
        "drug": item["drug"],
        "spoken": item["spoken"],
        "sentence": item["sentence"],
        "pin": pin,
        "gold_ipa": gold_ipa,
        "gold_ipa_unchanged": True,
        "idvynso_final_vowel": "unresolved" if slug_name == "idvynso" else None,
        "kokoro": {"voice": KOKORO_VOICE, "repo": KOKORO_REPO, "speed": KOKORO_SPEED},
        "cloud": {"voice": CLOUD_VOICE, "rate_hz": CLOUD_RATE},
        "scorer": SCORER,
        "boundary": {
            "method": "forced_align_text_grounded_v4",
            "listen_verify_pending": True,
            "note": "Not energy-trimmed. Verify crop against parent sentence WAV by ear.",
            "kokoro": kokoro_span_meta,
            "cloud": cloud_span_meta,
        },
        "reference": {
            "locked_gold_wav": str(gold_path.relative_to(ROOT)),
            "reference_sha256": sha256_file(gold_path),
            "manifest_teacher": man.get("teacher"),
        },
        "clips": clips,
        "comparisons": comparisons,
        "official_sentence_span_wavlm_f1_vs_cloud_iso": official_span_f1,
    }
    (slug_dir / "report.json").write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n")
    return row


def write_html(rows: list[dict], dest: Path) -> None:
    parts = []
    for row in rows:
        cmp_rows = "".join(
            f"<tr><td>{html.escape(c['comparison'])}</td>"
            f"<td>{html.escape(c['kind'])}</td>"
            f"<td><strong>{c['wavlm_f1']:.3f}</strong></td>"
            f"<td><code>{html.escape(c['left'])}</code> ↔ <code>{html.escape(c['right'])}</code></td></tr>"
            for c in row["comparisons"]
        )
        clip_blocks = []
        for c in row["clips"]:
            bounds = ""
            if c.get("bounds_s"):
                b = c["bounds_s"]
                bounds = f" · crop {b['start']:.3f}–{b['end']:.3f}s"
            clip_blocks.append(
                f"<div class='clip'><div class='clip-h'>{html.escape(c['label'])} "
                f"({html.escape(c['engine'])}/{html.escape(c['context'])}) · {c['duration_s']:.3f}s{bounds}</div>"
                f"<div class='hash'>sha256 {html.escape(c['sha256'][:16])}…</div>"
                f"<div class='ctc'><code>{html.escape(c['ctc_phones_raw'])}</code></div>"
                f"<audio controls preload='none' src='{html.escape(Path(c['mp3']).as_posix())}'></audio></div>"
            )
        parts.append(
            f"""<article class="card">
<h2>{html.escape(row['drug'])}</h2>
<p class="meta">pin <code>{html.escape(row['pin'])}</code> · gold IPA <code>/{html.escape(row['gold_ipa'])}/</code> (unchanged)</p>
<p class="meta">official sentence-span WavLM F1 (Kokoro crop vs Cloud iso): <strong>{row['official_sentence_span_wavlm_f1_vs_cloud_iso']:.3f}</strong></p>
<p class="meta">ref sha256 <code>{html.escape(row['reference']['reference_sha256'][:20])}…</code> · scorer {html.escape(row['scorer']['model'])} {html.escape(row['scorer']['metric'])}</p>
<p class="sent">{html.escape(row['sentence'])}</p>
<table><thead><tr><th>comparison</th><th>kind</th><th>WavLM F1</th><th>clips</th></tr></thead><tbody>{cmp_rows}</tbody></table>
<div class="clips">{''.join(clip_blocks)}</div>
</article>"""
        )
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Context-controlled comparison</title>
<style>
body{{font:15px/1.45 -apple-system,sans-serif;margin:24px auto;max-width:960px;background:#fafafa;color:#111}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;margin:16px 0}}
.meta,.sent{{color:#444;font-size:14px}} table{{width:100%;border-collapse:collapse;font-size:13px;margin:12px 0}}
th,td{{border-bottom:1px solid #eee;padding:6px 8px;text-align:left}}
.clips{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.clip{{border:1px solid #eee;border-radius:6px;padding:8px;background:#fcfcfc}}
.clip-h{{font-weight:600;font-size:13px}} .hash{{font-size:11px;color:#666}}
.ctc code{{font-size:11px}} audio{{width:100%;height:28px;margin-top:4px}}
</style></head><body>
<h1>Context-controlled comparison</h1>
<p class="meta">Symmetric WavLM F1 between clip pairs. CTC phones per clip. Crop boundaries are
<strong>forced-align text-grounded</strong> (listen-verify pending; not energy-trimmed).
No pass/fail threshold.</p>
{''.join(parts)}
</body></html>"""
    dest.write_text(page)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", action="append")
    ap.add_argument("--resynth", action="store_true")
    args = ap.parse_args()
    slugs = args.slug or list(PINS.keys())
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in slugs:
        pin = PINS[s]
        print(f"=== {s} pin={pin} ===", flush=True)
        row = run_slug(s, pin, resynth=args.resynth)
        rows.append(row)
        for c in row["comparisons"]:
            print(f"  {c['comparison']}: {c['wavlm_f1']:.3f}", flush=True)
        print(f"  official kokoro_span vs cloud_iso: {row['official_sentence_span_wavlm_f1_vs_cloud_iso']:.3f}", flush=True)

    manifest = {
        "run": "context-controlled",
        "scorer": SCORER,
        "kokoro": {"voice": KOKORO_VOICE, "repo": KOKORO_REPO, "speed": KOKORO_SPEED},
        "cloud": {"voice": CLOUD_VOICE, "rate_hz": CLOUD_RATE},
        "boundary_method": "forced_align_text_grounded_v4",
        "listen_verify_pending": True,
        "rows": rows,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    write_html(rows, OUT / "index.html")

    # Flat provenance file for listen page builder.
    prov_lines = []
    for row in rows:
        prov_lines.append(json.dumps(row, ensure_ascii=False))
    (ROOT / "runs/misaki-iter/scored-provenance.jsonl").write_text("\n".join(prov_lines) + "\n")
    print(f"WROTE {OUT / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
