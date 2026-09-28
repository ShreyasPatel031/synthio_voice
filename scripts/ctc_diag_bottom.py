#!/usr/bin/env python3
"""Bottom-up CTC diagnostic: gap-recovery sweep vs Standard IPA gold.

Order of attack for low names: CTC window first, then Misaki, then IPA.
"""
from __future__ import annotations

import io
import json
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# On the eval VM the checkout lives under shreyaspatel's home.
if not (ROOT / "runs" / "misaki-iter" / "lexicon-rank.json").exists():
    ROOT = Path("/home/shreyaspatel/synthio_voice")
sys.path.insert(0, str(ROOT))
STORE = ROOT / "runs" / "misaki-iter"

from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore


def dur_wav(b: bytes) -> float:
    with wave.open(io.BytesIO(b), "rb") as w:
        return w.getnframes() / w.getframerate()


def f1(a: bytes, b: bytes, cache: dict, key: str) -> float:
    if key not in cache:
        cache[key] = extract_frame_embeddings(b)
    return float(speech_bertscore(extract_frame_embeddings(a), cache[key])["f1"])


def gold_path(slug: str) -> Path | None:
    for folder in (STORE / "cloud-gold", ROOT / "data" / "gold_gemini_ipa" / "wavs"):
        for name in (f"{slug}.wav", f"{slug.replace('-', '_')}.wav"):
            p = folder / name
            if p.exists() and p.stat().st_size > 500:
                return p
    return None


def load_item(slug: str) -> dict:
    import re

    def s(n: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", n.lower()).strip("-")

    for line in (ROOT / "data" / "dose_v1.jsonl").read_text().splitlines():
        row = json.loads(line)
        for i, ing in enumerate(row.get("ingredients") or [row["name"]]):
            if s(ing) == slug:
                spans = row.get("spans") or []
                sentence = row["sentence"]
                if i < len(spans) and len(spans[i]) == 2:
                    a, b = spans[i]
                    spoken = sentence[a:b]
                else:
                    spoken = ing
                return {"drug": ing, "spoken": spoken, "sentence": sentence, "slug": slug}
    raise SystemExit(f"missing {slug}")


def extract_with_params(
    audio_bytes: bytes,
    sentence: str,
    drug: str,
    max_gap_s: float = 0.15,
    start_share: float = 0.5,
    end_share: float = 0.5,
):
    import librosa
    import torchaudio
    import wave as wavemod
    from io import BytesIO

    from dose_r.forced_align import (
        _PAD_S,
        _TARGET_SR,
        _build_target_sequence,
        _decode_segments,
        _locate_drug_word_indices,
    )
    from dose_r.scoring import phoneme_model

    torch_mod, processor, model = phoneme_model._get_model()
    vocab = processor.tokenizer.get_vocab()
    blank_id = processor.tokenizer.pad_token_id
    word_idx = _locate_drug_word_indices(sentence, drug)
    if word_idx is None:
        return None, {}
    target_ids, word_of_token = _build_target_sequence(sentence, vocab)
    drug_token_positions = [i for i, w in enumerate(word_of_token) if w in word_idx]
    if not drug_token_positions:
        return None, {}
    audio, _ = librosa.load(BytesIO(audio_bytes), sr=_TARGET_SR, mono=True)
    duration_s = len(audio) / _TARGET_SR
    inputs = processor(audio, sampling_rate=_TARGET_SR, return_tensors="pt")
    with torch_mod.no_grad():
        log_probs = torch_mod.log_softmax(model(inputs.input_values).logits, dim=-1)
    targets = torch_mod.tensor([target_ids], dtype=torch_mod.int32)
    input_lengths = torch_mod.tensor([log_probs.shape[1]])
    target_lengths = torch_mod.tensor([len(target_ids)])
    aligned_tokens, _scores = torchaudio.functional.forced_align(
        log_probs, targets, input_lengths, target_lengths, blank=blank_id,
    )
    frame_labels = aligned_tokens[0].tolist()
    segments = _decode_segments(frame_labels, blank_id)
    if len(segments) != len(target_ids):
        return None, {"error": "degenerate"}
    frame_stride = duration_s / log_probs.shape[1]
    first_pos, last_pos = min(drug_token_positions), max(drug_token_positions)
    drug_start_frame = segments[first_pos][1]
    drug_end_frame = segments[last_pos][2]
    meta = {
        "duration_s": round(duration_s, 3),
        "raw_drug_s": round((drug_end_frame - drug_start_frame) * frame_stride, 3),
        "frame_stride": round(frame_stride, 5),
    }
    if first_pos > 0:
        prev_end_frame = segments[first_pos - 1][2]
        gap_s = (drug_start_frame - prev_end_frame) * frame_stride
        recover = min(gap_s * start_share, max_gap_s)
        start_s = drug_start_frame * frame_stride - recover
        meta["prev_gap_s"] = round(gap_s, 3)
        meta["start_recover_s"] = round(recover, 3)
    else:
        start_s = max(0.0, drug_start_frame * frame_stride - _PAD_S)
        meta["prev_gap_s"] = None
    if last_pos < len(segments) - 1:
        next_start_frame = segments[last_pos + 1][1]
        gap_s = (next_start_frame - drug_end_frame) * frame_stride
        recover = min(gap_s * end_share, max_gap_s)
        end_s = drug_end_frame * frame_stride + recover
        meta["next_gap_s"] = round(gap_s, 3)
        meta["end_recover_s"] = round(recover, 3)
    else:
        end_s = min(duration_s, drug_end_frame * frame_stride + _PAD_S)
        meta["next_gap_s"] = None
    meta["crop_s"] = round(end_s - start_s, 3)
    meta["start_s"] = round(start_s, 3)
    meta["end_s"] = round(end_s, 3)
    with wavemod.open(BytesIO(audio_bytes), "rb") as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        w.setpos(max(0, int(start_s * rate)))
        n = int((end_s - start_s) * rate)
        frames = w.readframes(min(n, w.getnframes() - w.tell()))
    out = io.BytesIO()
    with wavemod.open(out, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(frames)
    return out.getvalue(), meta


def main() -> int:
    slugs = sys.argv[1:] or ["idvynso"]
    out_dir = STORE / "bottom-up-ctc"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    report = []
    for slug in slugs:
        it = load_item(slug)
        gold = gold_path(slug)
        sent = STORE / "lexicon-sent" / f"{slug}.wav"
        span_default = STORE / "lexicon-span" / f"{slug}.wav"
        iso = STORE / "cloud-iso" / f"{slug}.wav"
        print(f"\n===== {slug} =====", flush=True)
        print("sentence:", it["sentence"], flush=True)
        print("spoken:", it["spoken"], "gold:", gold, flush=True)
        if not gold or not sent.exists():
            print("MISSING assets", flush=True)
            continue
        gold_b = gold.read_bytes()
        sent_b = sent.read_bytes()
        gkey = str(gold)
        row: dict = {"slug": slug, "spoken": it["spoken"], "sentence": it["sentence"]}
        if span_default.exists():
            span_b = span_default.read_bytes()
            f = f1(span_b, gold_b, cache, gkey)
            row["current_span_f1"] = round(f, 4)
            row["current_span_dur"] = round(dur_wav(span_b), 3)
            print(
                f"current span F1={f:.4f} dur={row['current_span_dur']}s",
                flush=True,
            )
        if iso.exists():
            iso_b = iso.read_bytes()
            row["iso_f1"] = round(f1(iso_b, gold_b, cache, gkey), 4)
            row["iso_dur"] = round(dur_wav(iso_b), 3)
            print(f"iso F1={row['iso_f1']:.4f} dur={row['iso_dur']}s", flush=True)
        row["gold_self_f1"] = round(f1(gold_b, gold_b, cache, gkey), 4)

        # Run forced-align once; reuse log-probs by only varying boundary math.
        # extract_with_params re-runs align each time — fine for 3 names.
        sweeps = []
        for max_gap in (0.05, 0.10, 0.15, 0.25, 0.40):
            for start_share, end_share, tag in (
                (0.5, 0.5, "mid50"),
                (0.75, 0.5, "start75"),
                (0.0, 0.5, "no-start"),
                (0.25, 0.5, "start25"),
                (0.5, 0.75, "end75"),
                (0.0, 0.0, "raw"),
                (0.25, 0.25, "tight25"),
            ):
                span, meta = extract_with_params(
                    sent_b,
                    it["sentence"],
                    it["spoken"],
                    max_gap_s=max_gap,
                    start_share=start_share,
                    end_share=end_share,
                )
                if span is None:
                    continue
                score = f1(span, gold_b, cache, gkey)
                sweeps.append(
                    {
                        "max_gap": max_gap,
                        "tag": tag,
                        "start_share": start_share,
                        "end_share": end_share,
                        "f1": round(score, 4),
                        **meta,
                    }
                )
        sweeps.sort(key=lambda r: -r["f1"])
        row["best_ctc"] = sweeps[0] if sweeps else None
        row["default_ctc"] = next(
            (s for s in sweeps if s["max_gap"] == 0.15 and s["tag"] == "mid50"),
            None,
        )
        row["sweep_top5"] = sweeps[:5]
        if sweeps:
            best = sweeps[0]
            span_b, _ = extract_with_params(
                sent_b,
                it["sentence"],
                it["spoken"],
                max_gap_s=best["max_gap"],
                start_share=best["start_share"],
                end_share=best["end_share"],
            )
            (out_dir / f"{slug}-best.wav").write_bytes(span_b)
            def_span, _ = extract_with_params(
                sent_b,
                it["sentence"],
                it["spoken"],
                max_gap_s=0.15,
                start_share=0.5,
                end_share=0.5,
            )
            if def_span:
                (out_dir / f"{slug}-default.wav").write_bytes(def_span)
            d = row["default_ctc"]
            print(
                f"DEFAULT mid50/0.15 F1={d['f1'] if d else None} "
                f"dur={d.get('crop_s') if d else None} "
                f"prev_gap={d.get('prev_gap_s') if d else None} "
                f"next_gap={d.get('next_gap_s') if d else None}",
                flush=True,
            )
            print(
                f"BEST {best['tag']}/gap{best['max_gap']} F1={best['f1']} "
                f"dur={best['crop_s']}s "
                f"Δ={best['f1'] - (d['f1'] if d else 0):+.3f}",
                flush=True,
            )
            for s in sweeps[:8]:
                print(
                    f"  {s['f1']:.4f} {s['tag']:8} gap={s['max_gap']} "
                    f"crop={s['crop_s']}s prev={s.get('prev_gap_s')} "
                    f"next={s.get('next_gap_s')}",
                    flush=True,
                )
        if row.get("default_ctc") and row.get("best_ctc"):
            delta = row["best_ctc"]["f1"] - row["default_ctc"]["f1"]
            iso_gap = None
            if row.get("iso_f1") is not None:
                iso_gap = row["iso_f1"] - row["default_ctc"]["f1"]
            row["iso_minus_default"] = round(iso_gap, 4) if iso_gap is not None else None
            if delta >= 0.02 or (iso_gap is not None and iso_gap >= 0.15):
                row["ctc_verdict"] = "CTC_CULPRIT"
            elif delta >= 0.01:
                row["ctc_verdict"] = "CTC_MILD"
            else:
                row["ctc_verdict"] = "CTC_OK_TRY_MISAKI"
        report.append(row)
        (out_dir / f"{slug}.json").write_text(json.dumps(row, indent=2) + "\n")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nWROTE", out_dir / "report.json", flush=True)
    for r in report:
        print(
            r["slug"],
            r.get("ctc_verdict"),
            "cur",
            r.get("current_span_f1"),
            "iso",
            r.get("iso_f1"),
            "best",
            (r.get("best_ctc") or {}).get("f1"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
