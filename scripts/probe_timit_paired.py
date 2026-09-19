#!/usr/bin/env python
"""Paired test: TIMIT-phoneme CTC distance vs Path 2 on the SAME new audio.

40 items: the ear-checked set plus a spread of stored Path 2 scores.
Original Gemini WAVs were not in git, so both scores are recomputed here.
"""
from __future__ import annotations

import base64
import json
import math
import os
import sys
from io import BytesIO
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/.cache/huggingface")
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_B64", "")
    if raw:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"] = base64.b64decode(raw).decode()

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dose_r import dataset  # noqa: E402
from dose_r.adapters.gemini_tts import GeminiTTSAdapter  # noqa: E402
from dose_r.config import ALL_SYSTEMS  # noqa: E402
from dose_r.forced_align import extract_drug_span_forced_align  # noqa: E402
from dose_r.references.ipa_references import ipa_variants_for  # noqa: E402
from dose_r.references.reference_clips import available_clips  # noqa: E402
from dose_r.scoring.phoneme_distance import normalize_phonemes  # noqa: E402

MODEL = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
EAR = ["chantix", "adquey", "voranigo", "vyloy", "aripiprazole",
       "esomeprazole", "eliquis", "acoramidis", "talquetamab"]
AUDIO = REPO_ROOT / "runs" / "path3-timit-probe" / "audio"
SPAN = REPO_ROOT / "runs" / "path3-timit-probe" / "spans"
OUT = REPO_ROOT / "runs" / "path3-timit-probe" / "paired.json"
_MAP = {
    "tʃ": "ʧ", "dʒ": "ʤ", "ɡ": "g",
    "iː": "i", "uː": "u", "ɑː": "ɑ", "ɔː": "ɔ", "ɜː": "ɝ", "eː": "eɪ", "oː": "oʊ",
    "ɚ": "ɝ", "ɨ": "ɪ", "ᵻ": "ɪ", "ɐ": "ə", "r": "ɹ",
}


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def longest(s, symbols):
    i, out = 0, []
    while i < len(s):
        mapped = None
        step = 1
        for j in range(len(s), i, -1):
            piece = _MAP.get(s[i:j], s[i:j])
            if piece in symbols:
                mapped = piece
                step = j - i
                break
        if mapped is None:
            i += 1
        else:
            out.append(mapped)
            i += step
    return out


def leven(a, b):
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if a[i - 1] == b[j - 1] else 1))
            prev = cur
    return dp[m]


def ctc_tokens(ids, proc, blank):
    out, prev = [], None
    skip = {"<pad>", "<s>", "</s>", "<unk>", "|", " ", ""}
    for i in ids:
        if i != prev and i != blank:
            tok = proc.tokenizer.convert_ids_to_tokens(int(i))
            if tok not in skip:
                out.append(tok)
        prev = i
    return out


def choose_ids(items, clips, n=40):
    p2 = {}
    for line in (REPO_ROOT / "runs/gemini-flash-tts-v1-speech-similarity-v4/results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        sc = rec.get("score") or {}
        if sc.get("scoreable") and rec["item_id"] in items and rec["item_id"] in clips and ipa_variants_for(rec["item_id"]):
            p2[rec["item_id"]] = sc["score"]
    ranked = sorted(p2, key=lambda i: p2[i])
    picked = []
    for i in EAR:
        if i in p2 and i not in picked:
            picked.append(i)
    # worst, best, and a stride through the middle
    for i in ranked[:12] + ranked[-12:]:
        if i not in picked:
            picked.append(i)
    stride = max(1, len(ranked) // 20)
    for i in ranked[::stride]:
        if len(picked) >= n:
            break
        if i not in picked:
            picked.append(i)
    return picked[:n], p2


def main() -> int:
    items = {i.item_id: i for i in dataset.load_items()}
    clips = available_clips()
    ids, stored = choose_ids(items, clips)
    print(f"selected {len(ids)}", flush=True)
    AUDIO.mkdir(parents=True, exist_ok=True)
    SPAN.mkdir(parents=True, exist_ok=True)

    spec = next(v for v in ALL_SYSTEMS.values() if v.tier == "gemini-tts")
    adapter = GeminiTTSAdapter(spec, speaker="Kore")
    # reuse the 9 clips already synthesized in the gop probe
    reuse = REPO_ROOT / "runs" / "path3-gop-probe" / "audio"
    for n, item_id in enumerate(ids, 1):
        dest = AUDIO / f"{item_id}.wav"
        src = reuse / f"{item_id}.wav"
        if dest.exists():
            continue
        if src.exists():
            dest.write_bytes(src.read_bytes())
            continue
        item = items[item_id]
        print(f"synth {n}/{len(ids)} {item_id}", flush=True)
        result = adapter.synthesize(item.sentence, item_id)
        if not result.ok or not result.audio:
            print(f"  FAIL {result.error}")
            continue
        dest.write_bytes(result.audio)

    print("spans", flush=True)
    for item_id in ids:
        dest = SPAN / f"{item_id}.wav"
        src = AUDIO / f"{item_id}.wav"
        if dest.exists() or not src.exists():
            continue
        item = items[item_id]
        span = extract_drug_span_forced_align(src.read_bytes(), item.sentence, item.drug)
        if span is None:
            print("  no span", item_id)
            continue
        dest.write_bytes(span)
        print(" ", item_id, flush=True)

    import gc
    from dose_r.scoring.phoneme_model import _get_model
    _get_model.cache_clear()
    gc.collect()

    import librosa
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    print("timit", flush=True)
    proc = Wav2Vec2Processor.from_pretrained(MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL)
    model.eval()
    symbols = set(proc.tokenizer.get_vocab())
    blank = proc.tokenizer.pad_token_id
    rows = {}
    for item_id in ids:
        span_path = SPAN / f"{item_id}.wav"
        if not span_path.exists():
            continue
        audio, _ = librosa.load(BytesIO(span_path.read_bytes()), sr=16000, mono=True)
        inputs = proc(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        hyp = ctc_tokens(torch.argmax(logits, dim=-1)[0].tolist(), proc, blank)
        best, best_ref = None, None
        for v in ipa_variants_for(item_id):
            ref = longest(normalize_phonemes(v), symbols)
            if not ref:
                continue
            rate = leven(hyp, ref) / len(ref)
            if best is None or rate < best:
                best, best_ref = rate, ref
        rows[item_id] = {"hyp": hyp, "ref": best_ref, "rate": best, "stored_p2": stored.get(item_id)}
        print(f"  {item_id} rate={best:.3f}", flush=True)

    del model, proc
    gc.collect()

    print("path2", flush=True)
    from dose_r.scoring.speech_similarity import extract_frame_embeddings, score_speech_similarity
    for item_id, row in rows.items():
        span_path = SPAN / f"{item_id}.wav"
        clip = clips.get(items[item_id].drug)
        if clip is None:
            continue
        score, comp = score_speech_similarity(
            extract_frame_embeddings(span_path.read_bytes()),
            extract_frame_embeddings(clip.path),
        )
        row["path2"] = score
        row["f1"] = comp["f1"]
        print(f"  {item_id} path2={score:.3f} rate={row['rate']:.3f}", flush=True)

    paired = [i for i in rows if "path2" in rows[i] and rows[i]["rate"] is not None]
    rho = spearman([rows[i]["rate"] for i in paired], [rows[i]["path2"] for i in paired])
    print(f"\npaired {len(paired)} spearman rate vs path2 {rho:+.4f}")
    # ear items
    for name in EAR:
        if name in rows and "path2" in rows[name]:
            r = rows[name]
            print(f"  {name:16s} path2={r['path2']:.3f} rate={r['rate']:.3f} hyp={r['hyp']}")
    OUT.write_text(json.dumps({"rho": rho, "rows": rows}, indent=2))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
