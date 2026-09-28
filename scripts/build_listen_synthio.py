#!/usr/bin/env python3
"""Synthio listen page: Cloud gold, human (if any), Kokoro, Qwen, Qwen FT."""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dose_r.references.reference_clips import available_clips

LISTEN = ROOT / "runs" / "listen-hack-vs-gold"
STORE = ROOT / "runs" / "misaki-iter"
RANK = STORE / "lexicon-rank.json"
PINS = STORE / "user_pins.json"
HUMAN_F1 = STORE / "lexicon-vs-human.json"
SPAN = STORE / "lexicon-span"
PLAIN_SPAN = STORE / "kokoro-plain-span"
QWEN_PLAIN = ROOT / "runs" / "finetune-qwen-cloud-ipa" / "qwen17-plain-synth"
QWEN_FT = ROOT / "runs" / "finetune-qwen-cloud-ipa" / "qwen17-ft-synth"
QWEN_PLAIN_SCORES = ROOT / "runs" / "finetune-qwen-cloud-ipa" / "qwen17-plain-scores.jsonl"
QWEN_FT_SCORES = ROOT / "runs" / "finetune-qwen-cloud-ipa" / "qwen17-ft-scores.jsonl"
GOLD_WAV = ROOT / "data" / "gold_gemini_ipa" / "wavs"
DOSE = ROOT / "data" / "dose_v1.jsonl"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[slug(row["ingredient"])] = row
    return out


def dose_items() -> dict[str, dict]:
    items = {}
    for line in DOSE.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for i, ing in enumerate(row.get("ingredients") or [row["name"]]):
            items.setdefault(slug(ing), {"drug": ing, "sentence": row["sentence"]})
    return items


def gold_ipa() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in (
        ROOT / "data" / "gold_gemini_ipa" / "manifest.jsonl",
        STORE / "gold-ipa-labels.jsonl",
        ROOT / "data" / "finetune_cloud_ipa" / "manifest.jsonl",
    ):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = slug(row.get("ingredient") or row.get("spoken_text") or "")
            val = (row.get("ipa_used") or row.get("ipa") or "").strip().strip("/")
            if key and val:
                out.setdefault(key, val)
    return out


def to_mp3(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
        return
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-codec:a",
            "libmp3lame",
            "-qscale:a",
            "5",
            str(dest),
        ]
    )


def dur(path: Path) -> str:
    if not path.exists():
        return ""
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return f"{float(out):.2f}s"


def f1_label(value: float | None, vs: str) -> str:
    if value is None:
        return ""
    return f" · F1 {value:.3f} vs {vs}"


def audio_row(label: str, src: str, extra: str = "") -> str:
    return (
        f"<label>{html.escape(label)}{html.escape(extra)}</label>"
        f'<audio controls preload="none" src="{html.escape(src)}"></audio>'
    )


def main() -> int:
    rank = json.loads(RANK.read_text())
    pins = json.loads(PINS.read_text()) if PINS.exists() else {}
    items = dose_items()
    ipa = gold_ipa()
    clips = {slug(k): v for k, v in available_clips().items()}
    human_f1 = {}
    if HUMAN_F1.exists():
        human_f1 = {r["slug"]: r for r in json.loads(HUMAN_F1.read_text())["rows"]}
    qplain = load_jsonl(QWEN_PLAIN_SCORES)
    qft = load_jsonl(QWEN_FT_SCORES)

    mp3_gold = LISTEN / "mp3" / "gold"
    mp3_plain = LISTEN / "mp3" / "kokoro-plain"
    mp3_misaki = LISTEN / "mp3" / "misaki"
    mp3_human = LISTEN / "mp3" / "human"
    mp3_qwen = LISTEN / "mp3" / "qwen"
    mp3_ft = LISTEN / "mp3" / "qwen-ft"

    cards = []
    n_human = n_qwen = 0
    for row in sorted(rank["rows"], key=lambda r: float(r["ctc_f1"])):
        s = row["slug"]
        it = items.get(s, {"drug": row["drug"], "sentence": ""})
        spoken = it["drug"]
        sentence = it["sentence"]
        misaki = row.get("misaki") or pins.get(s, {}).get("misaki") or ""
        gold_phon = ipa.get(s, "")
        gold_src = GOLD_WAV / f"{s}.wav"
        if not gold_src.exists():
            gold_src = GOLD_WAV / f"{s.replace('-', '_')}.wav"
        gold_mp3 = mp3_gold / f"{s}.mp3"
        if gold_src.exists():
            to_mp3(gold_src, gold_mp3)
        span = SPAN / f"{s}.wav"
        misaki_mp3 = mp3_misaki / f"{s}.mp3"
        if span.exists():
            to_mp3(span, misaki_mp3)

        hf = human_f1.get(s, {}).get("human_f1")
        headline = f"Kokoro {row['ctc_f1']:.3f} vs Standard IPA"
        parts = [
            "<article class='card'>",
            f"<div class='name'>{html.escape(spoken)} "
            f"<span class='f1'>{html.escape(headline)}</span></div>",
            f"<p class='sent'>{html.escape(sentence)}</p>",
            f"<p class='pron'><span>Standard IPA</span> /{html.escape(gold_phon)}/ · "
            f"<span>Kokoro</span> {html.escape(misaki)}</p>",
        ]
        human = clips.get(s)
        if human is not None:
            n_human += 1
            h_mp3 = mp3_human / f"{s}.mp3"
            to_mp3(human.path, h_mp3)
            parts.append(
                audio_row(
                    f"Human · {human.source}",
                    f"mp3/human/{s}.mp3?v=6",
                    f" · {dur(h_mp3)}",
                )
            )
        if gold_mp3.exists():
            parts.append(
                audio_row(
                    "Standard IPA pronunciation",
                    f"mp3/gold/{s}.mp3?v=6",
                    f" · {dur(gold_mp3)}",
                )
            )

        plain_wav = PLAIN_SPAN / f"{s}.wav"
        if plain_wav.exists():
            plain_mp3 = mp3_plain / f"{s}.mp3"
            to_mp3(plain_wav, plain_mp3)
            parts.append(
                audio_row(
                    "Kokoro plain · sentence crop",
                    f"mp3/kokoro-plain/{s}.mp3?v=6",
                    f" · {dur(plain_mp3)}",
                )
            )

        if misaki_mp3.exists():
            pin = " · pin" if s in pins else ""
            extras = f" · {dur(misaki_mp3)} · F1 {row['ctc_f1']:.3f} vs Standard IPA{pin}"
            extras += f1_label(hf, "human")
            parts.append(
                audio_row(
                    "Kokoro + pronunciation · sentence crop",
                    f"mp3/misaki/{s}.mp3?v=6",
                    extras,
                )
            )

        qwav = QWEN_PLAIN / f"{s}.wav"
        if qwav.exists():
            n_qwen += 1
            q_mp3 = mp3_qwen / f"{s}.mp3"
            to_mp3(qwav, q_mp3)
            qs = qplain.get(s, {})
            extras = f" · {dur(q_mp3)}{f1_label(qs.get('teacher_f1'), 'Standard IPA')}"
            if qs.get("human_f1") is not None:
                extras += f1_label(qs["human_f1"], "human")
            parts.append(audio_row("Qwen + wav", f"mp3/qwen/{s}.mp3?v=6", extras))

        ftwav = QWEN_FT / f"{s}.wav"
        if ftwav.exists():
            ft_mp3 = mp3_ft / f"{s}.mp3"
            to_mp3(ftwav, ft_mp3)
            fs = qft.get(s, {})
            extras = f" · {dur(ft_mp3)}{f1_label(fs.get('teacher_f1'), 'Standard IPA')}"
            if fs.get("human_f1") is not None:
                extras += f1_label(fs["human_f1"], "human")
            parts.append(audio_row("Qwen fine-tuned", f"mp3/qwen-ft/{s}.mp3?v=6", extras))

        parts.append("</article>")
        cards.append("".join(parts))

    mean = rank["mean_f1"]
    n = len(cards)
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Synthio listen — Kokoro, Qwen, human</title>
<style>
body{{font:16px/1.45 -apple-system,BlinkMacSystemFont,sans-serif;margin:32px auto;max-width:820px;color:#1a1a1a;background:#fafafa;padding:0 16px 80px}}
h1{{font-size:22px;font-weight:650;margin-bottom:8px}}
.note{{color:#444;margin:0 0 20px}}
.card{{background:#fff;border:1px solid #e4e4e4;border-radius:8px;padding:16px 16px 8px;margin:14px 0}}
.name{{font-weight:650}}
.f1{{color:#9a3412;font-weight:500}}
.sent{{margin:8px 0 8px;color:#333}}
.pron{{margin:0 0 12px;color:#444;font-size:14px}}
.pron span{{font-weight:600;color:#222}}
audio{{width:100%;margin:2px 0 12px}}
label{{display:block;font-size:13px;color:#333}}
</style></head><body>
<h1>Kokoro, Qwen, and human references</h1>
<p class="note">n={n} DoSE sentences. Sorted lowest F1 first vs Standard IPA pronunciation (mean {mean:.3f}). Kokoro plain is the uninjected sentence crop. Kokoro + pronunciation is the lexicon crop. Human clips on {n_human} names. Qwen + wav is the isolated Qwen word ({n_qwen} names).</p>
{''.join(cards)}
</body></html>
"""
    LISTEN.mkdir(parents=True, exist_ok=True)
    (LISTEN / "index.html").write_text(page)
    print(f"WROTE {LISTEN / 'index.html'} cards={n} human={n_human} qwen={n_qwen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
