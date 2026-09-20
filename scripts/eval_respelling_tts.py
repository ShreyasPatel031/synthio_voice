"""Cloud TTS Standard-C: plain spelling vs homogenized dictionary respelling.

Path 2 gold: wavlm-large SpeechBERTScore F1 vs the same human clip.
No IPA, no SSML phoneme tags. The respelling is fed as ordinary English text
(the format DOSE itself publishes).

Two respelling renderings, because ALL-CAPS syllables and hyphens are TTS
footguns (Cloud TTS may spell ZOL letter-by-letter):

  respell_spaced  nu-ZOL-vence -> "nu zol vence"
  respell_hyphen  nu-ZOL-vence -> "nu-zol-vence"
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import wave
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_hf = ROOT / ".cache" / "huggingface"
_hf.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_hf)
os.environ["TRANSFORMERS_CACHE"] = str(_hf)
os.environ["HF_HUB_CACHE"] = str(_hf / "hub")

import requests  # noqa: E402

from dose_r.references.reference_clips import available_clips  # noqa: E402
from dose_r.references.respelling import is_canonical  # noqa: E402

ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
VOICE = "en-US-Standard-C"
RATE = 24000
OUT = ROOT / "runs" / "plain-vs-respelling-tts"
LISTEN = ROOT / "runs" / "listen-plain-vs-respelling"
RESPELLINGS = ROOT / "dose_r" / "references" / "respellings.jsonl"


def _sa_info() -> dict:
    blob = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_B64") or os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS_JSON", ""
    )
    if not blob:
        raise SystemExit("no GOOGLE_APPLICATION_CREDENTIALS_B64 / _JSON")
    blob = blob.strip()
    if blob.startswith("{"):
        return json.loads(blob)
    return json.loads(base64.b64decode(blob))


def token_and_project() -> tuple[str, str]:
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    info = _sa_info()
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(Request())
    return creds.token, info["project_id"]


def synth(tok: str, project: str, *, text: str | None = None, ssml: str | None = None) -> bytes:
    body = {
        "input": {"ssml": ssml} if ssml else {"text": text},
        "voice": {"languageCode": "en-US", "name": VOICE},
        "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": RATE},
    }
    resp = requests.post(
        ENDPOINT,
        headers={
            "Authorization": f"Bearer {tok}",
            "Content-Type": "application/json",
            "x-goog-user-project": project,
        },
        json=body,
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code}: {resp.text[:400]}")
    return base64.b64decode(resp.json()["audioContent"])


def cached(path: Path, tok: str, project: str, **kw) -> bytes:
    if path.exists() and path.stat().st_size > 1000:
        return path.read_bytes()
    audio = synth(tok, project, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio)
    return audio


def wav_duration_s(data: bytes) -> float:
    with wave.open(BytesIO(data)) as w:
        return w.getnframes() / float(w.getframerate())


def spoken_spaced(canon: str) -> str:
    return " ".join(canon.replace("-", " ").split()).lower()


def spoken_hyphen(canon: str) -> str:
    return canon.lower()


_EMB_CACHE: dict[int, object] = {}


def _embed(data: bytes):
    from dose_r.scoring.speech_similarity import extract_frame_embeddings

    key = hash(data)
    if key not in _EMB_CACHE:
        _EMB_CACHE[key] = extract_frame_embeddings(data)
    return _EMB_CACHE[key]


def f1(a: bytes, b: bytes) -> float:
    from dose_r.scoring.speech_similarity import speech_bertscore

    return float(speech_bertscore(_embed(a), _embed(b))["f1"])


def load_respellings() -> dict[str, dict]:
    out = {}
    for line in RESPELLINGS.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("respelling") and is_canonical(rec["respelling"]):
            out[rec["ingredient"].lower()] = rec
    return out


def write_listen(rows: list[dict], tts_dir: Path) -> None:
    LISTEN.mkdir(parents=True, exist_ok=True)
    ranked = sorted(rows, key=lambda r: r["delta_spaced"])
    worst = ranked[:8]
    best = list(reversed(ranked[-8:]))
    sections = [("Respelling worse than plain (largest losses)", worst),
                ("Respelling better than plain (largest gains)", best)]
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Human vs Cloud TTS — plain vs dictionary respelling</title>",
        "<style>body{font:16px/1.4 system-ui;max-width:720px;margin:2rem auto;padding:0 1rem}",
        "section{border:1px solid #ccc;border-radius:8px;padding:1rem 1.2rem;margin:1rem 0}",
        "h1{font-size:1.3rem} h2{font-size:1.05rem;margin:0 0 .4rem}",
        "p{margin:.2rem 0 .6rem;color:#333} .ipa{font-family:ui-monospace,monospace}",
        "label{display:block;font-weight:600;margin:.5rem 0 .15rem} audio{width:100%}",
        ".meta{font-size:.9rem;color:#555}</style>",
        "<h1>Cloud TTS Standard-C: plain spelling vs dictionary respelling</h1>",
        "<p>Same human clip (Path 2 wavlm F1). Respell arm is lowercase spaced ",
        "canonical form (<span class='ipa'>nu zol vence</span>), not IPA.</p>",
    ]
    for title, group in sections:
        parts.append(f"<h1>{title}</h1>")
        for rec in group:
            slug = rec["key"]
            for label, src in (
                ("1_human", rec["human_path"]),
                ("2_plain", tts_dir / f"{slug}.plain.wav"),
                ("3_respell", tts_dir / f"{slug}.spaced.wav"),
            ):
                dest = LISTEN / f"{slug}__{label}.wav"
                dest.write_bytes(Path(src).read_bytes())
            parts.append(f"<section id='{slug}'><h2>{rec['ingredient']}</h2>")
            parts.append(
                f"<p>canonical: <span class='ipa'>{rec['respelling']}</span><br>"
                f"fed as: <span class='ipa'>{rec['spaced']}</span><br>"
                f"source: {rec['source']} · human: {rec['clip_source']}</p>"
            )
            parts.append(
                f"<p class='meta'>plain {rec['plain']:.3f} &nbsp; spaced {rec['spaced_f1']:.3f} "
                f"&nbsp; hyphen {rec['hyphen_f1']:.3f} &nbsp; Δ spaced {rec['delta_spaced']:+.3f}</p>"
            )
            parts.append(f"<label>1. Human</label><audio controls src='{slug}__1_human.wav'></audio>")
            parts.append(f"<label>2. Plain spelling</label><audio controls src='{slug}__2_plain.wav'></audio>")
            parts.append(
                f"<label>3. Dictionary respelling (spaced)</label>"
                f"<audio controls src='{slug}__3_respell.wav'></audio></section>"
            )
    (LISTEN / "index.html").write_text("\n".join(parts))


def diagnose(rows: list[dict], tok: str, project: str, tts_dir: Path) -> list[dict]:
    """On the 12 worst spaced-vs-plain losses: SSML <sub alias> and CAPS-kept."""
    worst = sorted(rows, key=lambda r: r["delta_spaced"])[:12]
    out = []
    for rec in worst:
        slug = rec["key"]
        human = Path(rec["human_path"]).read_bytes()
        spoken = rec["ingredient"]
        alias = rec["spaced"]
        caps = rec["respelling"].replace("-", " ")
        ssml = (
            f'<speak><sub alias="{xml_escape(alias)}">{xml_escape(spoken)}</sub></speak>'
        )
        caps_wav = cached(tts_dir / f"{slug}.caps.wav", tok, project, text=caps)
        ssml_wav = cached(tts_dir / f"{slug}.ssml_sub.wav", tok, project, ssml=ssml)
        row = {
            "ingredient": rec["ingredient"],
            "plain": rec["plain"],
            "spaced": rec["spaced_f1"],
            "hyphen": rec["hyphen_f1"],
            "caps_kept": f1(caps_wav, human),
            "ssml_sub": f1(ssml_wav, human),
            "dur_human": rec["dur_human"],
            "dur_plain": rec["dur_plain"],
            "dur_spaced": rec["dur_spaced"],
            "dur_caps": round(wav_duration_s(caps_wav), 3),
            "dur_ssml": round(wav_duration_s(ssml_wav), 3),
        }
        out.append(row)
        print(
            f"DIAG {rec['ingredient']:20s} plain={row['plain']:.3f} "
            f"spaced={row['spaced']:.3f} caps={row['caps_kept']:.3f} "
            f"ssml={row['ssml_sub']:.3f} dur_p/s="
            f"{row['dur_plain']:.2f}/{row['dur_spaced']:.2f}",
            flush=True,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-listen", action="store_true")
    args = ap.parse_args()

    resp = load_respellings()
    clips = available_clips()
    items = []
    for ing, clip in sorted(clips.items(), key=lambda kv: kv[0].lower()):
        rec = resp.get(ing.lower())
        if not rec:
            continue
        items.append((ing, clip, rec))
    if args.limit:
        items = items[: args.limit]
    print(f"overlap clips×respelling: {len(items)}", flush=True)

    tok, project = token_and_project()
    tts_dir = OUT / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (ing, clip, rec) in enumerate(items, 1):
        slug = ing.lower().replace(" ", "_")
        human = clip.path.read_bytes()
        spaced = spoken_spaced(rec["respelling"])
        hyphen = spoken_hyphen(rec["respelling"])
        plain = cached(tts_dir / f"{slug}.plain.wav", tok, project, text=ing)
        sp = cached(tts_dir / f"{slug}.spaced.wav", tok, project, text=spaced)
        hy = cached(tts_dir / f"{slug}.hyphen.wav", tok, project, text=hyphen)
        scores = {
            "plain": f1(plain, human),
            "spaced": f1(sp, human),
            "hyphen": f1(hy, human),
        }
        row = {
            "key": slug,
            "ingredient": ing,
            "clip_source": clip.source,
            "source": rec["source"],
            "respelling": rec["respelling"],
            "spaced": spaced,
            "hyphen": hyphen,
            "human_path": str(clip.path),
            "plain": scores["plain"],
            "spaced_f1": scores["spaced"],
            "hyphen_f1": scores["hyphen"],
            "delta_spaced": round(scores["spaced"] - scores["plain"], 4),
            "delta_hyphen": round(scores["hyphen"] - scores["plain"], 4),
            "dur_human": round(clip.duration_s, 3),
            "dur_plain": round(wav_duration_s(plain), 3),
            "dur_spaced": round(wav_duration_s(sp), 3),
            "dur_hyphen": round(wav_duration_s(hy), 3),
        }
        rows.append(row)
        print(
            f"{i:3d}/{len(items)} {ing:28s} plain={scores['plain']:.3f} "
            f"spaced={scores['spaced']:.3f} ({row['delta_spaced']:+.3f}) "
            f"hyphen={scores['hyphen']:.3f} ({row['delta_hyphen']:+.3f})",
            flush=True,
        )

    def mean(key):
        return sum(r[key] for r in rows) / len(rows)

    n = len(rows)
    summary = {
        "n": n,
        "voice": VOICE,
        "mean_plain": round(mean("plain"), 4),
        "mean_spaced": round(mean("spaced_f1"), 4),
        "mean_hyphen": round(mean("hyphen_f1"), 4),
        "mean_delta_spaced": round(mean("delta_spaced"), 4),
        "mean_delta_hyphen": round(mean("delta_hyphen"), 4),
        "spaced_beats_plain": sum(1 for r in rows if r["delta_spaced"] > 0.01),
        "hyphen_beats_plain": sum(1 for r in rows if r["delta_hyphen"] > 0.01),
        "spaced_worse": sum(1 for r in rows if r["delta_spaced"] < -0.01),
        "hyphen_worse": sum(1 for r in rows if r["delta_hyphen"] < -0.01),
        "mean_dur_human": round(mean("dur_human"), 3),
        "mean_dur_plain": round(mean("dur_plain"), 3),
        "mean_dur_spaced": round(mean("dur_spaced"), 3),
        "mean_dur_hyphen": round(mean("dur_hyphen"), 3),
        "worst_spaced": sorted(rows, key=lambda r: r["delta_spaced"])[:8],
        "best_spaced": sorted(rows, key=lambda r: r["delta_spaced"], reverse=True)[:8],
    }
    print("\nMEANS", json.dumps({k: summary[k] for k in summary if k not in ("worst_spaced", "best_spaced")}), flush=True)

    diag = []
    if summary["mean_delta_spaced"] < 0.005:
        print("\nspaced did not beat plain — running SSML/CAPS diagnose on 12 worst", flush=True)
        diag = diagnose(rows, tok, project, tts_dir)
        summary["diagnose_n"] = len(diag)
        summary["diagnose_mean_caps"] = round(sum(d["caps_kept"] for d in diag) / len(diag), 4) if diag else None
        summary["diagnose_mean_ssml"] = round(sum(d["ssml_sub"] for d in diag) / len(diag), 4) if diag else None
        summary["diagnose"] = diag

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "rows": rows}
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))
    if not args.skip_listen:
        write_listen(rows, tts_dir)
        print("listen page", LISTEN / "index.html", flush=True)
    print("wrote", OUT / "results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
