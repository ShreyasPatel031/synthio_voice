"""Uniform pronunciation fields vs plain spelling, Path 2 gold.

Same Cloud TTS voice, same human clip, same wavlm F1 as the spaced-respelling
eval. Two arms that can be applied to every name, including the 40% with no
audio (they just cannot be *scored* here):

  compact   canonical respelling with hyphens stripped (weegohvee)
  ipa       spelling unchanged + customPronunciations IPA sidecar

Plain and spaced scores are reused from runs/plain-vs-respelling-tts when
present so this run does not move the baseline.
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
from dose_r.references.tts_pronunciation import (  # noqa: E402
    compact_ascii,
    custom_pronunciation,
    ipa_from_canonical,
)

ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
VOICE = "en-US-Standard-C"
RATE = 24000
PRIOR = ROOT / "runs" / "plain-vs-respelling-tts"
OUT = ROOT / "runs" / "uniform-pron-tts"
LISTEN = ROOT / "runs" / "listen-uniform-pron"
RESPELLINGS = ROOT / "dose_r" / "references" / "respellings.jsonl"
HOLDOUT = ROOT / "dose_r" / "references" / "plain_tts_gap_holdout.json"


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


def synth(
    tok: str,
    project: str,
    *,
    text: str,
    pronunciations: dict | None = None,
) -> bytes:
    inp: dict = {"text": text}
    if pronunciations:
        inp["customPronunciations"] = pronunciations
    body = {
        "input": inp,
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


def load_prior() -> dict[str, dict]:
    path = PRIOR / "results.json"
    if not path.exists():
        return {}
    return {r["key"]: r for r in json.loads(path.read_text())["rows"]}


def write_listen(rows: list[dict], tts_dir: Path, holdout: set[str]) -> None:
    LISTEN.mkdir(parents=True, exist_ok=True)
    focus = [r for r in rows if r["ingredient"] in holdout]
    focus.sort(key=lambda r: r["plain"])
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Uniform pronunciation vs human — worst 25</title>",
        "<style>body{font:16px/1.4 system-ui;max-width:760px;margin:2rem auto;padding:0 1rem}",
        "section{border:1px solid #ccc;border-radius:8px;padding:1rem 1.2rem;margin:1rem 0}",
        "h1{font-size:1.3rem} h2{font-size:1.05rem;margin:0 0 .4rem}",
        "p{margin:.2rem 0 .6rem;color:#333} .ipa{font-family:ui-monospace,monospace}",
        "label{display:block;font-weight:600;margin:.5rem 0 .15rem} audio{width:100%}",
        ".meta{font-size:.9rem;color:#555}</style>",
        "<h1>Same voice, same human clip. Text stays the spelling on IPA.</h1>",
        "<p>compact = one ASCII token from the dictionary string. ",
        "ipa = customPronunciations sidecar, spelling unchanged.</p>",
    ]
    for rec in focus:
        slug = rec["key"]
        for label, src in (
            ("1_human", rec["human_path"]),
            ("2_plain", tts_dir / f"{slug}.plain.wav"),
            ("3_compact", tts_dir / f"{slug}.compact.wav"),
            ("4_ipa", tts_dir / f"{slug}.ipa.wav"),
        ):
            dest = LISTEN / f"{slug}__{label}.wav"
            dest.write_bytes(Path(src).read_bytes())
        parts.append(f"<section id='{slug}'><h2>{rec['ingredient']}</h2>")
        parts.append(
            f"<p>canonical: <span class='ipa'>{rec['respelling']}</span><br>"
            f"compact: <span class='ipa'>{rec['compact']}</span><br>"
            f"ipa: <span class='ipa'>{rec['ipa']}</span></p>"
        )
        parts.append(
            f"<p class='meta'>plain {rec['plain']:.3f} &nbsp; "
            f"compact {rec['compact_f1']:.3f} ({rec['delta_compact']:+.3f}) &nbsp; "
            f"ipa {rec['ipa_f1']:.3f} ({rec['delta_ipa']:+.3f}) &nbsp; "
            f"spaced {rec['spaced_f1']:.3f}</p>"
        )
        parts.append(f"<label>1. Human</label><audio controls src='{slug}__1_human.wav'></audio>")
        parts.append(f"<label>2. Plain spelling</label><audio controls src='{slug}__2_plain.wav'></audio>")
        parts.append(f"<label>3. Compact ASCII</label><audio controls src='{slug}__3_compact.wav'></audio>")
        parts.append(
            f"<label>4. IPA sidecar (spelling unchanged)</label>"
            f"<audio controls src='{slug}__4_ipa.wav'></audio></section>"
        )
    (LISTEN / "index.html").write_text("\n".join(parts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-listen", action="store_true")
    args = ap.parse_args()

    resp = load_respellings()
    clips = available_clips()
    prior = load_prior()
    holdout = set()
    if HOLDOUT.exists():
        payload = json.loads(HOLDOUT.read_text())
        holdout = {r["ingredient"] for r in payload.get("core_holdout", [])}

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
    prior_tts = PRIOR / "tts"

    rows = []
    for i, (ing, clip, rec) in enumerate(items, 1):
        slug = ing.lower().replace(" ", "_")
        human = clip.path.read_bytes()
        canon = rec["respelling"]
        compact = compact_ascii(canon)
        ipa = ipa_from_canonical(canon)

        plain_path = tts_dir / f"{slug}.plain.wav"
        if not plain_path.exists() and (prior_tts / f"{slug}.plain.wav").exists():
            plain_path.write_bytes((prior_tts / f"{slug}.plain.wav").read_bytes())
        plain = cached(plain_path, tok, project, text=ing)
        comp = cached(tts_dir / f"{slug}.compact.wav", tok, project, text=compact)
        ipa_wav = cached(
            tts_dir / f"{slug}.ipa.wav",
            tok,
            project,
            text=ing,
            pronunciations=custom_pronunciation(ing, ipa),
        )

        prev = prior.get(slug, {})
        scores = {
            "plain": prev["plain"] if "plain" in prev else f1(plain, human),
            "compact": f1(comp, human),
            "ipa": f1(ipa_wav, human),
            "spaced": prev.get("spaced_f1"),
        }
        row = {
            "key": slug,
            "ingredient": ing,
            "clip_source": clip.source,
            "source": rec["source"],
            "respelling": canon,
            "compact": compact,
            "ipa": ipa,
            "human_path": str(clip.path),
            "plain": scores["plain"],
            "compact_f1": scores["compact"],
            "ipa_f1": scores["ipa"],
            "spaced_f1": scores["spaced"],
            "delta_compact": round(scores["compact"] - scores["plain"], 4),
            "delta_ipa": round(scores["ipa"] - scores["plain"], 4),
            "dur_human": round(clip.duration_s, 3),
            "dur_plain": round(wav_duration_s(plain), 3),
            "dur_compact": round(wav_duration_s(comp), 3),
            "dur_ipa": round(wav_duration_s(ipa_wav), 3),
            "in_holdout": ing in holdout,
        }
        rows.append(row)
        print(
            f"{i:3d}/{len(items)} {ing:28s} plain={scores['plain']:.3f} "
            f"compact={scores['compact']:.3f} ({row['delta_compact']:+.3f}) "
            f"ipa={scores['ipa']:.3f} ({row['delta_ipa']:+.3f})",
            flush=True,
        )

    def mean(key, subset=None):
        xs = subset if subset is not None else rows
        xs = [r[key] for r in xs if r.get(key) is not None]
        return sum(xs) / len(xs) if xs else None

    hold_rows = [r for r in rows if r["in_holdout"]]
    summary = {
        "n": len(rows),
        "voice": VOICE,
        "mean_plain": round(mean("plain"), 4),
        "mean_compact": round(mean("compact_f1"), 4),
        "mean_ipa": round(mean("ipa_f1"), 4),
        "mean_spaced": round(mean("spaced_f1"), 4) if mean("spaced_f1") else None,
        "mean_delta_compact": round(mean("delta_compact"), 4),
        "mean_delta_ipa": round(mean("delta_ipa"), 4),
        "compact_beats_plain": sum(1 for r in rows if r["delta_compact"] > 0.01),
        "ipa_beats_plain": sum(1 for r in rows if r["delta_ipa"] > 0.01),
        "compact_worse": sum(1 for r in rows if r["delta_compact"] < -0.01),
        "ipa_worse": sum(1 for r in rows if r["delta_ipa"] < -0.01),
        "holdout_n": len(hold_rows),
        "holdout_mean_plain": round(mean("plain", hold_rows), 4) if hold_rows else None,
        "holdout_mean_compact": round(mean("compact_f1", hold_rows), 4) if hold_rows else None,
        "holdout_mean_ipa": round(mean("ipa_f1", hold_rows), 4) if hold_rows else None,
        "mean_dur_human": round(mean("dur_human"), 3),
        "mean_dur_plain": round(mean("dur_plain"), 3),
        "mean_dur_compact": round(mean("dur_compact"), 3),
        "mean_dur_ipa": round(mean("dur_ipa"), 3),
        "best_compact": sorted(rows, key=lambda r: r["delta_compact"], reverse=True)[:8],
        "best_ipa": sorted(rows, key=lambda r: r["delta_ipa"], reverse=True)[:8],
        "worst_compact": sorted(rows, key=lambda r: r["delta_compact"])[:8],
        "worst_ipa": sorted(rows, key=lambda r: r["delta_ipa"])[:8],
    }
    skip = {"best_compact", "best_ipa", "worst_compact", "worst_ipa"}
    print("\nMEANS", json.dumps({k: summary[k] for k in summary if k not in skip}), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    if not args.skip_listen and holdout:
        write_listen(rows, tts_dir, holdout)
        print("listen page", LISTEN / "index.html", flush=True)
    print("wrote", OUT / "results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
