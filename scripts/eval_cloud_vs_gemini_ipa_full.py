#!/usr/bin/env python
"""Cloud Standard-C + IPA vs Gemini 3.1 + IPA on all DOSE names.

Reference:
  human clip when one exists (~60%)
  Gemini 3.1 compact (hyphens stripped) when no clip (~40%)

Same source IPA from runs/gemini3-ipa-query/ipa.json. Never G2P.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
_hf = ROOT / ".cache" / "huggingface"
_hf.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_hf))
os.environ.setdefault("HF_HUB_CACHE", str(_hf / "hub"))

from dose_r.references.reference_clips import available_clips  # noqa: E402
from dose_r.references.tts_pronunciation import (  # noqa: E402
    compact_ascii,
    custom_pronunciation,
    spoken_text,
)
from eval_gemini31_holdout import (  # noqa: E402
    MODEL,
    RATE,
    VOICE,
    f1,
    mean,
    slug,
)
from eval_gemini31_ipa_vs_cloud import cached, synth_gemini  # noqa: E402

IPA_JSON = ROOT / "runs" / "gemini3-ipa-query" / "ipa.json"
PRON = ROOT / "dose_r" / "references" / "pronunciations.jsonl"
VERIFY_TTS = ROOT / "runs" / "verify-google-ipa" / "tts"
RESPELL_TTS = ROOT / "runs" / "ipa-vs-respell" / "tts"
GEM_TTS = ROOT / "runs" / "gemini31-ipa-vs-cloud" / "tts"
COMPACT_TTS = ROOT / "runs" / "gemini31-respell-all" / "tts"
OUT = ROOT / "runs" / "cloud-vs-gemini-ipa-full"
WORKERS = int(os.environ.get("TTS_WORKERS", "4"))
GAP = 0.03
PROMPT_IPA = (
    "Pronounce this US drug name using the given IPA exactly. "
    "Do not spell letters. IPA: {ipa}"
)
PROMPT_COMPACT = "Pronounce this US drug name clearly as a single name."


def dose_names() -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    # Prefer jsonl so we do not need the parquet staged.
    dose = ROOT / "data" / "dose_v1.jsonl"
    for line in dose.read_text().splitlines():
        for name in json.loads(line)["ingredients"]:
            if name.lower() not in seen:
                seen.add(name.lower())
                names.append(name)
    return names


def unwrap_ipa(raw: str) -> str:
    s = (raw or "").strip().strip("/[]() ")
    return s.replace("'", "ˈ")


def cloud_ipa_wav(name: str) -> Path | None:
    s = slug(name)
    s2 = s.replace("-", "_")
    for p in (
        VERIFY_TTS / f"{s}.ipa.wav",
        VERIFY_TTS / f"{s2}.ipa.wav",
        RESPELL_TTS / f"{s}.ipa.wav",
        RESPELL_TTS / f"{s2}.ipa.wav",
    ):
        if p.exists() and p.stat().st_size > 500:
            return p
    return None


def gemini_ipa_wav(name: str) -> Path | None:
    s = slug(name)
    for p in (GEM_TTS / f"{s}.sidecar.wav", GEM_TTS / f"{s}.prompt.wav"):
        if p.exists() and p.stat().st_size > 500:
            return p
    return None


def load_pron() -> dict[str, dict]:
    out = {}
    for line in PRON.read_text().splitlines():
        rec = json.loads(line)
        out[rec["ingredient"].lower()] = rec
    return out


def compact_text(name: str, pron: dict[str, dict]) -> str:
    rec = pron.get(name.lower()) or {}
    canon = (rec.get("respelling") or "").strip()
    return compact_ascii(canon) or spoken_text(name)


def winner(a: float, b: float) -> str:
    if a >= b + GAP:
        return "gemini"
    if b >= a + GAP:
        return "cloud"
    return "tie"


def main() -> int:
    names = dose_names()
    ipa_map = {r["ingredient"].lower(): r for r in json.loads(IPA_JSON.read_text())}
    pron = load_pron()
    clips = available_clips()
    clip_key = {k.lower(): v for k, v in clips.items()}
    GEM_TTS.mkdir(parents=True, exist_ok=True)
    COMPACT_TTS.mkdir(parents=True, exist_ok=True)

    jobs = []
    compact_jobs = []
    for name in names:
        ipa = unwrap_ipa((ipa_map.get(name.lower()) or {}).get("ipa") or "")
        if not ipa:
            continue
        s = slug(name)
        if gemini_ipa_wav(name) is None:
            text = spoken_text(name)
            phrase = text if " " not in text else text.split()[0]
            if phrase not in text:
                phrase = text.split()[0]
            jobs.append((name, ipa, text, phrase, GEM_TTS / f"{s}.sidecar.wav"))
        if name.lower() not in clip_key:
            cp = COMPACT_TTS / f"{s}.compact.wav"
            if not cp.exists() or cp.stat().st_size < 500:
                compact_jobs.append((name, compact_text(name, pron), cp))

    print(
        f"{len(names)} names; synth gemini+IPA {len(jobs)}; synth compact {len(compact_jobs)}",
        flush=True,
    )

    def gem_job(name, ipa, text, phrase, path):
        try:
            return cached(
                path,
                synth_gemini,
                text,
                PROMPT_IPA.format(ipa=ipa),
                custom_pronunciation(phrase, ipa),
            )
        except Exception as exc:
            # Multi-word / invalid phrase: IPA in the prompt only.
            if "400" not in str(exc) and "INVALID_ARGUMENT" not in str(exc):
                raise
            alt = GEM_TTS / f"{slug(name)}.prompt.wav"
            return cached(
                alt,
                synth_gemini,
                text,
                PROMPT_IPA.format(ipa=ipa),
                None,
            )

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {
            ex.submit(gem_job, name, ipa, text, phrase, path): name
            for name, ipa, text, phrase, path in jobs
        }
        done = 0
        for fut in as_completed(futs):
            name = futs[fut]
            done += 1
            try:
                fut.result()
                print(f"  gemini-ipa {done}/{len(jobs)} {name}", flush=True)
            except Exception as exc:
                print(f"  FAIL gemini-ipa {name}: {exc}", flush=True)

        futs2 = {
            ex.submit(cached, path, synth_gemini, text, PROMPT_COMPACT, None): name
            for name, text, path in compact_jobs
        }
        done = 0
        for fut in as_completed(futs2):
            name = futs2[fut]
            done += 1
            try:
                fut.result()
                print(f"  compact {done}/{len(compact_jobs)} {name}", flush=True)
            except Exception as exc:
                print(f"  FAIL compact {name}: {exc}", flush=True)

    rows = []
    for name in names:
        rec = ipa_map.get(name.lower()) or {}
        ipa = unwrap_ipa(rec.get("ipa") or "")
        cloud = cloud_ipa_wav(name)
        gem = gemini_ipa_wav(name)
        clip = clip_key.get(name.lower())
        compact = COMPACT_TTS / f"{slug(name)}.compact.wav"
        if clip is not None:
            ref, ref_kind = clip.path, "human"
            ref_src = clip.source
        elif compact.exists() and compact.stat().st_size > 500:
            ref, ref_kind = compact, "gemini_compact"
            ref_src = "gemini-3.1-compact"
        else:
            rows.append(
                {
                    "ingredient": name,
                    "ipa": rec.get("ipa"),
                    "error": "no_reference",
                }
            )
            continue
        if cloud is None or gem is None:
            rows.append(
                {
                    "ingredient": name,
                    "ipa": rec.get("ipa"),
                    "ref": ref_kind,
                    "error": "missing_tts",
                    "has_cloud": cloud is not None,
                    "has_gemini": gem is not None,
                }
            )
            continue
        g = round(f1(gem, ref), 4)
        c = round(f1(cloud, ref), 4)
        gc = round(f1(gem, cloud), 4)
        row = {
            "ingredient": name,
            "ipa": rec.get("ipa"),
            "ref": ref_kind,
            "ref_source": ref_src,
            "gemini_f1": g,
            "cloud_f1": c,
            "gemini_vs_cloud": gc,
            "delta_gemini_minus_cloud": round(g - c, 4),
            "winner": winner(g, c),
        }
        rows.append(row)
        print(
            f"  {name} ref={ref_kind} G={g} C={c} Δ={g-c:+.4f} {row['winner']}",
            flush=True,
        )

    scored = [r for r in rows if r.get("gemini_f1") is not None]
    human = [r for r in scored if r["ref"] == "human"]
    compact_rows = [r for r in scored if r["ref"] == "gemini_compact"]

    def block(xs: list[dict]) -> dict:
        return {
            "n": len(xs),
            "mean_gemini": mean([r["gemini_f1"] for r in xs]),
            "mean_cloud": mean([r["cloud_f1"] for r in xs]),
            "mean_gemini_vs_cloud": mean(
                [r["gemini_vs_cloud"] for r in xs if r.get("gemini_vs_cloud") is not None]
            ),
            "gemini_wins": sum(1 for r in xs if r["winner"] == "gemini"),
            "cloud_wins": sum(1 for r in xs if r["winner"] == "cloud"),
            "ties": sum(1 for r in xs if r["winner"] == "tie"),
        }

    summary = {
        "n_dose": len(names),
        "n_scored": len(scored),
        "n_errors": len(rows) - len(scored),
        "model": MODEL,
        "gemini_voice": VOICE,
        "cloud_voice": "en-US-Standard-C",
        "gap": GAP,
        "all": block(scored),
        "vs_human": block(human),
        "vs_gemini_compact": block(compact_rows),
        "higher_f1_all": (
            "gemini"
            if (block(scored)["mean_gemini"] or 0) > (block(scored)["mean_cloud"] or 0)
            else "cloud"
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False)
        + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(OUT / "results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
