#!/usr/bin/env python3
"""Decode gold vs Misaki clips with wav2vec2-espeak CTC and compare to IPA targets."""

from __future__ import annotations

import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dose_r.forced_align import extract_drug_span_forced_align
from dose_r.scoring.phoneme_distance import normalize_phonemes, phoneme_distance_to_variant
from dose_r.scoring.phoneme_model import transcribe_phonemes

DEFAULT_SLUGS = ["icotyde", "idvynso", "vorasidenib", "advair"]

META = {
    "icotyde": {"gold_manifest": "aɪˈkoʊtaɪd", "gold_user": "aɪˈkoʊtaɪd"},
    "idvynso": {"gold_manifest": "ɪdˈvɪnsoʊ", "gold_user": "ɪdˈvɪnsoʊ"},
    "vorasidenib": {
        "gold_manifest": "vɔːrəˈsɪdənɪb",
        "gold_user": "ˌvɔːrəˈsɪdənɪb",
    },
    "advair": {"gold_manifest": "ˈædvɛr", "gold_user": "ˈædvɛɹ"},
}


def decode(path: Path) -> tuple[str, str]:
    raw = transcribe_phonemes(path)
    return raw, normalize_phonemes(raw)


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def dist(cand: str, ref: str) -> tuple[float, float]:
    d, n = phoneme_distance_to_variant(cand, ref)
    return d, d / n


def load_sentences() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for line in (ROOT / "data/dose_v1.jsonl").read_text().splitlines():
        row = json.loads(line)
        slug = row["name"].lower().replace(" ", "_")
        out[slug] = {"drug": row["name"], "sentence": row["sentence"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", action="append", dest="slugs")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()
    slugs = args.slugs or DEFAULT_SLUGS

    sentences = load_sentences()
    chosen_blob = json.loads((ROOT / "runs/misaki-iter/chosen.json").read_text())
    phones = chosen_blob.get("phones", chosen_blob)
    pins = json.loads((ROOT / "runs/misaki-iter/user_pins.json").read_text())
    rank = {
        r["slug"]: r
        for r in json.loads((ROOT / "runs/misaki-iter/cloud-rank.json").read_text())["rows"]
    }

    report: list[dict] = []
    for slug in slugs:
        meta = META[slug]
        sent = sentences[slug]
        pin = pins.get(slug, {}).get("misaki")
        stored = phones.get(slug)
        row: dict = {
            "slug": slug,
            "drug": sent["drug"],
            "sentence": sent["sentence"],
            "ctc_f1": rank.get(slug, {}).get("ctc_f1"),
            "misaki_pin": pin,
            "misaki_stored": stored,
            "gold_manifest": meta["gold_manifest"],
            "gold_user": meta["gold_user"],
            "clips": {},
            "comparisons": [],
        }

        print("=" * 72)
        print(f"{slug} ({sent['drug']})  CTC F1={row['ctc_f1']}")
        print(f"sentence: {sent['sentence']}")
        print(f"misaki pin:    {pin!r}")
        print(f"misaki stored: {stored!r}")
        print(f"gold manifest: /{meta['gold_manifest']}/")
        print(f"gold user:     /{meta['gold_user']}/")

        clip_paths = {
            "gold_iso": ROOT / "data/gold_gemini_ipa/wavs" / f"{slug}.wav",
            "gold_sent": ROOT / "runs/misaki-iter/cloud-gold" / f"{slug}.wav",
            "misaki_iso": ROOT / "runs/misaki-iter/cloud-iso" / f"{slug}.wav",
            "misaki_sent": ROOT / "runs/misaki-iter/user-sent" / f"{slug}.wav",
            "misaki_span_file": ROOT / "runs/misaki-iter/cloud-span" / f"{slug}.wav",
            "misaki_user_span": ROOT / "runs/misaki-iter/user-span" / f"{slug}.wav",
        }

        decoded: dict[str, tuple[str, str]] = {}
        for label, path in clip_paths.items():
            if not path.exists():
                print(f"\n[{label}] MISSING")
                continue
            raw, norm = decode(path)
            decoded[label] = (raw, norm)
            row["clips"][label] = {"raw": raw, "norm": norm, "path": str(path)}
            print(f"\n[{label}]")
            print(f"  raw:  {raw!r}")
            print(f"  norm: {norm}")

        sent_path = clip_paths["misaki_sent"]
        if sent_path.exists():
            span = extract_drug_span_forced_align(
                sent_path.read_bytes(), sent["sentence"], sent["drug"]
            )
            raw, norm = decode(span)
            decoded["misaki_span_live"] = (raw, norm)
            row["clips"]["misaki_span_live"] = {"raw": raw, "norm": norm}
            print("\n[misaki_span_live] forced align on user-sent")
            print(f"  raw:  {raw!r}")
            print(f"  norm: {norm}")

        for ref_label, ref in [
            ("manifest", meta["gold_manifest"]),
            ("user", meta["gold_user"]),
        ]:
            ref_norm = normalize_phonemes(ref)
            print(f"\n  -- vs gold {ref_label} norm={ref_norm} --")
            for clip_label, (_, norm) in decoded.items():
                d, rate = dist(norm, ref)
                cmp = {
                    "clip": clip_label,
                    "ref": ref_label,
                    "sim": round(sim(norm, ref_norm), 4),
                    "feat_dist": round(d, 3),
                    "rate": round(rate, 3),
                }
                row["comparisons"].append(cmp)
                print(
                    f"    {clip_label:18} sim={cmp['sim']:.3f} "
                    f"feat_dist={cmp['feat_dist']:.2f} rate={cmp['rate']:.3f}"
                )

        if "gold_iso" in decoded and "misaki_iso" in decoded:
            pair_sim = sim(decoded["gold_iso"][1], decoded["misaki_iso"][1])
            print(f"\n  gold_iso vs misaki_iso sim={pair_sim:.3f}")
            row["gold_vs_misaki_iso_sim"] = round(pair_sim, 4)
        if "gold_iso" in decoded and "misaki_span_live" in decoded:
            pair_sim = sim(decoded["gold_iso"][1], decoded["misaki_span_live"][1])
            print(f"  gold_iso vs misaki_span_live sim={pair_sim:.3f}")
            row["gold_vs_misaki_span_sim"] = round(pair_sim, 4)

        report.append(row)

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"\nWROTE {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
