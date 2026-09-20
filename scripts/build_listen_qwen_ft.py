#!/usr/bin/env python3
"""Build a listen page: Qwen plain vs FT vs teacher for debug examples."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "listen-qwen-ft-vs-plain"
PLAIN = ROOT / "runs" / "finetune-qwen-cloud-ipa" / "qwen17-plain-synth"
FT = ROOT / "runs" / "finetune-qwen-cloud-ipa" / "qwen17-ft-synth"
TEACH = ROOT / "data" / "finetune_cloud_ipa" / "wavs"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    manifest = {}
    for line in (ROOT / "data" / "finetune_cloud_ipa" / "manifest.jsonl").open():
        row = json.loads(line)
        manifest[row["ingredient"].lower()] = row
    plain, ft = {}, {}
    for line in (ROOT / "runs" / "finetune-qwen-cloud-ipa" / "qwen17-plain-scores.jsonl").open():
        row = json.loads(line)
        plain[row["ingredient"].lower()] = row
    for line in (ROOT / "runs" / "finetune-qwen-cloud-ipa" / "qwen17-ft-scores.jsonl").open():
        row = json.loads(line)
        ft[row["ingredient"].lower()] = row

    order = [
        "vorasidenib",
        "benadryl",
        "aripiprazole",
        "dupixent",
        "attruby",
        "humira",
        "otezla",
        "diphenhydramine",
    ]
    rows = []
    for key in order:
        if key not in plain or key not in ft:
            continue
        s = slug(key)
        for label, src in (
            ("1_plain", PLAIN / f"{s}.wav"),
            ("2_ft", FT / f"{s}.wav"),
            ("3_teacher", TEACH / f"{s}.wav"),
        ):
            if src.exists():
                shutil.copy2(src, OUT / f"{s}__{label}.wav")
        p, f = plain[key], ft[key]
        delta = None
        if p.get("human_f1") is not None and f.get("human_f1") is not None:
            delta = round(f["human_f1"] - p["human_f1"], 4)
        rows.append(
            {
                "drug": p["ingredient"],
                "slug": s,
                "arm": manifest.get(key, {}).get("arm"),
                "ipa_used": manifest.get(key, {}).get("ipa_used"),
                "plain_human": p.get("human_f1"),
                "ft_human": f.get("human_f1"),
                "delta": delta,
                "plain_teacher": p.get("teacher_f1"),
                "ft_teacher": f.get("teacher_f1"),
            }
        )

    parts = [
        "<!doctype html><html><head><meta charset=utf-8><title>Qwen plain vs FT</title>",
        "<style>body{font-family:system-ui;max-width:820px;margin:24px auto;padding:0 16px;line-height:1.4}",
        "h1{font-size:1.4rem} h2{margin-top:2rem;border-top:1px solid #ddd;padding-top:1rem}",
        ".note{color:#444;background:#f6f6f6;padding:12px;border-radius:8px;font-size:0.95rem}",
        "audio{width:100%;margin:4px 0 12px} .meta{font-size:0.9rem;color:#333}</style></head><body>",
        "<h1>Qwen 1.7B: plain vs fine-tuned (spoken name only)</h1>",
        "<div class=note><b>Debug finding — setup bugs, not proof that FT is useless:</b><ol>",
        "<li><b>Speaker mismatch:</b> teacher clips are Google <code>en-US-Standard-C</code>, "
        "but SFT baked the speaker embedding from Qwen <code>clone.wav</code>. "
        "Targets = Standard-C pronunciation; voice slot = a different speaker.</li>",
        "<li><b>Unfair comparison:</b> plain used <code>generate_voice_clone</code>; "
        "FT used <code>generate_custom_voice</code>. Different APIs.</li>",
        "<li>Training did run (loss ~15 → ~6.5). Wavs are not identical — audio changed; "
        "mean F1 just did not improve under this pairing.</li>",
        "<li>0.6B was skipped because SFT crashed (embed 2048 vs 1024). "
        "1.7B was chosen because it was the model that actually trained, "
        "not because overnight FT is known to be useless.</li>",
        "</ol>Order: gains first, then drops.</div>",
    ]
    for r in rows:
        drug = r["drug"]
        parts.append(f"<h2>{drug}</h2>")
        ipa = r["ipa_used"] or "(plain arm)"
        parts.append(
            f"<div class=meta>arm={r['arm']} · IPA: <code>{ipa}</code><br>"
            f"human F1 plain {r['plain_human']} → FT {r['ft_human']} "
            f"(Δ {r['delta']:+}) · teacher F1 {r['plain_teacher']} → {r['ft_teacher']}</div>"
        )
        for label, title in (
            ("1_plain", "1. Plain base (voice_clone)"),
            ("2_ft", "2. Fine-tuned (custom_voice)"),
            ("3_teacher", "3. Teacher Standard-C (train target)"),
        ):
            wav = f"{r['slug']}__{label}.wav"
            if (OUT / wav).exists():
                parts.append(f"<div>{title}</div><audio controls src=\"{wav}\"></audio>")
    parts.append("</body></html>")
    (OUT / "index.html").write_text("\n".join(parts))
    (OUT / "meta.json").write_text(json.dumps(rows, indent=2))
    print("wrote", OUT, "n_files", len(list(OUT.iterdir())))


if __name__ == "__main__":
    main()
