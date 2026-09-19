#!/usr/bin/env python
"""Select a small, failure-skewed subset of the 274-item benchmark for
Workstream 1 to run new candidate TTS models against, instead of always
scoring the full corpus.

Why this exists
----------------
Running a new candidate model through the full 274-item pipeline (synthesis
+ forced-alignment extraction + wavlm embeddings) costs real API spend and
~an hour of this box's CPU time per pass (see runs/gemini-flash-tts-v1-*
timing this session). Most of that time is spent on items every system
already gets right (Valium, Tylenol, ...) -- they carry almost no signal
about whether a NEW candidate is actually better or worse. A subset that
deliberately over-represents known-hard items surfaces regressions/wins
much faster per dollar and per minute, at the cost of not being a
population-representative accuracy estimate (it is not meant to be one --
see "Reading this correctly" in the output).

Tiers, each pulling from real data already collected this session -- no
tier is a guess:
  A. confirmed-bad-by-ear     -- items a human listened to and confirmed are
                                  genuinely mispronounced by Gemini (vyloy,
                                  adquey, voranigo). Small, always included.
  B. reference-flagged        -- items where the human REFERENCE clip itself
                                  was found (this session) to disagree with
                                  its own dictionary IPA (talquetamab,
                                  acoramidis). Included but tagged: a
                                  candidate model scoring badly here might be
                                  the reference's fault, not the model's.
  C. gemini-worst-quartile    -- bottom 25% of the real gemini-2.5-flash-tts
                                  v4 run (runs/gemini-flash-tts-v1-speech-
                                  similarity-v4), excluding tiers A/B.
  D. structurally-hard-asr    -- items where Cloud STT fails to transcribe
                                  even a HUMAN saying the name correctly
                                  (runs/reference-grounded-v1/reference_
                                  transcripts.jsonl, asr_recognizable=False).
                                  This is a naming-difficulty signal
                                  independent of any TTS system's quality.
  E. dual-reference-coverage  -- items with BOTH a Drugs.com and a
                                  Merriam-Webster human clip, needed to
                                  exercise/validate multi-reference (best-of-
                                  N) scoring itself, not just to test hard
                                  pronunciations.
  F. ceiling-control          -- top-quartile Gemini v4 scorers. Without
                                  these, a new candidate that fails on
                                  EVERYTHING would look identical to one that
                                  fails only on hard items -- need some easy
                                  items to confirm the model can pass at all.
  G. no-reference-coverage    -- a sample of items with NO human clip and no
                                  dictionary IPA disagreement checked (the
                                  99-item gap discussed this session). Cannot
                                  be scored by Path 2 today; included so the
                                  subset doesn't silently ignore 36% of the
                                  real benchmark's composition.

Brand/generic ratio and tier sizes are reported in the output, not silently
forced to match the full corpus -- see "Reading this correctly".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dose_r import dataset  # noqa: E402
from dose_r.references.reference_clips import available_clips  # noqa: E402

CONFIRMED_BAD_BY_EAR = {"vyloy", "adquey", "voranigo"}
REFERENCE_FLAGGED = {"talquetamab", "acoramidis"}

TARGET_SIZE = 55
QUARTILE_C_N = 20   # gemini-worst-quartile
QUARTILE_D_N = 8    # structurally-hard-asr
QUARTILE_E_N = 8    # dual-reference-coverage
QUARTILE_F_N = 8    # ceiling-control
QUARTILE_G_N = 8    # no-reference-coverage


def main() -> int:
    items = dataset.load_items()
    items_by_id = {i.item_id: i for i in items}
    clips = available_clips()

    v4_path = REPO_ROOT / "runs" / "gemini-flash-tts-v1-speech-similarity-v4" / "results.jsonl"
    v4_recs = [json.loads(l) for l in v4_path.read_text().splitlines() if l.strip()]
    v4_by_id = {r["item_id"]: r for r in v4_recs}
    scoreable = [r for r in v4_recs if r["score"] and r["score"].get("scoreable")]
    scoreable.sort(key=lambda r: r["score"]["score"])
    n = len(scoreable)

    transcripts_path = REPO_ROOT / "runs" / "reference-grounded-v1" / "reference_transcripts.jsonl"
    transcripts = [json.loads(l) for l in transcripts_path.read_text().splitlines() if l.strip()]
    unrecognizable = {t["ingredient"].lower() for t in transcripts if not t["asr_recognizable"]}

    manifest_path = REPO_ROOT / "data" / "reference_audio" / "manifest.jsonl"
    manifest_recs = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    dc = {r["ingredient"] for r in manifest_recs if r.get("source") == "drugs.com" and r.get("coverage") == "full"}
    mw = {r["ingredient"] for r in manifest_recs if r.get("source") == "merriam-webster" and r.get("coverage") == "full"}
    dual_ref = dc & mw

    selected: dict[str, dict] = {}

    def add(item_id: str, tier: str, rationale: str) -> None:
        if item_id in selected or item_id not in items_by_id:
            return
        item = items_by_id[item_id]
        v4 = v4_by_id.get(item_id, {}).get("score") or {}
        selected[item_id] = {
            "item_id": item_id, "drug": item.drug, "name_type": item.name_type,
            "tier": tier, "rationale": rationale,
            "has_human_ref": item.drug in clips,
            "dual_reference": item.drug in dual_ref,
            "gemini_v4_score": v4.get("score"),
            "gemini_v4_scoreable": v4.get("scoreable"),
        }

    for name in CONFIRMED_BAD_BY_EAR:
        add(name, "A_confirmed_bad_by_ear", "human listened, confirmed genuinely mispronounced")
    for name in REFERENCE_FLAGGED:
        add(name, "B_reference_flagged", "human reference clip disagrees with its own dictionary IPA")

    for r in scoreable:
        if len(selected) >= 0 and sum(1 for v in selected.values() if v["tier"] == "C_gemini_worst_quartile") >= QUARTILE_C_N:
            break
        add(r["item_id"], "C_gemini_worst_quartile", f"gemini v4 score={r['score']['score']:.2f}, bottom of {n} scoreable")

    hard_asr_candidates = [i.item_id for i in items if i.drug.lower() in unrecognizable]
    for item_id in hard_asr_candidates[:QUARTILE_D_N]:
        add(item_id, "D_structurally_hard_asr", "Cloud STT can't transcribe even a human saying this name")

    dual_candidates = [i.item_id for i in items if i.drug in dual_ref]
    for item_id in dual_candidates[:QUARTILE_E_N]:
        add(item_id, "E_dual_reference_coverage", "has both drugs.com AND merriam-webster clips -- exercises best-of-N scoring")

    for r in reversed(scoreable):
        if sum(1 for v in selected.values() if v["tier"] == "F_ceiling_control") >= QUARTILE_F_N:
            break
        add(r["item_id"], "F_ceiling_control", f"gemini v4 score={r['score']['score']:.2f}, top of {n} scoreable -- confirms model CAN pass")

    no_ref_candidates = [i.item_id for i in items if i.drug not in clips]
    for item_id in no_ref_candidates[:QUARTILE_G_N]:
        add(item_id, "G_no_reference_coverage", "no human clip at all -- cannot be scored by Path 2 today, tracked not ignored")

    print(f"Selected {len(selected)} items (target ~{TARGET_SIZE}) from {len(items)} total\n")

    by_tier: dict[str, list[str]] = {}
    for item_id, row in selected.items():
        by_tier.setdefault(row["tier"], []).append(item_id)
    for tier in sorted(by_tier):
        print(f"{tier} (n={len(by_tier[tier])}): {sorted(by_tier[tier])}")

    n_brand = sum(1 for v in selected.values() if v["name_type"] == "brand")
    n_generic = sum(1 for v in selected.values() if v["name_type"] == "generic")
    full_brand = sum(1 for i in items if i.name_type == "brand")
    print(f"\nbrand/generic in subset: {n_brand}/{n_generic} "
          f"({n_brand/len(selected)*100:.0f}% brand)")
    print(f"brand/generic in full corpus: {full_brand}/{len(items)-full_brand} "
          f"({full_brand/len(items)*100:.0f}% brand)")

    out_path = REPO_ROOT / "runs" / "hard-subset-v1.json"
    out_path.write_text(json.dumps({
        "n_items": len(selected), "n_total_corpus": len(items),
        "items": selected,
    }, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
