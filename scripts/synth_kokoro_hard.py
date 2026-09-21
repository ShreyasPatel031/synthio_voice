#!/usr/bin/env python3
"""Hard-subset Kokoro pronunciation arms.

Reads gold IPA from data/gold_gemini_ipa. Does not write that folder.
Kokoro has no reference-audio clone and no sampler. The deterministic
spelling run is the audio baseline. The other two arms inject the
published gold IPA, mapped into Misaki's alphabet (not a new G2P).

  plain    Misaki reads the spelling.
  ipa      After G2P, overwrite the drug token with the gold IPA.
  stress   Same as ipa, then move ˈ/ˌ to sit immediately before the vowel.
  docs     stress, plus the American rules in misaki EN_PHONES.md.
  lexicon  Put that IPA in Misaki's gold dictionary, then G2P normally.

Score later:
  python scripts/score_dose_ctc_vs_gemini_ipa.py \\
    --wav-dir runs/oss-eval/kokoro-hard-<arm> --condition kokoro-hard-<arm> --scope hard
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from kokoro import KPipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOLD_MANIFEST = ROOT / "data" / "gold_gemini_ipa" / "manifest.jsonl"
DOSE = ROOT / "data" / "dose_v1.jsonl"
HARD = ROOT / "runs" / "hard-subset-v1.json"
VOICE = "af_heart"
ARMS = ("plain", "ipa", "stress", "docs", "lexicon")
STRESSES = "ˌˈ"
VOWELS = frozenset("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def vocab() -> set[str]:
    path = hf_hub_download("hexgrad/Kokoro-82M", "config.json")
    return set(json.loads(Path(path).read_text())["vocab"])


def ipa_to_misaki(raw: str, symbols: set[str]) -> str:
    """Fold published IPA into Misaki's American symbols. Do not invent phones."""
    s = raw.strip().strip("/[]")
    # Slashes in the gold string are separators, not phonemes
    # (e.g. a two-word name stored as "ipa/ /ipa").
    s = s.replace("/", "")
    s = s.replace("'", "ˈ").replace(".", "")
    for old, new in (
        ("tʃ", "ʧ"),
        ("dʒ", "ʤ"),
        ("eɪ", "A"),
        ("aɪ", "I"),
        ("aʊ", "W"),
        ("ɔɪ", "Y"),
        ("oʊ", "O"),
        ("əʊ", "O"),
        ("ɝ", "ɜɹ"),
        ("ɚ", "əɹ"),
    ):
        s = s.replace(old, new)
    s = s.replace("ː", "").replace("r", "ɹ").replace("g", "ɡ").replace(" ", "")
    missing = sorted({ch for ch in s if ch not in symbols})
    if missing:
        raise ValueError(f"not in Misaki vocab: {missing} from {raw!r} -> {s!r}")
    return s


def ipa_pieces(raw: str) -> list[str]:
    return [p for p in re.split(r"[\s/]+", raw.strip()) if p]


def fold_documented(raw: str, symbols: set[str]) -> str:
    """American fold from misaki EN_PHONES.md (from_espeak) and en.G2P.

    https://github.com/hexgrad/misaki/blob/main/EN_PHONES.md
    Longest tie/diphthong first, then the American-only rewrites:
    bare e→A, leftover o→ɔ, ɜː→ɜɹ, ɪə→iə, ː dropped.
    British-only symbols become their American pair (a→æ, ɒ→ɑ).
    Live G2P then rewrites ɾ→T and ʔ→t unless version is 2.0.
    """
    s = raw.strip().strip("/[]").replace("'", "ˈ").replace(".", "")
    for old, new in (
        ("tʃ", "ʧ"),
        ("dʒ", "ʤ"),
        ("eɪ", "A"),
        ("aɪ", "I"),
        ("aʊ", "W"),
        ("ɔɪ", "Y"),
        ("oʊ", "O"),
        ("əʊ", "O"),
        ("ɝ", "ɜɹ"),
        ("ɚ", "əɹ"),
        ("ɜːɹ", "ɜɹ"),
        ("ɜː", "ɜɹ"),
        ("ɐ", "ə"),
        ("x", "k"),
        ("ç", "k"),
    ):
        s = s.replace(old, new)
    # After eɪ is gone, so this does not rewrite the ɪ inside eɪ.
    s = s.replace("ɪə", "iə")
    s = s.replace("ː", "")
    s = s.replace("r", "ɹ").replace("g", "ɡ")
    s = s.replace("e", "A").replace("o", "ɔ")
    s = s.replace("a", "æ").replace("ɒ", "ɑ")
    s = re.sub(r"(\S)\u0329", r"ᵊ\1", s)
    s = s.replace("ɾ", "T").replace("ʔ", "t")
    s = s.replace(" ", "")
    missing = sorted({ch for ch in s if ch not in symbols})
    if missing:
        raise ValueError(f"not in Misaki vocab: {missing} from {raw!r} -> {s!r}")
    return s


def restress(ps: str) -> str:
    """Misaki's restress: ˈ and ˌ move to immediately before the next vowel.

    IPA writes the tick at the start of the syllable (ˈɹɪn). Kokoro was
    trained on the tick sitting in front of the vowel (ɹˈɪn).
    """
    ips = list(enumerate(ps))
    moved = {}
    for i, p in ips:
        if p not in STRESSES:
            continue
        nxt = next((j for j, v in ips[i + 1 :] if v in VOWELS), None)
        if nxt is None:
            raise ValueError(f"stress with no following vowel: {ps!r}")
        moved[i] = nxt
    for i, j in moved.items():
        ips[i] = (j - 0.5, ips[i][1])
    return "".join(p for _, p in sorted(ips))


def load_gold() -> dict[str, str]:
    out = {}
    for line in GOLD_MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ipa = (rec.get("ipa") or "").strip()
        if ipa:
            out[rec["ingredient"].lower()] = ipa
    return out


def load_items() -> list[dict]:
    hard = json.loads(HARD.read_text())["items"]
    hard_drugs = {v["drug"].lower() for v in hard.values()}
    items = []
    for line in DOSE.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ingredients = row.get("ingredients") or [row["name"]]
        spans = row.get("spans") or []
        sentence = row["sentence"]
        for i, ing in enumerate(ingredients):
            if ing.lower() not in hard_drugs:
                continue
            if i < len(spans) and isinstance(spans[i], (list, tuple)) and len(spans[i]) == 2:
                a, b = spans[i]
                spoken = sentence[a:b]
            else:
                idx = sentence.lower().find(ing.lower())
                if idx < 0:
                    continue
                spoken = sentence[idx : idx + len(ing)]
            items.append(
                {
                    "drug": ing,
                    "spoken": spoken,
                    "sentence": sentence,
                    "slug": slug(ing),
                }
            )
    seen = {}
    for it in items:
        seen.setdefault(it["slug"], it)
    return list(seen.values())


def replace_drug(tokens, drug: str, phones: str | list[str]) -> None:
    words = drug.split()
    texts = [t.text for t in tokens]
    lowered = [t.lower() for t in texts]
    want = [w.lower() for w in words]
    pieces = [phones] if isinstance(phones, str) else list(phones)
    if len(pieces) == 1 and len(words) > 1:
        pieces = pieces + [""] * (len(words) - 1)
    if len(pieces) != len(words):
        raise ValueError(f"{drug!r} has {len(words)} words but {len(pieces)} phoneme pieces")
    for i in range(len(tokens) - len(words) + 1):
        if lowered[i : i + len(words)] == want:
            for offset, piece in enumerate(pieces):
                tokens[i + offset].phonemes = piece
            return
    raise ValueError(f"could not locate {drug!r} in tokens {texts}")


def write_audio(pipeline, tokens_or_sentence, dest: Path, *, from_tokens: bool) -> None:
    if from_tokens:
        result = next(pipeline.generate_from_tokens(tokens_or_sentence, voice=VOICE, speed=1))
    else:
        result = next(pipeline(tokens_or_sentence, voice=VOICE, speed=1))
    audio = result.audio
    if audio is None:
        raise RuntimeError("no audio")
    sf.write(dest, audio.detach().cpu().numpy(), 24000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=(*ARMS, "all"), default="all")
    args = ap.parse_args()
    arms = list(ARMS) if args.arm == "all" else [args.arm]

    gold = load_gold()
    items = load_items()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
    symbols = vocab()
    print(f"device={device} items={len(items)} arms={arms}", flush=True)

    for arm in arms:
        out = ROOT / "runs" / "oss-eval" / f"kokoro-hard-{arm}"
        out.mkdir(parents=True, exist_ok=True)
        log_path = out / "map.jsonl"
        failures = []
        n_ok = 0
        with log_path.open("w") as log:
            for it in items:
                dest = out / f"{it['slug']}.wav"
                if dest.exists() and dest.stat().st_size > 1000:
                    n_ok += 1
                    continue
                try:
                    phones = ""
                    ipa = gold.get(it["drug"].lower(), "")
                    if arm == "plain":
                        write_audio(pipeline, it["sentence"], dest, from_tokens=False)
                    else:
                        if not ipa:
                            raise ValueError("no gold IPA")
                        if arm == "docs":
                            words = it["spoken"].split()
                            pieces = [restress(fold_documented(p, symbols)) for p in ipa_pieces(ipa)]
                            if len(pieces) != len(words):
                                pieces = [restress(fold_documented(ipa, symbols))]
                            phones = " ".join(pieces)
                            _, tokens = pipeline.g2p(it["sentence"])
                            replace_drug(tokens, it["spoken"], pieces)
                            write_audio(pipeline, tokens, dest, from_tokens=True)
                        elif arm == "lexicon":
                            phones = ipa_to_misaki(ipa, symbols)
                            lex = pipeline.g2p.lexicon.golds
                            keys = {it["drug"], it["drug"].lower(), it["spoken"], it["spoken"].lower()}
                            saved = {k: lex.get(k) for k in keys}
                            try:
                                for k in keys:
                                    lex[k] = phones
                                _, tokens = pipeline.g2p(it["sentence"])
                                hit = any((t.phonemes or "") == phones for t in tokens)
                                write_audio(pipeline, tokens, dest, from_tokens=True)
                            finally:
                                for k, old in saved.items():
                                    if old is None:
                                        lex.pop(k, None)
                                    else:
                                        lex[k] = old
                            if not hit:
                                print("LEXICON_MISS", it["slug"], flush=True)
                        else:
                            phones = ipa_to_misaki(ipa, symbols)
                            if arm == "stress":
                                phones = restress(phones)
                            _, tokens = pipeline.g2p(it["sentence"])
                            replace_drug(tokens, it["spoken"], phones)
                            write_audio(pipeline, tokens, dest, from_tokens=True)
                    rec = {
                        "slug": it["slug"],
                        "drug": it["drug"],
                        "arm": arm,
                        "ipa": ipa,
                        "misaki": phones,
                    }
                    if arm == "lexicon":
                        rec["lexicon_hit"] = hit
                    log.write(json.dumps(rec) + "\n")
                    n_ok += 1
                    print("ok", arm, it["slug"], flush=True)
                except Exception as exc:
                    failures.append({"slug": it["slug"], "drug": it["drug"], "error": str(exc)})
                    print("FAIL", arm, it["slug"], exc, flush=True)
        (out / "failures.json").write_text(json.dumps(failures, indent=2))
        print(f"done {arm} ok={n_ok} fail={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
