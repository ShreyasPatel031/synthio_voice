#!/usr/bin/env python3
"""Bottom-half Kokoro search seeded from locked gold IPA.

Starts from plain Misaki vs the documented American fold of gold IPA.
Does not write data/gold_gemini_ipa. A stored string is kept only when
sentence CTC F1 beats the current one by more than 0.005. Candidates are
gold-IPA folds, not Gemini guesses or last-vowel roulette.

Every candidate is appended to runs/misaki-iter/iterations.jsonl.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dose_r.forced_align import extract_drug_span_forced_align
from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_score = _load("score_dose_ctc", ROOT / "scripts" / "score_dose_ctc_vs_gemini_ipa.py")
_kokoro = _load("synth_kokoro_hard", ROOT / "scripts" / "synth_kokoro_hard.py")
load_eval_items = _score.load_eval_items
score_span_vs_gemini = _score.score_span_vs_gemini
wav_bytes = _score.wav_bytes
VOWELS = _kokoro.VOWELS
fold_documented = _kokoro.fold_documented
ipa_pieces = _kokoro.ipa_pieces
load_gold = _kokoro.load_gold
replace_drug = _kokoro.replace_drug
restress = _kokoro.restress
slug = _kokoro.slug
vocab = _kokoro.vocab

from dose_r.misaki_fold import gold_misaki_candidates  # noqa: E402

VOICE = "af_heart"
ROUNDS = 3
N_VARIANTS = 5
STORE = ROOT / "runs" / "misaki-iter"
WAV_ROOT = ROOT / "runs" / "oss-eval"
NEIGHBORS = {
    "ɑ": ["O", "æ", "ɔ"],
    "æ": ["ɑ", "ɛ"],
    "ɔ": ["ɑ", "O"],
    "O": ["ɑ", "ɔ", "ʊ"],
    "ɪ": ["i", "ə"],
    "i": ["ɪ"],
    "ə": ["ɪ", "ʌ"],
    "ɛ": ["æ", "ɪ"],
    "ʊ": ["u", "O"],
    "u": ["ʊ"],
    "ʌ": ["ə", "ɑ"],
    "A": ["ɛ"],
    "I": ["ɑ"],
    "W": ["ɑ"],
    "Y": ["ɔ"],
    "ɜ": ["ə"],
    "ᵻ": ["ə", "ɪ"],
}


def load_full_items() -> list[dict]:
    """One wav per drug slug. Same span rules as the hard synth."""
    items = []
    dose = ROOT / "data" / "dose_v1.jsonl"
    for line in dose.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ingredients = row.get("ingredients") or [row["name"]]
        spans = row.get("spans") or []
        sentence = row["sentence"]
        for i, ing in enumerate(ingredients):
            if i < len(spans) and isinstance(spans[i], (list, tuple)) and len(spans[i]) == 2:
                a, b = spans[i]
                spoken = sentence[a:b]
            else:
                idx = sentence.lower().find(ing.lower())
                if idx < 0:
                    continue
                spoken = sentence[idx : idx + len(ing)]
            items.append(
                {"drug": ing, "spoken": spoken, "sentence": sentence, "slug": slug(ing)}
            )
    seen = {}
    for it in items:
        seen.setdefault(it["slug"], it)
    return list(seen.values())


def drug_phones(tokens, spoken: str) -> str:
    words = spoken.split()
    texts = [t.text for t in tokens]
    lowered = [t.lower() for t in texts]
    want = [w.lower() for w in words]
    for i in range(len(tokens) - len(words) + 1):
        if lowered[i : i + len(words)] == want:
            parts = [(tokens[i + j].phonemes or "") for j in range(len(words))]
            return " ".join(p for p in parts if p)
    raise ValueError(f"could not locate {spoken!r}")


def docs_phones(ipa: str, spoken: str, symbols: set[str]) -> str:
    words = spoken.split()
    pieces = [restress(fold_documented(p, symbols)) for p in ipa_pieces(ipa)]
    if len(pieces) != len(words):
        pieces = [restress(fold_documented(ipa, symbols))]
    return " ".join(pieces)


def phone_override(spoken: str, phones: str) -> str | list[str]:
    """Space inside one word is a syllable break, not a second G2P token."""
    words = spoken.split()
    parts = phones.split()
    if len(words) == 1:
        return phones
    if len(parts) == len(words):
        return parts
    return phones


def write_sentence(pipeline, item: dict, phones: str | None, dest: Path, speed: float = 1.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if phones is None:
        result = next(pipeline(item["sentence"], voice=VOICE, speed=speed))
    else:
        _, tokens = pipeline.g2p(item["sentence"])
        replace_drug(tokens, item["spoken"], phone_override(item["spoken"], phones))
        result = next(pipeline.generate_from_tokens(tokens, voice=VOICE, speed=speed))
    audio = result.audio
    if audio is None:
        raise RuntimeError("no audio")
    sf.write(dest, audio.detach().cpu().numpy(), 24000)


def write_isolated(pipeline, phones: str, dest: Path, speed: float = 1.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = next(pipeline.generate_from_tokens(phones.replace(" ", ""), voice=VOICE, speed=speed))
    audio = result.audio
    if audio is None:
        raise RuntimeError("no audio")
    sf.write(dest, audio.detach().cpu().numpy(), 24000)


def variants(ps: str, symbols: set[str]) -> list[str]:
    """Five Misaki rewrites: stress moved onto each vowel, then stressed-vowel neighbors."""
    out: list[str] = []

    def add(s: str) -> None:
        if not s or s == ps or s in out:
            return
        if any(ch not in symbols and not ch.isspace() for ch in s):
            return
        out.append(s)

    for piece in ps.split():
        bare = piece.replace("ˈ", "")
        for i, ch in enumerate(bare):
            if ch in VOWELS:
                add(bare[:i] + "ˈ" + bare[i:])
        if "ˈ" in piece:
            j = piece.index("ˈ")
            for k in range(j + 1, len(piece)):
                if piece[k] in NEIGHBORS:
                    for rep in NEIGHBORS[piece[k]]:
                        add(piece[:k] + rep + piece[k + 1 :])
                    break
        if len(out) >= N_VARIANTS:
            break
    return out[:N_VARIANTS]


def sentence_f1(wav: Path, item: dict, gold: Path) -> float | None:
    raw = wav_bytes(wav)
    span = extract_drug_span_forced_align(raw, item["sentence"], item["spoken"])
    if span is None:
        span = extract_drug_span_forced_align(raw, item["sentence"], item["drug"])
    if span is None:
        return None
    return float(score_span_vs_gemini(span, gold))


def gold_wav(drug: str) -> Path | None:
    p = ROOT / "data" / "gold_gemini_ipa" / "wavs" / f"{slug(drug)}.wav"
    if p.exists() and p.stat().st_size > 500:
        return p
    return None


def isolated_f1(wav: Path, gold_emb) -> float:
    emb = extract_frame_embeddings(wav.read_bytes())
    return float(speech_bertscore(emb, gold_emb)["f1"])


def log_row(row: dict) -> None:
    path = STORE / "iterations.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def gemini_proposal(gold: Path, attempt: Path, current: str, symbols: set[str]) -> str | None:
    """Ask Gemini, given both wavs, for one Misaki string. None if the call fails."""
    import base64

    try:
        tok = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    alphabet = "".join(sorted(symbols))
    prompt = (
        "You hear two clips of one drug name. Clip A is the target. "
        "Clip B is Kokoro saying the Misaki phoneme string below. "
        "Reply with JSON only: {\"phonemes\": \"...\"}. "
        "The string must use only these symbols, with ˈ immediately before the stressed vowel: "
        f"{alphabet}. Current string: {current}"
    )
    parts = [
        {"text": prompt},
        {"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(gold.read_bytes()).decode()}},
        {"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(attempt.read_bytes()).decode()}},
    ]
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.0},
    }
    url = (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/"
        "project-amer-scs-sandbox/locations/us-central1/publishers/google/"
        "models/gemini-2.5-flash:generateContent"
    )
    try:
        import requests

        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
            json=body,
            timeout=60,
        )
        if resp.status_code != 200:
            print("gemini", resp.status_code, resp.text[:200], flush=True)
            return None
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            return None
        phones = json.loads(text[start : end + 1]).get("phonemes") or ""
        phones = phones.replace(" ", "")
        if phones and all(ch in symbols for ch in phones):
            return phones
    except Exception as exc:
        print("gemini skip", exc, flush=True)
    return None


def score_arm(wav_dir: Path, by_slug: dict[str, dict]) -> dict[str, float]:
    """Min sentence F1 per drug slug. Official CTC cut vs the gold word."""
    out: dict[str, list[float]] = {}
    items = load_eval_items("full")
    for i, item in enumerate(items, 1):
        key = slug(item["drug"])
        base = by_slug.get(key)
        if base is None or item["sentence"] != base["sentence"]:
            continue
        wav = wav_dir / f"{key}.wav"
        gold = gold_wav(item["drug"])
        if not wav.exists() or gold is None:
            continue
        f1 = sentence_f1(wav, base, gold)
        if f1 is None:
            continue
        out.setdefault(key, []).append(f1)
        if i % 40 == 0:
            print(f"scored {i}/{len(items)}", flush=True)
    return {k: min(v) for k, v in out.items()}


def main() -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    gold_ipa = load_gold()
    items = load_full_items()
    by_slug = {it["slug"]: it for it in items}
    symbols = vocab() | set("T ")
    print(f"items={len(items)}", flush=True)

    from kokoro import KPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
    plain_dir = WAV_ROOT / "kokoro-full-plain"
    docs_dir = WAV_ROOT / "kokoro-full-docs"
    phones_plain: dict[str, str] = {}
    phones_docs: dict[str, str] = {}
    for it in items:
        ipa = gold_ipa.get(it["drug"].lower(), "")
        dest_p = plain_dir / f"{it['slug']}.wav"
        dest_d = docs_dir / f"{it['slug']}.wav"
        try:
            if not (dest_p.exists() and dest_p.stat().st_size > 1000):
                write_sentence(pipeline, it, None, dest_p)
            _, tokens = pipeline.g2p(it["sentence"])
            phones_plain[it["slug"]] = drug_phones(tokens, it["spoken"])
            if ipa:
                phones_docs[it["slug"]] = docs_phones(ipa, it["spoken"], symbols)
                if not (dest_d.exists() and dest_d.stat().st_size > 1000):
                    write_sentence(pipeline, it, phones_docs[it["slug"]], dest_d)
            print("synth", it["slug"], flush=True)
        except Exception as exc:
            print("SYNTH_FAIL", it["slug"], exc, flush=True)

    print("scoring plain", flush=True)
    f1_plain = score_arm(plain_dir, by_slug)
    print("scoring docs", flush=True)
    f1_docs = score_arm(docs_dir, by_slug)

    best_phones: dict[str, str] = {}
    best_f1: dict[str, float] = {}
    best_dir = WAV_ROOT / "kokoro-full-best"
    best_dir.mkdir(parents=True, exist_ok=True)
    for it in items:
        key = it["slug"]
        choices = []
        if key in f1_plain:
            choices.append((f1_plain[key], phones_plain.get(key, ""), plain_dir))
        if key in f1_docs and key in phones_docs:
            choices.append((f1_docs[key], phones_docs[key], docs_dir))
        if not choices:
            continue
        choices.sort(key=lambda row: -row[0])
        f1, phones, src = choices[0]
        best_f1[key] = f1
        best_phones[key] = phones
        target = best_dir / f"{key}.wav"
        if not target.exists():
            target.write_bytes((src / f"{key}.wav").read_bytes())
        log_row({
            "round": 0,
            "slug": key,
            "drug": it["drug"],
            "source": "plain" if src == plain_dir else "docs",
            "phonemes": phones,
            "sentence_f1": round(f1, 4),
            "plain_f1": None if key not in f1_plain else round(f1_plain[key], 4),
            "docs_f1": None if key not in f1_docs else round(f1_docs[key], 4),
            "kept": True,
        })
    (STORE / "chosen.json").write_text(json.dumps(
        {"phones": best_phones, "f1": best_f1}, indent=2
    ))
    ranked = sorted(best_f1, key=lambda k: best_f1[k])
    print(
        f"start n={len(ranked)} mean={sum(best_f1.values())/len(best_f1):.3f} "
        f"bottom={best_f1[ranked[0]]:.3f}",
        flush=True,
    )

    for rnd in range(1, ROUNDS + 1):
        ranked = sorted(best_f1, key=lambda k: best_f1[k])
        half = ranked[: max(1, len(ranked) // 2)]
        print(f"round {rnd} bottom {len(half)}", flush=True)
        trial_dir = STORE / f"round{rnd}"
        improved = 0
        for n, key in enumerate(half, 1):
            it = by_slug[key]
            current = best_phones[key]
            gold = gold_wav(it["drug"])
            if gold is None:
                continue
            ipa = gold_ipa.get(it["drug"].lower(), "")
            cands: list[str] = []
            for _, extra in gold_misaki_candidates(ipa, it["spoken"], symbols) if ipa else []:
                if extra and extra not in cands:
                    cands.append(extra)
            for extra in (phones_plain.get(key), phones_docs.get(key)):
                if extra and extra not in cands:
                    cands.append(extra)
            cands = [c for c in cands if c != current][:N_VARIANTS]
            scored = []
            for j, phones in enumerate(cands):
                sent_try = trial_dir / "sent" / f"{key}-{j}.wav"
                try:
                    write_sentence(pipeline, it, phones, sent_try, speed=1.0)
                    f1 = sentence_f1(sent_try, it, gold)
                except Exception as exc:
                    print("variant fail", key, exc, flush=True)
                    continue
                if f1 is None:
                    continue
                scored.append((f1, phones, sent_try))
                log_row({
                    "round": rnd,
                    "slug": key,
                    "drug": it["drug"],
                    "phonemes": phones,
                    "sentence_f1": round(f1, 4),
                    "source": "gold-seed",
                    "kept": False,
                })
            if not scored:
                continue
            scored.sort(key=lambda row: -row[0])
            f1, phones, sent = scored[0]
            kept = f1 > best_f1[key] + 0.005
            log_row({
                "round": rnd,
                "slug": key,
                "drug": it["drug"],
                "phonemes": phones,
                "sentence_f1": round(f1, 4),
                "previous_f1": round(best_f1[key], 4),
                "kept": kept,
            })
            if kept:
                best_f1[key] = f1
                best_phones[key] = phones
                (best_dir / f"{key}.wav").write_bytes(sent.read_bytes())
                improved += 1
            if n % 20 == 0:
                print(f"  {n}/{len(half)} improved={improved}", flush=True)
        mean = sum(best_f1.values()) / len(best_f1)
        print(f"round {rnd} done improved={improved} mean={mean:.3f}", flush=True)
        (STORE / "chosen.json").write_text(json.dumps(
            {"round": rnd, "phones": best_phones, "f1": {k: round(v, 4) for k, v in best_f1.items()}},
            indent=2,
        ))

    vals = list(best_f1.values())
    summary = {
        "n": len(vals),
        "mean_sentence_f1": round(sum(vals) / len(vals), 4),
        "median_sentence_f1": round(sorted(vals)[len(vals) // 2], 4),
        "pass_f1_ge_0_70": sum(v >= 0.70 for v in vals),
        "rounds": ROUNDS,
        "variants_per_name": N_VARIANTS,
    }
    (STORE / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
