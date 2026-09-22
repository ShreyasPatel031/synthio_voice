#!/usr/bin/env python3
"""Full-benchmark injection path experiment.

Same stored Misaki phones, three ways of putting them into the DoSE sentence:

  lexicon       pipeline.g2p.lexicon.golds[spoken] = phones, then pipeline(sentence)
  markdown      [spoken](/phones/) inside the sentence, then pipeline(text)
  token_splice  g2p(sentence), overwrite the drug token, generate_from_tokens

Scores each sentence-span against the locked Cloud gold wav with WavLM F1.
Does not write data/gold_gemini_ipa, chosen.json, or user_pins.json.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VOICE = "af_heart"
SPEED = 1.0
KEEP_DELTA = 0.005
ARMS = ("lexicon", "markdown", "token_splice")
STRESS = "ˈ"
SECONDARY = "ˌ"
STORE = ROOT / "runs" / "misaki-iter"
GOLD_DIR = ROOT / "data" / "gold_gemini_ipa" / "wavs"
CHOSEN = STORE / "chosen.json"
PINS = STORE / "user_pins.json"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_full_items() -> list[dict]:
    items = []
    for line in (ROOT / "data" / "dose_v1.jsonl").read_text().splitlines():
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
    seen: dict[str, dict] = {}
    for it in items:
        seen.setdefault(it["slug"], it)
    return list(seen.values())


def load_phones() -> dict[str, str]:
    if not CHOSEN.exists():
        raise SystemExit(f"missing {CHOSEN}")
    chosen = json.loads(CHOSEN.read_text())
    phones = {k: str(v) for k, v in (chosen.get("phones") or {}).items() if v}
    if PINS.exists():
        pins = json.loads(PINS.read_text())
        for key, pin in pins.items():
            misaki = (pin or {}).get("misaki")
            if misaki:
                phones[key] = str(misaki)
    return phones


def bare(text: str) -> str:
    """Drop stress marks and whitespace so a folded string still counts as present."""
    return "".join(ch for ch in text if ch not in f" \t{STRESS}{SECONDARY}")


def phone_override(spoken: str, phones: str) -> str | list[str]:
    """Space inside one word is a syllable break, not a second G2P token."""
    words = spoken.split()
    parts = phones.split()
    if len(words) == 1:
        return phones
    if len(parts) == len(words):
        return parts
    return phones


def markdown_text(sentence: str, spoken: str, phones: str) -> str:
    """Replace the spoken span once. Surrounding spaces stay put."""
    if spoken not in sentence:
        idx = sentence.lower().find(spoken.lower())
        if idx < 0:
            raise ValueError(f"cannot find {spoken!r} in sentence")
        spoken = sentence[idx : idx + len(spoken)]
    return sentence.replace(spoken, f"[{spoken}](/{phones}/)", 1)


def lexicon_keys(spoken: str) -> list[str]:
    keys = [spoken]
    lower = spoken.lower()
    if lower != spoken:
        keys.append(lower)
    return keys


def drug_token_phones(tokens, spoken: str) -> str | None:
    words = spoken.split()
    if not words:
        return None
    texts = [t.text for t in tokens]
    lowered = [t.lower() for t in texts]
    want = [w.lower() for w in words]
    for i in range(len(tokens) - len(words) + 1):
        if lowered[i : i + len(words)] == want:
            parts = [(tokens[i + j].phonemes or "") for j in range(len(words))]
            return " ".join(p for p in parts if p)
    return None


def replace_drug(tokens, spoken: str, phones: str | list[str]) -> str:
    words = spoken.split()
    texts = [t.text for t in tokens]
    lowered = [t.lower() for t in texts]
    want = [w.lower() for w in words]
    for i in range(len(tokens) - len(words) + 1):
        if lowered[i : i + len(words)] != want:
            continue
        if isinstance(phones, list):
            for j, part in enumerate(phones):
                tokens[i + j].phonemes = part
        else:
            tokens[i].phonemes = phones
            for extra in tokens[i + 1 : i + len(words)]:
                extra.phonemes = ""
        found = drug_token_phones(tokens, spoken)
        if found is None:
            raise RuntimeError(f"replaced {spoken!r} but could not read phonemes back")
        return found
    raise ValueError(f"could not locate {spoken!r} in tokens {[t.text for t in tokens]}")


def link_not_consumed(graphemes: str) -> bool:
    return "[" in graphemes or "](" in graphemes or "](/" in graphemes


def override_missing(intended: str, realized: str) -> bool:
    needle = bare(intended)
    if not needle:
        return True
    return needle not in bare(realized)


def stress_changed(intended: str, drug_phones: str | None) -> bool | None:
    if drug_phones is None:
        return None
    return intended.count(STRESS) != drug_phones.count(STRESS)


def version_at_least(version: str, minimum: tuple[int, int, int]) -> bool:
    parts: list[int] = []
    for piece in version.split("."):
        digits = ""
        for ch in piece:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            parts.append(int(digits))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3]) >= minimum


def paired(rows: list[dict], left: str, right: str) -> dict:
    wins = {left: 0, right: 0, "tie": 0}
    for row in rows:
        a = (row.get("arms") or {}).get(left) or {}
        b = (row.get("arms") or {}).get(right) or {}
        fa, fb = a.get("span_wavlm_f1"), b.get("span_wavlm_f1")
        if fa is None or fb is None:
            continue
        delta = float(fa) - float(fb)
        if delta > KEEP_DELTA:
            wins[left] += 1
        elif delta < -KEEP_DELTA:
            wins[right] += 1
        else:
            wins["tie"] += 1
    return wins


def arm_summary(rows: list[dict], arm: str) -> dict:
    vals = []
    for row in rows:
        f1 = ((row.get("arms") or {}).get(arm) or {}).get("span_wavlm_f1")
        if f1 is not None:
            vals.append(float(f1))
    if not vals:
        return {"n": 0, "mean": None, "median": None}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 4),
        "median": round(statistics.median(vals), 4),
    }


def flag_slugs(rows: list[dict], arm: str, flag: str) -> list[str]:
    out = []
    for row in rows:
        if ((row.get("arms") or {}).get(arm) or {}).get(flag) is True:
            out.append(row["slug"])
    return out


def _self_test() -> int:
    sentence = "Because Datroway functions today."
    spoken = "Datroway"
    phones = "dætɹOwˈA"
    text = markdown_text(sentence, spoken, phones)
    if text != "Because [Datroway](/dætɹOwˈA/) functions today.":
        raise SystemExit(f"markdown spaces lost: {text}")
    if lexicon_keys("Datroway") != ["Datroway", "datroway"]:
        raise SystemExit("lexicon keys")
    if lexicon_keys("advair") != ["advair"]:
        raise SystemExit("lower lexicon key")
    if phone_override("Datroway", "dæt ɹO wˈA") != "dæt ɹO wˈA":
        raise SystemExit("single-word spaces must stay one token")
    if phone_override("Los Angeles", "lɔs ˈændʒɛləs") != ["lɔs", "ˈændʒɛləs"]:
        raise SystemExit("two-word split")
    if not override_missing("dætɹO", "ðə nˈeɪm"):
        raise SystemExit("missing override not flagged")
    if override_missing("dætˈɹO", "xx dætɹO yy"):
        raise SystemExit("stress-stripped override should count as present")
    if not link_not_consumed("Because [Datroway](/dæt/) functions"):
        raise SystemExit("unconsumed link")
    if link_not_consumed("Because Datroway functions"):
        raise SystemExit("clean graphemes")
    if stress_changed("dætˈɹO", "dætɹO") is not True:
        raise SystemExit("stress change")
    if stress_changed("dætˈɹO", "dætˈɹO") is not False:
        raise SystemExit("stress same")
    rows = [
        {"slug": "a", "arms": {"lexicon": {"span_wavlm_f1": 0.80}, "markdown": {"span_wavlm_f1": 0.70}}},
        {"slug": "b", "arms": {"lexicon": {"span_wavlm_f1": 0.70}, "markdown": {"span_wavlm_f1": 0.701}}},
        {"slug": "c", "arms": {"lexicon": {"span_wavlm_f1": None}, "markdown": {"span_wavlm_f1": 0.9}}},
    ]
    wins = paired(rows, "lexicon", "markdown")
    if wins != {"lexicon": 1, "markdown": 0, "tie": 1}:
        raise SystemExit(f"paired {wins}")
    if not version_at_least("0.9.2", (0, 9, 2)):
        raise SystemExit("version")
    if version_at_least("0.7.3", (0, 9, 2)):
        raise SystemExit("old version should fail the floor")
    print("self-test ok", flush=True)
    return 0


def _dist_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _unpack(result) -> tuple:
    if hasattr(result, "audio") or hasattr(result, "phonemes"):
        audio = getattr(result, "audio", None)
        if audio is None:
            audio = getattr(result, "output", None)
        ps = getattr(result, "phonemes", "") or ""
        gs = getattr(result, "graphemes", "") or ""
        tokens = list(getattr(result, "tokens", None) or [])
        return audio, ps, gs, tokens
    if isinstance(result, tuple) and len(result) >= 3:
        return result[2], result[1] or "", result[0] or "", []
    raise TypeError(f"unexpected kokoro result {type(result)}")


def _as_audio(audio):
    import numpy as np

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    arr = np.asarray(audio).squeeze()
    if arr.size == 0:
        raise RuntimeError("empty audio")
    return arr


def _consume(generator) -> tuple:
    import numpy as np

    audios = []
    phonemes = []
    graphemes = []
    tokens = []
    for result in generator:
        audio, ps, gs, tks = _unpack(result)
        if audio is not None:
            audios.append(_as_audio(audio))
        phonemes.append(ps)
        graphemes.append(gs)
        tokens.extend(tks)
    if not audios:
        raise RuntimeError("no audio")
    audio = audios[0] if len(audios) == 1 else np.concatenate(audios)
    return audio, " ".join(p for p in phonemes if p), "".join(graphemes), tokens


def _write_wav(audio, dest: Path) -> None:
    import soundfile as sf

    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, audio, 24000)


class Scorer:
    def __init__(self) -> None:
        self._gold: dict[str, object] = {}

    def score(self, wav: Path, item: dict, gold: Path) -> float | None:
        from dose_r.forced_align import extract_drug_span_forced_align
        from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore

        span = extract_drug_span_forced_align(wav.read_bytes(), item["sentence"], item["spoken"])
        if span is None:
            span = extract_drug_span_forced_align(wav.read_bytes(), item["sentence"], item["drug"])
        if span is None:
            return None
        key = str(gold)
        emb_g = self._gold.get(key)
        if emb_g is None:
            emb_g = extract_frame_embeddings(gold.read_bytes())
            self._gold[key] = emb_g
        return float(speech_bertscore(extract_frame_embeddings(span), emb_g)["f1"])


def _arm_record(
    *,
    intended: str,
    realized: str,
    graphemes: str,
    drug_phones: str | None,
    lexicon_key_miss: bool | None,
    span_wavlm_f1: float | None,
    error: str | None = None,
) -> dict:
    return {
        "span_wavlm_f1": None if span_wavlm_f1 is None else round(float(span_wavlm_f1), 4),
        "realized": realized,
        "graphemes": graphemes,
        "drug_phones": drug_phones,
        "link_not_consumed": link_not_consumed(graphemes),
        "override_missing": override_missing(intended, realized) if error is None else None,
        "stress_changed": stress_changed(intended, drug_phones),
        "lexicon_key_miss": lexicon_key_miss,
        "scored": False,
        "error": error,
    }


def synth_lexicon(pipeline, item: dict, phones: str) -> dict:
    golds = pipeline.g2p.lexicon.golds
    saved = dict(golds)
    try:
        for key in lexicon_keys(item["spoken"]):
            golds[key] = phones
        try:
            _ps, tokens = pipeline.g2p(item["sentence"])
            looked = drug_token_phones(tokens, item["spoken"])
        except Exception:
            looked = None
        miss = looked is None or bare(looked) != bare(phones)
        audio, realized, graphemes, tokens = _consume(
            pipeline(item["sentence"], voice=VOICE, speed=SPEED)
        )
        heard = drug_token_phones(tokens, item["spoken"])
        return {
            "audio": audio,
            "record": _arm_record(
                intended=phones,
                realized=realized,
                graphemes=graphemes,
                drug_phones=heard if heard is not None else looked,
                lexicon_key_miss=miss,
                span_wavlm_f1=None,
            ),
        }
    finally:
        golds.clear()
        golds.update(saved)


def synth_markdown(pipeline, item: dict, phones: str) -> dict:
    text = markdown_text(item["sentence"], item["spoken"], phones)
    audio, realized, graphemes, tokens = _consume(pipeline(text, voice=VOICE, speed=SPEED))
    return {
        "audio": audio,
        "record": _arm_record(
            intended=phones,
            realized=realized,
            graphemes=graphemes,
            drug_phones=drug_token_phones(tokens, item["spoken"]),
            lexicon_key_miss=None,
            span_wavlm_f1=None,
        ),
    }


def synth_token_splice(pipeline, item: dict, phones: str) -> dict:
    _ps, tokens = pipeline.g2p(item["sentence"])
    injected = replace_drug(tokens, item["spoken"], phone_override(item["spoken"], phones))
    audio, realized, graphemes, out_tokens = _consume(
        pipeline.generate_from_tokens(tokens, voice=VOICE, speed=SPEED)
    )
    heard = drug_token_phones(out_tokens, item["spoken"]) if out_tokens else injected
    return {
        "audio": audio,
        "record": _arm_record(
            intended=phones,
            realized=realized or injected,
            graphemes=graphemes or item["sentence"],
            drug_phones=heard,
            lexicon_key_miss=None,
            span_wavlm_f1=None,
        ),
    }


SYNTH = {
    "lexicon": synth_lexicon,
    "markdown": synth_markdown,
    "token_splice": synth_token_splice,
}


def _meta_path(out: Path, arm: str, slug_name: str) -> Path:
    return out / arm / f"{slug_name}.json"


def _wav_path(out: Path, arm: str, slug_name: str) -> Path:
    return out / arm / f"{slug_name}.wav"


def _done(meta: Path, wav: Path) -> bool:
    if not meta.exists():
        return False
    try:
        row = json.loads(meta.read_text())
    except json.JSONDecodeError:
        return False
    if row.get("error") and not str(row["error"]).startswith("score:"):
        return True
    if not wav.exists() or wav.stat().st_size < 1000:
        return False
    return row.get("scored") is True


def lexicon_miss_shapes(rows: list[dict]) -> dict:
    misses = set(flag_slugs(rows, "lexicon", "lexicon_key_miss"))
    shapes = {"one_word": 0, "multi_word": 0, "hyphen": 0}
    for row in rows:
        if row["slug"] not in misses:
            continue
        if row.get("hyphen"):
            shapes["hyphen"] += 1
        elif int(row.get("n_words") or 1) > 1:
            shapes["multi_word"] += 1
        else:
            shapes["one_word"] += 1
    return shapes


def collect_rows(out: Path, items: dict[str, dict], phones: dict[str, str]) -> list[dict]:
    slugs: set[str] = set()
    for arm in ARMS:
        arm_dir = out / arm
        if not arm_dir.exists():
            continue
        for path in arm_dir.glob("*.json"):
            slugs.add(path.stem)
    rows = []
    for key in sorted(slugs):
        it = items.get(key)
        if it is None:
            continue
        arms = {}
        for arm in ARMS:
            meta = _meta_path(out, arm, key)
            if meta.exists():
                try:
                    arms[arm] = json.loads(meta.read_text())
                except json.JSONDecodeError:
                    continue
        if not arms:
            continue
        rows.append(
            {
                "slug": key,
                "drug": it["drug"],
                "spoken": it["spoken"],
                "intended": phones.get(key, ""),
                "n_words": len(it["spoken"].split()),
                "hyphen": "-" in it["spoken"],
                "arms": arms,
            }
        )
    return rows


def build_report(rows: list[dict], *, kokoro_v: str, misaki_v: str, missing: list) -> dict:
    flags = {}
    for arm in ARMS:
        flags[arm] = {
            "link_not_consumed": flag_slugs(rows, arm, "link_not_consumed"),
            "override_missing": flag_slugs(rows, arm, "override_missing"),
            "stress_changed": flag_slugs(rows, arm, "stress_changed"),
            "lexicon_key_miss": flag_slugs(rows, arm, "lexicon_key_miss"),
        }
    return {
        "metric": "sentence-span WavLM F1",
        "voice": VOICE,
        "speed": SPEED,
        "keep_delta": KEEP_DELTA,
        "kokoro": kokoro_v,
        "misaki": misaki_v,
        "kokoro_markdown_floor_ok": version_at_least(kokoro_v, (0, 9, 2)),
        "n": len(rows),
        "missing": missing,
        "arms": {arm: arm_summary(rows, arm) for arm in ARMS},
        "paired": {
            "lexicon_vs_markdown": paired(rows, "lexicon", "markdown"),
            "lexicon_vs_token_splice": paired(rows, "lexicon", "token_splice"),
            "markdown_vs_token_splice": paired(rows, "markdown", "token_splice"),
        },
        "flags": flags,
        "lexicon_miss_shapes": lexicon_miss_shapes(rows),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=STORE / "inject-ab")
    ap.add_argument("--only", nargs="*", default=())
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return _self_test()
    unknown = [a for a in args.arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}")
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise SystemExit(f"bad shard {args.shard}/{args.shards}")

    phones = load_phones()
    items = {it["slug"]: it for it in load_full_items()}
    keys = sorted(items)
    if args.only:
        keys = [k for k in args.only if k in items]
    keys = [k for i, k in enumerate(keys) if i % args.shards == args.shard]
    if args.limit:
        keys = keys[: args.limit]

    kokoro_v = _dist_version("kokoro")
    misaki_v = _dist_version("misaki")
    print(
        f"kokoro={kokoro_v} misaki={misaki_v} markdown_floor="
        f"{version_at_least(kokoro_v, (0, 9, 2))} n={len(keys)}",
        flush=True,
    )

    import torch
    from kokoro import KPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
    print(f"device={device}", flush=True)
    scorer = Scorer()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    missing = []
    for n, key in enumerate(keys, 1):
        it = items[key]
        ph = phones.get(key)
        gold = GOLD_DIR / f"{key}.wav"
        if not ph:
            missing.append([key, "no-phones"])
            print(f"[{n}/{len(keys)}] skip {key} no-phones", flush=True)
            continue
        if not gold.exists() or gold.stat().st_size < 500:
            missing.append([key, "no-gold"])
            print(f"[{n}/{len(keys)}] skip {key} no-gold", flush=True)
            continue
        for arm in args.arms:
            meta = _meta_path(out, arm, key)
            wav = _wav_path(out, arm, key)
            if _done(meta, wav):
                print(f"[{n}/{len(keys)}] resume {arm} {key}", flush=True)
                continue
            try:
                synth = SYNTH[arm](pipeline, it, ph)
                _write_wav(synth["audio"], wav)
                record = synth["record"]
            except Exception as exc:
                record = _arm_record(
                    intended=ph,
                    realized="",
                    graphemes="",
                    drug_phones=None,
                    lexicon_key_miss=True if arm == "lexicon" else None,
                    span_wavlm_f1=None,
                    error=str(exc),
                )
                print(f"[{n}/{len(keys)}] ERR {arm} {key} {exc}", flush=True)
            else:
                try:
                    record["span_wavlm_f1"] = scorer.score(wav, it, gold)
                    if record["span_wavlm_f1"] is not None:
                        record["span_wavlm_f1"] = round(float(record["span_wavlm_f1"]), 4)
                    record["scored"] = True
                except Exception as exc:
                    record["error"] = f"score: {exc}"
                    print(f"[{n}/{len(keys)}] SCORE_ERR {arm} {key} {exc}", flush=True)
            meta.parent.mkdir(parents=True, exist_ok=True)
            meta.write_text(json.dumps(record, ensure_ascii=False))
            f1 = record.get("span_wavlm_f1")
            print(f"[{n}/{len(keys)}] {arm} {key} f1={f1}", flush=True)

    miss_path = out / "missing.jsonl"
    if missing:
        with miss_path.open("a") as handle:
            for row in missing:
                handle.write(json.dumps(row) + "\n")
    seen_miss: dict[str, str] = {}
    if miss_path.exists():
        for line in miss_path.read_text().splitlines():
            if line.strip():
                key, why = json.loads(line)
                seen_miss[key] = why
    merged = collect_rows(out, items, phones)
    report = build_report(
        merged,
        kokoro_v=kokoro_v,
        misaki_v=misaki_v,
        missing=sorted(seen_miss.items()),
    )
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"arms": report["arms"], "paired": report["paired"]}, indent=2), flush=True)
    print(f"WROTE {out / 'report.json'} n={report['n']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
