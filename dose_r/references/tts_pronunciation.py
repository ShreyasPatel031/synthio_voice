"""Uniform TTS pronunciation fields derived only from the canonical respelling.

The 40% of the benchmark with no human clip cannot use a score-gated switch
(plain if the engine already knows the name, spaced syllables if it does not).
Every name has to carry the same kind of string.

Two renderings of the same canonical hyphenated respelling:

  compact_ascii  wee-GOH-vee -> weegohvee
                 One token per word. No syllable pauses. Letters change
                 when the dictionary does (Xeljanz -> zeljans).

  ipa_from_canonical  wee-GOH-vee -> wiːˈɡoʊviː
                 Sidecar IPA for Cloud TTS `customPronunciations` (or an
                 SSML <phoneme> tag). The spoken *text* stays the real
                 spelling, so the 40% without audio and the names the
                 engine already knows are not rewritten as many words.

Do not feed the hyphenated/spaced dictionary string itself to TTS. That
channel inserts pauses and lost 0.08 mean F1 on Standard-C.
"""

from __future__ import annotations

from .wiki_notation import respell_to_arpabet_ipa


def compact_ascii(canonical: str) -> str:
    """Drop hyphens inside each word; keep real word spaces.

    `ak-oh-RAM-id-is` -> `akohramidis`
    `TROSE-pee-um chloride` -> `trosepeeum chloride`
    """
    return " ".join(word.replace("-", "").lower() for word in canonical.split() if word)


def ipa_from_canonical(canonical: str) -> str:
    """Wikipedia-key IPA for each word of the canonical respelling.

    Uses the ye=/aɪ/ and silent-e rules in `wiki_notation`. Multi-word
    names keep a space between word IPAs, which Cloud TTS accepted as a
    phrase pronunciation.
    """
    parts: list[str] = []
    for word in canonical.split():
        syllables = [p for p in word.split("-") if p]
        if not syllables:
            continue
        _, ipa = respell_to_arpabet_ipa(syllables)
        parts.append(ipa)
    if not parts:
        raise ValueError(f"no IPA in {canonical!r}")
    return " ".join(parts)


# Cloud TTS en-US customPronunciations rejects British centering
# diphthongs (Lyrica `ˈlɪərɪkɑː` -> HTTP 400). Fold them to the rhotic
# sequences in Google's published en-US inventory. Other non-US symbols
# (`ɒ`, `ɜː`, `ɛə`) have been accepted on this voice and are left alone.
_CLOUD_EN_US = (
    ("ɪər", "ɪr"),
    ("ɪə", "ɪr"),
    ("ʊər", "ʊr"),
    ("ʊə", "ʊr"),
)


def to_cloud_en_us_ipa(ipa: str) -> str:
    """Rewrite IPA so Cloud TTS en-US customPronunciations will accept it."""
    out = ipa
    for src, dst in _CLOUD_EN_US:
        out = out.replace(src, dst)
    return out


def custom_pronunciation(phrase: str, ipa: str) -> dict:
    """Cloud TTS `SynthesisInput.customPronunciations` block.

    `phrase` must be an exact substring of the text being synthesized
    (the ingredient name, not the compact rendering).
    """
    return {
        "pronunciations": [
            {
                "phrase": phrase,
                "phoneticEncoding": "PHONETIC_ENCODING_IPA",
                "pronunciation": ipa,
            }
        ]
    }
