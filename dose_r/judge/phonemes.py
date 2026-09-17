"""ARPABET inventory, articulatory features, and the phoneme-pair cost model.

The phonetic scorer must not treat phonemes as opaque symbols. A TTS system that
says /d/ where the reference has /t/ has made a *voicing* slip that most
listeners will not even notice inside a drug name; a system that says /m/ where
the reference has /s/ has produced a different word. Raw string edit distance
scores both as "one substitution". This module is what makes them cost
differently.

Every phoneme carries a vector of articulatory features with values in [0, 1].
The cost of substituting `a` for `b` is the weighted mean absolute feature
difference, so a single-feature slip costs a fraction of a full substitution and
a maximally distant pair costs 1.0. One "phoneme error unit" (PEU) is therefore
a maximally distant substitution, and downstream scoring is denominated in PEU.

Stress is deliberately NOT a feature here. ARPABET stress digits are stripped
before feature lookup and handled as a separate, lightly weighted term in
`distance.py`, because saying AE1-T versus AE2-T is a prosody difference, not a
segmental error, and the two should never be conflated.
"""

from __future__ import annotations

from functools import lru_cache

STRESS_DIGITS = "012"

# --- feature axes -----------------------------------------------------------
#
# Consonant axes are ordered so that numeric adjacency tracks acoustic
# adjacency: neighbouring places of articulation and neighbouring manners come
# out close, distant ones come out far.

PLACE = {
    "bilabial": 0.00,
    "labiodental": 0.15,
    "dental": 0.30,
    "alveolar": 0.42,
    "postalveolar": 0.58,
    "palatal": 0.70,
    "velar": 0.85,
    "glottal": 1.00,
}

MANNER = {
    "stop": 0.00,
    "affricate": 0.22,
    "fricative": 0.42,
    "nasal": 0.62,
    "liquid": 0.82,
    "glide": 1.00,
}

HEIGHT = {"low": 0.00, "mid_low": 0.33, "mid": 0.50, "mid_high": 0.70, "high": 1.00}
BACKNESS = {"front": 0.00, "central": 0.50, "back": 1.00}

# Relative importance of each axis when comparing two phonemes of the same
# class. Manner and place dominate for consonants because they carry the bulk of
# the perceptual contrast; voicing is real but frequently neutralised by
# context, so it is worth a little over half a place/manner step.
CONSONANT_WEIGHTS = {"place": 1.00, "manner": 1.00, "voice": 0.60}

# For vowels, height and backness are the formant axes a listener actually
# tracks. Rounding and tenseness are secondary. The offglide axes only matter
# for diphthongs and are weighted so that AY/AA (same nucleus, offglide added)
# costs clearly less than AY/UW (different everything).
VOWEL_WEIGHTS = {
    "height": 1.00,
    "backness": 1.00,
    "round": 0.50,
    "rhotic": 0.80,
    "tense": 0.35,
    "off_height": 0.40,
    "off_backness": 0.40,
}

# A vowel and a consonant are different kinds of sound; no feature axis is
# shared, so the model assigns the maximum. The exception is the glide/liquid
# set, which is acoustically vocalic -- /r/ and /er/ especially are routinely
# interchanged by phoneme recognisers and by speakers.
CROSS_CLASS_COST = 1.00
CROSS_CLASS_SONORANT_COST = 0.72

# A handful of pairs the feature model gets wrong on its own. These are
# overrides, not a general escape hatch: each one is a documented, well attested
# confusion that no articulatory axis captures.
PAIR_OVERRIDES = {
    frozenset({"ER", "R"}): 0.12,  # syllabic /r/ vs /r/ -- often the same sound
    frozenset({"AH", "IH"}): 0.22,  # both reduce to schwa in unstressed position
    frozenset({"AH", "EH"}): 0.24,
    frozenset({"IH", "IY"}): 0.20,  # tense/lax high front, heavily dialectal
    frozenset({"UH", "UW"}): 0.20,
    frozenset({"AA", "AO"}): 0.16,  # cot/caught merger: merged for most speakers
    frozenset({"T", "DX"}): 0.10,  # flapping
    frozenset({"D", "DX"}): 0.10,
}

_C = "consonant"
_V = "vowel"


def _cons(place: str, manner: str, voiced: bool) -> dict:
    return {
        "class": _C,
        "place": PLACE[place],
        "manner": MANNER[manner],
        "voice": 1.0 if voiced else 0.0,
        "sonorant": manner in ("nasal", "liquid", "glide"),
    }


def _vowel(
    height: str,
    backness: str,
    rounded: bool,
    tense: bool,
    off: tuple[str, str] | None = None,
    rhotic: bool = False,
) -> dict:
    return {
        "class": _V,
        "height": HEIGHT[height],
        "backness": BACKNESS[backness],
        "round": 1.0 if rounded else 0.0,
        "tense": 1.0 if tense else 0.0,
        "rhotic": 1.0 if rhotic else 0.0,
        "off_height": HEIGHT[off[0]] if off else HEIGHT[height],
        "off_backness": BACKNESS[off[1]] if off else BACKNESS[backness],
    }


FEATURES: dict[str, dict] = {
    # stops
    "P": _cons("bilabial", "stop", False),
    "B": _cons("bilabial", "stop", True),
    "T": _cons("alveolar", "stop", False),
    "D": _cons("alveolar", "stop", True),
    "K": _cons("velar", "stop", False),
    "G": _cons("velar", "stop", True),
    # affricates
    "CH": _cons("postalveolar", "affricate", False),
    "JH": _cons("postalveolar", "affricate", True),
    # fricatives
    "F": _cons("labiodental", "fricative", False),
    "V": _cons("labiodental", "fricative", True),
    "TH": _cons("dental", "fricative", False),
    "DH": _cons("dental", "fricative", True),
    "S": _cons("alveolar", "fricative", False),
    "Z": _cons("alveolar", "fricative", True),
    "SH": _cons("postalveolar", "fricative", False),
    "ZH": _cons("postalveolar", "fricative", True),
    "HH": _cons("glottal", "fricative", False),
    # nasals
    "M": _cons("bilabial", "nasal", True),
    "N": _cons("alveolar", "nasal", True),
    "NG": _cons("velar", "nasal", True),
    # liquids and glides
    "L": _cons("alveolar", "liquid", True),
    "R": _cons("postalveolar", "liquid", True),
    "W": _cons("velar", "glide", True),
    "Y": _cons("palatal", "glide", True),
    # alveolar flap: not in strict CMU ARPABET but emitted by most phoneme
    # recognisers, and dropping it would silently mis-score every flapped /t/.
    "DX": _cons("alveolar", "stop", True),
    # monophthongs
    "IY": _vowel("high", "front", False, True),
    "IH": _vowel("mid_high", "front", False, False),
    "EH": _vowel("mid_low", "front", False, False),
    "AE": _vowel("low", "front", False, False),
    "AA": _vowel("low", "back", False, True),
    "AO": _vowel("mid_low", "back", True, True),
    "UH": _vowel("mid_high", "back", True, False),
    "UW": _vowel("high", "back", True, True),
    "AH": _vowel("mid", "central", False, False),
    "ER": _vowel("mid", "central", False, True, rhotic=True),
    # diphthongs: nucleus plus offglide target
    "EY": _vowel("mid_high", "front", False, True, off=("high", "front")),
    "AY": _vowel("low", "central", False, True, off=("high", "front")),
    "OY": _vowel("mid_low", "back", True, True, off=("high", "front")),
    "AW": _vowel("low", "central", False, True, off=("high", "back")),
    "OW": _vowel("mid_high", "back", True, True, off=("high", "back")),
}

VOWELS = frozenset(p for p, f in FEATURES.items() if f["class"] == _V)
CONSONANTS = frozenset(p for p, f in FEATURES.items() if f["class"] == _C)


class UnknownPhoneme(KeyError):
    """Raised for a symbol that is not ARPABET. Guessing would corrupt scores."""


def strip_stress(phoneme: str) -> str:
    """`AH0` -> `AH`. Leaves consonants untouched."""
    return phoneme.rstrip(STRESS_DIGITS) if phoneme[-1] in STRESS_DIGITS else phoneme


def stress_of(phoneme: str) -> int | None:
    """Stress level 0/1/2 for a vowel carrying a digit, else None."""
    return int(phoneme[-1]) if phoneme[-1] in STRESS_DIGITS else None


def is_vowel(phoneme: str) -> bool:
    return strip_stress(phoneme) in VOWELS


def parse(sequence: str | list[str]) -> list[str]:
    """Normalise an ARPABET sequence to an upper-case token list, stress kept."""
    tokens = sequence.split() if isinstance(sequence, str) else list(sequence)
    out = []
    for tok in tokens:
        tok = tok.strip().upper()
        if not tok:
            continue
        if strip_stress(tok) not in FEATURES:
            raise UnknownPhoneme(tok)
        out.append(tok)
    return out


@lru_cache(maxsize=None)
def substitution_cost(a: str, b: str) -> float:
    """Cost in [0, 1] of hearing `b` where the reference has `a`.

    Stress digits are ignored here by construction; `distance.py` scores stress
    separately.
    """
    a, b = strip_stress(a), strip_stress(b)
    if a == b:
        return 0.0

    override = PAIR_OVERRIDES.get(frozenset({a, b}))
    if override is not None:
        return override

    fa, fb = FEATURES[a], FEATURES[b]
    if fa["class"] != fb["class"]:
        sonorant_side = fa if fa["class"] == _C else fb
        return (
            CROSS_CLASS_SONORANT_COST
            if sonorant_side["sonorant"]
            else CROSS_CLASS_COST
        )

    weights = CONSONANT_WEIGHTS if fa["class"] == _C else VOWEL_WEIGHTS
    total = sum(w * abs(fa[k] - fb[k]) for k, w in weights.items())
    return total / sum(weights.values())


def cost_matrix() -> dict[frozenset[str], float]:
    """Full pairwise cost table, for inspection and regression testing."""
    symbols = sorted(FEATURES)
    return {
        frozenset({a, b}): substitution_cost(a, b)
        for i, a in enumerate(symbols)
        for b in symbols[i + 1 :]
    }
