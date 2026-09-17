import json
from pathlib import Path

import pytest

from dose_r.judge.references import ReferenceSet

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

# Hand-written, real ARPABET. Small enough to reason about by hand, which is
# what unit tests of a distance metric need; the machine-guessed 284-name
# fixture is for scale tests only.
HAND_REFERENCES = [
    {
        "ingredient": "atorvastatin",
        "name_type": "generic",
        "ipa_variants": ["əˈtɔɹvəstætɪn", "ˌætɔɹˈvæstətɪn"],
        "arpabet_variants": [
            "AH0 T AO1 R V AH0 S T AE2 T IH0 N",
            "AE2 T AO0 R V AE1 S T AH0 T IH0 N",
        ],
        "sources": [{"name": "hand", "raw": "atorvastatin", "url": ""}],
        "confidence": "high",
        "notes": "second variant is the stress pattern several US speakers use",
    },
    {
        "ingredient": "metformin",
        "name_type": "generic",
        "ipa_variants": ["mɛtˈfɔɹmɪn"],
        "arpabet_variants": ["M EH0 T F AO1 R M IH0 N"],
        "sources": [{"name": "hand", "raw": "metformin", "url": ""}],
        "confidence": "high",
        "notes": "",
    },
    {
        "ingredient": "metoprolol",
        "name_type": "generic",
        "ipa_variants": ["mɛˈtoʊpɹəlɔl"],
        "arpabet_variants": ["M EH0 T OW1 P R AH0 L AO2 L"],
        "sources": [{"name": "hand", "raw": "metoprolol", "url": ""}],
        "confidence": "high",
        "notes": "",
    },
    {
        "ingredient": "misoprostol",
        "name_type": "generic",
        "ipa_variants": ["ˌmaɪsoʊˈpɹɔstɔl"],
        "arpabet_variants": ["M AY2 S OW0 P R AO1 S T AO2 L"],
        "sources": [{"name": "hand", "raw": "misoprostol", "url": ""}],
        "confidence": "medium",
        "notes": "ISMP-listed confusion with metoprolol",
    },
    {
        "ingredient": "Celebrex",
        "name_type": "brand",
        "ipa_variants": ["ˈsɛləbɹɛks"],
        "arpabet_variants": ["S EH1 L AH0 B R EH2 K S"],
        "sources": [{"name": "hand", "raw": "Celebrex", "url": ""}],
        "confidence": "high",
        "notes": "",
    },
    {
        "ingredient": "Cerebyx",
        "name_type": "brand",
        "ipa_variants": ["ˈsɛɹəbɪks"],
        "arpabet_variants": ["S EH1 R AH0 B IH0 K S"],
        "sources": [{"name": "hand", "raw": "Cerebyx", "url": ""}],
        "confidence": "high",
        "notes": "ISMP-listed confusion with Celebrex",
    },
]


@pytest.fixture(scope="session")
def hand_references() -> ReferenceSet:
    from dose_r.judge.references import _reference_from_obj

    return ReferenceSet([_reference_from_obj(o) for o in HAND_REFERENCES])


@pytest.fixture(scope="session")
def synthetic_references() -> ReferenceSet:
    path = FIXTURES / "synthetic_references.jsonl"
    if not path.exists():
        pytest.skip("run python -m dose_r.judge.fixtures.build_synthetic first")
    return ReferenceSet.load(path)


@pytest.fixture
def write_references(tmp_path):
    def _write(objs):
        path = tmp_path / "references.jsonl"
        path.write_text("\n".join(json.dumps(o) for o in objs))
        return path

    return _write
