"""End-to-end judging run: both scorers, the guardrail, and the artifacts.

Input is one JSONL of system renderings, one object per ingredient span:

    {"item_id": "dose-056#0", "arpabet": "B IH0 K T EH1 G R AH0 V IH0 R",
     "audio_uri": "s3://.../dose-056.wav"}

`arpabet` is the recognised phoneme sequence for the drug-name span; producing
it is upstream of this package. `audio_uri` is what the audio-LLM panel needs
and may be omitted while the panel is mocked.

Outputs, all written under `artifacts/`:
    phonetic.jsonl      per-span phonetic score with full decomposition
    confusability.jsonl per-span nearest confusable drug and margin
    panel.jsonl         per-span panel verdict and individual judge scores
    rows.jsonl          row-level scores after combination aggregation
    agreement.json      agreement between the two scorers, overall and by stratum
    flagged.jsonl       the manual listen-through worklist
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .agreement import ItemAgreement, report as agreement_report, write_flagged
from .combination import DEFAULT_AGGREGATOR, aggregate_all, aggregator_sensitivity
from .confusability import ConfusabilityIndex
from .dataset import Row, Span, load_rows
from .llm_panel import JudgePanel, JudgeRequest, mock_panel
from .phonetic_scorer import PhoneticScorer
from .references import ReferenceSet

ARTIFACT_DIR = Path("artifacts")


@dataclass(frozen=True)
class Hypothesis:
    item_id: str
    arpabet: str
    audio_uri: str = ""


def load_hypotheses(path: str | Path) -> dict[str, Hypothesis]:
    with Path(path).open() as f:
        objs = [json.loads(line) for line in f if line.strip()]
    return {
        o["item_id"]: Hypothesis(o["item_id"], o["arpabet"], o.get("audio_uri", ""))
        for o in objs
    }


def reference_hypotheses(rows: list[Row], references: ReferenceSet) -> dict[str, Hypothesis]:
    """A perfect system: every span rendered as its preferred reference variant.

    Useful as the known-good baseline for the tripwire and as a smoke test that
    a run wires up end to end.
    """
    out = {}
    for row in rows:
        for span in row.spans:
            reference = references.get(span.ingredient)
            if reference:
                out[span.item_id] = Hypothesis(
                    span.item_id, " ".join(reference.preferred.arpabet), ""
                )
    return out


SIMULATION_MIX = (
    ("reference", 0.40),
    ("alternate_variant", 0.10),
    ("voicing_slip", 0.12),
    ("stress_shift", 0.10),
    ("swap_adjacent", 0.10),
    ("whole_phoneme_wrong", 0.08),
    ("dropped_syllable", 0.06),
    ("other_drug", 0.04),
)


def simulated_hypotheses(
    rows: list[Row], references: ReferenceSet, seed: int = 7, quality: float = 1.0
) -> dict[str, Hypothesis]:
    """A synthetic system under test, for exercising the reporting end to end.

    Draws a perturbation class per span from `SIMULATION_MIX`; `quality` scales
    the probability mass on the error classes, so two runs at different quality
    give a baseline/candidate pair for the tripwire. This produces no audio and
    is not a substitute for a real system.
    """
    import random

    from .stress_test import perturb

    index = ConfusabilityIndex(references)
    graph = index.neighbor_graph(k=1)
    weights = [(kind, w if kind == "reference" else w * quality) for kind, w in SIMULATION_MIX]

    out = {}
    for row in rows:
        for span in row.spans:
            reference = references.get(span.ingredient)
            if reference is None:
                continue
            rng = random.Random(f"{seed}:{span.item_id}")
            kinds = [k for k, _ in weights]
            chosen = rng.choices(kinds, weights=[w for _, w in weights])[0]
            neighbors = graph.get(reference.ingredient)
            neighbor = references.get(neighbors[0].ingredient) if neighbors else None
            sequence = perturb(reference, chosen, rng, neighbor) or list(
                reference.preferred.arpabet
            )
            out[span.item_id] = Hypothesis(span.item_id, " ".join(sequence), "")
    return out


def _stratum(span: Span) -> str:
    return f"{span.name_type}{'/combination' if span.is_combination else ''}"


@dataclass
class JudgeRun:
    phonetic: dict
    confusability: dict
    panel: dict
    rows: list
    items: list
    agreement: dict
    aggregator_sensitivity: dict
    skipped: list[str]

    def tripwire_input(self) -> dict[str, tuple[int, float]]:
        """`{item_id: (primary score, confusability margin)}` for `calibration.tripwire`."""
        return {
            item_id: (score.score, self.confusability[item_id].margin)
            for item_id, score in self.phonetic.items()
        }


def run(
    hypotheses: dict[str, Hypothesis],
    references: ReferenceSet,
    rows: list[Row] | None = None,
    panel: JudgePanel | None = None,
    aggregator: str = DEFAULT_AGGREGATOR,
) -> JudgeRun:
    rows = load_rows() if rows is None else rows
    spans = {s.item_id: s for row in rows for s in row.spans}

    scorer = PhoneticScorer(references)
    pool = sorted({s.ingredient for s in spans.values() if s.ingredient in references})
    index = ConfusabilityIndex(references, pool)

    phonetic, confusability, skipped = {}, {}, []
    for item_id, span in spans.items():
        hypothesis = hypotheses.get(item_id)
        if hypothesis is None or span.ingredient not in references:
            skipped.append(item_id)
            continue
        phonetic[item_id] = scorer.score(span.ingredient, hypothesis.arpabet, item_id)
        confusability[item_id] = index.report(
            span.ingredient, hypothesis.arpabet, item_id
        )

    if panel is None:
        panel = mock_panel({k: v.score for k, v in phonetic.items()})

    panel_verdicts = {}
    for item_id in phonetic:
        span = spans[item_id]
        panel_verdicts[item_id] = panel.judge(
            JudgeRequest(
                item_id=item_id,
                ingredient=span.ingredient,
                sentence=span.sentence,
                span=span.span,
                audio_uri=hypotheses[item_id].audio_uri,
                reference_ipa=phonetic[item_id].best_variant_ipa,
            )
        )

    scored_rows = [r for r in rows if all(s.item_id in phonetic for s in r.spans)]
    row_scores = aggregate_all(
        scored_rows, {k: v.score for k, v in phonetic.items()}, aggregator
    )

    items = [
        ItemAgreement(
            item_id=item_id,
            ingredient=spans[item_id].ingredient,
            stratum=_stratum(spans[item_id]),
            phonetic_score=phonetic[item_id].score,
            panel_score=panel_verdicts[item_id].score,
            panel_spread=panel_verdicts[item_id].spread,
        )
        for item_id in phonetic
    ]

    return JudgeRun(
        items=items,
        phonetic=phonetic,
        confusability=confusability,
        panel=panel_verdicts,
        rows=row_scores,
        agreement=agreement_report(items),
        aggregator_sensitivity=aggregator_sensitivity(
            scored_rows, {k: v.score for k, v in phonetic.items()}
        ),
        skipped=skipped,
    )


def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def write_artifacts(result: JudgeRun, out_dir: str | Path = ARTIFACT_DIR) -> Path:
    out = Path(out_dir)
    _write_jsonl(out / "phonetic.jsonl", (v.to_dict() for v in result.phonetic.values()))
    _write_jsonl(
        out / "confusability.jsonl", (v.to_dict() for v in result.confusability.values())
    )
    _write_jsonl(out / "panel.jsonl", (v.to_dict() for v in result.panel.values()))
    _write_jsonl(out / "rows.jsonl", (r.to_dict() for r in result.rows))
    (out / "agreement.json").write_text(
        json.dumps(
            {
                "agreement": result.agreement,
                "aggregator_sensitivity": result.aggregator_sensitivity,
                "skipped": result.skipped,
            },
            indent=2,
        )
    )
    write_flagged(result.items, out / "flagged.jsonl")
    return out
