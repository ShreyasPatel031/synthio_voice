"""Score a NEW candidate TTS/voice model against ALL available human
references for a drug name, crediting the candidate for matching ANY
accepted pronunciation -- built for Workstream 1's upcoming open-source
candidate model evaluations.

Why this exists, not just `SpeechSimilarityScorer`
----------------------------------------------------
`scoring.speech_similarity.SpeechSimilarityScorer` (Path 2, the shipped
production scorer) deliberately scores against ONE reference clip per
ingredient (`references.reference_clips.available_clips()`, Merriam-Webster
preferred) -- the right contract for scoring a FIXED candidate consistently
across repeated runs over time.

Evaluating a brand-new candidate model is a different question: is this
model's pronunciation an accepted one AT ALL, not "does it match this one
specific recording." Scoring against only one human reference already
produced a real false positive this session -- aripiprazole and acoramidis
scored as if mispronounced (top-5 worst of 171 items) when the user
confirmed by ear they were just different, valid pronunciation variants.
For the 79 ingredients with BOTH a Drugs.com and a Merriam-Webster
recording, using only one of them risks exactly that failure mode whenever
the candidate happens to match the OTHER one.

This module scores against every available clip and takes the maximum --
crediting a candidate for matching any one accepted human pronunciation,
not penalizing it for not matching a specific recording's incidental
choice among several correct options.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..references.reference_clips import ReferenceClip
from .speech_similarity import extract_frame_embeddings, speech_bertscore


@dataclass(frozen=True)
class MultiReferenceResult:
    best_f1: float
    best_source: str
    all_scores: dict[str, float]  # source -> f1, for every reference actually tried


def score_against_best_reference(
    candidate_span: bytes, clips: list[ReferenceClip],
) -> MultiReferenceResult:
    """candidate_span (extracted drug-name audio) vs. every clip in `clips`
    -- returns the best (max F1) match and every individual score, so a
    caller can see whether a candidate matched one reference clearly or was
    mediocre against all of them (a real difference worth keeping visible,
    not collapsing into a single number).
    """
    if not clips:
        raise ValueError("no reference clips given -- caller should check "
                         "coverage before calling this")

    feats_candidate = extract_frame_embeddings(candidate_span)
    all_scores: dict[str, float] = {}
    for clip in clips:
        feats_ref = extract_frame_embeddings(clip.path)
        result = speech_bertscore(feats_candidate, feats_ref)
        # A later clip from the same source (e.g. duplicate manifest rows)
        # overwrites; sources are otherwise distinct (drugs.com, merriam-
        # webster, umich) so this only dedupes exact repeats.
        all_scores[clip.source] = result["f1"]

    best_source = max(all_scores, key=all_scores.get)
    return MultiReferenceResult(
        best_f1=all_scores[best_source], best_source=best_source, all_scores=all_scores,
    )
