"""Era classification: established vs newly-approved, per DOSE-R row.

DOSE reports 128 traditional / 146 new and does not publish which rows are
which. This module rebuilds the split from openFDA Drugs@FDA approval dates
(`fda_lookup.py`), one external fact this project did not have to invent.

Two things this module refuses to do:

  - Fabricate a date. A row with no FDA hit gets an explicit heuristic label
    (`era_source="heuristic_no_fda_match"`), never an invented date.
  - Trust "not found" at face value. `fda_lookup.candidate_queries` originally
    queried only openFDA's harmonized `openfda.*` fields, which are missing on
    a surprising number of long-approved drugs (Eliquis, Benadryl, Biktarvy,
    Ubrelvy and Wegovy all miss every `openfda.*` field). Falling back to
    "not found implies new" on that query alone would have misclassified five
    well-established drugs as newly-approved. Adding the raw `products.*`
    fields as a fallback (done in `fda_lookup.py`) fixed all five; see
    STRATA.md for the before/after hit rate.

A combination row (e.g. "bictegravir, emtricitabine, and tenofovir
alafenamide") has no single FDA application of its own to query -- DOSE's
sentence names the ingredients, not the marketed combination product. This
module looks up each ingredient separately and takes the *latest* (max)
approval date among them: a combination enters clinical speech only once its
newest component does, so the newest ingredient bounds the row's true age.
That is an approximation, not the combination product's real approval date,
and is recorded as such.
"""

from __future__ import annotations

from dataclasses import dataclass

from .fda_lookup import FdaClient, Lookup, iso

CUTOFF_DATE = "2021-08-27"
CUTOFF_RATIONALE = (
    "5 years before 2026-08-27, the most recent approval date openFDA "
    "returned for any ingredient in this dataset -- a trailing window "
    "anchored to the data rather than to today's real-world date, so it "
    "does not silently drift as time passes. See STRATA.md for the "
    "sensitivity of the 128/146 split to this choice."
)


@dataclass(frozen=True)
class EraResult:
    era: str  # "established" | "new"
    era_confidence: str  # "high" | "medium" | "low"
    era_source: str
    max_approval_date: str | None
    ingredient_dates: dict[str, str | None]
    unresolved_ingredients: list[str]


def _confidence_for(lookup: Lookup) -> str:
    if lookup.status != "hit" or not lookup.matched_query:
        return "low"
    return "high" if lookup.matched_query.startswith("openfda.") else "medium"


def classify_row(ingredients: list[str], lookups: dict[str, Lookup]) -> EraResult:
    dates: dict[str, str | None] = {}
    confidences: list[str] = []
    unresolved: list[str] = []

    for ing in ingredients:
        lookup = lookups[ing]
        if lookup.status == "hit":
            dates[ing] = iso(lookup.approval_date)
            confidences.append(_confidence_for(lookup))
        else:
            dates[ing] = None
            unresolved.append(ing)

    resolved_dates = [d for d in dates.values() if d]

    if not resolved_dates:
        return EraResult(
            era="new",
            era_confidence="low",
            era_source="heuristic_no_fda_match",
            max_approval_date=None,
            ingredient_dates=dates,
            unresolved_ingredients=unresolved,
        )

    max_date = max(resolved_dates)
    era = "established" if max_date < CUTOFF_DATE else "new"
    # A row is only as confident as its least confident ingredient, and an
    # unresolved co-ingredient on an otherwise-dated combination row weakens
    # that further even though it did not change the max-date decision.
    confidence = min(confidences, key=["low", "medium", "high"].index)
    if unresolved and confidence != "low":
        confidence = "medium" if confidence == "high" else confidence
    source = "openfda_max_ingredient_approval" if len(ingredients) > 1 else "openfda_approval"

    return EraResult(
        era=era,
        era_confidence=confidence,
        era_source=source,
        max_approval_date=max_date,
        ingredient_dates=dates,
        unresolved_ingredients=unresolved,
    )


def lookup_all(ingredient_types: dict[str, str], client: FdaClient | None = None) -> dict[str, Lookup]:
    client = client or FdaClient()
    return {name: client.lookup(name, name_type) for name, name_type in ingredient_types.items()}
