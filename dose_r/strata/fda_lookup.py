"""FDA approval-date lookup against openFDA's Drugs@FDA endpoint.

The era stratum ("established" vs "newly-approved") is not a public DOSE column,
so it has to be inferred from an external approval date. This module is the
external half: given a drug name, return the earliest US approval date for it,
or an explicit miss.

Two properties of the source shape the design:

1. Drugs@FDA covers NDA/BLA applications only. Pipeline compounds that DOSE
   included but the FDA has not approved return nothing, and a miss there is
   real information (unapproved => certainly not "established"), not an error.
2. Biologic generic names carry a four-letter FDA suffix ("bevacizumab-vikg")
   that never appears in the application record, so the suffix must be stripped
   before querying.

Responses are cached on disk so a rebuild costs no requests; openFDA's
anonymous quota is 1000 requests/day.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ENDPOINT = "https://api.fda.gov/drug/drugsfda.json"
DEFAULT_CACHE = Path("dose_r/strata/.fda_cache.json")

BIOLOGIC_SUFFIX = re.compile(r"-[a-z]{4}$")
DATE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")

# Salt/ester/formulation words that appear in DOSE generic names but not in the
# Drugs@FDA active-ingredient string; dropping them widens an exact-name query
# into one that matches.
MODIFIERS = [
    "marboxil",
    "propionate",
    "fumarate dihydrate",
    "fumarate",
    "decanoate",
    "undecanoate",
    "bromide",
    "chloride",
    "pivoxil",
    "deruxtecan",
    "sunirine",
    "histidinate",
    "alfa",
    "autotemcel",
    "autoleucel",
    "zamikeracel",
]


@dataclass
class Lookup:
    """Result of one name lookup."""

    name: str
    approval_date: str | None
    application_number: str | None
    matched_query: str | None
    status: str  # "hit" | "not_found" | "network_error"
    detail: str = ""


@dataclass
class FdaClient:
    cache_path: Path = DEFAULT_CACHE
    timeout: float = 25.0
    pause: float = 0.3
    offline: bool = False
    cache: dict = field(default_factory=dict)
    network_failures: list[str] = field(default_factory=list)
    requests_made: int = 0

    def __post_init__(self) -> None:
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text())

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=1, sort_keys=True))

    def _fetch(self, query: str) -> dict | None:
        """Raw openFDA search. None means the request itself failed."""
        if query in self.cache:
            return self.cache[query]
        if self.offline:
            self.network_failures.append(f"{query}: offline mode, no cache entry")
            return None

        url = f"{ENDPOINT}?{urllib.parse.urlencode({'search': query, 'limit': 10})}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                payload = {"results": []}  # openFDA's "no match"
            else:
                self.network_failures.append(f"{query}: HTTP {e.code}")
                return None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            self.network_failures.append(f"{query}: {type(e).__name__}: {e}")
            return None

        self.requests_made += 1
        self.cache[query] = payload
        time.sleep(self.pause)
        return payload

    def lookup(self, name: str, name_type: str) -> Lookup:
        errored = False
        for query in candidate_queries(name, name_type):
            payload = self._fetch(query)
            if payload is None:
                errored = True
                continue
            date, appno = earliest_approval(payload)
            if date:
                return Lookup(name, date, appno, query, "hit")
        if errored:
            return Lookup(name, None, None, None, "network_error", "all queries failed")
        return Lookup(name, None, None, None, "not_found", "no approved application")


def normalize(name: str) -> str:
    return BIOLOGIC_SUFFIX.sub("", name.strip().lower())


def base_forms(name: str) -> list[str]:
    """Progressively simpler forms of a name, most specific first."""
    stripped = normalize(name)
    forms = [stripped]
    for mod in MODIFIERS:
        if stripped.endswith(" " + mod):
            forms.append(stripped[: -(len(mod) + 1)].strip())
    head = stripped.split()[0] if stripped.split() else stripped
    if head not in forms:
        forms.append(head)
    return forms


def candidate_queries(name: str, name_type: str) -> list[str]:
    """Query fields, `openfda.*` first then the raw `products.*` fallback.

    `openfda.*` is a harmonized enrichment block that openFDA computes after
    the fact, and it is missing on a surprising number of real, long-approved
    applications -- Eliquis, Benadryl, Biktarvy, Ubrelvy and Wegovy all miss
    every `openfda.*` field despite being unambiguously approved. `products.*`
    is the raw application data and finds all five. Trusting `openfda.*` alone
    would misclassify well-established drugs as "not found" and, under the
    obvious fallback heuristic, as newly-approved -- the opposite of correct.
    """

    fields = (
        [
            "openfda.brand_name",
            "openfda.generic_name",
            "openfda.substance_name",
            "products.brand_name",
            "products.active_ingredients.name",
        ]
        if name_type == "brand"
        else [
            "openfda.generic_name",
            "openfda.substance_name",
            "openfda.brand_name",
            "products.active_ingredients.name",
            "products.brand_name",
        ]
    )
    queries = []
    for form in base_forms(name):
        for f in fields:
            q = f'{f}:"{form}"'
            if q not in queries:
                queries.append(q)
    return queries


def earliest_approval(payload: dict) -> tuple[str | None, str | None]:
    """Earliest approved ORIGinal submission across the matched applications.

    Supplements are excluded: a 2006 labeling supplement on a 2002 drug says
    nothing about when the name entered clinical speech. Falling back to the
    earliest approved supplement only happens when no ORIG record exists, which
    occurs for a handful of pre-electronic applications.
    """
    best: tuple[str, str] | None = None
    fallback: tuple[str, str] | None = None
    for result in payload.get("results", []):
        appno = result.get("application_number", "")
        for sub in result.get("submissions", []):
            if sub.get("submission_status") != "AP":
                continue
            date = sub.get("submission_status_date")
            if not date or not DATE.match(date):
                continue
            slot = (date, appno)
            if sub.get("submission_type") == "ORIG":
                best = slot if best is None or slot < best else best
            else:
                fallback = slot if fallback is None or slot < fallback else fallback
    chosen = best or fallback
    return chosen if chosen else (None, None)


def iso(date: str) -> str:
    m = DATE.match(date)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else date
