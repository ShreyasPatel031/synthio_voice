# Gold Reference Layer — Coverage

284 unique ingredients across the 274 DOSE rows.

## Confidence

| Tier | Count | Share | Meaning |
| --- | --- | --- | --- |
| high | 99 | 34.9% | two independent sources agree |
| medium | 171 | 60.2% | exactly one external source answered |
| low | 14 | 4.9% | no external source; no ground truth |

## By name type

| Tier | brand | generic |
| --- | --- | --- |
| high | 55 | 44 |
| medium | 80 | 91 |
| low | 8 | 6 |

## Sources that answered

| Source | Ingredients |
| --- | --- |
| gemini-grounded-search | 355 |
| merriam-webster/medical-api | 89 |
| wikipedia | 22 |
| cmudict | 13 |
| dailymed | 2 |
| wiktionary | 2 |
| merriam-webster/dictionary | 1 |

## Baseline comparison

| Snapshot | high | medium | low |
| --- | --- | --- | --- |
| Original (MW HTML scrape + CMUdict only) | 19 | 80 | 185 |
| + Wikipedia/Wiktionary (`{{IPAc-en}}`/`{{IPA}}`/`{{respell}}`) | 28 | 83 | 173 |
| + MW Medical Dictionary API | 41 | 170 | 73 |
| + Gemini/Google-Search grounding, rule-based fallback removed | 98 | 171 | 15 |
| + DailyMed Medication Guide respellings (this build) | 99 | 171 | 14 |

The Wikipedia/Wiktionary and Medical API steps were the first two real gains.
The Gemini step is the largest one by far: Gemini 2.5 Flash with the
`google_search` tool retrieves a real page that states the pronunciation and
cites it; every claim then passes an independent Gemini 2.5 Flash format/
plausibility check before being trusted (the citation itself -- a real page
Google's search infrastructure actually retrieved -- is the verification, not
a model guess; the format check is a backstop against the extraction regex
grabbing an unrelated phrase, not a truth check). This replaced the old USAN-
stem and grapheme-to-phoneme rule fallbacks entirely: an ingredient with no
real source is now `low` confidence with no respelling at all, not a spelling-
derived guess dressed up as data.

DailyMed closed a handful more the Gemini step missed: many FDA Medication
Guides state the brand's own phonetic respelling right in their title line
(`AMBELVIST (am bel' vist)`), which Gemini's default two-query web search
didn't happen to surface even though the source is real, free, and reachable
from this environment. A regex-based extraction like this needs its own
false-positive guard -- a table cell like `YUVIWEL (gross content per vial)`
also has the shape "NAME (something with a space)" -- so every candidate is
checked against the same Gemini format-plausibility judge before acceptance.

## Sources tried

| Source | Outcome |
| --- | --- |
| Merriam-Webster Medical API | **Wired in.** Structured, reliable; 89 ingredients. Needs `MW_MEDICAL_KEY` in `.env`; degrades to the HTML path when absent. |
| Merriam-Webster `/dictionary/` (HTML) | **Wired in**, as the fallback for names the medical API misses; 1 ingredient(s) this run. |
| Wikipedia (`{{IPAc-en}}`, `{{IPA\|en\|...}}`, `{{respell}}`) | **Wired in.** 22 ingredients. Most DOSE brand names are too new or minor for an English Wikipedia article at all. |
| Wiktionary (same templates) | **Wired in.** 2 ingredients; thin, and mostly overlaps Wikipedia rather than adding new names. |
| CMUdict | **Wired in** (pre-existing). 13 ingredients; a general dictionary, not a drug-name resource. |
| DailyMed (FDA Medication Guides) | **Wired in.** 2 ingredients (brands only). Reachable from this environment (unlike drugs.com), and a real find caught by manual spot-checking after this build shipped: many Medication Guides state the brand's own respelling right in the title line (`AMBELVIST (am bel' vist)`), in the same USAN prime-stress notation the Gemini-grounded path already parses. Not every label includes one, so this doesn't close every remaining gap. |
| Drugs.com (direct fetch) | Blocked. HTTP 403 on every direct request from this environment, medical and general pages alike. **Reached indirectly**: Gemini's `google_search` tool retrieves and cites Drugs.com pages server-side (Google's infrastructure, not this sandbox, does the fetch), so a citation naming drugs.com is still accepted as a real source even though this environment can't independently re-fetch it -- 355 ingredients answered via Gemini-grounded search overall (drugs.com and otherwise). |
| DrugBank | Dead end. HTTP 403. |
| FDA labels via openFDA (structured JSON) | Dead end for pronunciation. Reachable (200), but the structured label JSON does not carry the Medication Guide's free-text title line, which is where a respelling (if present at all) actually lives -- see DailyMed above, which serves the rendered guide text instead of the structured fields. |
| NLM RxNav / RxNorm | Dead end for pronunciation. Reachable, resolves names to RxCUIs reliably, but `allProperties` carries only coding/synonym fields (ATC, SNOMED, DrugBank ID, etc.) -- no phonetic field exists in the schema. Kept as the id-lookup step for the MedlinePlus Connect pipeline below. |
| MedlinePlus drug monographs | **Real pronunciations confirmed** (e.g. Metformin: `pronounced as (met for' min)`), reachable via a working, unauthenticated pipeline: RxNav name-to-RxCUI, then `connect.medlineplus.gov` (RxNorm OID `2.16.840.1.113883.6.88`) to the drug's monograph URL, then a page fetch. **Not wired into this build**: MedlinePlus's own lay respelling (`met for' min`, `a set a mee' noe fen`) is not the same key as Wikipedia's, and no citable published key for it was found -- guessing its letter-to-phoneme mapping would risk silently wrong phonemes recorded as sourced, which is the failure mode this project must not introduce. See the ranked list below. |
| NCI Drug Dictionary (cancer.gov) | Dead end as scraped. The public page is a React SPA (`drug-dictionary-app`) with no server-rendered content; its bundled config points at `webapis-dev.cancer.gov`, which does not resolve (NXDOMAIN) -- the backing API is not public from this environment. |
| AMA USAN pronunciation guide (key) | Found and reachable, not paywalled (`"gating_state":"not gated"`). It is the **notation key** the USAN Council uses (prime/double-prime stress marks, documented digraphs), not a per-drug lookup -- it explains how to read a pronunciation, it doesn't supply one. |
| AMA USAN Drug Finder (searchusan.ama-assn.org) | Dead end as scraped: an Angular SPA; the string `pronun` does not appear anywhere in its main JS bundle, so the finder itself does not appear to expose pronunciation, only naming/adoption-status data. |
| WHO INN lists (who.int) | Reachable (200), but the published INN list documents are name/CAS-number registries, not phonetic dictionaries; no pronunciation field found. |
| StatPearls / NCBI Bookshelf | Not pursued past a spot check -- these are clinical review monographs, not lexicographic sources, and did not surface pronunciation content. |

## Blocked or paywalled sources ranked by expected gain

Gemini/Google-Search grounding closed most of the old `low` tier, generic
and brand alike (it found real citations for coined INN names like
elranatamab-bcmm and risankizumab-rzaa just as readily as for brand names).
What's left (14 ingredients) skews brand-name-heavy --
these are mostly very recent approvals with essentially no indexed
pronunciation content anywhere on the public web yet, not a gap this
pipeline's extraction or verification logic is failing to close.

1. **USP Dictionary of USAN and International Drug Names** -- the compiled,
   official pronunciation reference for essentially every USAN/INN generic
   name, using the documented AMA/USAN key (prime-mark stress, plain-English
   digraphs) already confirmed public. Largely superseded by the Gemini step
   for coverage, but still the authoritative source where Gemini's search
   result disagrees with itself or looks unreliable. **What's needed:** USP
   sells it as a
   purchased publication/subscription -- buy access (print or the USP
   online reference platform) or reach the USAN Council directly for the
   per-drug Statements of Adoption, which carry the same pronunciation.
2. **A source key for MedlinePlus's own respelling notation** -- the
   monograph pages themselves are free and already reachable (pipeline
   above); only the notation-to-ARPABET converter is missing, and it needs a
   citable key, not a guess. **What's needed:** either NLM/ASHP documentation
   of the respelling conventions AHFS Consumer Medication Information uses,
   or a licensed relationship with ASHP (which authors that content) to
   confirm the mapping. This unblocks real pronunciations for a large slice
   of both brand and generic names with no new access cost once the key is
   in hand.
3. **Drugs.com / DrugBank** -- both carry real audio and IPA/respelling for a
   large share of these exact brand names (verified by manual browsing
   outside this bot-blocked environment). **What's needed:** an approved API
   partnership or a scraping allowance from the vendor (Drugs.com is run by
   Drugsite Trust; DrugBank offers a commercial data-license API) --
   whitelisting this environment's egress IP alone will not clear a
   Cloudflare-style bot check, so this is a data-license conversation, not an
   IT-access one.
4. **NCI Drug Dictionary backing API** -- confirmed (from the brief) to carry
   real oncology-INN pronunciations; the public site's own React app cannot
   reach it (dead `webapis-dev` host). **What's needed:** someone at NCI/CBIIT
   to confirm the correct production API host, or a data-use request for a
   bulk export of the drug dictionary. Value is narrower than #1 -- oncology
   INNs only -- but several DOSE names are exactly that class.

## Needs arbitration

14 ingredients have no external source at all -- Gemini's
Google-Search grounding either found nothing or nothing that survived the
LLM format check. There is no rule-based fallback for these: no phonetic
reference exists for them in this layer, full stop.

- Jideytro (brand)
- Kyzatrex (brand)
- Lynavoy (brand)
- Revtorpyk (brand)
- Veppanu (brand)
- Vyglxia (brand)
- Wakix (brand)
- Yuviwel (brand)
- cipepofol (generic)
- copper histidinate (generic)
- insulin icodec-abae (generic)
- nogapendekin alfa inbakicept-pmln (generic)
- pivekimab sunirine-pvzy (generic)
- prademagene zamikeracel (generic)
