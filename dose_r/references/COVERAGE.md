# Gold Reference Layer — Coverage

> **Do not convert** DailyMed/USAN/NCI respelling in this layer into IPA.
> Keep `sources[].raw`. See `README.md`.

284 unique ingredients across the 274 DOSE rows.

## Confidence

| Tier | Count | Share | Meaning |
| --- | --- | --- | --- |
| high | 114 | 40.1% | two independent sources agree |
| medium | 164 | 57.7% | exactly one external source answered |
| low | 6 | 2.1% | no external source; no ground truth |

## By name type

| Tier | brand | generic |
| --- | --- | --- |
| high | 60 | 54 |
| medium | 78 | 86 |
| low | 5 | 1 |

## Sources that answered

| Source | Ingredients |
| --- | --- |
| usan-official | 108 |
| dailymed | 96 |
| merriam-webster/medical-api | 87 |
| gemini-grounded-search | 62 |
| nci-dictionary-of-cancer-terms | 50 |
| wikipedia | 20 |
| cmudict | 14 |
| merriam-webster/dictionary | 4 |
| wiktionary | 2 |

## Trust tiers

Every citation (not just Gemini-grounded ones) is classified into one of
three buckets, and a `third_party_unverified` one is never counted as a
citation at all -- it is dropped before a respelling is even extracted from
it, the same treatment as a claim that fails the format-plausibility check.
MW/Wikipedia/Wiktionary/CMUdict/DailyMed/the AMA USAN Statement are each a
single known kind of source and are tagged directly; only an arbitrary
Gemini-grounded web citation is classified per-domain, by asking Gemini 2.5
Flash to bucket the domain with a few worked examples per bucket (not a
hardcoded domain list) -- see `classify_source_trust` in gemini_grounded.py.

| Tier | Meaning | Ingredients citing at least one |
| --- | --- | --- |
| official_medical | Government health agency, national regulator, or the drug-naming body itself (FDA, DailyMed, MedlinePlus, USAN/AMA) | 272 |
| verified_secondary | Editorially-maintained reference, not a primary authority but not open to public submission either (Drugs.com, WebMD, Wikipedia, Merriam-Webster, a university hospital's patient site) | 171 |
| third_party_unverified | Crowdsourced/user-generated, no editorial review (howtopronounce.com, a YouTube upload, a blog) -- **excluded**, never counted | 0 |

That last row should always read 0: it is what `_variants_from_claims` in
build.py exists to guarantee, not a live count of something still present in
the data. 7 ingredients (Avlayah, Blujepa, Simtriyo, TNKase, Toujeo, Zaiidra,
tenecteplase) had their *only* source turn out to be `third_party_unverified`
once this classification was added -- 5 (Blujepa, Simtriyo, TNKase, Zaiidra,
tenecteplase) were recovered by a second, more restrictive Gemini query that
explicitly excludes crowdsourced sites (`verified_claims(..., mode='tight')`),
and 2 (Avlayah, Toujeo) had no official/secondary source to find at all even
under that restriction and reverted to `low`.

## Baseline comparison

| Snapshot | high | medium | low |
| --- | --- | --- | --- |
| Original (MW HTML scrape + CMUdict only) | 19 | 80 | 185 |
| + Wikipedia/Wiktionary (`{{IPAc-en}}`/`{{IPA}}`/`{{respell}}`) | 28 | 83 | 173 |
| + MW Medical Dictionary API | 41 | 170 | 73 |
| + Gemini/Google-Search grounding, rule-based fallback removed | 98 | 171 | 15 |
| + DailyMed Medication Guide respellings | 86 | 194 | 4 |
| + NCI Dictionary of Cancer Terms (this build) | 114 | 164 | 6 |

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

The NCI Dictionary of Cancer Terms was the single largest jump in this build: 50 ingredients, and every one of them a second, independent citation for a name that usually only had one source before -- 27 previously-`medium` records moved straight to `high`. It was found by directly checking a source the user pointed at (cancer.gov's own Dictionary of Cancer Terms page), confirming the public page itself is a React SPA with no server-rendered pronunciation text (a dead end as scraped, like the separate NCI Drug Dictionary already documented above), then reverse-engineering its real backing JSON API from the app's own JS bundle the same way the AMA USAN search API was found. That API returns a structured record per term with both a text respelling and a real hosted audio recording -- an official U.S. federal government source (NIH's National Cancer Institute), not a secondhand citation of one. Coverage is deliberately oncology-skewed: it's a *cancer* terms dictionary, so it only lists the DOSE ingredients that come up in oncology or supportive-care contexts, confirmed directly by querying its own search endpoint for the names it doesn't have (valsartan, clopidogrel, etc.) and finding nothing, not just a spelling mismatch.

## Sources tried

| Source | Outcome |
| --- | --- |
| Merriam-Webster Medical API | **Wired in.** Structured, reliable; 87 ingredients. Needs `MW_MEDICAL_KEY` in `.env`; degrades to the HTML path when absent. |
| Merriam-Webster `/dictionary/` (HTML) | **Wired in**, as the fallback for names the medical API misses; 4 ingredient(s) this run. |
| Wikipedia (`{{IPAc-en}}`, `{{IPA\|en\|...}}`, `{{respell}}`) | **Wired in.** 20 ingredients. Most DOSE brand names are too new or minor for an English Wikipedia article at all. A brand/generic conflation bug was found and fixed here by spot-checking the rebuilt data, not by a user report: a naive "first pronunciation template on the page" search wrongly attributed a BRAND's own pronunciation to the GENERIC ingredient's record whenever the generic's lead sentence also named its brand in prose (confirmed on two live articles: empagliflozin's old source was actually Jardiance's IPA, `/dʒɑːrdiəns/`; perfluorohexyloctane's was actually Miebo's, `/maɪboʊ/`) -- both articles' own `pronounce` infobox field for the generic is genuinely empty, so a naive search fell through to the nearest IPA template, which belonged to the brand name bolded right next to it in the lead ("Empagliflozin, sold under the brand name Jardiance (...)"). Fixed in two layers: (1) try the `{{Infobox drug}}` `pronounce` field first, since it is Wikipedia's own explicit, unambiguous label for the article subject's own pronunciation; (2) failing that, restrict the prose search to the span between the article subject's own bolded mention and the next *different* bolded term, rather than the whole page. A third infobox shape was found the same way: for monoclonal antibodies (`mab_type` set), `{{Infobox drug}}` repurposes its `source` field -- normally the antibody's source-organism code ("u"/"o"/"xi"/"zu") -- to carry the pronunciation instead, confirmed directly on secukinumab's own article (`| source = {{IPAc-en|...}}` above an empty `| pronounce =`); checked against the other three mAb-infobox articles in this dataset (concizumab, lecanemab, talquetamab) to confirm `source` genuinely does hold an organism code, not a mis-filed pronunciation, on all of them. |
| Wiktionary (same templates) | **Wired in.** 2 ingredients; thin, and mostly overlaps Wikipedia rather than adding new names. Deliberately exempted from the Wikipedia brand/generic bold-region restriction above: Wiktionary has no `{{Infobox drug}}`-style lead-sentence bolding convention and is one page per exact spelling, so it carries no brand/generic dual-naming ambiguity for that restriction to guard against -- applying it anyway broke Benadryl, whose real `===Pronunciation===` section sits near the top of the page with no bolded "Benadryl" nearby at all; the restriction instead locked onto an unrelated *later* bolded mention inside a "Benadryl challenge" trivia section and searched only after it, missing the real section entirely. |
| CMUdict | **Wired in** (pre-existing). 14 ingredients; a general dictionary, not a drug-name resource. |
| DailyMed (FDA Medication Guides) | **Wired in.** 96 ingredients (brands only). Reachable from this environment (unlike drugs.com), and a real find caught by manual spot-checking after this build shipped: many Medication Guides state the brand's own respelling right in the title line (`AMBELVIST (am bel' vist)`), in the same USAN prime-stress notation the Gemini-grounded path already parses. Not every label includes one, so this doesn't close every remaining gap. |
| NCI Dictionary of Cancer Terms (cancer.gov) | **Wired in** (`sources.nci_pronunciation`). 50 ingredients, brand and generic alike. The public page (`cancer.gov/publications/dictionaries/cancer-terms/def/{name}`) is a React SPA with no server-rendered pronunciation text -- a dead end as scraped -- but its real backing API (`webapis.cancer.gov/glossary/v1/Terms/Cancer.gov/Patient/en/{name}`), reverse-engineered from the app's own JS bundle the same way the AMA USAN search API was found, returns a structured JSON record with both a text respelling (`pronunciation.key`, e.g. `(uh-see-tuh-MIH-nuh-fen)` for acetaminophen — keep that original string, do not G2P it) and a real hosted audio recording (`pronunciation.audio`, kept in the citation alongside the text). An official U.S. federal government source (NIH's National Cancer Institute), not a secondhand citation of one -- and it covers plenty of brand names too, not just generics (Xanax, Lipitor, Nexium, Crestor, Zoloft, Advil, Ambien, Valium among them). Coverage is deliberately oncology-skewed: confirmed directly against its own search endpoint, names with no connection to cancer care or its supportive treatments (valsartan, clopidogrel, atorvastatin, quetiapine) aren't in this dictionary at all, which is a real characteristic of what this source covers, not a bug in how it's queried. |
| Drugs.com (direct fetch) | Blocked. HTTP 403 on every direct request from this environment, medical and general pages alike. **Reached indirectly**: Gemini's `google_search` tool retrieves and cites Drugs.com pages server-side (Google's infrastructure, not this sandbox, does the fetch), so a citation naming drugs.com is still accepted as a real source even though this environment can't independently re-fetch it -- 62 ingredients answered via Gemini-grounded search overall (drugs.com and otherwise). |
| DrugBank | Dead end. HTTP 403. |
| FDA labels via openFDA (structured JSON) | Dead end for pronunciation. Reachable (200), but the structured label JSON does not carry the Medication Guide's free-text title line, which is where a respelling (if present at all) actually lives -- see DailyMed above, which serves the rendered guide text instead of the structured fields. |
| NLM RxNav / RxNorm | Dead end for pronunciation. Reachable, resolves names to RxCUIs reliably, but `allProperties` carries only coding/synonym fields (ATC, SNOMED, DrugBank ID, etc.) -- no phonetic field exists in the schema. Kept as the id-lookup step for the MedlinePlus Connect pipeline below. |
| MedlinePlus drug monographs | **Real pronunciations confirmed** (e.g. Metformin: `pronounced as (met for' min)`), reachable via a working, unauthenticated pipeline: RxNav name-to-RxCUI, then `connect.medlineplus.gov` (RxNorm OID `2.16.840.1.113883.6.88`) to the drug's monograph URL, then a page fetch. **Not wired into this build**: MedlinePlus's own lay respelling (`met for' min`, `a set a mee' noe fen`) is not the same key as Wikipedia's, and no citable published key for it was found -- guessing its letter-to-phoneme mapping would risk silently wrong phonemes recorded as sourced, which is the failure mode this project must not introduce. See the ranked list below. |
| NCI Drug Dictionary (cancer.gov) | Dead end as scraped. The public page is a React SPA (`drug-dictionary-app`) with no server-rendered content; its bundled config points at `webapis-dev.cancer.gov`, which does not resolve (NXDOMAIN) -- the backing API is not public from this environment. |
| AMA USAN Statement PDFs (searchusan.ama-assn.org) | **Wired in as the primary source for generics** (`sources.usan_pronunciation`), correcting an earlier claim in this table that this was only a notation key, not a per-drug lookup. The rendered Angular UI (`/usan/`) is indeed a dead end as scraped, but its real backing search API isn't -- reverse-engineered from the app's own JS bundle (`this.searchUrl = "/" + this.collection + "/search/" + term + "/" + sort + "/" + pageNum`), it's a MarkLogic full-text index (`GET /usan/search/{term}/relevant/1`) that returns each match's real title and document URI directly -- no filename to guess, and no HTTP-200-with-an-error-body ambiguity the document-download endpoint alone has. (An earlier version of this source tried constructing PDF filenames directly instead of searching; that missed several real documents outright, e.g. elranatamab's actual file has a trailing hyphen -- "elranatamab-.pdf" -- that isn't guessable, and searching finds it immediately.) Matched by similarity, not exact string equality, since USAN's own title field can itself contain a typo ("PRADEMEGENE ZAMIKERACEL" for a query of "prademagene zamikeracel") and a search can return an unrelated but textually-similar document (querying a brand name like "Wakix" returns its generic ingredient pitolisant's statement, not one for Wakix itself -- USAN doesn't register brand names at all, and a low similarity score correctly rejects that mismatch rather than attributing pitolisant's pronunciation to Wakix). 108 ingredients this build via direct search, plus a small number more via a Gemini web-search fallback targeted specifically at USAN documents when the direct index search itself returns nothing (`gemini_grounded.verified_claims(..., mode='usan')`) -- e.g. a name spelled differently in the index than in this dataset. Coverage is generic-only and modern-name-skewed: older, pre-digital-archive generics (acetaminophen, diazepam, valsartan) have no USAN Statement in this system at all and fall through to Merriam-Webster or Gemini-grounded search instead, which is a real gap in USAN's archive, not a bug in how this source is queried -- confirmed directly, not just assumed, by re-querying the raw MarkLogic index for every one of these names and finding zero documents anywhere in the whole USAN collection contain that word at all (a stemmed full-text search, so even an unrelated salt form mentioning the parent name in its own Statement would have surfaced). Three more names recovered a real citation this way that a plain similarity match alone would have missed: some plain generics have no bare-stem Statement in the index at all, only a salt/ester-qualified one (`aripiprazole` only exists as "aripiprazole-lauroxil.pdf"/"aripiprazole-cavoxil.pdf", `esomeprazole` only as "esomeprazole-strontium.pdf"/"...-potassium.pdf"/"...-sodium.pdf", `ibuprofen` only as "ibuprofen-sodium.pdf"/"...-lysine.pdf"/"...-trelamine...pdf") -- `_is_salt_form_of` accepts an exact `{stem} {qualifier}` prefix match even when the qualifier drags the overall similarity score under the acceptance threshold, and `_first_word_group` isolates just the stem's own portion of that document's PRONUNCIATION field (which, when it respells the qualifier at all, does so right after a whitespace gap detectably wider than the gaps between the stem's own syllables -- confirmed directly against real documents using two different conventions for that gap). A single salt-form document whose own field turns out to be nothing but the stem already (no detectable gap at all, e.g. "varenicline_tartrate.pdf", the only USAN document that exists anywhere for `varenicline`) is left unrecovered rather than guessed at: nothing here can algorithmically prove that document doesn't also silently omit part of the stem's own respelling, so it does not become this project's citation. |
| WHO INN lists (who.int) | Reachable (200), but the published INN list documents are name/CAS-number registries, not phonetic dictionaries; no pronunciation field found. |
| StatPearls / NCBI Bookshelf | Not pursued past a spot check -- these are clinical review monographs, not lexicographic sources, and did not surface pronunciation content. |

## Blocked or paywalled sources ranked by expected gain

Gemini/Google-Search grounding, DailyMed, and directly fetching the AMA USAN
Statement PDFs closed all but a handful of the old `low` tier, generic and
brand alike. cipepofol and copper histidinate in particular were both wrongly
written off in an earlier version of this report as needing a **purchased**
USP Dictionary subscription -- the actual per-drug USAN Statement (the same
record USP compiles from) is free and individually findable through AMA's
own real search index (see the Sources tried table above), confirmed
directly and now the primary source for both.
What's left (6 ingredients) was checked individually,
not just left to the pipeline's word: Vyglxia (troriluzole) has no FDA
approval at all yet (a Complete Response Letter, not approval, as of this
build) so no official pronunciation can exist; Wakix's full FDA label text
contains no pronunciation anywhere (confirmed by a direct openFDA full-text
search) and its USAN Statement doesn't state one either, so the only web hit
-- an unreliable YouTube auto-caption ("wake cakes") -- is correctly
discarded rather than recorded as data; Avlayah and Toujeo have no
official/verified-secondary source even under a second, more restrictive
query that explicitly excludes crowdsourced sites (`verified_claims(...,
mode='tight')`) -- the only hits for both are third_party_unverified (a
crowdsourced pronunciation site), correctly excluded rather than counted
as a citation (see the Trust tiers section above).

1. **USP Dictionary of USAN and International Drug Names** -- superseded for
   this benchmark: it compiles the same per-drug USAN Statements this build
   fetches directly and for free, so there is no remaining gain from buying
   access to it specifically. Kept as a reference for anyone reproducing this
   layer without hitting AMA's document store directly.
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

6 ingredients have no external source at all -- Gemini's
Google-Search grounding either found nothing or nothing that survived the
LLM format check. There is no rule-based fallback for these: no phonetic
reference exists for them in this layer, full stop.

- Avlayah (brand)
- Retatrutide (brand)
- Toujeo (brand)
- Vyglxia (brand)
- Wakix (brand)
- formoterol fumarate dihydrate (generic)
