# Gold Reference Layer — Coverage

284 unique ingredients across the 274 DOSE rows.

## Confidence

| Tier | Count | Share | Meaning |
| --- | --- | --- | --- |
| high | 28 | 9.9% | two independent sources agree |
| medium | 83 | 29.2% | exactly one external source answered |
| low | 173 | 60.9% | no external source; derived from spelling by rule |

## By name type

| Tier | brand | generic |
| --- | --- | --- |
| high | 13 | 15 |
| medium | 33 | 50 |
| low | 97 | 76 |

## Sources that answered

| Source | Ingredients |
| --- | --- |
| merriam-webster/medical-api | 100 |
| wikipedia | 22 |
| cmudict | 18 |
| wiktionary | 2 |
| merriam-webster/dictionary | 1 |

## Baseline comparison

| Snapshot | high | medium | low |
| --- | --- | --- | --- |
| Original (MW HTML scrape + CMUdict only) | 19 | 80 | 185 |
| + Wikipedia/Wiktionary (`{{IPAc-en}}`/`{{IPA}}`/`{{respell}}`) | 28 | 83 | 173 |
| + MW Medical Dictionary API (this build) | 28 | 83 | 173 |

The Wikipedia/Wiktionary step is the real gain here: it answered 24
ingredients no other source had, and independently corroborated several
Merriam-Webster entries into `high` confidence (e.g. Metformin, previously
`medium` on Merriam-Webster alone). The Medical API step is a reliability
swap, not a coverage one: it replaced HTML scraping of `/medical/` (bot-block
risk, brittle markup) with a structured JSON call, at parity on this dataset
(a name or two moves between the API and the `/dictionary/` HTML fallback,
but the combined total is effectively unchanged) -- exactly as predicted
before wiring it in, since the API is medical-only and this benchmark's
brand names lean on the general dictionary.

## Sources tried

| Source | Outcome |
| --- | --- |
| Merriam-Webster Medical API | **Wired in.** Structured, reliable; 100 ingredients. Needs `MW_MEDICAL_KEY` in `.env`; degrades to the HTML path when absent. |
| Merriam-Webster `/dictionary/` (HTML) | **Wired in**, as the fallback for names the medical API misses; 1 ingredient(s) this run. |
| Wikipedia (`{{IPAc-en}}`, `{{IPA\|en\|...}}`, `{{respell}}`) | **Wired in.** 22 ingredients. Most DOSE brand names are too new or minor for an English Wikipedia article at all. |
| Wiktionary (same templates) | **Wired in.** 2 ingredients; thin, and mostly overlaps Wikipedia rather than adding new names. |
| CMUdict | **Wired in** (pre-existing). 18 ingredients; a general dictionary, not a drug-name resource. |
| Drugs.com | Dead end. HTTP 403 on every request from this environment (bot-blocked), medical and general pages alike. |
| DrugBank | Dead end. HTTP 403. |
| FDA labels (openFDA, DailyMed) | Dead end. Reachable (200), but label text carries no pronunciation respellings -- nothing to extract. |
| NLM RxNav / RxNorm | Dead end for pronunciation. Reachable, resolves names to RxCUIs reliably, but `allProperties` carries only coding/synonym fields (ATC, SNOMED, DrugBank ID, etc.) -- no phonetic field exists in the schema. Kept as the id-lookup step for the MedlinePlus Connect pipeline below. |
| MedlinePlus drug monographs | **Real pronunciations confirmed** (e.g. Metformin: `pronounced as (met for' min)`), reachable via a working, unauthenticated pipeline: RxNav name-to-RxCUI, then `connect.medlineplus.gov` (RxNorm OID `2.16.840.1.113883.6.88`) to the drug's monograph URL, then a page fetch. **Not wired into this build**: MedlinePlus's own lay respelling (`met for' min`, `a set a mee' noe fen`) is not the same key as Wikipedia's, and no citable published key for it was found -- guessing its letter-to-phoneme mapping would risk silently wrong phonemes recorded as sourced, which is the failure mode this project must not introduce. See the ranked list below. |
| NCI Drug Dictionary (cancer.gov) | Dead end as scraped. The public page is a React SPA (`drug-dictionary-app`) with no server-rendered content; its bundled config points at `webapis-dev.cancer.gov`, which does not resolve (NXDOMAIN) -- the backing API is not public from this environment. |
| AMA USAN pronunciation guide (key) | Found and reachable, not paywalled (`"gating_state":"not gated"`). It is the **notation key** the USAN Council uses (prime/double-prime stress marks, documented digraphs), not a per-drug lookup -- it explains how to read a pronunciation, it doesn't supply one. |
| AMA USAN Drug Finder (searchusan.ama-assn.org) | Dead end as scraped: an Angular SPA; the string `pronun` does not appear anywhere in its main JS bundle, so the finder itself does not appear to expose pronunciation, only naming/adoption-status data. |
| WHO INN lists (who.int) | Reachable (200), but the published INN list documents are name/CAS-number registries, not phonetic dictionaries; no pronunciation field found. |
| StatPearls / NCBI Bookshelf | Not pursued past a spot check -- these are clinical review monographs, not lexicographic sources, and did not surface pronunciation content. |

## Blocked or paywalled sources ranked by expected gain

1. **USP Dictionary of USAN and International Drug Names** -- the compiled,
   official pronunciation reference for essentially every USAN/INN generic
   name, using the documented AMA/USAN key (prime-mark stress, plain-English
   digraphs) already confirmed public. This is the single best lead: most of
   this benchmark's `low` tier is coined INN generics (suzetrigine,
   ensartinib, deutivacaftor, ...) that are exactly what this dictionary
   covers and Merriam-Webster does not. **What's needed:** USP sells it as a
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

173 ingredients have no external source and are currently
rule-derived. These are the layer's weak spot and must not be read as gold:

- Adquey (brand)
- Alhemo (brand)
- Alyftrek (brand)
- Ambelvist (brand)
- Anktiva (brand)
- Attruby (brand)
- Aucatzyl (brand)
- Avlayah (brand)
- Awiqli (brand)
- Baxfendy (brand)
- Beqalzi (brand)
- Biktarvy (brand)
- Bizengri (brand)
- Blujepa (brand)
- Byetta (brand)
- Bysanti (brand)
- Casgevy (brand)
- Cobenfy (brand)
- Cosentyx (brand)
- Cypsedo (brand)
- Datroway (brand)
- Decnupaz (brand)
- Dupixent (brand)
- Ebglyss (brand)
- Elrexfio (brand)
- Ensacove (brand)
- Entresto (brand)
- Foundayo (brand)
- Hepcludex (brand)
- Icotyde (brand)
- Idvynso (brand)
- Ilaris (brand)
- Imaavy (brand)
- Imfinzi (brand)
- Ingrezza (brand)
- Inpefa (brand)
- Jardiance (brand)
- Jideytro (brand)
- Journavx (brand)
- Kisqali (brand)
- Kyzatrex (brand)
- Leqembi (brand)
- Lifyorli (brand)
- Lipfendra (brand)
- Loargys (brand)
- Lumvoa (brand)
- Lynavoy (brand)
- Lytenava (brand)
- Meibo (brand)
- Nurtec (brand)
- Nuzolvence (brand)
- Obicetrapib (brand)
- Ojemda (brand)
- Orzeyful (brand)
- Otezla (brand)
- Palynziq (brand)
- Plozasiran (brand)
- Qulipta (brand)
- Retatrutide (brand)
- Revtorpyk (brand)
- Revuforj (brand)
- Rezdiffra (brand)
- Rhapsido (brand)
- Rinvoq (brand)
- Simtriyo (brand)
- Skyrizi (brand)
- TNKase (brand)
- Tagrisso (brand)
- Talvey (brand)
- Toujeo (brand)
- Trikafta (brand)
- Trutakna (brand)
- Tryngolza (brand)
- Tryptyr (brand)
- Tzield (brand)
- Ubrelvy (brand)
- Utebzi (brand)
- Vabysmo (brand)
- Veozah (brand)
- Veppanu (brand)
- Voranigo (brand)
- Vraylar (brand)
- Vyglxia (brand)
- Vyloy (brand)
- Vyvgart (brand)
- Wakix (brand)
- Wayrilz (brand)
- Winrevair (brand)
- Xocova (brand)
- Xofluza (brand)
- Yuviwel (brand)
- Zaiidra (brand)
- Zaynich (brand)
- Zevaskyn (brand)
- Zipalertinib (brand)
- Zorevunersen (brand)
- Zycubo (brand)
- acoltremon (generic)
- acoramidis (generic)
- apremilast (generic)
- atacicept-vymj (generic)
- atogepant (generic)
- baxdrostat (generic)
- bevacizumab-vikg (generic)
- bictegravir (generic)
- bulevirtide-gmod (generic)
- canakinumab (generic)
- cariprazine (generic)
- centanafadine (generic)
- cipepofol (generic)
- concizumab (generic)
- datopotamab deruxtecan (generic)
- deutivacaftor (generic)
- difamilast (generic)
- doravirine (generic)
- dupilumab (generic)
- durvalumab (generic)
- elexacaftor (generic)
- elranatamab-bcmm (generic)
- ensartinib (generic)
- ensitrelvir (generic)
- exagamglogene autotemcel (generic)
- faricimab-svoa (generic)
- fezolinetant (generic)
- gadoquatrane (generic)
- gedatolisib (generic)
- gepotidacin (generic)
- icotrokinra (generic)
- islatravir (generic)
- lebrikizumab-lbkz (generic)
- lecanemab (generic)
- linerixibat (generic)
- milsaperidone (generic)
- navepegritide (generic)
- nipocalimab-aahu (generic)
- obecabtagene autoleucel (generic)
- olezarsen (generic)
- orforglipron (generic)
- osimertinib (generic)
- oveporexton (generic)
- pegvaliase-pqpz (generic)
- pegzilarginase-nbln (generic)
- pivekimab sunirine-pvzy (generic)
- prademagene zamikeracel (generic)
- relacorilant (generic)
- remibrutinib (generic)
- resmetirom (generic)
- revumenib (generic)
- ribociclib (generic)
- rilzabrutinib (generic)
- rimegepant (generic)
- risankizumab-rzaa (generic)
- sotatercept-csrk (generic)
- talquetamab (generic)
- tebipenem pivoxil (generic)
- tenecteplase (generic)
- teplizumab-mzwv (generic)
- tezacaftor (generic)
- tividenofusp alfa-eknm (generic)
- tovorafenib (generic)
- troriluzole (generic)
- ubrogepant (generic)
- upadacitinib (generic)
- valbenazine (generic)
- vanzacaftor (generic)
- veligrotug-vvze (generic)
- vepdegestrant (generic)
- vorasidenib (generic)
- zenocutuzumab (generic)
- zidebactam (generic)
- zidesamtinib (generic)
- zolbetuximab (generic)
- zoliflodacin (generic)
