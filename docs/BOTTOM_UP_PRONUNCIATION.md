# Bottom-up pronunciation improvement

Order for each low name (lowest Standard-IPA F1 first):

1. **CTC window** — compare iso F1 vs sentence-crop F1; sweep gap recovery.
   If end recovery helps without regressing highs, adjust `dose_r/forced_align.py`.
2. **Misaki** — trial lexicon pins with `scripts/misaki_trial_one.py`; keep only if F1 rises.
3. **IPA** — cross-check Standard IPA against Merck/FDA/MedlinePlus/Drugs.com.
   Do not silently rewrite gold; flag mismatches and trial alternate teachers if needed.

Tools: `scripts/ctc_diag_bottom.py`, `scripts/ctc_asymmetric_test.py`,
`scripts/rescore_ctc_window.py`, `scripts/misaki_trial_one.py`.

## 2026-09-25 pass (bottom 3)

| name | before | after | lever |
| --- | --- | --- | --- |
| Idvynso | 0.547 | **0.687** | Misaki `ɪdvˈɪnsoʊ` (Merck ihd-VIHN-soh); gold IPA OK |
| Advair | 0.578 | **0.604** | CTC end-share 0.75 (crop was 0.33s) |
| vorasidenib | 0.580 | **0.592** | CTC end75; Misaki next |

Global CTC change: start share 0.5, end share 0.75, cap 0.15s.
