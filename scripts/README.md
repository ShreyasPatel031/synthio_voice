"""IPA conversion scripts in this folder are dead.

Do not convert DailyMed/USAN/NCI respelling to IPA.
Do not revive eval_uniform_pron_tts IPA arm via ipa_from_canonical.
`eval_google_ipa_five.py` exists only as proof that source IPA beats G2P.

`eval_ipa_vs_respell_voice.py` is the two-voice check: Google IPA per name
part vs Cloud reading the original source respelling as English syllables.
It does not G2P `a-TA-ki-sept` / `zoe-li-floe-DAY-sin`.

`eval_gemini31_respell_all.py` feeds that same original respelling
(`zye-de-SAM-ti-nib`) to Gemini 3.1 Flash TTS and scores it against
the human clip and Cloud Standard-C + source IPA.

`eval_gemini31_ipa_vs_cloud.py` is Gemini 3.1 Flash TTS + the same
source IPA Cloud already accepted, vs Cloud Standard-C + IPA.

See `dose_r/references/README.md` and `.cursor/rules/no-respelling-to-ipa.mdc`.
"""
