# Agent rules

## Pronunciation: original source only

**Never convert dictionary respelling to IPA.** That converter is banned
(`RespellToIpaBanned`). It turned Dupixent `DU-pix-ent` into `ˈdʌpɪksɛnt`
("duh") and Attruby `ah-troo-be` into `ˈɑːtruːbɛ`. Real IPA from Google beat
it on every name we tested.

- Use the string the source published.
- IPA is allowed only when the source actually published IPA. Ask for IPA;
  do not G2P `DU-pix-ent` through Wikipedia's respelling key.
- Do not revive `ipa_from_canonical` or `respell_to_arpabet_ipa`.
- Do not fill `pronunciations.jsonl` `ipa` / `ipa_cloud` from conversion.

Full write-up: `dose_r/references/README.md`.
Rule: `.cursor/rules/no-respelling-to-ipa.mdc`.

## Gold IPA is locked

`data/gold_gemini_ipa/` is the only IPA + teacher wav set. Never regenerate it. Never rescore by calling TTS. Change a name only after the user explicitly says yes, and append the reason to `changes.jsonl` first. Files are read-only. Rule: `.cursor/rules/gold-gemini-ipa-lock.mdc`.

## Listen results are HTML

Any pronunciation, clip, IPA, or comparison is an HTML page under `runs/listen-<name>/index.html` with `<audio>` controls. Reuse existing audio. Do not resynthesize gold to fill the page. Rule: `.cursor/rules/listen-results-html.mdc`.
