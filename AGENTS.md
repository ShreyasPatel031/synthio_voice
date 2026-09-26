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
