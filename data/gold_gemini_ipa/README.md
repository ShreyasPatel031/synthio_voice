# Gold Gemini IPA

This is the only pronunciation dataset. It is the gold for the official DoSE benchmark (`benchmark/README.md`).

284 names. Each row is the user-validated IPA plus the Gemini 3.1 Flash TTS (Kore) teacher wav of that IPA. Not Cloud Standard-C. Not a respelling converted to IPA.

- `manifest.jsonl` — `ingredient`, `spoken_text`, `ipa`, `audio`
- `ipa.json` — same IPA strings
- `wavs/{slug}.wav` — teacher audio
- `LOCK` — write lock. `ipa.json`, `manifest.jsonl`, `summary.json`, and `wavs/` are read-only.

## Write lock — do not regenerate

**Never regenerate this dataset.** Do not resynthesize the wavs. Do not rescore F1 and then replace audio or IPA because a score looks low. Do not rebuild listen pages by calling TTS. A missing listen page is not a reason to synth.

`ipa.json`, `manifest.jsonl`, `summary.json`, and every file under `wavs/` are mode `0444`. Do not `chmod` them.

A change is allowed only when all of these are true:

1. The user explicitly asked to change **that name** (or that file) in this conversation. "Give me the list again" is not permission to synth or rescore.
2. You stop and ask before writing, and the user says yes.
3. You append one line to `changes.jsonl` **before** the edit, with `ingredient`, `why` (the reason, in your words), and `user_ok`.

If the user did not name the change, do nothing.

`dose_r/references/pronunciations.jsonl` `ipa` / `ipa_cloud` stay empty. Do not read old `runs/listen-*` pages or `data/finetune_cloud_ipa` (removed; that pack was a stale Cloud snapshot).
