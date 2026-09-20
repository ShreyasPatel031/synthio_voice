// DOSE-R : Forvo audio collector — exploratory pass on 5 confirmed hits
// Run at https://forvo.com in DevTools console (F12).
//
// Forvo's play buttons are usually JS-driven (onclick="Play(...)"), not a
// plain <audio src>, and the real mp3 lives behind an obfuscated/hashed path.
// This grabs every signal it can find per page — <audio> tags, .mp3/.ogg
// text references, and any onclick attribute mentioning Play — and tries to
// fetch whatever looks like a real audio URL. If nothing fetches cleanly,
// the raw onclick strings are included so a second pass can be built from
// real markup instead of guessing.

const NAMES = ["canakinumab", "osimertinib", "tenecteplase", "ivacaftor", "sacubitril"];
const DELAY_MS = 2000;

const sleep = ms => new Promise(r => setTimeout(r, ms));
const b64 = buf => { let s=''; const b=new Uint8Array(buf);
  for (let i=0;i<b.length;i++) s+=String.fromCharCode(b[i]); return btoa(s); };

window.FORVO_AUDIO = [];
window.dumpForvoAudio = () => {
  const blob = new Blob([JSON.stringify(window.FORVO_AUDIO, null, 1)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `forvo_audio_${window.FORVO_AUDIO.length}.json`;
  a.click();
};

async function tryFetchAudio(url) {
  try {
    const r = await fetch(url, {credentials:'include'});
    if (!r.ok) return null;
    const buf = await r.arrayBuffer();
    if (buf.byteLength < 500) return null; // too small to be real audio
    return b64(buf);
  } catch (e) { return null; }
}

(async () => {
  for (let i = 0; i < NAMES.length; i++) {
    const name = NAMES[i];
    const rec = { name, page_status: null, audio_b64: null, found_urls: [], onclick_snippets: [] };
    try {
      const r = await fetch(`https://forvo.com/word/${name}/`, {credentials:'include'});
      rec.page_status = r.status;
      if (r.ok) {
        const t = await r.text();

        // Direct <audio src="..."> or explicit .mp3/.ogg links
        const direct = [...t.matchAll(/(?:src|href)=["']([^"']+\.(?:mp3|ogg))["']/gi)].map(m => m[1]);
        // Forvo's classic play-button pattern: onclick="Play(id,'checksum','path',...)"
        const onclicks = [...t.matchAll(/onclick=["']Play\(([^)]+)\)["']/gi)].map(m => m[1]);

        rec.found_urls = [...new Set(direct)];
        rec.onclick_snippets = onclicks.slice(0, 5);

        for (const u of rec.found_urls) {
          const full = u.startsWith('http') ? u : `https://forvo.com${u.startsWith('/') ? '' : '/'}${u}`;
          const audio = await tryFetchAudio(full);
          if (audio) { rec.audio_b64 = audio; rec.audio_url = full; break; }
        }
      }
    } catch (e) { rec.page_status = 'ERR ' + e.message; }

    window.FORVO_AUDIO.push(rec);
    console.log(
      `${i+1}/${NAMES.length}`, name, rec.page_status,
      rec.audio_b64 ? `AUDIO ${Math.round(rec.audio_b64.length/1366)}kb` : 'no audio fetched',
      `| urls found: ${rec.found_urls.length}`, `| onclick samples: ${rec.onclick_snippets.length}`
    );
    await sleep(DELAY_MS);
  }
  console.log('\ndone. If audio_b64 is null for everyone, check the onclick_snippets field');
  console.log('in the dump — that is the real markup and tells us the actual mechanism.');
  window.dumpForvoAudio();
})();
