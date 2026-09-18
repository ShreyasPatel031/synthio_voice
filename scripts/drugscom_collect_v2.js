// DOSE-R : Drugs.com collector v2 -- multi-pattern
// Run at https://www.drugs.com in DevTools console (F12), signed in.
//
// v1 only tried /{slug}.html and 404'd on 76 names that live elsewhere
// (e.g. osimertinib is at /mtm/osimertinib.html). v2 tries each known path,
// then falls back to site search.
//
// Stop anytime and run  dumpDoseR()  to download what you have.

const DELAY_MS = 1500;
const NAMES = ["Ambelvist","Loargys","Meibo","Nurtec","Obicetrapib","Plozasiran","Vyglxia","Zaiidra","Zipalertinib","Zorevunersen","acoltremon","acoramidis","atacicept-vymj","baloxavir marboxil","baxdrostat","bevacizumab-vikg","bictegravir","bulevirtide-gmod","canakinumab","cefepime","centanafadine","cipepofol","concizumab","copper histidinate","datopotamab deruxtecan","deutivacaftor","difamilast","doravirine","elexacaftor","elranatamab-bcmm","emtricitabine","enlicitide decanoate","ensartinib","ensitrelvir","exagamglogene autotemcel","exenatide","faricimab-svoa","fezolinetant","gadoquatrane","gedatolisib","gepotidacin","icotrokinra","islatravir","lebrikizumab-lbkz","linerixibat","milsaperidone","navepegritide","nipocalimab-aahu","obecabtagene autoleucel","osimertinib","oveporexton","pegvaliase-pqpz","pegzilarginase-nbln","pivekimab sunirine-pvzy","prademagene zamikeracel","relacorilant","remibrutinib","revumenib","rilzabrutinib","risankizumab-rzaa","sotatercept-csrk","tebipenem pivoxil","tenecteplase","teplizumab-mzwv","tezacaftor","tividenofusp alfa-eknm","tovorafenib","troriluzole","vanzacaftor","veligrotug-vvze","vepdegestrant","zenocutuzumab","zidebactam","zidesamtinib","zolbetuximab","zoliflodacin","fluticasone propionate","formoterol fumarate dihydrate","ibuprofen","insulin glargine","insulin icodec-abae","ivacaftor","levothyroxine","lisdexamfetamine","loratadine","lurasidone","nogapendekin alfa inbakicept-pmln","omalizumab","omeprazole","perfluorohexyloctane","pitolisant","pregabalin","quetiapine","rivaroxaban","rosuvastatin","sacubitril","salmeterol","secukinumab","semaglutide","sertraline","sitagliptin","sonrotoclax","sotagliflozin","suzetrigine","tenofovir alafenamide","testosterone undecanoate","tiotropium bromide","tirzepatide","tofacitinib","trospium chloride","valsartan","varenicline","xanomeline","zolpidem"];

const PATHS = ['', 'mtm/', 'pro/', 'cdi/', 'monograph/', 'npp/', 'international/'];
const slug = n => n.toLowerCase().trim().replace(/[^a-z0-9 -]/g,'').replace(/\s+/g,'-');
const sleep = ms => new Promise(r => setTimeout(r, ms));
const b64 = buf => { let s=''; const b=new Uint8Array(buf);
  for (let i=0;i<b.length;i++) s+=String.fromCharCode(b[i]); return btoa(s); };

window.DOSER = [];
window.dumpDoseR = () => {
  const blob = new Blob([JSON.stringify(window.DOSER)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `drugscom_v2_${window.DOSER.length}.json`;
  a.click();
  console.log(`downloaded ${window.DOSER.length}`);
};

async function tryUrl(u) {
  try {
    const r = await fetch(u, {credentials:'include'});
    if (!r.ok) return null;
    const t = await r.text();
    const m = t.match(/\/audio\/wav\/[\w.-]+\.wav/);
    return m ? {url:u, audio:'https://www.drugs.com'+m[0]} : null;
  } catch (e) { return null; }
}

async function findViaSearch(name) {
  try {
    const r = await fetch(`https://www.drugs.com/search.php?searchterm=${encodeURIComponent(name)}`,
                          {credentials:'include'});
    if (!r.ok) return null;
    const t = await r.text();
    const m = t.match(/href="(\/(?:mtm|pro|cdi|monograph)?\/?[a-z0-9-]+\.html)"/);
    return m ? await tryUrl('https://www.drugs.com' + m[1]) : null;
  } catch (e) { return null; }
}

(async () => {
  console.log(`v2: ${NAMES.length} names, trying ${PATHS.length} paths each + search`);
  for (let i = 0; i < NAMES.length; i++) {
    const name = NAMES[i];
    const rec = {name, found_url:null, audio_url:null, audio_b64:null, via:null};
    let hit = null;
    for (const p of PATHS) {
      hit = await tryUrl(`https://www.drugs.com/${p}${slug(name)}.html`);
      if (hit) { rec.via = p || 'root'; break; }
      await sleep(250);
    }
    if (!hit) { hit = await findViaSearch(name); if (hit) rec.via = 'search'; }
    if (hit) {
      rec.found_url = hit.url; rec.audio_url = hit.audio;
      try {
        const ar = await fetch(hit.audio, {credentials:'include'});
        if (ar.ok) rec.audio_b64 = b64(await ar.arrayBuffer());
      } catch (e) {}
    }
    window.DOSER.push(rec);
    console.log(`${i+1}/${NAMES.length}`, name, rec.audio_b64 ? `AUDIO via ${rec.via}` : (rec.via||'MISS'));
    await sleep(DELAY_MS);
  }
  console.log(`\n=== ${window.DOSER.filter(r=>r.audio_b64).length}/${window.DOSER.length} with audio ===`);
  window.dumpDoseR();
})();
