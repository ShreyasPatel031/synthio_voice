// DOSE-R : Forvo coverage check
// Run at https://forvo.com in DevTools console (F12). Read-only, checks
// which of our 101 remaining uncovered names Forvo actually has.
//
// A real hit is HTTP 200 AND the audio-player markup is present -- Forvo's
// 404 page still contains the word "pronunciation" in its own boilerplate,
// so a text search alone (like the last quick check) gives false positives.

const DELAY_MS = 1200;
const NAMES = ["Ambelvist","Avlayah","Awiqli","Baxfendy","Beqalzi","Bysanti","Casgevy","Cypsedo","Decnupaz","Foundayo","Hepcludex","Icotyde","Idvynso","Ilaris","Inpefa","Jideytro","Lifyorli","Lipfendra","Loargys","Lumvoa","Lynavoy","Lytenava","Obicetrapib","Orzeyful","Plozasiran","Retatrutide","Revtorpyk","Simtriyo","TNKase","Trikafta","Trutakna","Tzield","Utebzi","Veppanu","Vyglxia","Wakix","Xocova","Yuviwel","Zaynich","Zipalertinib","Zorevunersen","atacicept-vymj","baxdrostat","bictegravir","bulevirtide-gmod","canakinumab","centanafadine","cipepofol","concizumab","deutivacaftor","difamilast","doravirine","durvalumab","elexacaftor","elranatamab-bcmm","ensartinib","ensitrelvir","exagamglogene autotemcel","faricimab-svoa","gadoquatrane","gedatolisib","icotrokinra","islatravir","ivacaftor","lecanemab","linerixibat","milsaperidone","navepegritide","olezarsen","orforglipron","osimertinib","oveporexton","pegvaliase-pqpz","pegzilarginase-nbln","perfluorohexyloctane","pitolisant","pivekimab sunirine-pvzy","relacorilant","resmetirom","ribociclib","risankizumab-rzaa","sacubitril","secukinumab","sonrotoclax","sotagliflozin","tebipenem pivoxil","tenecteplase","teplizumab-mzwv","tezacaftor","tividenofusp alfa-eknm","troriluzole","ubrogepant","valbenazine","vanzacaftor","veligrotug-vvze","vepdegestrant","xanomeline","zenocutuzumab","zidebactam","zidesamtinib","zoliflodacin"];

const slug = n => n.toLowerCase().trim().replace(/[^a-z0-9]+/g,'-').replace(/(^-|-$)/g,'');
const sleep = ms => new Promise(r => setTimeout(r, ms));

window.FORVO = [];
window.dumpForvo = () => {
  const blob = new Blob([JSON.stringify(window.FORVO, null, 1)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `forvo_coverage_${window.FORVO.length}.json`;
  a.click();
};

(async () => {
  console.log(`checking ${NAMES.length} names against Forvo...`);
  for (let i = 0; i < NAMES.length; i++) {
    const name = NAMES[i];
    const rec = { name, status: null, has_player: false, play_count: 0 };
    try {
      const r = await fetch(`https://forvo.com/word/${slug(name)}/`);
      rec.status = r.status;
      if (r.ok) {
        const t = await r.text();
        // Forvo's real pronunciation entries render a play control per
        // recording with this class; the 404 page has none.
        rec.play_count = (t.match(/class="[^"]*play[^"]*"/gi) || []).length;
        rec.has_player = rec.play_count > 0;
      }
    } catch (e) { rec.status = 'ERR ' + e.message; }
    window.FORVO.push(rec);
    if (rec.has_player) console.log(`${i+1}/${NAMES.length}`, name, 'HAS AUDIO, plays=' + rec.play_count);
    else if (i % 20 === 0) console.log(`${i+1}/${NAMES.length}`, name, rec.status, 'no audio');
    await sleep(DELAY_MS);
  }
  const hits = window.FORVO.filter(r=>r.has_player).length;
  console.log(`\n=== ${hits}/${window.FORVO.length} of our GAP names have real Forvo audio ===`);
  window.dumpForvo();
})();
