// DOSE-R : Drugs.com TEXT PRONUNCIATION collector (respelling, not audio)
//
// The prior collectors' respelling extraction was broken: it required 3+
// characters of plain text between "Pronunciation:</b>" and the next tag,
// but the real markup nests the respelling inside an immediately-following
// <i> tag with nothing in between:
//
//   <b>Pronunciation:</b> <i>Ah-wik-lee</i><br>
//
// That regex silently failed and fell through to a dangerous fallback that
// grabbed ANY parenthesized text on the page -- which is how Awiqli's real
// respelling was replaced with "insulin icodec" (a generic-name mention
// from a different line) in two prior collection runs, discarded, and
// never flagged as wrong. This version matches the real structure only,
// with no unsafe fallback: a miss returns null, never a guess.
//
// Run at https://www.drugs.com signed in, DevTools console (F12). Targets
// the 101 audio-gap names first (same list `drugscom_collect_v2.js` used),
// then the rest, since this source may not depend on audio existing.

const DELAY_MS = 1200;
const NAMES = ["Ambelvist","Avlayah","Awiqli","Baxfendy","Beqalzi","Bysanti","Casgevy","Cypsedo","Decnupaz","Foundayo","Hepcludex","Icotyde","Idvynso","Ilaris","Inpefa","Jideytro","Lifyorli","Lipfendra","Loargys","Lumvoa","Lynavoy","Lytenava","Obicetrapib","Orzeyful","Plozasiran","Retatrutide","Revtorpyk","Simtriyo","TNKase","Trikafta","Trutakna","Tzield","Utebzi","Veppanu","Vyglxia","Wakix","Xocova","Yuviwel","Zaynich","Zipalertinib","Zorevunersen","atacicept-vymj","baxdrostat","bictegravir","bulevirtide-gmod","canakinumab","centanafadine","cipepofol","concizumab","deutivacaftor","difamilast","doravirine","durvalumab","elexacaftor","elranatamab-bcmm","ensartinib","ensitrelvir","exagamglogene autotemcel","faricimab-svoa","gadoquatrane","gedatolisib","icotrokinra","islatravir","ivacaftor","lecanemab","linerixibat","milsaperidone","navepegritide","olezarsen","orforglipron","osimertinib","oveporexton","pegvaliase-pqpz","pegzilarginase-nbln","perfluorohexyloctane","pitolisant","pivekimab sunirine-pvzy","relacorilant","resmetirom","ribociclib","risankizumab-rzaa","sacubitril","secukinumab","sonrotoclax","sotagliflozin","tebipenem pivoxil","tenecteplase","teplizumab-mzwv","tezacaftor","tividenofusp alfa-eknm","troriluzole","ubrogepant","valbenazine","vanzacaftor","veligrotug-vvze","vepdegestrant","xanomeline","zenocutuzumab","zidebactam","zidesamtinib","zoliflodacin","Abilify","Adquey","Advair","Advil","Alhemo","Alyftrek","Ambien","Anktiva","Aspirin","Attruby","Aucatzyl","Benadryl","Biktarvy","Bizengri","Blujepa","Byetta","Chantix","Claritin","Cobenfy","Cosentyx","Crestor","Cymbalta","Datroway","Dupixent","Ebglyss","Eliquis","Elrexfio","Enbrel","Ensacove","Entresto","Farxiga","Flonase","Humira","Imaavy","Imfinzi","Ingrezza","Januvia","Jardiance","Journavx","Kisqali","Kyzatrex","Latuda","Leqembi","Lipitor","Lyrica","Meibo","Metformin","Motrin","Mounjaro","Nexium","Nurtec","Nuzolvence","Ojemda","Otezla","Ozempic","Palynziq","Plavix","Prilosec","Prozac","Qulipta","Revuforj","Rezdiffra","Rhapsido","Rinvoq","Rybelsus","Seroquel","Skyrizi","Spiriva","Symbicort","Synthroid","Tagrisso","Talvey","Tecfidera","Toujeo","Trulicity","Tryngolza","Tryptyr","Tylenol","Ubrelvy","Vabysmo","Valium","Veozah","Voranigo","Vraylar","Vyloy","Vyvanse","Vyvgart","Wayrilz","Wegovy","Winrevair","Xanax","Xarelto","Xeljanz","Xofluza","Xolair","Zaiidra","Zantac","Zepbound","Zevaskyn","Zoloft","Zycubo","Zyrtec","acetaminophen","acoltremon","acoramidis","adalimumab","alprazolam","apixaban","apremilast","aripiprazole","atogepant","atorvastatin","baloxavir marboxil","bevacizumab-vikg","budesonide","cariprazine","cefepime","cetirizine","clopidogrel","copper histidinate","dapagliflozin","datopotamab deruxtecan","diazepam","dimethyl fumarate","diphenhydramine","dulaglutide","duloxetine","dupilumab","efgartigimod alfa","empagliflozin","emtricitabine","enlicitide decanoate","esomeprazole","etanercept","exenatide","famotidine","fezolinetant","fluoxetine","fluticasone propionate","formoterol fumarate dihydrate","gepotidacin","ibuprofen","insulin glargine","insulin icodec-abae","lebrikizumab-lbkz","levothyroxine","lisdexamfetamine","loratadine","lurasidone","nipocalimab-aahu","nogapendekin alfa inbakicept-pmln","obecabtagene autoleucel","omalizumab","omeprazole","prademagene zamikeracel","pregabalin","quetiapine","remibrutinib","revumenib","rilzabrutinib","rimegepant","rivaroxaban","rosuvastatin","salmeterol","semaglutide","sertraline","sitagliptin","sotatercept-csrk","suzetrigine","talquetamab","tenofovir alafenamide","testosterone undecanoate","tiotropium bromide","tirzepatide","tofacitinib","tovorafenib","trospium chloride","upadacitinib","valsartan","varenicline","vorasidenib","zolbetuximab","zolpidem"];

const slug = n => n.toLowerCase().trim().replace(/[^a-z0-9 -]/g,'').replace(/\s+/g,'-');
const sleep = ms => new Promise(r => setTimeout(r, ms));
const PATHS = ['', 'mtm/', 'pro/', 'cdi/', 'monograph/', 'npp/', 'international/'];

// <b>Pronunciation:</b> <i>...</i>  -- the confirmed real markup.
const RESPELL = /<b>\s*Pronunciation:?\s*<\/b>\s*<i>\s*([^<]{2,80}?)\s*<\/i>/i;

window.DOSER_RESPELL = [];
window.dumpRespell = () => {
  const blob = new Blob([JSON.stringify(window.DOSER_RESPELL, null, 1)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `drugscom_respell_${window.DOSER_RESPELL.length}.json`;
  a.click();
};

async function findRespelling(name) {
  for (const p of PATHS) {
    try {
      const r = await fetch(`https://www.drugs.com/${p}${slug(name)}.html`, {credentials:'include'});
      if (!r.ok) continue;
      const t = await r.text();
      const m = t.match(RESPELL);
      if (m) return { path: p || 'root', respelling: m[1].trim(), page_url: r.url };
    } catch (e) {}
    await sleep(300);
  }
  return null;
}

(async () => {
  console.log(`checking ${NAMES.length} names for a real text respelling...`);
  for (let i = 0; i < NAMES.length; i++) {
    const name = NAMES[i];
    const hit = await findRespelling(name);
    const rec = { name, respelling: hit ? hit.respelling : null,
                  path: hit ? hit.path : null, page_url: hit ? hit.page_url : null };
    window.DOSER_RESPELL.push(rec);
    console.log(`${i+1}/${NAMES.length}`, name, rec.respelling ? `"${rec.respelling}"` : 'none found');
    await sleep(DELAY_MS);
  }
  const hits = window.DOSER_RESPELL.filter(r => r.respelling).length;
  console.log(`\n=== ${hits}/${window.DOSER_RESPELL.length} have a real text respelling ===`);
  window.dumpRespell();
})();
