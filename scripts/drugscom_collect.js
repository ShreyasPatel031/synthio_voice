// DOSE-R : Drugs.com pronunciation collector
// Run at https://www.drugs.com with DevTools console (F12), signed in.
//
// Ordered most-valuable-first: the 173 names with NO pronunciation source yet
// come before the 111 we already have, so stopping early still helps.
//
// Stop anytime and run   dumpDoseR()   to download what you have so far.

const DELAY_MS = 1500;      // please don't lower this
const GET_AUDIO = true;     // audio is the real prize; false = URLs only

const NAMES = ["Adquey","Alhemo","Alyftrek","Ambelvist","Anktiva","Attruby","Aucatzyl","Avlayah","Awiqli","Baxfendy","Beqalzi","Biktarvy","Bizengri","Blujepa","Byetta","Bysanti","Casgevy","Cobenfy","Cosentyx","Cypsedo","Datroway","Decnupaz","Dupixent","Ebglyss","Elrexfio","Ensacove","Entresto","Foundayo","Hepcludex","Icotyde","Idvynso","Ilaris","Imaavy","Imfinzi","Ingrezza","Inpefa","Jardiance","Jideytro","Journavx","Kisqali","Kyzatrex","Leqembi","Lifyorli","Lipfendra","Loargys","Lumvoa","Lynavoy","Lytenava","Meibo","Nurtec","Nuzolvence","Obicetrapib","Ojemda","Orzeyful","Otezla","Palynziq","Plozasiran","Qulipta","Retatrutide","Revtorpyk","Revuforj","Rezdiffra","Rhapsido","Rinvoq","Simtriyo","Skyrizi","TNKase","Tagrisso","Talvey","Toujeo","Trikafta","Trutakna","Tryngolza","Tryptyr","Tzield","Ubrelvy","Utebzi","Vabysmo","Veozah","Veppanu","Voranigo","Vraylar","Vyglxia","Vyloy","Vyvgart","Wakix","Wayrilz","Winrevair","Xocova","Xofluza","Yuviwel","Zaiidra","Zaynich","Zevaskyn","Zipalertinib","Zorevunersen","Zycubo","acoltremon","acoramidis","apremilast","atacicept-vymj","atogepant","baxdrostat","bevacizumab-vikg","bictegravir","bulevirtide-gmod","canakinumab","cariprazine","centanafadine","cipepofol","concizumab","datopotamab deruxtecan","deutivacaftor","difamilast","doravirine","dupilumab","durvalumab","elexacaftor","elranatamab-bcmm","ensartinib","ensitrelvir","exagamglogene autotemcel","faricimab-svoa","fezolinetant","gadoquatrane","gedatolisib","gepotidacin","icotrokinra","islatravir","lebrikizumab-lbkz","lecanemab","linerixibat","milsaperidone","navepegritide","nipocalimab-aahu","obecabtagene autoleucel","olezarsen","orforglipron","osimertinib","oveporexton","pegvaliase-pqpz","pegzilarginase-nbln","pivekimab sunirine-pvzy","prademagene zamikeracel","relacorilant","remibrutinib","resmetirom","revumenib","ribociclib","rilzabrutinib","rimegepant","risankizumab-rzaa","sotatercept-csrk","talquetamab","tebipenem pivoxil","tenecteplase","teplizumab-mzwv","tezacaftor","tividenofusp alfa-eknm","tovorafenib","troriluzole","ubrogepant","upadacitinib","valbenazine","vanzacaftor","veligrotug-vvze","vepdegestrant","vorasidenib","zenocutuzumab","zidebactam","zidesamtinib","zolbetuximab","zoliflodacin","Abilify","Advair","Advil","Ambien","Aspirin","Benadryl","Chantix","Claritin","Crestor","Cymbalta","Eliquis","Enbrel","Farxiga","Flonase","Humira","Januvia","Latuda","Lipitor","Lyrica","Metformin","Motrin","Mounjaro","Nexium","Ozempic","Plavix","Prilosec","Prozac","Rybelsus","Seroquel","Spiriva","Symbicort","Synthroid","Tecfidera","Trulicity","Tylenol","Valium","Vyvanse","Wegovy","Xanax","Xarelto","Xeljanz","Xolair","Zantac","Zepbound","Zoloft","Zyrtec","acetaminophen","adalimumab","alprazolam","apixaban","aripiprazole","atorvastatin","baloxavir marboxil","budesonide","cefepime","cetirizine","clopidogrel","copper histidinate","dapagliflozin","diazepam","dimethyl fumarate","diphenhydramine","dulaglutide","duloxetine","efgartigimod alfa","empagliflozin","emtricitabine","enlicitide decanoate","esomeprazole","etanercept","exenatide","famotidine","fluoxetine","fluticasone propionate","formoterol fumarate dihydrate","ibuprofen","insulin glargine","insulin icodec-abae","ivacaftor","levothyroxine","lisdexamfetamine","loratadine","lurasidone","nogapendekin alfa inbakicept-pmln","omalizumab","omeprazole","perfluorohexyloctane","pitolisant","pregabalin","quetiapine","rivaroxaban","rosuvastatin","sacubitril","salmeterol","secukinumab","semaglutide","sertraline","sitagliptin","sonrotoclax","sotagliflozin","suzetrigine","tenofovir alafenamide","testosterone undecanoate","tiotropium bromide","tirzepatide","tofacitinib","trospium chloride","valsartan","varenicline","xanomeline","zolpidem"];

const slug = n => n.toLowerCase().trim().replace(/[^a-z0-9 -]/g,'').replace(/\s+/g,'-');
const sleep = ms => new Promise(r => setTimeout(r, ms));
window.DOSER = [];

window.dumpDoseR = () => {
  const blob = new Blob([JSON.stringify(window.DOSER)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `drugscom_${window.DOSER.length}.json`;
  a.click();
  console.log(`downloaded ${window.DOSER.length} records`);
};

const b64 = buf => {
  let s = ''; const bytes = new Uint8Array(buf);
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return btoa(s);
};

(async () => {
  console.log(`starting ${NAMES.length} names -- stop anytime, then run dumpDoseR()`);
  for (let i = 0; i < NAMES.length; i++) {
    const name = NAMES[i];
    const rec = { name, status:null, audio_url:null, respell:null, audio_b64:null };
    try {
      const r = await fetch(`https://www.drugs.com/${slug(name)}.html`, {credentials:'include'});
      rec.status = r.status;
      if (r.ok) {
        const t = await r.text();
        const a = t.match(/\/audio\/wav\/[\w.-]+\.wav/);
        if (a) {
          rec.audio_url = 'https://www.drugs.com' + a[0];
          if (GET_AUDIO) {
            try {
              const ar = await fetch(rec.audio_url, {credentials:'include'});
              if (ar.ok) rec.audio_b64 = b64(await ar.arrayBuffer());
            } catch (e) {}
          }
        }
        const p = t.match(/[Pp]ronunciation[^<]{0,30}<[^>]*>\s*([^<]{3,60})</)
               || t.match(/\(\s*([a-z]{2,}(?:\s+[A-Za-z]{2,}){1,8})\s*\)/);
        if (p) rec.respell = p[1].trim();
      }
    } catch (e) { rec.status = 'ERR ' + e.message; }

    window.DOSER.push(rec);
    if (i % 10 === 0 || rec.audio_b64)
      console.log(`${i+1}/${NAMES.length}`, name, rec.status,
                  rec.audio_b64 ? `AUDIO ${Math.round(rec.audio_b64.length/1366)}kb` : (rec.audio_url?'url-only':'-'),
                  rec.respell||'');
    await sleep(DELAY_MS);
  }
  const hits = window.DOSER.filter(r=>r.audio_b64).length;
  console.log(`\n=== done: ${hits}/${window.DOSER.length} with audio ===`);
  window.dumpDoseR();
})();
