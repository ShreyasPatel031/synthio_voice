import base64
import io
import json
import struct
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dose_r.references import (
    audio_manifest,
    audio_sources_clincalc,
    audio_sources_drugscom,
    audio_sources_umich,
    audio_verify,
)
from scripts import fetch_clincalc_reference_audio as fcc
from scripts import fetch_drugscom_reference_audio as fdr
from scripts import fetch_umich_reference_audio as fum

MANIFEST = ROOT / "data" / "reference_audio" / "manifest.jsonl"
DATASET = ROOT / "data" / "dose_v1.jsonl"


def make_mp3(duration_s: float, bitrate_kbps: int = 128, samplerate: int = 44100) -> bytes:
    frame_size = (144 * bitrate_kbps * 1000) // samplerate
    frame = bytes([0xFF, 0xFB, 0x90, 0xC0]) + bytes(frame_size - 4)
    n_frames = max(1, round(duration_s / (1152 / samplerate)))
    return frame * n_frames


def make_wav(duration_s: float, rate: int = 16000, channels: int = 1) -> bytes:
    n_frames = int(duration_s * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{n_frames * channels}h", *([0] * (n_frames * channels))))
    return buf.getvalue()


def wav_b64(duration_s: float, **kw) -> str:
    return base64.b64encode(make_wav(duration_s, **kw)).decode()


# --- audio_verify -----------------------------------------------------------


def test_probe_wav_reads_duration_rate_channels():
    probe = audio_verify.probe_wav(make_wav(1.0, rate=22050, channels=2))
    assert probe.ok
    assert probe.format == "wav"
    assert probe.duration_s == pytest.approx(1.0, abs=0.01)
    assert probe.sample_rate_hz == 22050
    assert probe.channels == 2


def test_probe_wav_rejects_garbage():
    probe = audio_verify.probe_wav(b"not a wav file")
    assert not probe.ok
    assert probe.error


def test_duration_flag_bounds():
    assert audio_verify.duration_flag(0.29) is not None
    assert audio_verify.duration_flag(0.3) is None
    assert audio_verify.duration_flag(4.0) is None
    assert audio_verify.duration_flag(4.01) is not None
    assert audio_verify.duration_flag(None) is None


def test_estimate_syllables():
    assert audio_verify.estimate_syllables("Advil") == 2
    assert audio_verify.estimate_syllables("insulin glargine") == 6
    assert audio_verify.estimate_syllables("") == 1


def test_syllable_outliers_flags_disproportionate_duration():
    durations = {"Short": 0.4, "Also": 0.42, "Names": 0.38, "Suspect": 3.5}
    outliers = audio_verify.syllable_outliers(durations)
    assert "Suspect" in outliers
    assert "Short" not in outliers


def test_syllable_outliers_empty_input():
    assert audio_verify.syllable_outliers({}) == {}


# --- audio_manifest -----------------------------------------------------------


def test_record_key_defaults_query_to_ingredient():
    key = audio_manifest.record_key({"ingredient": "Advil", "source": "drugs.com"})
    assert key == ("Advil", "drugs.com", "Advil")


def test_save_load_round_trip(tmp_path):
    path = tmp_path / "manifest.jsonl"
    records = [
        {"ingredient": "B", "source": "s", "query": "B"},
        {"ingredient": "A", "source": "s", "query": "A"},
    ]
    audio_manifest.save(records, path)
    loaded = audio_manifest.load(path)
    assert [r["ingredient"] for r in loaded] == ["A", "B"]


def test_replace_source_only_touches_named_source():
    existing = [
        {"ingredient": "A", "source": "merriam-webster", "query": "A"},
        {"ingredient": "B", "source": "drugs.com", "query": "B", "old": True},
    ]
    fresh = [{"ingredient": "B", "source": "drugs.com", "query": "B", "old": False}]
    merged = audio_manifest.replace_source(existing, fresh, "drugs.com")
    assert len(merged) == 2
    by_ing = {r["ingredient"]: r for r in merged}
    assert by_ing["A"]["source"] == "merriam-webster"
    assert by_ing["B"]["old"] is False


def test_available_clips_prefers_nci_over_drugs_com(tmp_path):
    from dose_r.references.reference_clips import available_clips

    nci = tmp_path / "nci.mp3"
    drugs = tmp_path / "drugs.wav"
    nci.write_bytes(make_mp3(1.0))
    drugs.write_bytes(make_wav(0.5))
    man = tmp_path / "manifest.jsonl"
    rows = [
        {
            "ingredient": "Voranigo",
            "name_type": "brand",
            "source": "drugs.com",
            "local_path": str(drugs),
            "format": "wav",
            "sample_rate_hz": 16000,
            "duration_s": 0.5,
            "coverage": "full",
        },
        {
            "ingredient": "Voranigo",
            "name_type": "brand",
            "source": "nci",
            "local_path": str(nci),
            "format": "mp3",
            "sample_rate_hz": 48000,
            "duration_s": 1.9,
            "coverage": "full",
            "respelling": "voh-rah-NEE-goh",
        },
    ]
    man.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    clips = available_clips(man)
    assert clips["Voranigo"].source == "nci"
    assert clips["Voranigo"].path == nci


def test_available_clips_ignores_component_coverage(tmp_path):
    from dose_r.references.reference_clips import available_clips

    nci = tmp_path / "nci.mp3"
    nci.write_bytes(make_mp3(1.0))
    man = tmp_path / "manifest.jsonl"
    rows = [
        {
            "ingredient": "osimertinib",
            "name_type": "generic",
            "source": "nci",
            "local_path": str(nci),
            "format": "mp3",
            "sample_rate_hz": 48000,
            "duration_s": 1.5,
            "coverage": "component",
        }
    ]
    man.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert "osimertinib" not in available_clips(man)


def test_available_clips_skips_nci_when_file_missing(tmp_path):
    from dose_r.references.reference_clips import available_clips

    drugs = tmp_path / "drugs.wav"
    drugs.write_bytes(make_wav(0.5))
    man = tmp_path / "manifest.jsonl"
    rows = [
        {
            "ingredient": "Voranigo",
            "name_type": "brand",
            "source": "nci",
            "local_path": str(tmp_path / "missing.mp3"),
            "format": "mp3",
            "sample_rate_hz": 48000,
            "duration_s": 1.9,
            "coverage": "full",
        },
        {
            "ingredient": "Voranigo",
            "name_type": "brand",
            "source": "drugs.com",
            "local_path": str(drugs),
            "format": "wav",
            "sample_rate_hz": 16000,
            "duration_s": 0.5,
            "coverage": "full",
        },
    ]
    man.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    clips = available_clips(man)
    assert clips["Voranigo"].source == "drugs.com"


def test_duplicate_groups_detects_shared_hash():
    records = [
        {"ingredient": "A", "sha256": "same"},
        {"ingredient": "B", "sha256": "same"},
        {"ingredient": "C", "sha256": "other"},
    ]
    dupes = audio_manifest.duplicate_groups(records)
    assert dupes == {"same": ["A", "B"]}


# --- audio_sources_drugscom ---------------------------------------------------


def test_merge_records_last_file_wins(tmp_path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    f1.write_text(json.dumps([{"name": "X", "status": 404, "audio_url": None, "audio_b64": None}]))
    f2.write_text(json.dumps([{"name": "X", "status": 200, "audio_url": "u", "audio_b64": "Zg=="}]))
    merged = audio_sources_drugscom.merge_records([f1, f2])
    assert merged["X"]["status"] == 200
    assert merged["X"]["audio_b64"] == "Zg=="


def test_audio_conflicts_detects_disagreement(tmp_path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    f1.write_text(json.dumps([{"name": "X", "status": 200, "audio_url": "u1", "audio_b64": "AAAA"}]))
    f2.write_text(json.dumps([{"name": "X", "status": 200, "audio_url": "u2", "audio_b64": "BBBB"}]))
    assert audio_sources_drugscom.audio_conflicts([f1, f2]) == ["X"]


def test_audio_conflicts_none_when_only_one_file_has_audio(tmp_path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    f1.write_text(json.dumps([{"name": "X", "status": 404, "audio_url": None, "audio_b64": None}]))
    f2.write_text(json.dumps([{"name": "X", "status": 200, "audio_url": "u", "audio_b64": "AAAA"}]))
    assert audio_sources_drugscom.audio_conflicts([f1, f2]) == []


def test_decode_audio_round_trips():
    raw = make_wav(0.5)
    record = {"audio_b64": base64.b64encode(raw).decode()}
    assert audio_sources_drugscom.decode_audio(record) == raw


# --- fetch_drugscom_reference_audio (the driver) ------------------------------


@pytest.fixture
def audio_dir(tmp_path, monkeypatch):
    scratch = ROOT / "data" / "reference_audio" / f"_test_scratch_{tmp_path.name}"
    monkeypatch.setattr(fdr, "AUDIO_DIR", scratch)
    yield scratch
    if scratch.exists():
        import shutil

        shutil.rmtree(scratch)


@pytest.fixture
def collected_file(tmp_path):
    path = tmp_path / "collected.json"
    path.write_text(
        json.dumps(
            [
                {"name": "Abilify", "status": 200, "audio_url": "https://x/1.wav",
                 "respell": "junk", "audio_b64": wav_b64(1.0)},
                {"name": "Advair", "status": 404, "audio_url": None,
                 "respell": None, "audio_b64": None},
                {"name": "Aspirin", "status": 200, "audio_url": None,
                 "respell": None, "audio_b64": None},
            ]
        )
    )
    return path


def test_run_builds_one_clip_and_two_misses(collected_file, audio_dir):
    clips, misses, conflicts = fdr.run([collected_file])
    assert conflicts == []
    assert [c["ingredient"] for c in clips] == ["Abilify"]
    assert clips[0]["source"] == "drugs.com"
    assert clips[0]["coverage"] == "full"
    assert clips[0]["respelling"] is None
    assert clips[0]["status"] == "ok"

    by_reason = {m["ingredient"]: m["reason"] for m in misses}
    assert by_reason["Advair"] == "url_not_found_not_absent"
    assert by_reason["Aspirin"] == "no_audio_on_page"


def test_run_ignores_respell_field(collected_file, audio_dir):
    clips, _, _ = fdr.run([collected_file])
    assert "junk" not in json.dumps(clips)


def test_run_is_idempotent_on_disk(collected_file, audio_dir):
    fdr.run([collected_file])
    written = audio_dir / "1.wav"
    assert written.exists()
    mtime_before = written.stat().st_mtime_ns

    clips_again, _, _ = fdr.run([collected_file])
    assert written.stat().st_mtime_ns == mtime_before
    assert len(clips_again) == 1


def test_run_flags_a_syllable_outlier(tmp_path, audio_dir):
    path = tmp_path / "collected.json"
    path.write_text(
        json.dumps(
            [
                {"name": "Advil", "status": 200, "audio_url": "https://x/a.wav",
                 "respell": None, "audio_b64": wav_b64(0.4)},
                {"name": "Aspirin", "status": 200, "audio_url": "https://x/b.wav",
                 "respell": None, "audio_b64": wav_b64(0.45)},
                {"name": "Zepbound", "status": 200, "audio_url": "https://x/c.wav",
                 "respell": None, "audio_b64": wav_b64(3.9)},
            ]
        )
    )
    clips, _, _ = fdr.run([path])
    by_ing = {c["ingredient"]: c for c in clips}
    assert by_ing["Zepbound"]["status"] == "flagged"
    assert any("possible name mismatch" in f for f in by_ing["Zepbound"]["flags"])
    assert by_ing["Advil"]["status"] == "ok"


# --- audio_sources_umich -------------------------------------------------------


SAMPLE_UMICH_HTML = (
    '<source src="https://web.archive.org/web/20251018234448im_/'
    'https://pharmacy.umich.edu/wp-content/uploads/cefepime.wav" type="audio/wav">'
    '<source src="https://web.archive.org/web/20251018234448im_/'
    'https://pharmacy.umich.edu/wp-content/uploads/Lipitor.wav" type="audio/wav">'
)


def test_parse_audio_urls_extracts_name_and_im_url():
    urls = audio_sources_umich.parse_audio_urls(SAMPLE_UMICH_HTML)
    assert urls == {
        "cefepime": "https://web.archive.org/web/20251018234448im_/"
        "https://pharmacy.umich.edu/wp-content/uploads/cefepime.wav",
        "Lipitor": "https://web.archive.org/web/20251018234448im_/"
        "https://pharmacy.umich.edu/wp-content/uploads/Lipitor.wav",
    }


def test_parse_audio_urls_ignores_non_source_tags():
    html = '<a href="https://pharmacy.umich.edu/wp-content/uploads/cefepime.wav">cefepime</a>'
    assert audio_sources_umich.parse_audio_urls(html) == {}


# --- fetch_umich_reference_audio (the driver) -----------------------------------


def test_match_ingredients_is_case_insensitive():
    page_names = {"cefepime": "url1", "Lipitor": "url2", "Diamox": "url3"}
    matches = fum.match_ingredients(page_names, ["Cefepime", "lipitor", "Aspirin"])
    assert matches == {"Cefepime": ("cefepime", "url1"), "lipitor": ("Lipitor", "url2")}


def test_match_ingredients_no_overlap():
    assert fum.match_ingredients({"acebutolol": "url"}, ["Aspirin"]) == {}


@pytest.fixture
def umich_audio_dir(tmp_path, monkeypatch):
    scratch = ROOT / "data" / "reference_audio" / f"_test_umich_scratch_{tmp_path.name}"
    monkeypatch.setattr(fum, "AUDIO_DIR", scratch)
    yield scratch
    if scratch.exists():
        import shutil

        shutil.rmtree(scratch)


def _umich_html_for(*names: str) -> str:
    return "".join(
        f'<source src="https://web.archive.org/web/20251018234448im_/'
        f'https://pharmacy.umich.edu/wp-content/uploads/{name}.wav" type="audio/wav">'
        for name in names
    )


def test_run_builds_a_clip_for_a_matching_ingredient(umich_audio_dir, monkeypatch):
    calls = []

    def fake_fetch(url, delay_s):
        calls.append(url)
        return make_wav(1.2, rate=22050)

    monkeypatch.setattr(audio_sources_umich, "polite_fetch_audio", fake_fetch)
    html = _umich_html_for("cefepime", "notarealdrug")

    clips, misses = fum.run(html)

    assert [c["ingredient"] for c in clips] == ["cefepime"]
    assert clips[0]["source"] == "umich"
    assert clips[0]["source_name"] == "umich/wayback-machine"
    assert clips[0]["coverage"] == "full"
    assert clips[0]["respelling"] is None
    assert clips[0]["status"] == "ok"
    assert misses == []
    assert len(calls) == 1


def test_run_is_idempotent_on_disk(umich_audio_dir, monkeypatch):
    calls = []

    def fake_fetch(url, delay_s):
        calls.append(url)
        return make_wav(1.2, rate=22050)

    monkeypatch.setattr(audio_sources_umich, "polite_fetch_audio", fake_fetch)
    html = _umich_html_for("cefepime")

    fum.run(html)
    written = umich_audio_dir / "cefepime.wav"
    assert written.exists()

    fum.run(html)
    assert len(calls) == 1


def test_run_records_a_miss_on_download_failure(umich_audio_dir, monkeypatch):
    def failing_fetch(url, delay_s):
        raise ConnectionError("connection reset by peer")

    monkeypatch.setattr(audio_sources_umich, "polite_fetch_audio", failing_fetch)
    html = _umich_html_for("cefepime")

    clips, misses = fum.run(html)
    assert clips == []
    assert misses == [
        {
            "ingredient": "cefepime",
            "name_type": "generic",
            "reason": "download_failed",
            "detail": "connection reset by peer",
        }
    ]


# --- audio_sources_clincalc ------------------------------------------------


def _clincalc_page_html(generic=None, generic_file=None, brand=None, brand_file=None):
    parts = []
    if generic:
        parts.append(
            f"<h2 class=\"pTitle\">The generic name '{generic}' is pronounced:</h2>"
            f'<audio><source src="../mp3/{generic_file}.ogg" type="audio/ogg">'
            f'<source src="../mp3/{generic_file}.mp3" type="audio/mpeg"></audio>'
        )
    if brand:
        parts.append(
            f"<h2 class=\"pTitle\">The brand name '{brand}' is pronounced:</h2>"
            f'<audio><source src="../mp3/{brand_file}.ogg" type="audio/ogg">'
            f'<source src="../mp3/{brand_file}.mp3" type="audio/mpeg"></audio>'
        )
    return "\n".join(parts)


def test_split_names_strips_many_more_suffixes():
    assert audio_sources_clincalc.split_names("Vicodin; Norco; Lortab (many more)") == [
        "Vicodin", "Norco", "Lortab",
    ]
    assert audio_sources_clincalc.split_names("Cardizem CD (and many more)") == ["Cardizem CD"]
    assert audio_sources_clincalc.split_names("atorvastatin") == ["atorvastatin"]


def test_strip_route_annotation():
    assert audio_sources_clincalc.strip_route_annotation("Fluticasone (inhaled)") == "Fluticasone"
    assert audio_sources_clincalc.strip_route_annotation("Advil") == "Advil"


def test_name_matches_exact_case_insensitive():
    assert audio_sources_clincalc.name_matches("Advair", "advair")


def test_name_matches_salt_suffix_prefix():
    assert audio_sources_clincalc.name_matches("fluticasone", "fluticasone propionate")
    assert audio_sources_clincalc.name_matches("Fluticasone (inhaled)", "fluticasone propionate")


def test_name_matches_rejects_word_boundary_crossing_prefix():
    assert not audio_sources_clincalc.name_matches("form", "formoterol fumarate dihydrate")
    assert not audio_sources_clincalc.name_matches("albuterol", "atorvastatin")


def test_name_matches_does_not_match_a_longer_clincalc_name():
    assert not audio_sources_clincalc.name_matches("fluticasone propionate", "fluticasone")


def test_parse_index_simple_entry():
    html = '<a href="HowToPronounce/atorvastatin">atorvastatin (Lipitor)</a>'
    [entry] = audio_sources_clincalc.parse_index(html)
    assert entry.slug == "atorvastatin"
    assert entry.generics == ["atorvastatin"]
    assert entry.brands == ["Lipitor"]


def test_parse_index_combo_with_many_brands():
    html = (
        '<a href="HowToPronounce/acetaminophenhydrocodone">'
        "acetaminophen; hydrocodone (Vicodin; Norco; Lortab (many more))</a>"
    )
    [entry] = audio_sources_clincalc.parse_index(html)
    assert entry.generics == ["acetaminophen", "hydrocodone"]
    assert entry.brands == ["Vicodin", "Norco", "Lortab"]


def test_parse_index_route_annotation_before_brand_group():
    # The brand group is the *last* top-level "(...)", not the first -- a
    # route-annotated generic ("(inhaled)") must not be mistaken for it.
    html = '<a href="HowToPronounce/fluticasone">fluticasone (inhaled) (Flovent)</a>'
    [entry] = audio_sources_clincalc.parse_index(html)
    assert entry.generics == ["fluticasone (inhaled)"]
    assert entry.brands == ["Flovent"]


def test_merge_by_slug_unions_two_labels_for_the_same_page():
    html = (
        '<a href="HowToPronounce/fluticasone">fluticasone (inhaled) (Flovent)</a>'
        '<a href="HowToPronounce/fluticasone">fluticasone (nasal) (Flonase)</a>'
    )
    merged = audio_sources_clincalc.merge_by_slug(audio_sources_clincalc.parse_index(html))
    assert merged["fluticasone"].brands == ["Flovent", "Flonase"]


def test_candidate_slugs_filters_to_matching_ingredients():
    html = (
        '<a href="HowToPronounce/atorvastatin">atorvastatin (Lipitor)</a>'
        '<a href="HowToPronounce/acyclovir">acyclovir (Zovirax)</a>'
    )
    entries = audio_sources_clincalc.parse_index(html)
    slugs = audio_sources_clincalc.candidate_slugs(entries, ["Lipitor"])
    assert set(slugs) == {"atorvastatin"}


def test_parse_page_extracts_both_blocks_with_resolved_urls():
    html = _clincalc_page_html("Atorvastatin", "atorvastatin", "Lipitor", "lipitor")
    page = audio_sources_clincalc.parse_page(html, audio_sources_clincalc.page_url("atorvastatin"))
    assert page.generic_name == "Atorvastatin"
    assert page.generic_url == "https://clincalc.com/pronouncetop200drugs/mp3/atorvastatin.mp3"
    assert page.brand_name == "Lipitor"
    assert page.brand_url == "https://clincalc.com/pronouncetop200drugs/mp3/lipitor.mp3"


def test_parse_page_handles_generic_only():
    html = _clincalc_page_html(generic="Acyclovir", generic_file="acyclovir")
    page = audio_sources_clincalc.parse_page(html, audio_sources_clincalc.page_url("acyclovir"))
    assert page.brand_name is None
    assert page.brand_url is None


# --- fetch_clincalc_reference_audio (the driver) -----------------------------


@pytest.fixture
def clincalc_audio_dir(tmp_path, monkeypatch):
    scratch = ROOT / "data" / "reference_audio" / f"_test_clincalc_scratch_{tmp_path.name}"
    monkeypatch.setattr(fcc, "AUDIO_DIR", scratch)
    yield scratch
    if scratch.exists():
        import shutil

        shutil.rmtree(scratch)


def test_run_builds_generic_and_brand_clips_as_separate_rows(clincalc_audio_dir, monkeypatch):
    index_html = '<a href="HowToPronounce/atorvastatin">atorvastatin (Lipitor)</a>'
    page_html = _clincalc_page_html("Atorvastatin", "atorvastatin", "Lipitor", "lipitor")
    monkeypatch.setattr(audio_sources_clincalc, "polite_fetch", lambda url, delay_s: page_html)
    monkeypatch.setattr(audio_sources_clincalc, "polite_fetch_audio", lambda url, delay_s: make_mp3(1.0))

    clips, misses = fcc.run(index_html)
    by_ing = {c["ingredient"]: c for c in clips}

    assert set(by_ing) == {"atorvastatin", "Lipitor"}
    assert by_ing["atorvastatin"]["name_type"] == "generic"
    assert by_ing["Lipitor"]["name_type"] == "brand"
    assert by_ing["atorvastatin"]["coverage"] == "full"
    assert by_ing["atorvastatin"]["status"] == "ok"
    assert "Lipitor" in by_ing["atorvastatin"]["respelling"]
    assert "atorvastatin" in by_ing["Lipitor"]["respelling"]
    assert misses == []


def test_run_flags_component_coverage_for_a_multi_name_clip(clincalc_audio_dir, monkeypatch):
    index_html = '<a href="HowToPronounce/ibuprofen">ibuprofen (Advil; Motrin)</a>'
    page_html = _clincalc_page_html("Ibuprofen", "ibuprofen", "Advil; Motrin", "advilmotrin")
    monkeypatch.setattr(audio_sources_clincalc, "polite_fetch", lambda url, delay_s: page_html)
    monkeypatch.setattr(audio_sources_clincalc, "polite_fetch_audio", lambda url, delay_s: make_mp3(1.0))

    clips, _ = fcc.run(index_html)
    by_ing = {c["ingredient"]: c for c in clips}

    assert by_ing["Advil"]["coverage"] == "component"
    assert by_ing["Motrin"]["coverage"] == "component"
    assert by_ing["Advil"]["status"] == "flagged"
    assert any("partial coverage" in f for f in by_ing["Advil"]["flags"])
    assert by_ing["ibuprofen"]["coverage"] == "full"


def test_run_reports_index_page_mismatch_as_a_miss(clincalc_audio_dir, monkeypatch):
    # The index lists this page under two labels (inhaled/Flovent and
    # nasal/Flonase) but the page itself only ever names Flovent -- Flonase
    # must come back as a miss, not a silently-wrong clip.
    index_html = (
        '<a href="HowToPronounce/fluticasone">fluticasone (inhaled) (Flovent)</a>'
        '<a href="HowToPronounce/fluticasone">fluticasone (nasal) (Flonase)</a>'
    )
    page_html = _clincalc_page_html("Fluticasone (inhaled)", "fluticasone", "Flovent", "flovent")
    monkeypatch.setattr(audio_sources_clincalc, "polite_fetch", lambda url, delay_s: page_html)
    monkeypatch.setattr(audio_sources_clincalc, "polite_fetch_audio", lambda url, delay_s: make_mp3(1.0))

    clips, misses = fcc.run(index_html)

    assert "Flonase" not in {c["ingredient"] for c in clips}
    assert any(
        m["ingredient"] == "Flonase" and m["reason"] == "index_page_mismatch" for m in misses
    )


def test_run_dedupes_the_same_clip_named_on_both_sides(clincalc_audio_dir, monkeypatch):
    # aspirin's own page heads both its generic and brand block "Aspirin",
    # linking the same file -- one recording, one row, not two.
    index_html = '<a href="HowToPronounce/aspirin">aspirin (Aspirin)</a>'
    page_html = _clincalc_page_html("Aspirin", "aspirin", "Aspirin", "aspirin")
    monkeypatch.setattr(audio_sources_clincalc, "polite_fetch", lambda url, delay_s: page_html)
    calls = []

    def fake_audio(url, delay_s):
        calls.append(url)
        return make_mp3(1.0)

    monkeypatch.setattr(audio_sources_clincalc, "polite_fetch_audio", fake_audio)

    clips, _ = fcc.run(index_html)
    assert [c["ingredient"] for c in clips] == ["Aspirin"]
    assert len(calls) == 1


def test_run_is_idempotent_on_disk(clincalc_audio_dir, monkeypatch):
    index_html = '<a href="HowToPronounce/atorvastatin">atorvastatin (Lipitor)</a>'
    page_html = _clincalc_page_html("Atorvastatin", "atorvastatin", "Lipitor", "lipitor")
    monkeypatch.setattr(audio_sources_clincalc, "polite_fetch", lambda url, delay_s: page_html)
    calls = []

    def fake_audio(url, delay_s):
        calls.append(url)
        return make_mp3(1.0)

    monkeypatch.setattr(audio_sources_clincalc, "polite_fetch_audio", fake_audio)

    fcc.run(index_html)
    assert len(calls) == 2
    fcc.run(index_html)
    assert len(calls) == 2


# --- the real, checked-in manifest ---------------------------------------------


@pytest.fixture(scope="module")
def manifest_records():
    return audio_manifest.load(MANIFEST)


@pytest.fixture(scope="module")
def dataset_ingredients():
    with DATASET.open() as f:
        return {i for line in f for i in json.loads(line)["ingredients"]}


def test_manifest_has_known_sources(manifest_records):
    sources = {r["source"] for r in manifest_records}
    assert {"merriam-webster", "drugs.com", "umich", "clincalc", "nci"} <= sources


def test_manifest_drugs_com_rows_are_unique_ingredients(manifest_records):
    # The exact count grows with each new collection batch, so this checks the
    # invariant that matters -- one row per ingredient, no re-import duplicates
    # -- rather than a specific number that would break on the next import.
    drugscom = [r for r in manifest_records if r["source"] == "drugs.com"]
    assert len(drugscom) >= 143
    assert len({r["ingredient"] for r in drugscom}) == len(drugscom)


def test_manifest_umich_rows_match_the_known_overlap(manifest_records):
    # UMich's list predates DOSE and is mostly unrelated drugs, so the overlap
    # with DOSE's 284 ingredients is small and specific, not something that
    # grows with future collection batches the way drugs.com's does.
    umich = [r for r in manifest_records if r["source"] == "umich"]
    assert {r["ingredient"] for r in umich} == {
        "Crestor", "Lipitor", "Plavix", "atorvastatin", "clopidogrel", "valsartan", "cefepime",
    }
    assert all(r["coverage"] == "full" for r in umich)
    assert len({r["ingredient"] for r in umich}) == len(umich)


def test_manifest_clincalc_rows_have_no_key_collisions_from_shared_pages(manifest_records):
    # A page's audio can name several DOSE ingredients at once (a combo's
    # generics, or several brands on one clip) -- each still needs its own
    # unique (ingredient, query) key, not a silently-overwritten duplicate.
    clincalc = [r for r in manifest_records if r["source"] == "clincalc"]
    keys = [(r["ingredient"], r["query"]) for r in clincalc]
    assert len(keys) == len(set(keys))
    assert len(clincalc) > 0


def test_manifest_clincalc_component_rows_are_flagged(manifest_records):
    for r in manifest_records:
        if r["source"] == "clincalc" and r["coverage"] == "component":
            assert r["status"] == "flagged"
            assert any("partial coverage" in f for f in r["flags"])


def test_manifest_no_duplicate_keys(manifest_records):
    keys = [audio_manifest.record_key(r) for r in manifest_records]
    assert len(keys) == len(set(keys))


def test_manifest_ingredients_are_in_dataset(manifest_records, dataset_ingredients):
    for r in manifest_records:
        assert r["ingredient"] in dataset_ingredients


def test_manifest_drugscom_rows_have_no_respelling(manifest_records):
    for r in manifest_records:
        if r["source"] == "drugs.com":
            assert r["respelling"] is None


def test_manifest_local_files_exist_and_hash_matches(manifest_records):
    for r in manifest_records:
        if r["source"] != "drugs.com":
            continue
        local_path = ROOT / r["local_path"]
        assert local_path.exists(), r["ingredient"]
        data = local_path.read_bytes()
        assert len(data) == r["bytes"]
        import hashlib

        assert hashlib.sha256(data).hexdigest() == r["sha256"]


def test_manifest_drugscom_durations_within_bounds_or_flagged(manifest_records):
    # A clip outside the expected single-name duration band is flagged rather
    # than dropped -- nogapendekin alfa inbakicept-pmln (4.35s) is a genuine
    # recording of a long generic name, not a bad decode. An out-of-band
    # duration must show up in `flags`, not be silently kept as if unremarkable.
    for r in manifest_records:
        if r["source"] == "drugs.com":
            in_bounds = audio_verify.MIN_DURATION_S <= r["duration_s"] <= audio_verify.MAX_DURATION_S
            assert in_bounds or r["flags"], r["ingredient"]


def test_manifest_flagged_rows_carry_a_reason(manifest_records):
    for r in manifest_records:
        if r["status"] == "flagged":
            assert r["flags"]
