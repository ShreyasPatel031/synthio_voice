"""University of Michigan College of Pharmacy drug-pronunciation audio, via
the Wayback Machine.

The live page and its audio files (`https://pharmacy.umich.edu/wp-content/
uploads/{name}.wav`) both return HTTP 403 to this environment -- a Cloudflare
JS challenge, not a normal block page. The Wayback Machine bypasses this
entirely: a captured snapshot of the page serves clean, and its embedded
`im_` audio URLs redirect (302) to whichever snapshot actually captured that
file's bytes. Following redirects is what makes this work; the `im_` URL
itself never serves the audio directly.

The page is a 450-name teaching list, mostly unrelated to DOSE's set, so
`parse_audio_urls` returns everything it finds and callers filter down to the
handful of names that match a DOSE ingredient.
"""

from __future__ import annotations

import re
import time
import urllib.request

PAGE_URL = (
    "https://web.archive.org/web/20251018234448/"
    "https://pharmacy.umich.edu/pharmacy-student-experience/resources/drug-pronunciations/"
)
UA = {"User-Agent": "Mozilla/5.0 (compatible; DOSE-R research benchmark)"}
TIMEOUT = 25

_SOURCE_TAG = re.compile(
    r'<source src="(https://web\.archive\.org/web/\d+im_/'
    r'https://pharmacy\.umich\.edu/wp-content/uploads/([^"]+?)\.wav)" type="audio/wav">'
)


def fetch_page(url: str = PAGE_URL) -> str:
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=TIMEOUT).read().decode("utf-8", "ignore")


def parse_audio_urls(html: str) -> dict[str, str]:
    """name (as filed on the page, original case) -> its `im_` Wayback audio URL."""
    return {name: url for url, name in _SOURCE_TAG.findall(html)}


def fetch_audio(url: str) -> bytes:
    """The clip's bytes, following the `im_` URL's redirect to a real snapshot."""
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=TIMEOUT).read()


def polite_fetch_audio(url: str, delay_s: float = 1.0) -> bytes:
    time.sleep(delay_s)
    return fetch_audio(url)
