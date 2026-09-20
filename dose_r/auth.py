"""Credential handling for the GCP sandbox.

The service-account key is read from the GOOGLE_APPLICATION_CREDENTIALS_JSON
environment variable and decoded in memory. It is deliberately never written to
disk: the harness persists audio, timings and costs, never credentials.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import threading
from typing import Any

import google.auth.transport.requests as google_requests
from google.oauth2 import service_account

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_ENV_VAR = "GOOGLE_APPLICATION_CREDENTIALS_JSON"

_lock = threading.Lock()
_credentials: service_account.Credentials | None = None


class CredentialsUnavailable(RuntimeError):
    """Raised when no usable service-account credential is present."""


def _decode_key_material(raw: str) -> dict[str, Any]:
    """Accept either raw JSON or base64-wrapped JSON."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(base64.b64decode(raw, validate=True).decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CredentialsUnavailable(
            f"{_ENV_VAR} is set but is neither JSON nor base64-encoded JSON"
        ) from exc


def get_credentials() -> service_account.Credentials:
    """Return process-wide cached credentials, refreshing when expired."""
    global _credentials
    with _lock:
        if _credentials is None:
            raw = os.environ.get(_ENV_VAR)
            if not raw:
                raise CredentialsUnavailable(
                    f"{_ENV_VAR} is not set; cannot reach the GCP sandbox."
                )
            _credentials = service_account.Credentials.from_service_account_info(
                _decode_key_material(raw), scopes=_SCOPES
            )
        if not _credentials.valid:
            _credentials.refresh(google_requests.Request())
        return _credentials


def auth_headers() -> dict[str, str]:
    creds = get_credentials()
    return {"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"}


def identity() -> str:
    """Service-account email, for run provenance."""
    return getattr(get_credentials(), "service_account_email", "unknown")
