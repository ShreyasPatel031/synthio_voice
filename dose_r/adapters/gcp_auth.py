"""Google Cloud access tokens, minted in memory.

The service-account key lives in GOOGLE_APPLICATION_CREDENTIALS_JSON (base64 or
raw JSON) and is never written to disk -- google-auth accepts the parsed dict
directly, so there is no reason to materialise a key file.

Note for anyone hitting `ModuleNotFoundError: _cffi_backend` here: the system
`cryptography` install is missing its cffi backend, which breaks all of
google.auth.crypt. `pip install cffi` repairs it; see requirements.txt.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import threading

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

_lock = threading.Lock()
_credentials = None


class CredentialError(RuntimeError):
    pass


def _decode(blob: str) -> dict:
    blob = blob.strip()
    if blob.startswith("{"):
        return json.loads(blob)
    try:
        return json.loads(base64.b64decode(blob))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CredentialError(
            "GOOGLE_APPLICATION_CREDENTIALS_JSON is neither JSON nor base64-encoded JSON"
        ) from exc


def service_account_info() -> dict:
    blob = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not blob:
        raise CredentialError("GOOGLE_APPLICATION_CREDENTIALS_JSON is not set")
    info = _decode(blob)
    if info.get("type") != "service_account":
        raise CredentialError(f"expected a service_account key, got {info.get('type')!r}")
    return info


def credentials():
    global _credentials
    with _lock:
        if _credentials is None:
            from google.oauth2 import service_account

            _credentials = service_account.Credentials.from_service_account_info(
                service_account_info(), scopes=SCOPES
            )
        return _credentials


def access_token() -> str:
    """A pre-minted DOSE_R_GCP_ACCESS_TOKEN wins, so a run can be driven by an
    externally supplied token without the key ever entering this process."""
    token = os.environ.get("DOSE_R_GCP_ACCESS_TOKEN")
    if token:
        return token
    import google.auth.transport.requests

    creds = credentials()
    if not creds.valid:
        creds.refresh(google.auth.transport.requests.Request())
    return creds.token


def project_id() -> str:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project:
        return project
    return service_account_info()["project_id"]
