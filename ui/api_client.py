"""HTTP client for the Income Insight FastAPI service.

This is the UI's only path to the model. It knows the endpoint contract from
api/main.py and turns every failure (no connection, timeout, 4xx/5xx) into an
ApiError with a message a person can act on. No model code lives here.

Endpoints used:
    GET  /healthz        GET /version      GET /schema       GET /audit
    POST /predict        (JSON: {"features": {...10 fields...}, "run_id"?: int})
    POST /predict_batch  (multipart form, field name "file", a CSV)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

# Render's free tier sleeps when idle; the first request can take ~a minute.
DEFAULT_TIMEOUT = 90

STATUS_HINTS = {
    409: "The API serves a different model run than the one requested.",
    413: "The upload is too large (the API accepts at most 5 MB and 10,000 rows).",
    422: "The API rejected the input. Nothing was scored or logged.",
    503: "The API or its Supabase database is temporarily unavailable.",
}


class ApiError(RuntimeError):
    """A failed API call, with the API's own explanation when it gave one."""

    def __init__(self, message: str, status_code: Optional[int] = None, detail: Any = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail

    @classmethod
    def from_response(cls, status_code: int, body: Any, text: str = "") -> "ApiError":
        detail = body.get("detail") if isinstance(body, dict) else None
        if isinstance(detail, dict):
            message = detail.get("message") or str(detail)
        elif isinstance(detail, list):  # FastAPI/pydantic request-validation error
            message = "The request did not match the API's expected format."
        elif isinstance(detail, str):
            message = detail
        else:
            message = (text or "").strip()[:300] or "no details were returned"
        hint = STATUS_HINTS.get(status_code)
        full = f"{hint} {message}" if hint and hint not in message else message
        return cls(f"HTTP {status_code}: {full}", status_code=status_code, detail=detail)

    @property
    def errors(self) -> List[Dict[str, Any]]:
        """Per-value problems as rows of {row, column, value, error}."""
        if isinstance(self.detail, dict) and isinstance(self.detail.get("errors"), list):
            return [
                {
                    "row": e.get("row"),
                    "column": e.get("column"),
                    "value": e.get("value"),
                    "error": e.get("error"),
                }
                for e in self.detail["errors"]
            ]
        if isinstance(self.detail, list):  # pydantic: [{loc, msg, input, ...}]
            out = []
            for e in self.detail:
                loc = [str(p) for p in e.get("loc", []) if p not in ("body",)]
                out.append({
                    "row": None,
                    "column": ".".join(loc) or None,
                    "value": e.get("input"),
                    "error": e.get("msg"),
                })
            return out
        return []

    @property
    def missing_columns(self) -> List[str]:
        if isinstance(self.detail, dict):
            return list(self.detail.get("missing_columns") or [])
        return []

    @property
    def n_errors(self) -> int:
        if isinstance(self.detail, dict) and isinstance(self.detail.get("n_errors"), int):
            return self.detail["n_errors"]
        return len(self.errors)


def _json_or_none(resp) -> Any:
    try:
        return resp.json()
    except ValueError:
        return None


class ApiClient:
    """Thin wrapper over the FastAPI endpoints."""

    def __init__(self, base_url: str, timeout: float = DEFAULT_TIMEOUT):
        if not base_url:
            raise ValueError("API_URL is not set")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, ok_statuses=(), **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = requests.request(method, url, timeout=self.timeout, **kwargs)
        except requests.exceptions.Timeout as e:
            raise ApiError(
                f"The API at {self.base_url} did not answer within {self.timeout:.0f} seconds. "
                "Render's free tier sleeps when idle, so wait a minute and try again."
            ) from e
        except requests.exceptions.RequestException as e:
            raise ApiError(
                f"Could not reach the API at {self.base_url} ({type(e).__name__}). "
                "Check API_URL and that the API is running."
            ) from e

        body = _json_or_none(resp)
        if resp.status_code >= 400 and resp.status_code not in ok_statuses:
            raise ApiError.from_response(resp.status_code, body, getattr(resp, "text", ""))
        if body is None:
            raise ApiError(f"{path} returned a response that is not JSON", resp.status_code)
        return body

    # -- ops / metadata ------------------------------------------------------
    def healthz(self) -> dict:
        # /healthz answers 503 (with the same body) when the model files failed
        # to load; the UI still wants that body to show what is wrong.
        return self._request("GET", "/healthz", ok_statuses=(503,))

    def version(self) -> dict:
        return self._request("GET", "/version")

    def schema(self) -> dict:
        return self._request("GET", "/schema")

    def audit(self) -> dict:
        return self._request("GET", "/audit")

    # -- prediction ----------------------------------------------------------
    def predict(self, features: Dict[str, Any], run_id: Optional[int] = None) -> dict:
        payload: Dict[str, Any] = {"features": features}
        if run_id is not None:
            payload["run_id"] = int(run_id)
        return self._request("POST", "/predict", json=payload)

    def predict_batch(self, filename: str, content: bytes) -> dict:
        files = {"file": (filename or "upload.csv", content, "text/csv")}
        return self._request("POST", "/predict_batch", files=files)
