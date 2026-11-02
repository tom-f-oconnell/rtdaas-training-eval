"""Read-only client for the robot-telemetry-daas data API."""

from __future__ import annotations

import os

import httpx
import pandas as pd

DEFAULT_API_URL = "http://localhost:29398"
ARMS = ("psml", "psmr", "mtml", "mtmr")


def default_api_url() -> str:
    return os.environ.get("RTDAAS_API_URL", DEFAULT_API_URL)


class ApiClient:
    """The procedure routes of the data API.

    Args:
        base_url: API root, e.g. ``http://localhost:29398``.
        transport: Optional httpx transport, for tests.
    """

    def __init__(self, base_url: str | None = None, transport: httpx.BaseTransport | None = None):
        self.base_url = (base_url or default_api_url()).rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=300.0, transport=transport)

    def _get(self, path: str, **params) -> dict:
        response = self._http.get(path, params=params or None)
        response.raise_for_status()
        return response.json()

    def trials(self) -> list[dict]:
        return self._get("/api/trials")["trials"]

    def trial(self, trial_id: str) -> dict:
        """One trial, with its ``gestures`` list."""
        return self._get(f"/api/trials/{trial_id}")

    def kinematics(self, trial_id: str) -> pd.DataFrame:
        """All four arms at the full 30 Hz: ``frame_idx, t_s, gesture, <arm>_<field>...``."""
        rows = self._get(f"/api/trials/{trial_id}/kinematics", arms=",".join(ARMS))["samples"]
        return pd.DataFrame(rows)

    def frames(self, trial_id: str) -> pd.DataFrame:
        """Per-frame video features, possibly empty: the table is optional in the pipeline."""
        return pd.DataFrame(self._get(f"/api/trials/{trial_id}/frames")["frames"])
