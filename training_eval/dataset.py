"""Build the feature table from the API, keeping raw responses in a local cache."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from training_eval.client import ApiClient
from training_eval.features import trial_row

log = logging.getLogger(__name__)


def _load_trial(client: ApiClient, trial_id: str, cache_dir: Path | None) -> dict:
    path = cache_dir / f"{trial_id}.pkl.gz" if cache_dir else None
    if path is not None and path.exists():
        return pd.read_pickle(path)
    detail = client.trial(trial_id)
    raw = {
        "gestures": detail.get("gestures", []),
        "kinematics": client.kinematics(trial_id),
        "frames": client.frames(trial_id),
    }
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(raw, path)
    return raw


def trial_features(client: ApiClient, meta: dict, cache_dir: Path | None = None) -> dict:
    """The feature row of one trial from the list of ``/api/trials``."""
    raw = _load_trial(client, meta["trial_id"], cache_dir)
    return trial_row(meta, raw["kinematics"], raw["gestures"], raw["frames"])


def build_table(
    client: ApiClient, cache_dir: Path | None = None, tasks: list[str] | None = None
) -> pd.DataFrame:
    """One row per trial with a usable kinematics series.

    Args:
        client: The API to read.
        cache_dir: Where to keep raw responses, so that changing a feature does not mean
            downloading the dataset again.
        tasks: Restrict to these tasks.

    Raises:
        RuntimeError: If the API has no trials.
    """
    trials = client.trials()
    if tasks:
        trials = [t for t in trials if t["task"] in tasks]
    if not trials:
        raise RuntimeError(f"{client.base_url} has no matching trials; has jigsaws_ingest run?")
    rows = []
    for i, meta in enumerate(trials, start=1):
        rows.append(trial_features(client, meta, cache_dir))
        log.info("%d/%d %s", i, len(trials), meta["trial_id"])
    return pd.DataFrame(rows).sort_values("trial_id").reset_index(drop=True)
