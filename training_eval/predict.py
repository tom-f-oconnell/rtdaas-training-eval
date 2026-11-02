"""Train a final model, and score trials from the API with it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from training_eval.evaluate import labels_for
from training_eval.features import feature_columns
from training_eval.models import make_classifier


class OperatorInTrainingData(ValueError):
    """Raised when asked to score an operator the model trained on, which proves nothing."""


@dataclass
class ModelBundle:
    pipeline: object
    features: list[str]
    target: str
    subjects: list[str]
    n_trials: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> ModelBundle:
        return joblib.load(path)


def train(
    table: pd.DataFrame,
    groups: tuple[str, ...] | list[str] | None = None,
    model: str = "logreg",
    target: str = "skill",
    exclude_subjects: list[str] | None = None,
) -> ModelBundle:
    """Fit on every trial except those of ``exclude_subjects``."""
    features = feature_columns(groups) if groups else feature_columns()
    kept = table[~table["subject"].isin(exclude_subjects or [])]
    if kept.empty:
        raise ValueError("no trials left to train on")
    pipeline = make_classifier(model).fit(kept[["task", *features]], labels_for(kept, target))
    return ModelBundle(pipeline, features, target, sorted(kept["subject"].unique()), len(kept))


def predict_rows(
    bundle: ModelBundle, rows: pd.DataFrame, allow_known: bool = False
) -> pd.DataFrame:
    """Class probabilities for feature rows.

    Raises:
        OperatorInTrainingData: If a row's operator was trained on and ``allow_known`` is false.
    """
    known = sorted(set(rows["subject"]) & set(bundle.subjects))
    if known and not allow_known:
        raise OperatorInTrainingData(
            f"operators {known} are in this model's training data, so its output would not be a "
            "prediction. Train with --exclude-subject, or read their held-out score from "
            "out_of_fold_predictions.csv."
        )
    proba = bundle.pipeline.predict_proba(rows[["task", *bundle.features]])
    classes = list(bundle.pipeline.classes_)
    out = rows[["trial_id", "subject"]].copy()
    for j, cls in enumerate(classes):
        out[f"p_{cls}"] = proba[:, j]
    out["predicted"] = np.array(classes)[proba.argmax(axis=1)]
    return out


def top_reasons(bundle: ModelBundle, rows: pd.DataFrame, n: int = 5) -> list[dict]:
    """Features pushing the mean prediction toward its most likely class (logistic models only).

    Returns:
        ``[{"feature", "z", "contribution"}]``: ``z`` is the value in standard deviations from the
        training mean for its task, and ``contribution`` its effect on the class's log-odds.
    """
    model = bundle.pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return []
    X = rows[["task", *bundle.features]]
    z = bundle.pipeline[:-1].transform(X).mean(axis=0)
    mean_proba = bundle.pipeline.predict_proba(X).mean(axis=0)
    cls = int(mean_proba.argmax())
    coef = (
        model.coef_[0] * (1 if cls == 1 else -1) if model.coef_.shape[0] == 1 else model.coef_[cls]
    )
    contributions = coef * z
    order = np.argsort(-contributions)[:n]
    return [
        {
            "feature": bundle.features[i],
            "z": round(float(z[i]), 2),
            "contribution": round(float(contributions[i]), 3),
        }
        for i in order
    ]
