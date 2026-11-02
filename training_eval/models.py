"""Models, and the scaling that lets one model cover three different tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

SKILL_ORDER = ("novice", "intermediate", "expert")
CLASSIFIERS = ("majority", "logreg", "forest")
REGRESSORS = ("mean", "ridge")


class TaskScaler(BaseEstimator, TransformerMixin):
    """Z-score each feature within its task, and drop the task column.

    Knot tying and suturing differ more in duration and path than novices and experts do, so a
    single scale would mostly encode the task. Fit on training folds only, like any scaler.
    """

    def fit(self, X: pd.DataFrame, y=None):
        features = X.drop(columns="task")
        self.columns_ = list(features.columns)
        self.global_ = features.agg(["mean", "std"])
        self.by_task_ = {
            task: part.agg(["mean", "std"]) for task, part in features.groupby(X["task"])
        }
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        out = X[self.columns_].to_numpy(dtype=np.float64, copy=True)
        for task in X["task"].unique():
            stats = self.by_task_.get(task, self.global_)
            std = stats.loc["std"].to_numpy(dtype=np.float64)
            std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
            mask = (X["task"] == task).to_numpy()
            out[mask] = (out[mask] - stats.loc["mean"].to_numpy(dtype=np.float64)) / std
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)


def make_classifier(name: str) -> Pipeline:
    """A pipeline from a table with a ``task`` column to skill classes.

    Heavily regularised: 8 operators is the effective sample size, whatever the number of trials.
    """
    estimators = {
        "majority": DummyClassifier(strategy="prior"),
        "logreg": LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000),
        "forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=0
        ),
    }
    return Pipeline(
        [
            ("scale", TaskScaler()),
            ("impute", SimpleImputer(strategy="median")),
            ("model", estimators[name]),
        ]
    )


def make_regressor(name: str) -> Pipeline:
    """A pipeline from a table with a ``task`` column to a GRS score."""
    estimators = {"mean": DummyRegressor(strategy="mean"), "ridge": Ridge(alpha=30.0)}
    return Pipeline(
        [
            ("scale", TaskScaler()),
            ("impute", SimpleImputer(strategy="median")),
            ("model", estimators[name]),
        ]
    )
