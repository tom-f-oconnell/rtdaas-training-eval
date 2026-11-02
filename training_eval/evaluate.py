"""Cross-validation that holds whole operators out, and the metrics to report.

Skill is a property of the operator: in JIGSAWS each one has a single skill level. Splitting by
trial would let a model recognise the operator's style and read off the label, so the headline
split here is leave-one-user-out (LOUO). Leave-one-supertrial-out, common in the literature, is
kept so that the size of that leak can be measured instead of assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import LeaveOneGroupOut

from training_eval.models import SKILL_ORDER, make_classifier, make_regressor

TARGETS = ("skill", "novice_vs_rest")
SPLITS = {"louo": "subject", "supertrial": "repetition"}


def labels_for(table: pd.DataFrame, target: str) -> pd.Series:
    """The classification labels for ``target``: three skill levels, or novice against the rest."""
    if target == "skill":
        return table["skill_level"]
    if target == "novice_vs_rest":
        return table["skill_level"].map(lambda s: "novice" if s == "novice" else "trained")
    raise ValueError(f"unknown target {target!r}; have {TARGETS}")


def _splits(table: pd.DataFrame, split: str):
    return LeaveOneGroupOut().split(table, groups=table[SPLITS[split]])


@dataclass
class ClassifierResult:
    model: str
    target: str
    split: str
    features: list[str]
    classes: list[str]
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    operators_correct: int
    operators_total: int
    confusion: list[list[int]]
    out_of_fold: pd.DataFrame = field(repr=False)
    p_value: float | None = None
    n_permutations: int = 0

    def summary(self) -> dict:
        out = {k: v for k, v in self.__dict__.items() if k not in ("out_of_fold", "features")}
        out["n_features"] = len(self.features)
        return out


def _class_order(values) -> list[str]:
    present = set(values)
    known = [s for s in (*SKILL_ORDER, "trained") if s in present]
    return known + sorted(present - set(known))


def out_of_fold_probabilities(
    table: pd.DataFrame, features: list[str], labels: pd.Series, model: str, split: str
) -> tuple[np.ndarray, list[str]]:
    """Each trial's class probabilities from a model that never saw its held-out group."""
    classes = _class_order(labels)
    proba = np.zeros((len(table), len(classes)))
    X = table[["task", *features]]
    for train, test in _splits(table, split):
        fitted = make_classifier(model).fit(X.iloc[train], labels.iloc[train])
        for j, cls in enumerate(fitted.classes_):
            proba[test, classes.index(cls)] = fitted.predict_proba(X.iloc[test])[:, j]
    return proba, classes


def _operator_predictions(oof: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """An operator's skill: their trials' probabilities averaged, then the most likely class."""
    means = oof.groupby("subject")[[f"p_{c}" for c in classes]].mean()
    out = pd.DataFrame(
        {
            "true": oof.groupby("subject")["true"].first(),
            "predicted": means.to_numpy().argmax(axis=1),
            "n_trials": oof.groupby("subject").size(),
        }
    )
    out["predicted"] = [classes[i] for i in out["predicted"]]
    return out.join(means)


def evaluate_classifier(
    table: pd.DataFrame,
    features: list[str],
    model: str = "logreg",
    target: str = "skill",
    split: str = "louo",
) -> ClassifierResult:
    """Cross-validate one classifier.

    Args:
        table: The feature table.
        features: Columns to use.
        model: A name from :data:`training_eval.models.CLASSIFIERS`.
        target: A name from :data:`TARGETS`.
        split: ``louo`` or ``supertrial``.
    """
    labels = labels_for(table, target)
    proba, classes = out_of_fold_probabilities(table, features, labels, model, split)
    predicted = np.array(classes)[proba.argmax(axis=1)]
    oof = table[["trial_id", "task", "subject"]].copy()
    oof["true"] = labels.to_numpy()
    oof["predicted"] = predicted
    for j, cls in enumerate(classes):
        oof[f"p_{cls}"] = proba[:, j]
    operators = _operator_predictions(oof, classes)
    return ClassifierResult(
        model=model,
        target=target,
        split=split,
        features=list(features),
        classes=classes,
        accuracy=float(accuracy_score(labels, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
        macro_f1=float(f1_score(labels, predicted, average="macro", labels=classes)),
        operators_correct=int((operators["true"] == operators["predicted"]).sum()),
        operators_total=len(operators),
        confusion=confusion_matrix(labels, predicted, labels=classes).tolist(),
        out_of_fold=oof,
    )


def permutation_test(
    table: pd.DataFrame,
    features: list[str],
    observed: float,
    model: str = "logreg",
    target: str = "skill",
    n_permutations: int = 200,
    seed: int = 0,
) -> float:
    """p-value for balanced accuracy under shuffled operator labels (LOUO).

    Shuffles the operator-to-label mapping, not trials: labels belong to operators, and with only
    8 of them the chance of a good score is much higher than trial counts suggest.
    ``sklearn.model_selection.permutation_test_score`` permutes *within* groups, which would change
    nothing here, because all of an operator's trials share one label.
    """
    rng = np.random.default_rng(seed)
    labels = labels_for(table, target)
    per_subject = labels.groupby(table["subject"]).first()
    hits = 0
    for _ in range(n_permutations):
        mapping = dict(zip(per_subject.index, rng.permutation(per_subject.to_numpy()), strict=True))
        shuffled = table["subject"].map(mapping)
        proba, classes = out_of_fold_probabilities(table, features, shuffled, model, "louo")
        predicted = np.array(classes)[proba.argmax(axis=1)]
        hits += balanced_accuracy_score(shuffled, predicted) >= observed
    return (hits + 1) / (n_permutations + 1)


def evaluate_regressor(
    table: pd.DataFrame, features: list[str], model: str = "ridge", split: str = "louo"
) -> dict:
    """Cross-validate a GRS regression; unlike skill level, GRS varies between one's own trials."""
    known = table[table["grs_total"].notna()].reset_index(drop=True)
    X = known[["task", *features]]
    y = known["grs_total"].astype(float)
    predicted = np.zeros(len(known))
    for train, test in _splits(known, split):
        predicted[test] = (
            make_regressor(model).fit(X.iloc[train], y.iloc[train]).predict(X.iloc[test])
        )
    # A leave-one-out mean is lower exactly when the held-out value is higher, so ranking it would
    # show a strong and meaningless negative correlation.
    return {
        "model": model,
        "split": split,
        "mae": float(mean_absolute_error(y, predicted)),
        "r2": float(r2_score(y, predicted)),
        "spearman": None if model == "mean" else float(spearmanr(y, predicted).statistic),
    }
