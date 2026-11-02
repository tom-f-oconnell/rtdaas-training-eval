from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training_eval.evaluate import (
    evaluate_classifier,
    evaluate_regressor,
    labels_for,
    out_of_fold_probabilities,
    permutation_test,
)
from training_eval.features import FEATURE_GROUPS, feature_columns
from training_eval.models import TaskScaler, make_classifier
from training_eval.predict import OperatorInTrainingData, predict_rows, top_reasons, train
from training_eval.report import run_evaluation, write_report

SUBJECTS = {"B": "novice", "G": "novice", "H": "novice", "I": "novice",
            "C": "intermediate", "F": "intermediate", "D": "expert", "E": "expert"}  # fmt: skip
EFFECT = {"novice": -1.0, "intermediate": 0.0, "expert": 1.0}
FEATURES = ["signal", "noise1", "noise2"]


def _table(strength=2.0, seed=0, per_subject=5):
    """Trials with the JIGSAWS shape: each operator has one label and several trials per task."""
    rng = np.random.default_rng(seed)
    rows = []
    for subject, skill in SUBJECTS.items():
        style = rng.normal(0, 1)  # what a leaky split can recognise an operator by
        for task, offset in (("Suturing", 10.0), ("Knot_Tying", -5.0)):
            for rep in range(1, per_subject + 1):
                rows.append(
                    {
                        "trial_id": f"{task}_{subject}{rep:03d}",
                        "task": task,
                        "subject": subject,
                        "repetition": rep,
                        "skill_level": skill,
                        "grs_total": 15 + 5 * EFFECT[skill] + rng.normal(0, 1),
                        "signal": offset + strength * EFFECT[skill] + rng.normal(0, 0.3),
                        "noise1": style + rng.normal(0, 0.1),
                        "noise2": rng.normal(0, 1),
                    }
                )
    return pd.DataFrame(rows)


def test_a_planted_signal_is_found_holding_operators_out():
    results = [evaluate_classifier(_table(seed=seed), FEATURES) for seed in range(5)]
    assert np.mean([r.balanced_accuracy for r in results]) > 0.7
    for result in results:
        novice, expert = result.classes.index("novice"), result.classes.index("expert")
        assert result.confusion[novice][expert] == 0 and result.confusion[expert][novice] == 0


def test_no_signal_scores_near_chance_holding_operators_out():
    scores = [
        evaluate_classifier(_table(strength=0.0, seed=s), FEATURES).balanced_accuracy
        for s in range(5)
    ]
    assert np.mean(scores) < 0.5  # 3 classes: chance is 0.33


def test_louo_never_trains_on_the_held_out_operator(monkeypatch):
    table = _table()
    seen = []
    real = make_classifier

    def spy(name):
        pipeline = real(name)
        original = pipeline.fit

        def fit(X, y):
            seen.append(set(table.loc[X.index, "subject"]))
            return original(X, y)

        pipeline.fit = fit
        return pipeline

    monkeypatch.setattr("training_eval.evaluate.make_classifier", spy)
    out_of_fold_probabilities(table, FEATURES, labels_for(table, "skill"), "logreg", "louo")
    assert len(seen) == 8
    assert all(len(s) == 7 for s in seen)
    assert {next(iter(set(SUBJECTS) - s)) for s in seen} == set(SUBJECTS)


def test_holding_out_a_repetition_leaks_the_operators_style():
    """The same operator stays in training, so a pure identity feature predicts their label."""
    leaky, honest = [], []
    for seed in range(4):
        table = _table(strength=0.0, seed=seed)
        leaky.append(evaluate_classifier(table, ["noise1"], "forest", split="supertrial"))
        honest.append(evaluate_classifier(table, ["noise1"], "forest", split="louo"))
    leaky_score = np.mean([r.balanced_accuracy for r in leaky])
    assert leaky_score > np.mean([r.balanced_accuracy for r in honest]) + 0.2


def test_task_scaler_removes_task_offset_using_training_statistics_only():
    table = _table()
    X = table[["task", *FEATURES]]
    scaled = TaskScaler().fit(X).transform(X)
    assert abs(scaled[:, 0][(table["task"] == "Suturing").to_numpy()].mean()) < 1e-9
    assert abs(scaled[:, 0][(table["task"] == "Knot_Tying").to_numpy()].mean()) < 1e-9
    fitted = TaskScaler().fit(X[table["subject"] != "B"])
    b_suturing = X[(table["subject"] == "B") & (table["task"] == "Suturing")]
    stats = table[(table["subject"] != "B") & (table["task"] == "Suturing")]["signal"]
    expected = (b_suturing["signal"].iloc[0] - stats.mean()) / stats.std()
    assert fitted.transform(b_suturing)[0, 0] == pytest.approx(expected)


def test_task_scaler_handles_an_unseen_task_and_a_constant_feature():
    table = _table()
    table["constant"] = 1.0
    X = table[["task", "signal", "constant"]]
    scaler = TaskScaler().fit(X[table["task"] == "Suturing"])
    out = scaler.transform(X[table["task"] == "Knot_Tying"])
    assert np.isfinite(out).all()


def test_permutation_test_separates_real_signal_from_none():
    real = _table()
    observed = evaluate_classifier(real, FEATURES).balanced_accuracy
    assert permutation_test(real, FEATURES, observed, n_permutations=30) < 0.2
    fake = _table(strength=0.0, seed=1)
    observed = evaluate_classifier(fake, FEATURES).balanced_accuracy
    assert permutation_test(fake, FEATURES, observed, n_permutations=30) > 0.2


def test_regression_beats_the_mean_on_a_planted_signal():
    table = _table()
    ridge, mean = evaluate_regressor(table, FEATURES), evaluate_regressor(table, FEATURES, "mean")
    assert ridge["mae"] < mean["mae"] and ridge["spearman"] > 0.7
    assert mean["spearman"] is None


def test_novice_vs_rest_labels():
    assert set(labels_for(_table(), "novice_vs_rest")) == {"novice", "trained"}
    with pytest.raises(ValueError):
        labels_for(_table(), "nope")


def _with_real_feature_names(table):
    """Every real feature column as noise, except that duration carries the planted signal."""
    columns = feature_columns(list(FEATURE_GROUPS))
    rng = np.random.default_rng(0)
    extra = {c: 0.1 * rng.normal(0, 1, len(table)) for c in columns}
    extra["log_duration_s"] = table["signal"].to_numpy()
    return table.assign(**extra)


def test_predict_refuses_an_operator_the_model_trained_on():
    table = _with_real_feature_names(_table())
    bundle = train(table)
    with pytest.raises(OperatorInTrainingData, match="training data"):
        predict_rows(bundle, table[table["subject"] == "D"])
    assert len(predict_rows(bundle, table[table["subject"] == "D"], allow_known=True)) == 10


def test_predict_a_held_out_operator_with_reasons():
    table = _with_real_feature_names(_table())
    bundle = train(table, exclude_subjects=["D"])
    assert "D" not in bundle.subjects
    rows = table[table["subject"] == "D"]
    predicted = predict_rows(bundle, rows)
    assert predicted["p_expert"].mean() > predicted["p_novice"].mean()
    reasons = top_reasons(bundle, rows)
    assert reasons[0]["contribution"] >= reasons[-1]["contribution"]
    assert "log_duration_s" in [r["feature"] for r in reasons]


def test_model_bundle_round_trip(tmp_path):
    table = _with_real_feature_names(_table())
    bundle = train(table, exclude_subjects=["D"])
    path = tmp_path / "models" / "m.joblib"
    bundle.save(path)
    loaded = type(bundle).load(path)
    rows = table[table["subject"] == "D"]
    pd.testing.assert_frame_equal(predict_rows(loaded, rows), predict_rows(bundle, rows))


def test_full_report_runs_and_writes_its_files(tmp_path):
    table = _with_real_feature_names(_table())
    results = run_evaluation(table, n_permutations=3)
    report = write_report(results, tmp_path)
    text = report.read_text()
    assert "Skill level, three classes" in text and "| majority |" in text
    assert (tmp_path / "evaluation.json").exists()
    assert len(pd.read_csv(tmp_path / "out_of_fold_predictions.csv")) == len(table)
