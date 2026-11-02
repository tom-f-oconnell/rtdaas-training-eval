"""Run the full evaluation and write it as Markdown and JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from training_eval.evaluate import (
    ClassifierResult,
    _operator_predictions,
    evaluate_classifier,
    evaluate_regressor,
    permutation_test,
)
from training_eval.features import DEFAULT_GROUPS, feature_columns

ABLATIONS: dict[str, tuple[str, ...]] = {
    "duration only": ("time",),
    "duration + path": ("time", "path"),
    "motion quality (no duration or path)": ("motion", "bimanual"),
    "default (all kinematics)": DEFAULT_GROUPS,
    "gesture annotations only": ("gesture",),
    "video features only": ("video",),
    "kinematics + annotations + video": (*DEFAULT_GROUPS, "gesture", "video"),
}


def _row(label: str, r: ClassifierResult) -> dict:
    return {
        "features": label,
        "model": r.model,
        "split": r.split,
        "accuracy": round(r.accuracy, 3),
        "balanced_accuracy": round(r.balanced_accuracy, 3),
        "macro_f1": round(r.macro_f1, 3),
        "operators": f"{r.operators_correct}/{r.operators_total}",
    }


def _markdown_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(r[h]) for h in headers) + " |" for r in rows]
    return "\n".join(lines)


def run_evaluation(table: pd.DataFrame, n_permutations: int = 200) -> dict:
    """Every comparison the report needs, as plain data."""
    default = feature_columns()
    baseline = evaluate_classifier(table, default, "majority")
    main = evaluate_classifier(table, default, "logreg")
    main.p_value = permutation_test(
        table, default, main.balanced_accuracy, n_permutations=n_permutations
    )
    main.n_permutations = n_permutations
    binary = evaluate_classifier(table, default, "logreg", target="novice_vs_rest")
    binary.p_value = permutation_test(
        table,
        default,
        binary.balanced_accuracy,
        target="novice_vs_rest",
        n_permutations=n_permutations,
    )
    binary.n_permutations = n_permutations

    ablation = [
        _row(label, evaluate_classifier(table, feature_columns(groups), "logreg"))
        for label, groups in ABLATIONS.items()
    ]
    models = [
        _row("default", r) for r in (baseline, main, evaluate_classifier(table, default, "forest"))
    ]
    leak = [
        _row("default", main),
        _row("default", evaluate_classifier(table, default, "logreg", split="supertrial")),
    ]
    return {
        "n_trials": len(table),
        "n_operators": int(table["subject"].nunique()),
        "main": main,
        "binary": binary,
        "models": models,
        "ablation": ablation,
        "leak": leak,
        "grs": [
            evaluate_regressor(table, default, "mean"),
            evaluate_regressor(table, default, "ridge"),
        ],
        "operators": _operator_predictions(main.out_of_fold, main.classes),
        "binary_operators": _operator_predictions(binary.out_of_fold, binary.classes),
    }


def write_report(results: dict, out_dir: Path) -> Path:
    """Write ``evaluation.md``, ``evaluation.json`` and the out-of-fold predictions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    main: ClassifierResult = results["main"]
    binary: ClassifierResult = results["binary"]
    confusion = [
        {"true \\ predicted": main.classes[i], **dict(zip(main.classes, row, strict=True))}
        for i, row in enumerate(main.confusion)
    ]
    lines = [
        "# Operator skill evaluation",
        "",
        f"{results['n_trials']} trials from {results['n_operators']} operators. Scores hold each "
        "operator out of training unless a row says otherwise.",
        "",
        "## Skill level, three classes",
        "",
        _markdown_table(results["models"]),
        "",
        f"Permutation test on balanced accuracy, shuffling labels across operators: "
        f"p = {main.p_value:.3f} ({main.n_permutations} permutations).",
        "",
        _markdown_table(confusion),
        "",
        "### By operator",
        "",
        _markdown_table(results["operators"].round(2).reset_index().to_dict("records")),
        "",
        "## Novice against trained",
        "",
        _markdown_table([_row("default", binary)]),
        "",
        f"Permutation test: p = {binary.p_value:.3f} ({binary.n_permutations} permutations).",
        "",
        _markdown_table(results["binary_operators"].round(2).reset_index().to_dict("records")),
        "",
        "## What each kind of evidence adds",
        "",
        _markdown_table(results["ablation"]),
        "",
        "## How much holding out only a repetition inflates the score",
        "",
        _markdown_table(results["leak"]),
        "",
        "## GRS score (expert rating), regression",
        "",
        _markdown_table(results["grs"]),
        "",
    ]
    report = out_dir / "evaluation.md"
    report.write_text("\n".join(lines))
    (out_dir / "evaluation.json").write_text(
        json.dumps(
            {
                k: (v.summary() if isinstance(v, ClassifierResult) else v)
                for k, v in results.items()
                if k not in ("operators", "binary_operators")
            },
            indent=2,
        )
    )
    main.out_of_fold.to_csv(out_dir / "out_of_fold_predictions.csv", index=False)
    return report
