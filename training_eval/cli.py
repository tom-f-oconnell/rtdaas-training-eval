"""Command line: ``rtdaas-training-eval features|evaluate|train|predict``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from training_eval.client import ApiClient
from training_eval.dataset import build_table, trial_features
from training_eval.evaluate import TARGETS
from training_eval.features import FEATURE_GROUPS
from training_eval.models import CLASSIFIERS
from training_eval.predict import (
    ModelBundle,
    OperatorInTrainingData,
    predict_rows,
    top_reasons,
    train,
)
from training_eval.report import run_evaluation, write_report

DEFAULT_TABLE = Path("data/features.csv")
DEFAULT_CACHE = Path("data/raw")
DEFAULT_MODEL = Path("models/skill.joblib")


def _cmd_features(args) -> int:
    client = ApiClient(args.api)
    cache = None if args.no_cache else args.cache
    table = build_table(client, cache_dir=cache, tasks=args.task)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"wrote {len(table)} trials from {table['subject'].nunique()} operators to {args.out}")
    return 0


def _cmd_evaluate(args) -> int:
    table = pd.read_csv(args.features)
    results = run_evaluation(table, n_permutations=args.permutations)
    report = write_report(results, args.out)
    print(report.read_text())
    print(f"wrote {report}")
    return 0


def _cmd_train(args) -> int:
    table = pd.read_csv(args.features)
    groups = args.groups.split(",") if args.groups else None
    bundle = train(table, groups, args.model, args.target, args.exclude_subject)
    bundle.save(args.out)
    print(f"trained on {bundle.n_trials} trials of operators {bundle.subjects}; saved {args.out}")
    return 0


def _cmd_predict(args) -> int:
    client = ApiClient(args.api)
    bundle = ModelBundle.load(args.model)
    trials = client.trials()
    if args.trial:
        chosen = [t for t in trials if t["trial_id"] in set(args.trial)]
        missing = set(args.trial) - {t["trial_id"] for t in chosen}
        if missing:
            print(f"unknown trials: {sorted(missing)}", file=sys.stderr)
            return 1
    else:
        chosen = [t for t in trials if t["subject"] == args.subject]
        if not chosen:
            print(f"operator {args.subject!r} has no trials", file=sys.stderr)
            return 1
    cache = None if args.no_cache else args.cache
    rows = pd.DataFrame([trial_features(client, meta, cache) for meta in chosen])
    try:
        predictions = predict_rows(bundle, rows, allow_known=args.allow_known_operator)
    except OperatorInTrainingData as exc:
        print(exc, file=sys.stderr)
        return 1
    print(predictions.round(3).to_string(index=False))
    probabilities = [c for c in predictions.columns if c.startswith("p_")]
    mean = predictions[probabilities].mean()
    print(f"\nOverall: {mean.idxmax()[2:]} ({mean.max():.0%}) across {len(rows)} trial(s)")
    reasons = top_reasons(bundle, rows)
    if reasons:
        print(
            "\nWhat pushed it there (z = standard deviations from the training mean for the task):"
        )
        for r in reasons:
            print(f"  {r['feature']:<28} z={r['z']:+.2f}  log-odds {r['contribution']:+.3f}")
    print("\nA research estimate from 8 operators, not an assessment of a person.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rtdaas-training-eval", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("features", help="download trials from the API and build the feature table")
    p.add_argument("--api", help="API root; default $RTDAAS_API_URL, then http://localhost:29398")
    p.add_argument("--out", type=Path, default=DEFAULT_TABLE)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--task", action="append", help="only this task (repeatable)")
    p.set_defaults(func=_cmd_features)

    p = sub.add_parser("evaluate", help="cross-validate, holding whole operators out")
    p.add_argument("--features", type=Path, default=DEFAULT_TABLE)
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--permutations", type=int, default=200)
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("train", help="fit a final model")
    p.add_argument("--features", type=Path, default=DEFAULT_TABLE)
    p.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--model", choices=CLASSIFIERS, default="logreg")
    p.add_argument("--target", choices=TARGETS, default="skill")
    p.add_argument("--groups", help=f"comma-separated, from {sorted(FEATURE_GROUPS)}")
    p.add_argument("--exclude-subject", action="append", help="hold this operator out (repeatable)")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("predict", help="score an operator's trials from the API")
    p.add_argument("--api")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--allow-known-operator", action="store_true", help="score even if trained on")
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--trial", action="append", help="trial id (repeatable)")
    who.add_argument("--subject", help="operator letter: all of their trials")
    p.set_defaults(func=_cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
