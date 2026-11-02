# rtdaas-training-eval

Predicts a surgical operator's skill from their da Vinci kinematics, using the JIGSAWS data served
by [robot-telemetry-daas](https://github.com/tom-f-oconnell/robot-telemetry-daas). It holds every
operator out of training when scoring them, and reports how far to trust the result. It is a
research tool, not an assessment of anyone: the dataset has only 8 operators.

## Use

You need Python 3.11 or newer, and robot-telemetry-daas running with JIGSAWS loaded.

```bash
python3 -m venv venv
venv/bin/pip install -e .

venv/bin/rtdaas-training-eval features    # download the trials (about 5 minutes, 120 MB)
venv/bin/rtdaas-training-eval evaluate    # write reports/evaluation.md (about 3 minutes)
```

To score an operator, train a model without them and predict:

```bash
venv/bin/rtdaas-training-eval train --exclude-subject D --out models/without_D.joblib
venv/bin/rtdaas-training-eval predict --model models/without_D.joblib --subject D
```

`predict` refuses an operator the model trained on, because that would not be a prediction. For the
operators in the dataset, `reports/out_of_fold_predictions.csv` has each trial's held-out score.

Keep `data/`, `models/` and `reports/` to yourself, as git does here: they derive from JIGSAWS, whose
licence restricts use to academic research and forbids redistribution.

## Options

The API defaults to `http://localhost:29398`. Set `RTDAAS_API_URL` or pass `--api` to change it.

| Command | Useful options |
|---|---|
| `features` | `--task Suturing` (repeatable), `--cache DIR`, `--no-cache`, `--out FILE` |
| `evaluate` | `--permutations N`, `--features FILE`, `--out DIR` |
| `train` | `--model logreg\|forest`, `--target skill\|novice_vs_rest`, `--groups time,path,motion,bimanual,gesture,video`, `--exclude-subject X` (repeatable) |
| `predict` | `--trial ID` (repeatable) or `--subject X`, `--allow-known-operator` |

## Development

```bash
python3 -m venv venv
venv/bin/pip install -e ".[dev]"
```

[design.md](design.md) explains the choice of split and features, and what the results on the real
data do and do not show.

```bash
venv/bin/pytest
venv/bin/ruff check . && venv/bin/ruff format --check .
RTDAAS_API_URL=http://localhost:29398 venv/bin/pytest -m integration   # against a running API
```
