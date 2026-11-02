# Design

## Holding out whole operators

JIGSAWS gives each operator one skill level, and has 8 operators. A model that sees any of an
operator's trials in training can recognise their style and read off the label, so scoring it on
their other trials measures identity, not skill. The headline split is therefore leave-one-user-out
(LOUO). Leave-one-supertrial-out is kept only to measure that leak.

The effective sample size is 8, not 103. Two consequences:

- Models are small and strongly regularised, and were not tuned against the LOUO score. Tuning on
  it would make the score optimistic; the one regularisation check used synthetic tables only.
- The permutation test shuffles labels across operators. scikit-learn's `permutation_test_score`
  shuffles within groups, which does nothing here, since an operator's trials share a label.

## Features

Per-trial, from the API's kinematics at 30 Hz:

| Group | What | Why it is separate |
|---|---|---|
| `time` | log duration | Experts tend to be faster. A model must beat this to be more than a stopwatch |
| `path` | total tool and master path | Scales with duration |
| `motion` | speed, idle time, jerk, workspace volume, grip rate, tremor band | Rates, independent of duration |
| `bimanual` | left/right balance and correlation, masters moving while tools do not (mostly clutching) | |
| `gesture` | gesture rate and durations | Needs hand annotation, which a live system would not have |
| `video` | motion energy, sharpness | May reflect the recording setup rather than the operator |

The default set is the four kinematic groups. Details that matter on the real data:

- Real recordings have single-sample position jumps (0.7 m/s on `Suturing_D001`), so positions pass a
  median filter before differentiation.
- Whole-trial dimensionless jerk grows with duration, so it is averaged over 5 s windows.
- Gripper values differ by arm and trial, so grip changes use hysteresis relative to each signal.
- `TaskScaler` standardises within each task, on training folds only. Tasks differ more in duration
  than skill levels do.

## Results on the real data (2026-11-02, 103 trials, 8 operators)

From `evaluate` with 500 permutations, holding each operator out:

| Question | Result | Chance or baseline |
|---|---|---|
| Three skill levels, balanced accuracy | 0.56 (p = 0.02) | 0.33 |
| Three skill levels, operators correct | 5 of 8 | 4 of 8, always saying novice |
| Novice against trained, balanced accuracy | 0.71 (p = 0.09) | 0.50 |
| Novice against trained, operators correct | 7 of 8 | |
| GRS score, mean absolute error | 4.3 points (Spearman 0.43) | 4.9, predicting the mean |
| Same three-class model, holding out only a repetition | 0.85, 8 of 8 | |

What this does and does not show:

- **There is some real signal, and it is modest.** The novice-against-trained result does not reach
  p < 0.05.
- **It does not recognise experts.** Only 3 of 29 expert trials were called expert: neither expert was
  labelled correctly, and one was called a novice (76 %). The training data has one other expert to
  learn from each time.
- **Intermediates look too good.** 25 of 28 were right, from one other intermediate operator to learn
  from. That may be a real style or something the two share about how they were recorded. Eight
  operators cannot separate those.
- **It does not clearly beat simple measures.** Duration and path alone scored 0.60 against 0.56 for
  every kinematic feature; 8 operators cannot resolve a gap that small.
- **Holding out a repetition inflates the score from 0.56 to 0.85.** Any published or future result
  should be read with its split in mind.

Skill level is self-reported experience, not measured ability, which is one reason GRS, an expert
rating of each trial, is reported too.

## Not done

- Per-gesture or sequence models. They need more operators to be evaluated, not more parameters.
- Video models. The stored features are per-frame summaries; `video` scored no better than chance
  would suggest on its own (0.38 against 0.33).
- Coaching feedback. `predict` lists which features moved the score, but those are correlations from
  8 people, not advice.
