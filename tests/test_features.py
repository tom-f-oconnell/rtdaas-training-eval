from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training_eval import features as f

FS = f.SAMPLE_RATE_HZ


def _line(speed=0.01, seconds=20.0):
    t = np.arange(0, seconds, 1 / FS)
    return np.column_stack([speed * t, np.zeros_like(t), np.zeros_like(t)])


def test_path_length_and_speed_of_a_straight_line():
    assert f.path_length(_line(0.01, 20.0)) == pytest.approx(0.2, rel=0.02)
    assert np.median(f.speed_profile(_line(0.02))) == pytest.approx(0.02, rel=0.01)


def test_single_sample_position_spikes_do_not_count_as_motion():
    clean = _line(0.01, 20.0)
    spiked = clean.copy()
    spiked[100, 0] += 0.5  # 50 cm in one sample, as a real recording can have
    assert f.path_length(spiked) == pytest.approx(f.path_length(clean), rel=0.01)


def test_jerk_ranks_a_jittery_movement_as_less_smooth():
    t = np.arange(0, 30, 1 / FS)
    smooth = np.column_stack(
        [0.02 * np.sin(2 * np.pi * 0.3 * t), np.zeros_like(t), np.zeros_like(t)]
    )
    rng = np.random.default_rng(0)
    jittery = smooth + rng.normal(0, 0.0008, smooth.shape)
    assert f.log_jerk_rms(jittery) > f.log_jerk_rms(smooth) + 1
    assert f.log_dimensionless_jerk(jittery) < f.log_dimensionless_jerk(smooth) - 1


def test_dimensionless_jerk_does_not_grow_with_duration():
    t = np.arange(0, 120, 1 / FS)
    p = np.column_stack([0.02 * np.sin(2 * np.pi * 0.3 * t), np.zeros_like(t), np.zeros_like(t)])
    short, long = f.log_dimensionless_jerk(p[: int(30 * FS)]), f.log_dimensionless_jerk(p)
    assert long == pytest.approx(short, abs=0.2)


def test_dimensionless_jerk_ignores_a_still_tool():
    assert np.isnan(f.log_dimensionless_jerk(np.zeros((300, 3))))


def test_grip_changes_uses_each_signals_own_range():
    t = np.arange(0, 20, 1 / FS)
    square = np.sign(
        np.sin(2 * np.pi * 0.25 * t)
    )  # 5 periods: 10 transitions, less the first state
    for low, high in ((-1.2, 0.0), (-0.5, 0.5), (0.0, 1.0)):
        scaled = low + (square + 1) / 2 * (high - low)
        assert f.grip_changes(scaled) == pytest.approx(9, abs=1)
    assert f.grip_changes(np.full_like(t, 0.3)) == 0


def test_grip_changes_ignores_noise_within_a_level():
    rng = np.random.default_rng(1)
    noisy_closed = -1.0 + rng.normal(0, 0.01, 300)
    signal = np.concatenate([noisy_closed, noisy_closed + 1.0])
    assert f.grip_changes(signal) == 1


def test_tremor_fraction_finds_a_tremor():
    t = np.arange(0, 30, 1 / FS)
    slow = 0.02 + 0.01 * np.sin(2 * np.pi * 0.5 * t)
    assert f.tremor_fraction(slow + 0.005 * np.sin(2 * np.pi * 9 * t)) > 0.15  # 20 % of the power
    assert f.tremor_fraction(slow) < 0.01


def _kinematics(n=900, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    data = {"frame_idx": np.arange(n), "t_s": t, "gesture": ["G1"] * (n - 90) + [None] * 90}
    for i, arm in enumerate(("psml", "psmr", "mtml", "mtmr")):
        scale = 3.0 if arm.startswith("mtm") else 1.0
        for j, axis in enumerate("xyz"):
            data[f"{arm}_{axis}"] = scale * 0.02 * np.sin(2 * np.pi * (0.2 + 0.05 * j) * t + i)
        for axis in ("wx", "wy", "wz"):
            data[f"{arm}_{axis}"] = rng.normal(0, 0.1, n)
        data[f"{arm}_gripper"] = np.sign(np.sin(2 * np.pi * 0.2 * t + i))
    return pd.DataFrame(data)


def test_a_trial_row_has_every_feature_and_no_label_leaks_into_features():
    meta = {"trial_id": "Suturing_B001", "task": "Suturing", "subject": "B", "repetition": 1,
            "skill_level": "novice", "grs_total": 13}  # fmt: skip
    gestures = [
        {"gesture": "G1", "duration_s": 10.0},
        {"gesture": "G2", "duration_s": 20.0},
    ]
    frames = pd.DataFrame({"motion_energy": [0.1, 0.2], "sharpness": [100.0, 200.0]})
    row = f.trial_row(meta, _kinematics(), gestures, frames)
    for name in f.feature_columns(list(f.FEATURE_GROUPS)):
        assert np.isfinite(row[name]), name
    assert row["log_duration_s"] == pytest.approx(np.log(30.0))
    assert row["gestures_per_min"] == pytest.approx(4.0)
    assert row["unlabelled_frac"] == pytest.approx(0.1)
    assert row["motion_scale"] == pytest.approx(3.0, rel=0.05)
    assert not {"skill_level", "grs_total", "subject"} & set(
        f.feature_columns(list(f.FEATURE_GROUPS))
    )


def test_missing_gestures_and_video_are_nan_not_an_error():
    meta = {"trial_id": "x", "task": "t", "subject": "B", "repetition": 1, "skill_level": "novice"}
    row = f.trial_row(meta, _kinematics(), [], pd.DataFrame())
    assert np.isnan(row["gestures_per_min"]) and np.isnan(row["video_motion_mean"])


def test_unknown_feature_group_is_rejected():
    with pytest.raises(ValueError, match="unknown feature groups"):
        f.feature_columns(["nope"])
