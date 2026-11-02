"""Per-trial features from one trial's kinematics, gesture segments and video features.

Everything here is a pure function of its inputs. Features are grouped so that an evaluation can
ask what each kind of evidence adds, above all whether a model is more than a stopwatch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter, welch

SAMPLE_RATE_HZ = 30.0
IDLE_SPEED_M_S = {"psm": 0.005, "mtm": 0.005}
TREMOR_BAND_HZ = (6.0, 12.0)
SMOOTH_WINDOW = 15  # 0.5 s, comfortably shorter than a deliberate movement
LDLJ_WINDOW_S = 5.0

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "time": ("log_duration_s",),
    "path": ("psm_path_m", "mtm_path_m"),
    "motion": (
        "psm_speed_mean",
        "psm_speed_p95",
        "psm_idle_frac",
        "psm_log_jerk",
        "psm_ldlj",
        "psm_log_volume",
        "psm_angular_speed",
        "psm_grip_changes_per_min",
        "mtm_speed_mean",
        "mtm_speed_p95",
        "mtm_idle_frac",
        "mtm_log_jerk",
        "mtm_ldlj",
        "mtm_log_volume",
        "mtm_tremor_frac",
    ),
    "bimanual": (
        "psm_path_balance",
        "psm_both_moving_frac",
        "psm_speed_corr",
        "mtm_only_frac",
        "motion_scale",
    ),
    "gesture": (
        "gestures_per_min",
        "gesture_mean_s",
        "gesture_cv",
        "gesture_distinct",
        "unlabelled_frac",
    ),
    "video": ("video_motion_mean", "video_motion_std", "video_sharpness_log"),
}
DEFAULT_GROUPS: tuple[str, ...] = ("time", "path", "motion", "bimanual")


def feature_columns(groups: tuple[str, ...] | list[str] = DEFAULT_GROUPS) -> list[str]:
    """The feature names in the given groups, in a stable order."""
    unknown = sorted(set(groups) - set(FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"unknown feature groups {unknown}; have {sorted(FEATURE_GROUPS)}")
    return [name for group in groups for name in FEATURE_GROUPS[group]]


def _position(df: pd.DataFrame, arm: str) -> np.ndarray:
    return df[[f"{arm}_x", f"{arm}_y", f"{arm}_z"]].to_numpy(dtype=np.float64)


def _despike(p: np.ndarray) -> np.ndarray:
    """Remove single-sample position jumps, which real recordings have and jerk would amplify."""
    return np.column_stack([medfilt(p[:, axis], 5) for axis in range(p.shape[1])])


def _derivative(p: np.ndarray, order: int) -> np.ndarray:
    window = min(SMOOTH_WINDOW, len(p) - (1 - len(p) % 2))
    return savgol_filter(
        p, window, polyorder=max(3, order + 1), deriv=order, delta=1 / SAMPLE_RATE_HZ, axis=0
    )


def speed_profile(p: np.ndarray) -> np.ndarray:
    """Tool speed in m/s from positions (n, 3), smoothed."""
    return np.linalg.norm(_derivative(_despike(p), 1), axis=1)


def path_length(p: np.ndarray) -> float:
    """Path length in metres, integrated from the smoothed speed so sensor jitter does not count."""
    return float(speed_profile(p).sum() / SAMPLE_RATE_HZ)


def log_jerk_rms(p: np.ndarray) -> float:
    """Log RMS jerk: how abruptly the tool's acceleration changes, regardless of duration."""
    jerk = np.linalg.norm(_derivative(_despike(p), 3), axis=1)
    return float(np.log(np.sqrt(np.mean(jerk**2)) + 1e-9))


def log_dimensionless_jerk(p: np.ndarray, window_s: float = LDLJ_WINDOW_S) -> float:
    """Mean log dimensionless jerk over fixed windows; higher (less negative) is smoother.

    Dimensionless jerk grows with a movement's duration, so a whole-trial value would mostly
    measure time. Averaging over fixed windows keeps it about smoothness.
    """
    n = int(window_s * SAMPLE_RATE_HZ)
    jerk = np.linalg.norm(_derivative(_despike(p), 3), axis=1)
    speed = np.linalg.norm(_derivative(_despike(p), 1), axis=1)
    values = []
    for start in range(0, max(1, len(p) - n + 1), n):
        v_peak = speed[start : start + n].max()
        if v_peak < 1e-3:  # nothing moved, so smoothness says nothing
            continue
        integral = np.sum(jerk[start : start + n] ** 2) / SAMPLE_RATE_HZ
        values.append(-np.log(window_s**3 / v_peak**2 * integral + 1e-12))
    return float(np.mean(values)) if values else float("nan")


def log_motion_volume(p: np.ndarray) -> float:
    """Log volume (m^3) of the box holding the central 90 % of positions on each axis."""
    extent = np.percentile(p, 95, axis=0) - np.percentile(p, 5, axis=0)
    return float(np.log(np.prod(extent) + 1e-12))


def grip_changes(gripper: np.ndarray) -> int:
    """Open and close transitions, with hysteresis relative to this signal's own range.

    Relative because what closed and open read varies by arm and trial on the real data.
    """
    low, high = np.percentile(gripper, 5), np.percentile(gripper, 95)
    if high - low < 1e-3:
        return 0
    lo_th, hi_th = low + 0.3 * (high - low), low + 0.7 * (high - low)
    state, changes = None, 0
    for value in gripper:
        new = "open" if value > hi_th else "closed" if value < lo_th else state
        if state is not None and new != state:
            changes += 1
        state = new
    return changes


def tremor_fraction(speed: np.ndarray) -> float:
    """Fraction of the speed signal's power in the band of physiological hand tremor."""
    if len(speed) < 64:
        return float("nan")
    freqs, power = welch(speed - speed.mean(), fs=SAMPLE_RATE_HZ, nperseg=64)
    band = (freqs >= TREMOR_BAND_HZ[0]) & (freqs <= TREMOR_BAND_HZ[1])
    return float(power[band].sum() / (power.sum() + 1e-18))


def _mean_of_sides(left: float, right: float) -> float:
    return float(np.nanmean([left, right]))


def _pair_features(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    """Features for one pair of arms, averaged over the two sides."""
    per_side: list[dict[str, float]] = []
    for side in ("l", "r"):
        arm = f"{prefix}{side}"
        p = _position(df, arm)
        speed = speed_profile(p)
        duration_min = len(p) / SAMPLE_RATE_HZ / 60
        angular = df[[f"{arm}_wx", f"{arm}_wy", f"{arm}_wz"]].to_numpy(dtype=np.float64)
        per_side.append(
            {
                "path_m": float(speed.sum() / SAMPLE_RATE_HZ),
                "speed_mean": float(speed.mean()),
                "speed_p95": float(np.percentile(speed, 95)),
                "idle_frac": float(np.mean(speed < IDLE_SPEED_M_S[prefix])),
                "log_jerk": log_jerk_rms(p),
                "ldlj": log_dimensionless_jerk(p),
                "log_volume": log_motion_volume(p),
                "angular_speed": float(np.median(np.linalg.norm(angular, axis=1))),
                "grip_changes_per_min": grip_changes(df[f"{arm}_gripper"].to_numpy())
                / duration_min,
                "tremor_frac": tremor_fraction(speed),
            }
        )
    left, right = per_side
    out = {f"{prefix}_{k}": _mean_of_sides(left[k], right[k]) for k in left}
    out[f"{prefix}_path_m"] = left["path_m"] + right["path_m"]
    return out


def kinematic_features(df: pd.DataFrame) -> dict[str, float]:
    """The ``time``, ``path``, ``motion`` and ``bimanual`` features of one trial."""
    out: dict[str, float] = {"log_duration_s": float(np.log(len(df) / SAMPLE_RATE_HZ))}
    out.update(_pair_features(df, "psm"))
    out.update(_pair_features(df, "mtm"))

    speeds = {arm: speed_profile(_position(df, arm)) for arm in ("psml", "psmr", "mtml", "mtmr")}
    paths = [speeds["psml"].sum(), speeds["psmr"].sum()]
    moving_l = speeds["psml"] >= IDLE_SPEED_M_S["psm"]
    moving_r = speeds["psmr"] >= IDLE_SPEED_M_S["psm"]
    out["psm_path_balance"] = float(min(paths) / max(max(paths), 1e-9))
    out["psm_both_moving_frac"] = float(np.mean(moving_l & moving_r))
    still = speeds["psml"].std() < 1e-9 or speeds["psmr"].std() < 1e-9
    out["psm_speed_corr"] = (
        float("nan") if still else float(np.corrcoef(speeds["psml"], speeds["psmr"])[0, 1])
    )

    # A master moving while its tool does not: mostly clutching to reposition the hands.
    only = [
        (speeds[m] >= 2 * IDLE_SPEED_M_S["mtm"]) & ~(speeds[s] >= IDLE_SPEED_M_S["psm"])
        for m, s in (("mtml", "psml"), ("mtmr", "psmr"))
    ]
    out["mtm_only_frac"] = float(np.mean(only[0] | only[1]))
    out["motion_scale"] = float(out["mtm_path_m"] / max(out["psm_path_m"], 1e-9))
    return out


def gesture_features(gestures: list[dict], kinematics: pd.DataFrame) -> dict[str, float]:
    """The ``gesture`` features. They need hand annotation, which a live system would not have."""
    duration_min = len(kinematics) / SAMPLE_RATE_HZ / 60
    durations = np.array([g["duration_s"] for g in gestures], dtype=np.float64)
    labels = kinematics["gesture"] if "gesture" in kinematics else pd.Series(dtype=object)
    if len(durations) == 0:
        return dict.fromkeys(FEATURE_GROUPS["gesture"], float("nan"))
    return {
        "gestures_per_min": float(len(durations) / duration_min),
        "gesture_mean_s": float(durations.mean()),
        "gesture_cv": float(durations.std() / max(durations.mean(), 1e-9)),
        "gesture_distinct": float(len({g["gesture"] for g in gestures})),
        "unlabelled_frac": float(labels.isna().mean()) if len(labels) else float("nan"),
    }


def video_features(frames: pd.DataFrame) -> dict[str, float]:
    """The ``video`` features, which may reflect the recording setup rather than the operator."""
    if frames.empty:
        return dict.fromkeys(FEATURE_GROUPS["video"], float("nan"))
    return {
        "video_motion_mean": float(frames["motion_energy"].mean()),
        "video_motion_std": float(frames["motion_energy"].std()),
        "video_sharpness_log": float(np.log(frames["sharpness"].mean() + 1e-9)),
    }


def trial_row(
    meta: dict, kinematics: pd.DataFrame, gestures: list[dict], frames: pd.DataFrame
) -> dict:
    """One row of the feature table: labels and identifiers, then every feature."""
    row = {
        "trial_id": meta["trial_id"],
        "task": meta["task"],
        "subject": meta["subject"],
        "repetition": meta["repetition"],
        "skill_level": meta["skill_level"],
        "grs_total": meta.get("grs_total"),
    }
    row.update(kinematic_features(kinematics))
    row.update(gesture_features(gestures, kinematics))
    row.update(video_features(frames))
    return row
