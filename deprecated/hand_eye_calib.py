"""Hand-eye calibration utilities (load/save + hand-authored fallback).

The canonical way to produce `hand_eye_calib.npz` is now
`calibrate_hand_eye.py`, which runs `cv2.calibrateHandEye` over ~25
captured poses. This module just holds the data format used by both
calibrate_hand_eye.py (writer) and pose.py (reader).

The `build_T_flange_camera()` helper is retained as a debugging fallback
for hand-authoring T_flange_camera from a ruler measurement plus a
rotation description (cases A/B/C/D/freeform).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

OUTPUT_FILE = "hand_eye_calib.npz"


# ---------------------------------------------------------------------------
# Save/load
# ---------------------------------------------------------------------------
def save_hand_eye_calib(T_flange_camera: np.ndarray, path: str = OUTPUT_FILE) -> None:
    if T_flange_camera.shape != (4, 4):
        raise ValueError(f"expected 4x4, got {T_flange_camera.shape}")
    np.savez(path, T_flange_camera=T_flange_camera.astype(np.float64))


def load_hand_eye_calib(path: str = OUTPUT_FILE) -> np.ndarray:
    data = np.load(path)
    T = data["T_flange_camera"].astype(np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"corrupt hand-eye file: expected 4x4, got {T.shape}")
    return T


# ---------------------------------------------------------------------------
# Optional hand-authored rotation cases (debugging / fallback)
# ---------------------------------------------------------------------------
def _case_a_aligned() -> np.ndarray:
    return np.eye(3)


def _case_b_back() -> np.ndarray:
    return np.diag([-1.0, 1.0, -1.0])


def _case_c_down_neg_z() -> np.ndarray:
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ])


def _case_d_tilt_x(theta_rad: float) -> np.ndarray:
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,  -s],
        [0.0,   s,   c],
    ])


def _case_freeform_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


ROTATION_CASES: dict[str, Callable[..., np.ndarray]] = {
    "A_aligned": _case_a_aligned,
    "B_back": _case_b_back,
    "C_down_neg_z": _case_c_down_neg_z,
    "D_tilt_x": _case_d_tilt_x,
    "freeform_rpy": _case_freeform_rpy,
}


def build_T_flange_camera(
    tx_mm: float,
    ty_mm: float,
    tz_mm: float,
    rotation_case: str,
    tilt_x_deg: float = 0.0,
    freeform_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Compose T_flange_camera from a ruler measurement + rotation description.

    Intended as a debugging fallback only. The AX=XB solver in
    `calibrate_hand_eye.py` is the preferred way to obtain this transform.
    """
    if rotation_case not in ROTATION_CASES:
        raise ValueError(
            f"unknown rotation_case {rotation_case!r}; "
            f"expected one of {sorted(ROTATION_CASES)}"
        )

    if rotation_case == "D_tilt_x":
        R = ROTATION_CASES[rotation_case](math.radians(tilt_x_deg))
    elif rotation_case == "freeform_rpy":
        r, p, y = (math.radians(a) for a in freeform_rpy_deg)
        R = ROTATION_CASES[rotation_case](r, p, y)
    else:
        R = ROTATION_CASES[rotation_case]()

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([tx_mm, ty_mm, tz_mm], dtype=np.float64) / 1000.0
    return T
