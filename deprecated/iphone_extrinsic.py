"""Per-session iPhone extrinsic solver.

Solves T_base_iphoneCam -- the fixed transform that puts the iPhone
overhead camera into the SO-101 base frame -- by averaging PnP results
over `n_frames` detections of a stationary reference AprilTag whose
base-frame pose `T_base_tag_ref` is known (measured with a ruler and tape).

Called once at startup by pose.py; not persisted to disk.

Rotation averaging uses quaternion-space averaging (Markley et al., 2007):
stack quaternions into a 4xN matrix M, compute the eigenvector of M M^T
with the largest eigenvalue. Much more robust than averaging matrix entries.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from detector import TAG_SIZE_M, TagDetection, detect_tags, make_detector, render_overlay


class ReferenceTagNotFound(RuntimeError):
    """Raised when the reference tag isn't visible during the startup solve."""


def _rvec_to_quat(rvec: np.ndarray) -> np.ndarray:
    """Rodrigues -> unit quaternion [w, x, y, z]."""
    R, _ = cv2.Rodrigues(rvec)
    return _matrix_to_quat(R)


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion [w, x, y, z].

    Shepperd's method; numerically stable.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def average_quaternions(quats: np.ndarray) -> np.ndarray:
    """Markley-averaged unit quaternion from an Nx4 stack [w,x,y,z]."""
    if quats.shape[0] == 0:
        raise ValueError("cannot average zero quaternions")
    q = quats.copy()

    ref = q[0]
    flipped = np.einsum("ij,j->i", q, ref) < 0.0
    q[flipped] = -q[flipped]

    M = q.T @ q
    eigvals, eigvecs = np.linalg.eigh(M)
    mean = eigvecs[:, -1]
    if mean[0] < 0:
        mean = -mean
    return mean / np.linalg.norm(mean)


def _invert_transform(T: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def solve_iphone_extrinsic_avg(
    cap: cv2.VideoCapture,
    K: np.ndarray,
    dist: np.ndarray,
    tag_id: int,
    T_base_tag_ref: np.ndarray,
    n_frames: int = 10,
    tag_size_m: float = TAG_SIZE_M,
    timeout_s: float = 10.0,
    detector: "object | None" = None,
    verbose: bool = True,
    preview_window: str | None = None,
) -> np.ndarray:
    """Solve T_base_iphoneCam by averaging PnP over n_frames sightings.

    Blocks up to `timeout_s` seconds trying to find the reference tag
    in `n_frames` distinct frames.

    Args:
        cap: opened OpenCV VideoCapture for the iPhone camera.
        K, dist: iPhone intrinsics.
        tag_id: id of the reference AprilTag taped at a known base-frame pose.
        T_base_tag_ref: 4x4 known pose of the reference tag in the base frame.
        n_frames: how many frames to average.
        tag_size_m: AprilTag edge length in metres.
        timeout_s: abort if we can't get `n_frames` detections in this long.
        detector: optional pre-built apriltag detector.
        verbose: print progress lines.
        preview_window: if set, each frame is drawn with detection overlays
            into a cv2 window with this name so the user can see the camera
            and confirm the reference tag is visible. Pressing 'q' or ESC
            during the solve aborts with ReferenceTagNotFound.

    Returns:
        4x4 T_base_iphoneCam.

    Raises:
        ReferenceTagNotFound: if the reference tag isn't seen `n_frames` times
            before `timeout_s` elapses, or the user aborts via the preview.
    """
    if T_base_tag_ref.shape != (4, 4):
        raise ValueError(f"T_base_tag_ref must be 4x4, got {T_base_tag_ref.shape}")

    det = detector if detector is not None else make_detector()

    quats: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    start = time.monotonic()
    frames_processed = 0
    missed_frames = 0

    while len(quats) < n_frames:
        if (time.monotonic() - start) > timeout_s:
            raise ReferenceTagNotFound(
                f"only captured {len(quats)}/{n_frames} sightings of tag id={tag_id} "
                f"within {timeout_s:.1f}s (processed {frames_processed} frames, "
                f"{missed_frames} had no matching tag)"
            )

        ok, frame = cap.read()
        if not ok:
            continue
        frames_processed += 1

        detections: list[TagDetection] = detect_tags(frame, det, K, dist, tag_size_m)
        hit = next((d for d in detections if d.id == tag_id), None)

        if hit is not None:
            T_iphone_tag = hit.T_camera_tag
            T_iphone_base_candidate = T_iphone_tag @ _invert_transform(T_base_tag_ref)
            T_base_iphone_candidate = _invert_transform(T_iphone_base_candidate)

            quats.append(_matrix_to_quat(T_base_iphone_candidate[:3, :3]))
            translations.append(T_base_iphone_candidate[:3, 3].copy())

            if verbose:
                print(
                    f"[iphone_extrinsic] sighting {len(quats)}/{n_frames}"
                    f"  tvec(iphone<-tag) = {np.array2string(hit.tvec.ravel(), precision=3)}"
                )
        else:
            missed_frames += 1

        if preview_window is not None:
            render_overlay(frame, detections, K, dist, tag_size_m)
            status = (
                f"solving T_base_iphone   sightings {len(quats)}/{n_frames}   "
                f"tag {tag_id}: {'VISIBLE' if hit is not None else 'MISSING'}"
            )
            color = (0, 255, 0) if hit is not None else (0, 0, 255)
            cv2.putText(
                frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )
            cv2.imshow(preview_window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise ReferenceTagNotFound(
                    f"user aborted extrinsic solve at "
                    f"{len(quats)}/{n_frames} sightings"
                )

        if hit is None:
            continue

    mean_q = average_quaternions(np.stack(quats, axis=0))
    mean_t = np.mean(np.stack(translations, axis=0), axis=0)

    T_base_iphone = np.eye(4, dtype=np.float64)
    T_base_iphone[:3, :3] = _quat_to_matrix(mean_q)
    T_base_iphone[:3, 3] = mean_t

    if verbose:
        t_std = np.std(np.stack(translations, axis=0), axis=0)
        print(
            f"[iphone_extrinsic] solved over {n_frames} frames; "
            f"translation std = {np.array2string(t_std * 1000, precision=2)} mm"
        )

    return T_base_iphone
