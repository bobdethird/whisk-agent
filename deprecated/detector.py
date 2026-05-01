"""AprilTag detection and PnP pose estimation.

Exposes `detect_tags()` and `render_overlay()` as importable helpers so the
same detection logic is reused by the iPhone overhead camera, the wrist
camera, and the startup iPhone-extrinsic solver.

The module retains a standalone `__main__` for quick debugging with the
iPhone overhead camera (visualises tags in camera frame).

Important convention note:
    OpenCV's `SOLVEPNP_IPPE_SQUARE` expects a square-local frame with
    x right, y up, z out of the tag plane. The AprilTag docs use
    x right, y down, z into the tag plane. We solve with IPPE's required
    frame, then convert the returned rotation into the AprilTag convention
    before exposing it to the rest of the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from apriltag import apriltag

from camera_utils import disable_autofocus

TAG_SIZE_M = 0.027
DEFAULT_TAG_FAMILY = "tagStandard41h12"
R_IPPE_TO_APRILTAG = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


@dataclass
class TagDetection:
    """Single AprilTag detection with pose in the camera frame.

    The tag-local axes follow the AprilTag documentation:
        x right, y down, z into the tag.
    """

    id: int
    corners: np.ndarray       # (4, 2) pixel corners in detector order (lb, rb, rt, lt)
    center: tuple[float, float]
    rvec: np.ndarray          # (3, 1) Rodrigues rotation, camera <- tag
    tvec: np.ndarray          # (3, 1) translation, camera <- tag (metres)
    T_camera_tag: np.ndarray  # (4, 4) convenience form of (rvec, tvec)

    @property
    def z_m(self) -> float:
        """Forward distance along the camera optical axis (+Z in OpenCV)."""
        return float(self.tvec.ravel()[2])

    @property
    def range_m(self) -> float:
        """Euclidean camera-to-tag distance."""
        return float(np.linalg.norm(self.tvec))


def _tag_object_points(tag_size_m: float) -> np.ndarray:
    """Square geometry required by OpenCV's SOLVEPNP_IPPE_SQUARE.

    This is not the final public tag-frame convention. IPPE requires:
        x right, y up, z out of the tag
    and we convert the solved rotation into the AprilTag convention after
    pose estimation.
    """
    half = tag_size_m / 2.0
    return np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)


def _corners_for_ippe_square(detector_corners: np.ndarray) -> np.ndarray:
    """Map apriltag's (lb, rb, rt, lt) image corners to IPPE's order.

    OpenCV's SOLVEPNP_IPPE_SQUARE expects image points that correspond to the
    3D square corners returned by `_tag_object_points()`, namely:
        (lt, rt, rb, lb)
    The apriltag wrapper used here exposes corners as:
        (lb, rb, rt, lt)
    so we reorder before calling solvePnP.
    """
    if detector_corners.shape != (4, 2):
        raise ValueError(f"expected (4, 2) corners, got {detector_corners.shape}")
    return detector_corners[[3, 2, 1, 0]]


def _convert_ippe_pose_to_apriltag(
    rvec_ippe: np.ndarray,
    tvec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an IPPE-square pose into the AprilTag axis convention."""
    R_ippe, _ = cv2.Rodrigues(rvec_ippe)
    R_apriltag = R_ippe @ R_IPPE_TO_APRILTAG
    rvec_apriltag, _ = cv2.Rodrigues(R_apriltag)

    T_camera_tag = np.eye(4, dtype=np.float64)
    T_camera_tag[:3, :3] = R_apriltag
    T_camera_tag[:3, 3] = tvec.ravel()
    return rvec_apriltag, tvec, T_camera_tag


def make_detector(
    tag_family: str = DEFAULT_TAG_FAMILY,
    decimate: float = 1.0,
    threads: int = 4,
    refine_edges: bool = True,
) -> apriltag:
    """Construct an apriltag detector with the matcha-bot's tuning defaults."""
    return apriltag(
        tag_family,
        decimate=decimate,
        threads=threads,
        refine_edges=refine_edges,
    )


def detect_tags(
    frame: np.ndarray,
    detector: apriltag,
    K: np.ndarray,
    dist: np.ndarray,
    tag_size_m: float = TAG_SIZE_M,
) -> list[TagDetection]:
    """Run AprilTag detection + PnP on a BGR frame.

    Returns a list of TagDetection; tags for which solvePnP fails are
    silently dropped (consistent with the prior detector.py behaviour).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    obj = _tag_object_points(tag_size_m)

    detections: list[TagDetection] = []
    for det in detector.detect(gray):
        corners = np.array(det["lb-rb-rt-lt"], dtype=np.float64)
        corners_ippe = _corners_for_ippe_square(corners)
        ok, rvec, tvec = cv2.solvePnP(
            obj, corners_ippe, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            continue

        rvec, tvec, T = _convert_ippe_pose_to_apriltag(rvec, tvec)

        detections.append(TagDetection(
            id=int(det["id"]),
            corners=corners,
            center=tuple(float(c) for c in det["center"]),
            rvec=rvec,
            tvec=tvec,
            T_camera_tag=T,
        ))
    return detections


def origin_center_error_px(
    det: TagDetection,
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    """Pixel distance between detector center and projected PnP tag origin."""
    projected_origin, _ = cv2.projectPoints(
        np.zeros((1, 3), dtype=np.float64),
        det.rvec,
        det.tvec,
        K,
        dist,
    )
    ox, oy = projected_origin.reshape(2)
    cx, cy = det.center
    return float(np.linalg.norm(np.array([ox - cx, oy - cy])))


def render_overlay(
    frame: np.ndarray,
    detections: list[TagDetection],
    K: np.ndarray,
    dist: np.ndarray,
    tag_size_m: float = TAG_SIZE_M,
) -> np.ndarray:
    """Draw tag outlines, axes, id/z labels in place on `frame`. Returns frame."""
    for det in detections:
        pts = det.corners.astype(np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        cv2.drawFrameAxes(frame, K, dist, det.rvec, det.tvec, tag_size_m * 0.5, 2)
        cx, cy = (int(det.center[0]), int(det.center[1]))
        projected_origin, _ = cv2.projectPoints(
            np.zeros((1, 3), dtype=np.float64),
            det.rvec,
            det.tvec,
            K,
            dist,
        )
        ox, oy = projected_origin.reshape(2)
        origin_px = (int(round(ox)), int(round(oy)))
        center_err_px = origin_center_error_px(det, K, dist)
        cv2.drawMarker(
            frame,
            origin_px,
            (255, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=2,
        )
        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
        label = (
            f"id={det.id} range={det.range_m:.4f}m z={det.z_m:.4f}m "
            f"origin-center={center_err_px:.1f}px"
        )
        cv2.putText(frame, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


# ---------------------------------------------------------------------------
# Standalone debugging loop (iPhone overhead camera in camera frame)
# ---------------------------------------------------------------------------
def _debug_main() -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    CALIB_FILE = "camera_calib.npz"
    CAMERA_INDEX = 1
    PLOT_EVERY = 3

    calib = np.load(CALIB_FILE)
    K = calib["K"].astype(np.float64)
    DIST = calib["dist"].astype(np.float64)
    calib_size = (int(calib["image_size"][0]), int(calib["image_size"][1]))

    def cam_to_plot(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)

    def draw_camera(ax, K_, img_size, depth=0.25):
        w, h = img_size
        fx, fy, cx, cy = K_[0, 0], K_[1, 1], K_[0, 2], K_[1, 2]
        px = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
        rays = np.stack([
            (px[:, 0] - cx) / fx,
            (px[:, 1] - cy) / fy,
            np.ones(4),
        ], axis=1) * depth
        corners = cam_to_plot(rays)
        origin = cam_to_plot(np.zeros(3))
        segs = [[origin, c] for c in corners]
        segs += [[corners[i], corners[(i + 1) % 4]] for i in range(4)]
        ax.add_collection3d(Line3DCollection(segs, colors="k", linewidths=1))
        ax.scatter(*origin, color="red", s=40, zorder=10)

    def draw_tag(ax, det: TagDetection, tag_size: float):
        R, _ = cv2.Rodrigues(det.rvec)
        t = det.tvec.ravel()
        half = tag_size / 2
        local = np.array([
            [-half, -half, 0],
            [half, -half, 0],
            [half, half, 0],
            [-half, half, 0],
        ])
        corners_cam = (R @ local.T).T + t
        ax.add_collection3d(Poly3DCollection(
            [cam_to_plot(corners_cam)],
            alpha=0.3, facecolor="cornflowerblue", edgecolor="navy",
        ))
        axis_len = tag_size * 0.6
        center = cam_to_plot(t)
        for i, color in enumerate("rgb"):
            end = cam_to_plot(t + R[:, i] * axis_len)
            ax.plot([center[0], end[0]],
                    [center[1], end[1]],
                    [center[2], end[2]],
                    color=color, lw=2)
        ax.text(*center, f" id={det.id}", fontsize=8)

    def refresh_3d(ax, detections: list[TagDetection]):
        ax.cla()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(0.0, 3.0)
        ax.set_zlim(-1.0, 1.0)
        ax.set_box_aspect((3, 3, 2))
        ax.set_xlabel("X right (m)")
        ax.set_ylabel("Z forward (m)")
        ax.set_zlabel("-Y up (m)")
        draw_camera(ax, K, calib_size)
        for det in detections:
            draw_tag(ax, det, TAG_SIZE_M)

    detector = make_detector()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    disable_autofocus(cap, label="iphone")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, calib_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calib_size[1])

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("camera open failed")

    h, w = frame.shape[:2]
    if (w, h) != calib_size:
        raise SystemExit(
            f"camera delivered {(w, h)} but calibration is for {calib_size}; "
            "recalibrate at the runtime resolution or force this resolution."
        )

    plt.ion()
    fig = plt.figure("world (camera at origin)", figsize=(6, 6))
    ax3d = fig.add_subplot(111, projection="3d")

    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detect_tags(frame, detector, K, DIST, TAG_SIZE_M)
        render_overlay(frame, detections, K, DIST, TAG_SIZE_M)
        cv2.imshow("apriltag", frame)

        if frame_i % PLOT_EVERY == 0:
            refresh_3d(ax3d, detections)
            plt.pause(0.001)
        frame_i += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.close("all")


if __name__ == "__main__":
    _debug_main()
