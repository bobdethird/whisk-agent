import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from apriltag import apriltag

TAG_SIZE_M = 0.10
CALIB_FILE = "camera_calib.npz"
PLOT_EVERY = 3  # redraw the 3D view every N camera frames

calib = np.load(CALIB_FILE)
K = calib["K"].astype(np.float64)
DIST = calib["dist"].astype(np.float64)
CALIB_SIZE = (int(calib["image_size"][0]), int(calib["image_size"][1]))

OBJ = (TAG_SIZE_M / 2.0) * np.array([
    [-1,  1, 0],
    [ 1,  1, 0],
    [ 1, -1, 0],
    [-1, -1, 0],
], dtype=np.float64)


def cam_to_plot(p):
    """OpenCV cam frame (X right, Y down, Z fwd) -> plot frame (X right, Y fwd, Z up).

    Lets matplotlib's default 3D view angle render the scene intuitively: the
    camera sits at the origin looking "into the page".
    """
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def draw_camera(ax, K, img_size, depth=0.25):
    w, h = img_size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    px = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    rays_cam = np.stack([
        (px[:, 0] - cx) / fx,
        (px[:, 1] - cy) / fy,
        np.ones(4),
    ], axis=1) * depth
    corners = cam_to_plot(rays_cam)
    origin = cam_to_plot(np.zeros(3))
    segs = [[origin, c] for c in corners]
    segs += [[corners[i], corners[(i + 1) % 4]] for i in range(4)]
    ax.add_collection3d(Line3DCollection(segs, colors="k", linewidths=1))
    ax.scatter(*origin, color="red", s=40, zorder=10)


def draw_tag(ax, rvec, tvec, tag_size, tag_id):
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.ravel()
    half = tag_size / 2
    local = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
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
    ax.text(*center, f" id={tag_id}", fontsize=8)


def refresh_3d(ax, tag_poses):
    ax.cla()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(0.0, 3.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_box_aspect((3, 3, 2))
    ax.set_xlabel("X right (m)")
    ax.set_ylabel("Z forward (m)")
    ax.set_zlabel("-Y up (m)")
    draw_camera(ax, K, CALIB_SIZE)
    for rvec, tvec, tid in tag_poses:
        draw_tag(ax, rvec, tvec, TAG_SIZE_M, tid)


detector = apriltag(
    "tagStandard41h12",
    decimate=1.0,   # full-resolution quad search (default 2.0); ~2x detection distance
    threads=4,      # parallelize to compensate for decimate=1.0 cost
    refine_edges=True,
)
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CALIB_SIZE[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CALIB_SIZE[1])

ok, frame = cap.read()
if not ok:
    raise SystemExit("camera open failed")

h, w = frame.shape[:2]
if (w, h) != CALIB_SIZE:
    raise SystemExit(
        f"camera delivered {(w, h)} but calibration is for {CALIB_SIZE}; "
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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tag_poses = []
    for det in detector.detect(gray):
        corners = np.array(det["lb-rb-rt-lt"], dtype=np.float64)
        ok_pnp, rvec, tvec = cv2.solvePnP(
            OBJ, corners, K, DIST, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 0), 2)
        cx, cy = map(int, det["center"])
        if ok_pnp:
            cv2.drawFrameAxes(frame, K, DIST, rvec, tvec, TAG_SIZE_M * 0.5, 2)
            label = f"id={det['id']} z={tvec[2, 0]:.2f}m"
            tag_poses.append((rvec, tvec, det["id"]))
        else:
            label = f"id={det['id']}"
        cv2.putText(frame, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("apriltag", frame)

    if frame_i % PLOT_EVERY == 0:
        refresh_3d(ax3d, tag_poses)
        plt.pause(0.001)
    frame_i += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
plt.close("all")
