"""Chessboard camera calibration.

Usage:
    1. Print a chessboard (default pattern below is 9x6 internal corners, e.g.
       https://github.com/opencv/opencv/blob/master/doc/pattern.png printed on A4).
    2. python calibrate.py
    3. Hold the board in view. When corners turn green press SPACE to capture.
       Vary angle, distance, tilt, and position in the frame. Aim for 15-25 shots.
    4. Press ENTER to run calibration. Output is written to camera_calib.npz.
"""

import cv2
import numpy as np

from camera_utils import disable_autofocus

PATTERN = (9, 6)          # internal corners (cols, rows). Use an asymmetric size.
CAMERA_INDEX = 1
MIN_FRAMES = 15
OUTPUT_FILE = "camera_calib.npz"

SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
FIND_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    | cv2.CALIB_CB_NORMALIZE_IMAGE
    | cv2.CALIB_CB_FAST_CHECK
)

# 3D model of the board in its own frame: (0,0,0), (1,0,0), ... (8,5,0).
# Units are "one square" -- intrinsics are scale-invariant so this is fine.
objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)

objpoints: list[np.ndarray] = []
imgpoints: list[np.ndarray] = []

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise SystemExit(f"failed to open camera {CAMERA_INDEX}")
disable_autofocus(cap, label="calibrate")

print(__doc__)
image_size: tuple[int, int] | None = None

while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_size = (gray.shape[1], gray.shape[0])

    found, corners = cv2.findChessboardCorners(gray, PATTERN, FIND_FLAGS)
    refined = None
    display = frame.copy()
    if found:
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRIT)
        cv2.drawChessboardCorners(display, PATTERN, refined, True)

    color = (0, 255, 0) if found else (0, 0, 255)
    status = f"captured {len(objpoints)}/{MIN_FRAMES}+  [SPACE] add  [ENTER] calibrate  [ESC] quit"
    cv2.putText(display, status, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.imshow("calibrate", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit("aborted")
    if key in (10, 13):
        if len(objpoints) >= MIN_FRAMES:
            break
        print(f"need at least {MIN_FRAMES} frames, have {len(objpoints)}")
    if key == ord(" ") and found and refined is not None:
        objpoints.append(objp.copy())
        imgpoints.append(refined)
        print(f"captured frame {len(objpoints)}")

cap.release()
cv2.destroyAllWindows()

if not objpoints or image_size is None:
    raise SystemExit("no frames captured")

print(f"\nrunning cv2.calibrateCamera on {len(objpoints)} frames ...")
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

per_view = []
for i in range(len(objpoints)):
    proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    per_view.append(cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / np.sqrt(len(proj)))

print(f"overall RMS reprojection error : {rms:.4f} px")
print(f"mean per-view reprojection err : {np.mean(per_view):.4f} px")
print(f"worst per-view reprojection err: {np.max(per_view):.4f} px")
print(f"\nimage size = {image_size}")
print("K =")
print(K)
print("dist =", dist.ravel())

np.savez(OUTPUT_FILE, K=K, dist=dist, image_size=np.array(image_size))
print(f"\nsaved calibration to {OUTPUT_FILE}")
print("load in detector.py with:")
print("    calib = np.load('camera_calib.npz')")
print("    K, DIST = calib['K'], calib['dist']")
