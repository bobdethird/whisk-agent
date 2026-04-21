import cv2
import numpy as np
from apriltag import apriltag

TAG_SIZE_M = 0.10
CALIB_FILE = "camera_calib.npz"

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

detector = apriltag("tagStandard41h12")
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

while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        else:
            label = f"id={det['id']}"
        cv2.putText(frame, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("apriltag", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
