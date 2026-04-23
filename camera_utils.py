"""Small shared helpers for any script that opens a camera.

Anything touching `cv2.VideoCapture` in this repo should go through
`disable_autofocus()` right after `cap.isOpened()` succeeds. Autofocus
silently varies the lens focal length at runtime, which invalidates the
intrinsic calibration in `*_camera_calib.npz` and makes PnP ranges drift
with scene depth. We want a single fixed focus per session so that the
`K` we load from disk is the `K` the camera is actually using.
"""

from __future__ import annotations

import cv2


def disable_autofocus(cap: cv2.VideoCapture, label: str = "camera") -> bool:
    """Attempt to turn off autofocus (and auto-exposure) on an open capture.

    Returns whether the backend reports success. Some backends (notably
    macOS AVFoundation + Continuity Camera) ignore UVC-style control
    requests and return True anyway, so treat the ack as advisory -- if
    you suspect the focal length is still drifting, compare the
    `det.range_m` reported at a fixed, ruler-measured distance across a
    full session.
    """
    ok = bool(cap.set(cv2.CAP_PROP_AUTOFOCUS, 0))
    if ok:
        print(f"[{label}] autofocus disable requested (backend ack: ok)")
    else:
        print(
            f"[{label}] WARNING: backend rejected CAP_PROP_AUTOFOCUS=0; "
            f"focal length may drift at runtime. Check that the camera "
            f"driver supports UVC focus controls."
        )
    return ok
