"""
Download the SO-101 URDF from the LeRobot GitHub repository.

The URDF describes the arm's kinematic structure (link lengths, joint axes,
limits).  ikpy reads it to build the IK chain used in LeRobotExecutor.

Run once before first hardware use:
    python scripts/download_urdf.py
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

_ROOT   = Path(__file__).parent.parent
_ASSETS = _ROOT / "assets"
_OUT    = _ASSETS / "so101.urdf"

# LeRobot repo URL for the SO-101 URDF.
# If this URL breaks, find the current path at:
#   https://github.com/huggingface/lerobot/tree/main/lerobot/configs/robot
URDF_URL = (
    "https://raw.githubusercontent.com/huggingface/lerobot/"
    "main/lerobot/configs/robot/so101.urdf"
)


def download() -> None:
    _ASSETS.mkdir(parents=True, exist_ok=True)

    if _OUT.exists():
        print(f"[download_urdf] Already exists: {_OUT}")
        print("  Delete the file and re-run to force a fresh download.")
        return

    print(f"[download_urdf] Downloading SO-101 URDF from LeRobot repo...")
    print(f"  → {_OUT}")

    try:
        urllib.request.urlretrieve(URDF_URL, _OUT)
        size_kb = _OUT.stat().st_size // 1024
        print(f"[download_urdf] Done. ({size_kb} KB)")
    except Exception as exc:
        print(f"[download_urdf] ERROR: {exc}")
        print()
        print("If the URL is broken, manually download the URDF from:")
        print("  https://github.com/huggingface/lerobot")
        print(f"and save it to:  {_OUT}")
        sys.exit(1)


if __name__ == "__main__":
    download()
