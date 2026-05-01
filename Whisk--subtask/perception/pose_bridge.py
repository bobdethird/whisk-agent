"""
Pose bridge — pluggable pose source for the agent loop.

Backends (all expose the same interface: no-arg callable → pose dict)
----------------------------------------------------------------------
MockPoseProvider      — Gaussian-noised hardcoded poses (development)
FilePoseProvider      — reads poses.json; edit by hand or have any process write it
ServerPoseProvider    — polls the local HTTP pose server (teammate integration)
ManualPoseProvider    — prompts the user to type poses in the terminal

Switching backends requires changing one line in main.py.  The agent loop
itself never needs to know which backend is active.

Usage
-----
    from perception.pose_bridge import FilePoseProvider
    provider = FilePoseProvider("poses.json")
    history = run_agent_loop(..., get_poses_fn=provider.get_poses, ...)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

from perception.mock_poses import MOCK_POSES, get_mock_poses


# ---------------------------------------------------------------------------
# Base / type alias
# ---------------------------------------------------------------------------

PoseMap = dict[str, list[float]]   # {object_name: [x, y, z]}


# ---------------------------------------------------------------------------
# Mock backend (already working)
# ---------------------------------------------------------------------------

class MockPoseProvider:
    """
    Gaussian-noised hardcoded poses.  Zero dependencies, always works.
    Use this when you have no camera and don't need real positions.
    """

    def __init__(self, noise_std: float = 0.002) -> None:
        self.noise_std = noise_std

    def get_poses(self) -> PoseMap:
        """Return noisy mock poses."""
        return get_mock_poses(self.noise_std)


# ---------------------------------------------------------------------------
# File backend
# ---------------------------------------------------------------------------

_DEFAULT_POSES_FILE = Path(__file__).parent.parent / "poses.json"


class FilePoseProvider:
    """
    Reads object poses from a JSON file on disk.

    The file is re-read on every call so it can be updated live —
    by hand in a text editor, by the teammate's detection script,
    or by any other process.

    File format (same structure as MOCK_POSES)::

        {
          "matcha_cup":   [0.30, 0.02, 0.40],
          "matcha_bowl":  [0.10, 0.02, 0.40],
          "matcha_whisk": [0.20, 0.02, 0.50],
          "matcha_scoop": [0.35, 0.02, 0.45],
          "cup_of_ice":   [0.00, 0.10, 0.35],
          "cup_of_water": [0.05, 0.10, 0.35]
        }

    Missing keys fall back to the corresponding MOCK_POSES value so
    partial updates (e.g. only some objects detected) still work.
    """

    def __init__(
        self,
        path: str | Path = _DEFAULT_POSES_FILE,
        fallback_to_mock: bool = True,
    ) -> None:
        self.path = Path(path)
        self.fallback_to_mock = fallback_to_mock
        self._last_warn: float = 0.0

        # Create the file with mock defaults if it doesn't exist
        if not self.path.exists():
            self._write_defaults()
            print(f"[PoseBridge] Created {self.path} with mock defaults.")

    # ------------------------------------------------------------------
    def get_poses(self) -> PoseMap:
        """Read and return poses from the JSON file."""
        try:
            with open(self.path) as f:
                data: dict = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            now = time.time()
            if now - self._last_warn > 5.0:
                print(f"[PoseBridge] WARNING: cannot read {self.path}: {exc}")
                self._last_warn = now
            if self.fallback_to_mock:
                return get_mock_poses()
            raise

        # Merge with mock defaults so missing keys don't crash the agent
        result: PoseMap = {}
        for name in MOCK_POSES:
            if name in data:
                result[name] = [float(v) for v in data[name]]
            elif self.fallback_to_mock:
                result[name] = [float(v) for v in MOCK_POSES[name]]
        return result

    def _write_defaults(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({k: list(v) for k, v in MOCK_POSES.items()}, f, indent=2)


# ---------------------------------------------------------------------------
# HTTP server backend (teammate integration)
# ---------------------------------------------------------------------------

class ServerPoseProvider:
    """
    Polls the local pose HTTP server started by ``pose_server.py``.

    Your teammate's detection script POSTs new poses to the server;
    this provider fetches the latest snapshot on every agent step.

    Falls back to the last known good poses if the server is unreachable,
    and falls back to mock poses if no real poses have arrived yet.
    """

    def __init__(
        self,
        url: str = "http://localhost:5050/poses",
        timeout: float = 0.5,
        fallback_to_mock: bool = True,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.fallback_to_mock = fallback_to_mock
        self._last_known: PoseMap | None = None
        self._last_warn: float = 0.0

    def get_poses(self) -> PoseMap:
        """GET the latest poses from the server."""
        try:
            import urllib.request
            with urllib.request.urlopen(self.url, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
            self._last_known = {k: [float(v) for v in vals]
                                 for k, vals in data.items()}
            return dict(self._last_known)
        except Exception as exc:
            now = time.time()
            if now - self._last_warn > 5.0:
                print(f"[PoseBridge] Server unreachable ({exc}); using fallback.")
                self._last_warn = now
            if self._last_known is not None:
                return dict(self._last_known)
            if self.fallback_to_mock:
                return get_mock_poses()
            raise


# ---------------------------------------------------------------------------
# Manual / terminal backend
# ---------------------------------------------------------------------------

class ManualPoseProvider:
    """
    Prompts the user to enter or confirm poses in the terminal each step.

    Useful for very early hardware testing when you want to manually specify
    where objects are.  Press Enter to accept the displayed default.
    """

    def __init__(self, initial_poses: PoseMap | None = None) -> None:
        self._poses: PoseMap = dict(initial_poses or MOCK_POSES)

    def get_poses(self) -> PoseMap:
        """Display current poses and let the user edit any of them."""
        print("\n[ManualPoseProvider] Current poses (press Enter to keep, or type x,y,z):")
        for name, pose in self._poses.items():
            default = f"{pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}"
            raw = input(f"  {name} [{default}]: ").strip()
            if raw:
                try:
                    parts = [float(p) for p in raw.split(",")]
                    if len(parts) == 3:
                        self._poses[name] = parts
                    else:
                        print(f"    Expected 3 values, got {len(parts)} — keeping old value.")
                except ValueError:
                    print("    Could not parse — keeping old value.")
        return dict(self._poses)
