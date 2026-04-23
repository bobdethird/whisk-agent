"""PoseMap: fuses AprilTag observations from the iPhone overhead camera
and the SO-101 wrist camera into a single best-known base-frame pose
per tag id.

Fusion rule (PRD section 4.3.1):
    - Weighted-average translations: wrist 60%, iPhone 40%.
    - Take the wrist camera's rotation whenever both are fresh.
    - Stale observations (> STALE_S seconds old) are treated as absent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

STALE_S = 2.0
WRIST_WEIGHT = 0.6
IPHONE_WEIGHT = 0.4

TAG_OBJECT_MAP: dict[int, str] = {
    0: "reference",
    1: "bowl",
    2: "chawan",
    3: "whisk",
    4: "spoon",
    5: "canister",
    6: "kettle",
}


@dataclass(frozen=True)
class Observation:
    """One camera's view of a tag, timestamped."""

    T_base_tag: np.ndarray
    timestamp: float

    def is_fresh(self, now: float | None = None, stale_s: float = STALE_S) -> bool:
        t = time.monotonic() if now is None else now
        return (t - self.timestamp) <= stale_s


class PoseMap:
    """Shared registry of tag poses in the SO-101 base frame."""

    def __init__(self, stale_s: float = STALE_S):
        self._wrist: dict[int, Observation] = {}
        self._iphone: dict[int, Observation] = {}
        self._stale_s = stale_s

    def update_wrist(self, tag_id: int, T_base_tag: np.ndarray) -> None:
        self._wrist[tag_id] = Observation(
            T_base_tag=np.asarray(T_base_tag, dtype=np.float64),
            timestamp=time.monotonic(),
        )

    def update_iphone(self, tag_id: int, T_base_tag: np.ndarray) -> None:
        self._iphone[tag_id] = Observation(
            T_base_tag=np.asarray(T_base_tag, dtype=np.float64),
            timestamp=time.monotonic(),
        )

    def get_pose(self, tag_id: int) -> np.ndarray | None:
        """Return the fused 4x4 T_base_tag for `tag_id`, or None if no fresh view."""
        now = time.monotonic()
        w = self._wrist.get(tag_id)
        i = self._iphone.get(tag_id)
        w_ok = w is not None and w.is_fresh(now, self._stale_s)
        i_ok = i is not None and i.is_fresh(now, self._stale_s)

        if not w_ok and not i_ok:
            return None
        if w_ok and not i_ok:
            return w.T_base_tag.copy()
        if i_ok and not w_ok:
            return i.T_base_tag.copy()

        # Both fresh: weighted-average translation, wrist rotation wins.
        fused = np.eye(4, dtype=np.float64)
        fused[:3, :3] = w.T_base_tag[:3, :3]
        fused[:3, 3] = (
            WRIST_WEIGHT * w.T_base_tag[:3, 3]
            + IPHONE_WEIGHT * i.T_base_tag[:3, 3]
        )
        return fused

    def get_all_poses(self) -> dict[int, np.ndarray]:
        """Snapshot the fused poses of every tag seen by either camera."""
        now = time.monotonic()
        ids = set()
        for tid, obs in self._wrist.items():
            if obs.is_fresh(now, self._stale_s):
                ids.add(tid)
        for tid, obs in self._iphone.items():
            if obs.is_fresh(now, self._stale_s):
                ids.add(tid)

        out: dict[int, np.ndarray] = {}
        for tid in ids:
            pose = self.get_pose(tid)
            if pose is not None:
                out[tid] = pose
        return out

    def sources(self, tag_id: int) -> tuple[bool, bool]:
        """Return (wrist_fresh, iphone_fresh) for `tag_id`. Useful for viz/logs."""
        now = time.monotonic()
        w = self._wrist.get(tag_id)
        i = self._iphone.get(tag_id)
        return (
            w is not None and w.is_fresh(now, self._stale_s),
            i is not None and i.is_fresh(now, self._stale_s),
        )

    def object_name(self, tag_id: int) -> str:
        return TAG_OBJECT_MAP.get(tag_id, f"tag_{tag_id}")
