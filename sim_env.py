from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco  # type: ignore[import-not-found]

from mujoco_sim.apriltag_world_config import DEFAULT_RENDER_HEIGHT, DEFAULT_RENDER_WIDTH
from so101_kinematics import SO101Kinematics
from so101_mujoco_utils import convert_to_dictionary, set_initial_pose


ROOT_DIR = Path(__file__).parent
MODEL_PATH = ROOT_DIR / "simulation_code" / "model" / "scene.xml"
DEFAULT_CAMERA = "table_observer"

STARTING_POSITION = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -45.0,
    "elbow_flex": 90.0,
    "wrist_flex": -45.0,
    "wrist_roll": 0.0,
    "gripper": 50.0,
}


@dataclass
class SimEnv:
    model: mujoco.MjModel
    data: mujoco.MjData
    scene_path: Path
    camera_name: str = DEFAULT_CAMERA
    render_width: int = DEFAULT_RENDER_WIDTH
    render_height: int = DEFAULT_RENDER_HEIGHT
    kinematics: SO101Kinematics = field(default_factory=SO101Kinematics)
    current_position: dict[str, float] = field(default_factory=lambda: dict(STARTING_POSITION))
    viewer: Any | None = None

    def sync_current_position(self) -> None:
        self.current_position = convert_to_dictionary(self.data.qpos.copy())


def create_env(
    scene_path: Path = MODEL_PATH,
    camera_name: str = DEFAULT_CAMERA,
    render_width: int = DEFAULT_RENDER_WIDTH,
    render_height: int = DEFAULT_RENDER_HEIGHT,
) -> SimEnv:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    set_initial_pose(model, data, STARTING_POSITION)
    return SimEnv(
        model=model,
        data=data,
        scene_path=scene_path,
        camera_name=camera_name,
        render_width=render_width,
        render_height=render_height,
    )
