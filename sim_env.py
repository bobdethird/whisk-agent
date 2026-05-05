from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from mujoco_sim.apriltag_world_config import APRILTAGS, DEFAULT_RENDER_HEIGHT, DEFAULT_RENDER_WIDTH, TABLE_TAG_Z_M
from so101_kinematics import SO101Kinematics
from so101_mujoco_utils import convert_to_dictionary, set_initial_pose


ROOT_DIR = Path(__file__).parent
MODEL_PATH = ROOT_DIR / "simulation_code" / "model" / "scene.xml"
DEFAULT_CAMERA = "table_observer"
DEFAULT_TAG_X_RANGE = (0.30, 0.46)
DEFAULT_TAG_Y_RANGE = (-0.12, 0.08)
DEFAULT_TAG_MIN_SPACING_M = 0.055
HORIZONTAL_WRIST_ROLL_DEGREES = -90.0

STARTING_POSITION = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -45.0,
    "elbow_flex": 90.0,
    "wrist_flex": -45.0,
    "wrist_roll": HORIZONTAL_WRIST_ROLL_DEGREES,
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


def _require_tag_body_id(model: mujoco.MjModel, tag_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tag_name)
    if body_id < 0:
        raise ValueError(f"Could not find AprilTag body named {tag_name!r}.")
    return body_id


def _sample_spaced_positions(
    rng: np.random.Generator,
    count: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    min_spacing: float,
) -> list[np.ndarray]:
    positions: list[np.ndarray] = []
    max_attempts = 2000
    for _ in range(count):
        for _ in range(max_attempts):
            candidate = np.array(
                [
                    rng.uniform(*x_range),
                    rng.uniform(*y_range),
                    TABLE_TAG_Z_M,
                ],
                dtype=float,
            )
            if all(np.linalg.norm(candidate[:2] - position[:2]) >= min_spacing for position in positions):
                positions.append(candidate)
                break
        else:
            raise RuntimeError("Could not place AprilTags without overlap; widen the randomization ranges.")
    return positions


def randomize_apriltag_positions(
    env: SimEnv,
    seed: int | None = None,
    x_range: tuple[float, float] = DEFAULT_TAG_X_RANGE,
    y_range: tuple[float, float] = DEFAULT_TAG_Y_RANGE,
    min_spacing: float = DEFAULT_TAG_MIN_SPACING_M,
) -> None:
    rng = np.random.default_rng(seed)
    positions = _sample_spaced_positions(rng, len(APRILTAGS), x_range, y_range, min_spacing)

    print("Randomized AprilTag positions:")
    for tag, position in zip(APRILTAGS, positions):
        body_id = _require_tag_body_id(env.model, tag.name)
        env.model.body_pos[body_id] = position
        print(f"  tag {tag.tag_id}: x={position[0]:.4f} y={position[1]:.4f} z={position[2]:.4f}")

    mujoco.mj_forward(env.model, env.data)
