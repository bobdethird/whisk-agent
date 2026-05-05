from pathlib import Path
from typing import Literal

import numpy as np  # type: ignore[import-not-found]
from lerobot.model.kinematics import RobotKinematics  # type: ignore[import-not-found]

from so101_mujoco_utils import JOINT_ORDER


MODEL_DIR = Path(__file__).parent / "simulation_code" / "model"
DEFAULT_URDF_PATH = MODEL_DIR / "so101_new_calib.urdf"
TARGET_FRAME_NAME = "gripper_frame_link"
MUJOCO_SITE_NAME = "gripperframe"
ARM_JOINT_ORDER = JOINT_ORDER[:-1]
GRIPPER_JOINT = JOINT_ORDER[-1]
ToolPointName = Literal["claw_center", "fixed_jaw_tip"]
CLAW_CENTER_TOOL_POINT: ToolPointName = "claw_center"
FIXED_JAW_TOOL_POINT: ToolPointName = "fixed_jaw_tip"

# LeRobot's URDF frame and MuJoCo's gripperframe site use different local axes.
URDF_TO_MUJOCO_GRIPPERFRAME_ROTATION = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=float,
)
# The Menagerie SO-101 model places gripperframe at the pinch/contact frame,
# about 2 cm forward from LeRobot's URDF target frame in MuJoCo-frame +Z.
URDF_TO_MUJOCO_GRIPPERFRAME_OFFSET = np.array([0.0, 0.0, 0.0199], dtype=float)
# Offset from Menagerie's gripperframe site to the center of the claw contact
# points when the gripper is closed. Motion commands use this as the user-facing
# tool point so the visible claw, not the internal site, lands on targets.
MUJOCO_GRIPPERFRAME_TO_CLAW_TARGET_OFFSET = np.array([0.0007, 0.0001, -0.0120], dtype=float)
# Offset from gripperframe to the stationary jaw's central fingertip sphere
# (`fixed_jaw_sph_tip1`) in gripperframe-local axes.
MUJOCO_GRIPPERFRAME_TO_FIXED_JAW_TARGET_OFFSET = np.array([0.002873, 0.000218, -0.0201], dtype=float)
MUJOCO_GRIPPERFRAME_TOOL_OFFSETS: dict[ToolPointName, np.ndarray] = {
    CLAW_CENTER_TOOL_POINT: MUJOCO_GRIPPERFRAME_TO_CLAW_TARGET_OFFSET,
    FIXED_JAW_TOOL_POINT: MUJOCO_GRIPPERFRAME_TO_FIXED_JAW_TARGET_OFFSET,
}


def _require_pose_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 pose matrix, got shape {pose.shape}.")
    return pose


def joint_dict_to_array(position_dict: dict[str, float], joint_names=ARM_JOINT_ORDER) -> np.ndarray:
    missing = [joint for joint in joint_names if joint not in position_dict]
    if missing:
        raise KeyError(f"Missing joint value(s): {', '.join(missing)}")
    return np.array([position_dict[joint] for joint in joint_names], dtype=float)


def array_to_joint_dict(
    joint_values: np.ndarray,
    base_position_dict: dict[str, float] | None = None,
    joint_names=ARM_JOINT_ORDER,
    gripper: float | None = None,
) -> dict[str, float]:
    joint_values = np.asarray(joint_values, dtype=float)
    if len(joint_values) != len(joint_names):
        raise ValueError(f"Expected {len(joint_names)} joint values, got {len(joint_values)}.")

    result = dict(base_position_dict or {})
    for joint, value in zip(joint_names, joint_values):
        result[joint] = float(value)

    if gripper is not None:
        result[GRIPPER_JOINT] = float(gripper)
    elif base_position_dict is not None and GRIPPER_JOINT in base_position_dict:
        result[GRIPPER_JOINT] = float(base_position_dict[GRIPPER_JOINT])

    return result


def pose_from_position_rotation(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    if position.shape != (3,):
        raise ValueError(f"Expected position shape (3,), got {position.shape}.")
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected rotation shape (3, 3), got {rotation.shape}.")

    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    return pose


def target_relative_gripper_rotation(
    target_position: np.ndarray,
    gripper_angle_degrees: float,
    base_position: np.ndarray | None = None,
) -> np.ndarray:
    target_position = np.asarray(target_position, dtype=float)
    if target_position.shape != (3,):
        raise ValueError(f"Expected target position shape (3,), got {target_position.shape}.")

    if base_position is None:
        base_position = np.zeros(3)
    base_position = np.asarray(base_position, dtype=float)
    if base_position.shape != (3,):
        raise ValueError(f"Expected base position shape (3,), got {base_position.shape}.")

    radial = target_position - base_position
    radial[2] = 0.0
    radial_norm = np.linalg.norm(radial)
    if radial_norm == 0.0:
        raise ValueError("Target position must not be directly above the base origin.")
    radial /= radial_norm

    world_up = np.array([0.0, 0.0, 1.0])
    angle_rad = np.deg2rad(float(gripper_angle_degrees))
    approach_axis = np.cos(angle_rad) * radial + np.sin(angle_rad) * world_up
    approach_axis /= np.linalg.norm(approach_axis)

    sideways_axis = np.cross(world_up, radial)
    sideways_axis /= np.linalg.norm(sideways_axis)
    normal_axis = np.cross(approach_axis, sideways_axis)
    normal_axis /= np.linalg.norm(normal_axis)

    return np.column_stack((approach_axis, sideways_axis, normal_axis))


def pose_from_target_relative_gripper_angle(
    target_position: np.ndarray,
    gripper_angle_degrees: float,
    base_position: np.ndarray | None = None,
) -> np.ndarray:
    rotation = target_relative_gripper_rotation(
        target_position,
        gripper_angle_degrees,
        base_position=base_position,
    )
    return pose_from_position_rotation(target_position, rotation)


def translated_pose(pose: np.ndarray, offset_xyz: np.ndarray) -> np.ndarray:
    pose = _require_pose_matrix(pose).copy()
    offset_xyz = np.asarray(offset_xyz, dtype=float)
    if offset_xyz.shape != (3,):
        raise ValueError(f"Expected offset shape (3,), got {offset_xyz.shape}.")
    pose[:3, 3] += offset_xyz
    return pose


def tool_point_offset(tool_point: ToolPointName = CLAW_CENTER_TOOL_POINT) -> np.ndarray:
    try:
        return MUJOCO_GRIPPERFRAME_TOOL_OFFSETS[tool_point]
    except KeyError as exc:
        raise ValueError(f"Unknown tool point {tool_point!r}.") from exc


def gripperframe_pose_to_tool_target_pose(
    pose: np.ndarray,
    tool_point: ToolPointName = CLAW_CENTER_TOOL_POINT,
) -> np.ndarray:
    pose = _require_pose_matrix(pose).copy()
    pose[:3, 3] += pose[:3, :3] @ tool_point_offset(tool_point)
    return pose


def tool_target_pose_to_gripperframe_pose(
    pose: np.ndarray,
    tool_point: ToolPointName = CLAW_CENTER_TOOL_POINT,
) -> np.ndarray:
    pose = _require_pose_matrix(pose).copy()
    pose[:3, 3] -= pose[:3, :3] @ tool_point_offset(tool_point)
    return pose


def gripperframe_pose_to_claw_target_pose(pose: np.ndarray) -> np.ndarray:
    return gripperframe_pose_to_tool_target_pose(pose, CLAW_CENTER_TOOL_POINT)


def claw_target_pose_to_gripperframe_pose(pose: np.ndarray) -> np.ndarray:
    return tool_target_pose_to_gripperframe_pose(pose, CLAW_CENTER_TOOL_POINT)


def rotation_error_rad(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    rotation_a = np.asarray(rotation_a, dtype=float)
    rotation_b = np.asarray(rotation_b, dtype=float)
    if rotation_a.shape != (3, 3) or rotation_b.shape != (3, 3):
        raise ValueError("Expected both rotations to be 3x3 matrices.")

    delta = rotation_a.T @ rotation_b
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def urdf_pose_to_mujoco_gripperframe(pose: np.ndarray) -> np.ndarray:
    pose = _require_pose_matrix(pose).copy()
    pose[:3, :3] = pose[:3, :3] @ URDF_TO_MUJOCO_GRIPPERFRAME_ROTATION
    pose[:3, 3] += pose[:3, :3] @ URDF_TO_MUJOCO_GRIPPERFRAME_OFFSET
    return pose


def mujoco_gripperframe_to_urdf_pose(pose: np.ndarray) -> np.ndarray:
    pose = _require_pose_matrix(pose).copy()
    pose[:3, 3] -= pose[:3, :3] @ URDF_TO_MUJOCO_GRIPPERFRAME_OFFSET
    pose[:3, :3] = pose[:3, :3] @ URDF_TO_MUJOCO_GRIPPERFRAME_ROTATION.T
    return pose


class SO101Kinematics:
    def __init__(
        self,
        urdf_path: Path | str = DEFAULT_URDF_PATH,
        target_frame_name: str = TARGET_FRAME_NAME,
        joint_names: tuple[str, ...] = ARM_JOINT_ORDER,
    ):
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"SO-101 URDF not found: {self.urdf_path}")

        self.target_frame_name = target_frame_name
        self.joint_names = tuple(joint_names)
        self.backend_name = "lerobot"
        self._kinematics = RobotKinematics(
            str(self.urdf_path),
            target_frame_name=self.target_frame_name,
            joint_names=list(self.joint_names),
        )

    def forward_kinematics(self, position_dict: dict[str, float], frame: str = "mujoco") -> np.ndarray:
        joint_values = joint_dict_to_array(position_dict, self.joint_names)
        urdf_pose = self._kinematics.forward_kinematics(joint_values)
        if frame == "urdf":
            return urdf_pose
        if frame == "mujoco":
            return urdf_pose_to_mujoco_gripperframe(urdf_pose)
        raise ValueError("frame must be either 'mujoco' or 'urdf'.")

    def inverse_kinematics(
        self,
        current_position_dict: dict[str, float],
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
        frame: str = "mujoco",
        gripper: float | None = None,
        max_iterations: int = 25,
        position_tolerance_m: float = 1e-4,
    ) -> dict[str, float]:
        current_joint_values = joint_dict_to_array(current_position_dict, self.joint_names)
        desired_ee_pose = _require_pose_matrix(desired_ee_pose)

        if frame == "mujoco":
            desired_ee_pose = mujoco_gripperframe_to_urdf_pose(desired_ee_pose)
        elif frame != "urdf":
            raise ValueError("frame must be either 'mujoco' or 'urdf'.")

        result = current_joint_values
        for _ in range(max_iterations):
            result = self._kinematics.inverse_kinematics(
                result,
                desired_ee_pose,
                position_weight=position_weight,
                orientation_weight=orientation_weight,
            )
            solved_pose = self._kinematics.forward_kinematics(result)
            position_error = np.linalg.norm(solved_pose[:3, 3] - desired_ee_pose[:3, 3])
            if position_error <= position_tolerance_m:
                break

        return array_to_joint_dict(
            result,
            base_position_dict=current_position_dict,
            joint_names=self.joint_names,
            gripper=gripper,
        )
