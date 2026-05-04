from pathlib import Path

import numpy as np  # type: ignore[import-not-found]

from so101_mujoco_utils import JOINT_ORDER


MODEL_DIR = Path(__file__).parent / "simulation_code" / "model"
DEFAULT_URDF_PATH = MODEL_DIR / "so101_new_calib.urdf"
TARGET_FRAME_NAME = "gripper_frame_link"
MUJOCO_SITE_NAME = "gripperframe"
ARM_JOINT_ORDER = JOINT_ORDER[:-1]
GRIPPER_JOINT = JOINT_ORDER[-1]

# LeRobot's URDF frame and MuJoCo's gripperframe site share the same origin,
# but their local axes differ by this fixed rotation.
URDF_TO_MUJOCO_GRIPPERFRAME_ROTATION = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=float,
)


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


def translated_pose(pose: np.ndarray, offset_xyz: np.ndarray) -> np.ndarray:
    pose = _require_pose_matrix(pose).copy()
    offset_xyz = np.asarray(offset_xyz, dtype=float)
    if offset_xyz.shape != (3,):
        raise ValueError(f"Expected offset shape (3,), got {offset_xyz.shape}.")
    pose[:3, 3] += offset_xyz
    return pose


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
    return pose


def mujoco_gripperframe_to_urdf_pose(pose: np.ndarray) -> np.ndarray:
    pose = _require_pose_matrix(pose).copy()
    pose[:3, :3] = pose[:3, :3] @ URDF_TO_MUJOCO_GRIPPERFRAME_ROTATION.T
    return pose


class _DirectPlacoKinematics:
    def __init__(self, urdf_path: Path, target_frame_name: str, joint_names: tuple[str, ...]):
        try:
            import placo  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Direct Placo fallback requires `placo`. Install `lerobot[kinematics]` "
                "or `placo` in the active environment."
            ) from exc

        self.robot = placo.RobotWrapper(str(urdf_path))
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.target_frame_name = target_frame_name
        self.joint_names = list(joint_names)
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        joint_pos_rad = np.deg2rad(joint_pos_deg[: len(self.joint_names)])
        for joint_name, joint_value in zip(self.joint_names, joint_pos_rad):
            self.robot.set_joint(joint_name, joint_value)
        self.robot.update_kinematics()
        return self.robot.get_T_world_frame(self.target_frame_name)

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray:
        current_joint_rad = np.deg2rad(current_joint_pos[: len(self.joint_names)])
        for joint_name, joint_value in zip(self.joint_names, current_joint_rad):
            self.robot.set_joint(joint_name, joint_value)

        self.tip_frame.T_world_frame = desired_ee_pose
        self.tip_frame.configure(self.target_frame_name, "soft", position_weight, orientation_weight)
        self.solver.solve(True)
        self.robot.update_kinematics()

        return np.rad2deg([self.robot.get_joint(joint_name) for joint_name in self.joint_names])


class SO101Kinematics:
    def __init__(
        self,
        urdf_path: Path | str = DEFAULT_URDF_PATH,
        target_frame_name: str = TARGET_FRAME_NAME,
        joint_names: tuple[str, ...] = ARM_JOINT_ORDER,
        backend: str = "auto",
    ):
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"SO-101 URDF not found: {self.urdf_path}")

        self.target_frame_name = target_frame_name
        self.joint_names = tuple(joint_names)
        self.backend_name, self._backend = self._build_backend(backend)

    def _build_backend(self, backend: str):
        if backend not in {"auto", "lerobot", "placo"}:
            raise ValueError("backend must be one of: 'auto', 'lerobot', 'placo'.")

        errors = []
        if backend in {"auto", "lerobot"}:
            try:
                from lerobot.model.kinematics import RobotKinematics  # type: ignore[import-not-found]

                return (
                    "lerobot",
                    RobotKinematics(
                        str(self.urdf_path),
                        target_frame_name=self.target_frame_name,
                        joint_names=list(self.joint_names),
                    ),
                )
            except Exception as exc:
                if backend == "lerobot":
                    raise
                errors.append(f"LeRobot unavailable: {exc}")

        if backend in {"auto", "placo"}:
            try:
                return (
                    "placo",
                    _DirectPlacoKinematics(self.urdf_path, self.target_frame_name, self.joint_names),
                )
            except Exception as exc:
                if backend == "placo":
                    raise
                errors.append(f"direct Placo unavailable: {exc}")

        raise ImportError("Could not initialize SO-101 kinematics backend. " + " | ".join(errors))

    def forward_kinematics(self, position_dict: dict[str, float], frame: str = "mujoco") -> np.ndarray:
        joint_values = joint_dict_to_array(position_dict, self.joint_names)
        urdf_pose = self._backend.forward_kinematics(joint_values)
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
            result = self._backend.inverse_kinematics(
                result,
                desired_ee_pose,
                position_weight=position_weight,
                orientation_weight=orientation_weight,
            )
            solved_pose = self._backend.forward_kinematics(result)
            position_error = np.linalg.norm(solved_pose[:3, 3] - desired_ee_pose[:3, 3])
            if position_error <= position_tolerance_m:
                break

        return array_to_joint_dict(
            result,
            base_position_dict=current_position_dict,
            joint_names=self.joint_names,
            gripper=gripper,
        )
