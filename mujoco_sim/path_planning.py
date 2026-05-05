from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

import mujoco  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from sim_env import SimEnv
from so101_kinematics import (
    DEFAULT_URDF_PATH,
    TARGET_FRAME_NAME,
    urdf_pose_to_mujoco_gripperframe,
)
from so101_mujoco_utils import JOINT_ORDER, convert_to_dictionary, convert_to_list


ARM_JOINT_ORDER = JOINT_ORDER[:-1]
GRIPPER_JOINT = JOINT_ORDER[-1]
DEFAULT_GRIPPER_BODY_NAMES = ("gripper", "moving_jaw_so101_v1")
DEFAULT_OBJECT_BODY_NAMES = ("work_table", "cup", "second_cup")
DEFAULT_FLOOR_GEOM_NAME = "floor"

PlannerName = Literal["direct", "pyroboplan", "mujoco-rrt"]


@dataclass(frozen=True)
class MotionPlannerConfig:
    planner: PlannerName = "direct"
    timeout: float = 5.0
    step_size: float = 0.05
    collision_padding: float = 0.0
    rng_seed: int | None = None
    goal_bias: float = 0.2
    debug: bool = False
    pyroboplan_fallback: bool = True


@dataclass(frozen=True)
class CollisionPlanningContext:
    allowed_gripper_contact_geom_ids: frozenset[int] = frozenset()
    gripper_body_names: tuple[str, ...] = DEFAULT_GRIPPER_BODY_NAMES
    object_body_names: tuple[str, ...] = DEFAULT_OBJECT_BODY_NAMES
    attached_body_name: str | None = None
    attached_freejoint_name: str | None = None
    allowed_support_body_names: tuple[str, ...] = ()


class PlanningError(RuntimeError):
    pass


class PlannerUnavailableError(PlanningError):
    pass


class RuntimeCollisionGuard:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        collision_context: CollisionPlanningContext,
    ) -> None:
        self.model = model
        self.context = collision_context
        self.gripper_body_ids = _body_ids(model, collision_context.gripper_body_names)
        self.allowed_support_body_ids = _body_ids(model, collision_context.allowed_support_body_names)
        self.robot_geom_ids, self.obstacle_geom_ids = _partition_collision_geoms(model, collision_context.object_body_names)
        self.attached_body_id = _optional_body_id(model, collision_context.attached_body_name)
        self.initial_contact_pairs = _contact_pairs(data)

    def violation(self, data: mujoco.MjData) -> str | None:
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            pair = frozenset((geom1, geom2))
            if pair in self.initial_contact_pairs:
                continue
            if self._contact_is_allowed(geom1, geom2):
                continue
            if self._contact_is_relevant(geom1, geom2):
                return f"{self._geom_name(geom1)} <-> {self._geom_name(geom2)}"
        return None

    def _contact_is_allowed(self, geom1: int, geom2: int) -> bool:
        if geom1 in self.context.allowed_gripper_contact_geom_ids and _body_in_subtree(
            self.model, _geom_body_id(self.model, geom2), self.gripper_body_ids
        ):
            return True
        if geom2 in self.context.allowed_gripper_contact_geom_ids and _body_in_subtree(
            self.model, _geom_body_id(self.model, geom1), self.gripper_body_ids
        ):
            return True
        if self._is_allowed_support_contact(geom1, geom2):
            return True
        return False

    def _is_allowed_support_contact(self, geom1: int, geom2: int) -> bool:
        if self.attached_body_id is None or not self.allowed_support_body_ids:
            return False
        body1 = _geom_body_id(self.model, geom1)
        body2 = _geom_body_id(self.model, geom2)
        attached1 = _is_descendant_body(self.model, body1, self.attached_body_id)
        attached2 = _is_descendant_body(self.model, body2, self.attached_body_id)
        support1 = _body_in_subtree(self.model, body1, self.allowed_support_body_ids)
        support2 = _body_in_subtree(self.model, body2, self.allowed_support_body_ids)
        return (attached1 and support2) or (attached2 and support1)

    def _contact_is_relevant(self, geom1: int, geom2: int) -> bool:
        robot1, robot2 = geom1 in self.robot_geom_ids, geom2 in self.robot_geom_ids
        obstacle1, obstacle2 = geom1 in self.obstacle_geom_ids, geom2 in self.obstacle_geom_ids
        attached1 = self.attached_body_id is not None and _geom_body_id(self.model, geom1) == self.attached_body_id
        attached2 = self.attached_body_id is not None and _geom_body_id(self.model, geom2) == self.attached_body_id
        return (robot1 and obstacle2) or (robot2 and obstacle1) or (attached1 and obstacle2) or (attached2 and obstacle1)

    def _geom_name(self, geom_id: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or f"geom_{geom_id}"


def is_arm_motion(start_position: dict[str, float], target_position: dict[str, float], tolerance_degrees: float = 1e-5) -> bool:
    return any(abs(start_position[joint] - target_position[joint]) > tolerance_degrees for joint in ARM_JOINT_ORDER)


def interpolate_joint_path(
    start_position: dict[str, float],
    target_position: dict[str, float],
    max_step_size_rad: float,
) -> list[dict[str, float]]:
    max_step_degrees = max(1e-6, math.degrees(max_step_size_rad))
    distance = _configuration_distance(_arm_array_degrees(start_position), _arm_array_degrees(target_position))
    steps = max(1, math.ceil(distance / max_step_degrees))
    return [_interpolate_position(start_position, target_position, step / steps) for step in range(steps + 1)]


def plan_joint_path(
    env: SimEnv,
    target_position: dict[str, float],
    config: MotionPlannerConfig,
    collision_context: CollisionPlanningContext | None = None,
) -> list[dict[str, float]]:
    start_position = convert_to_dictionary(env.data.qpos.copy())
    if not is_arm_motion(start_position, target_position):
        return [start_position, dict(target_position)]

    if config.planner == "direct":
        return interpolate_joint_path(start_position, target_position, config.step_size)

    collision_context = collision_context or CollisionPlanningContext()
    if config.planner == "pyroboplan":
        try:
            path = _plan_with_pyroboplan(env, start_position, target_position, config)
            validator = MujocoPathValidator(env, config, collision_context)
            validator.validate_path(path)
            return path
        except PlannerUnavailableError:
            if not config.pyroboplan_fallback:
                raise
            print("pyroboplan is unavailable; falling back to MuJoCo RRT planner")
        except PlanningError as exc:
            if not config.pyroboplan_fallback:
                raise
            print(f"pyroboplan path was rejected ({exc}); falling back to MuJoCo RRT planner")

    return _plan_with_mujoco_rrt(env, start_position, target_position, config, collision_context)


def _plan_with_pyroboplan(
    env: SimEnv,
    start_position: dict[str, float],
    target_position: dict[str, float],
    config: MotionPlannerConfig,
) -> list[dict[str, float]]:
    try:
        import pinocchio  # type: ignore[import-not-found]
        from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions  # type: ignore[import-not-found]
        from pyroboplan.planning.utils import discretize_joint_space_path  # type: ignore[import-not-found]
    except ImportError as exc:
        raise PlannerUnavailableError("pyroboplan and Pinocchio are not installed in this environment") from exc

    model, collision_model, _ = pinocchio.buildModelsFromUrdf(str(DEFAULT_URDF_PATH))
    collision_model.addAllCollisionPairs()
    q_start = pinocchio.neutral(model)
    q_goal = pinocchio.neutral(model)
    _write_pinocchio_arm_q(model, q_start, start_position)
    _write_pinocchio_arm_q(model, q_goal, target_position)
    _verify_pyroboplan_fk(model, q_start, env.kinematics.forward_kinematics(start_position, frame="mujoco"))

    options = RRTPlannerOptions(
        max_step_size=config.step_size,
        max_connection_dist=max(config.step_size * 8.0, config.step_size),
        rrt_connect=True,
        bidirectional_rrt=True,
        max_planning_time=config.timeout,
        rng_seed=config.rng_seed,
        fast_return=True,
        goal_biasing_probability=config.goal_bias,
        collision_distance_padding=config.collision_padding,
    )
    planner = RRTPlanner(model, collision_model, options=options)
    raw_path = planner.plan(q_start, q_goal)
    if not raw_path:
        raise PlanningError("pyroboplan did not find a path")

    q_path = discretize_joint_space_path(raw_path, config.step_size)
    return [_position_from_pinocchio_q(model, q, target_position) for q in q_path]


def _write_pinocchio_arm_q(model: object, q: np.ndarray, position: dict[str, float]) -> None:
    for joint_name in ARM_JOINT_ORDER:
        joint_id = model.getJointId(joint_name)
        if joint_id == 0:
            raise PlannerUnavailableError(f"URDF model does not contain joint {joint_name!r}")
        q[int(model.idx_qs[joint_id])] = math.radians(position[joint_name])


def _position_from_pinocchio_q(model: object, q: np.ndarray, base_position: dict[str, float]) -> dict[str, float]:
    position = dict(base_position)
    for joint_name in ARM_JOINT_ORDER:
        joint_id = model.getJointId(joint_name)
        position[joint_name] = math.degrees(float(q[int(model.idx_qs[joint_id])]))
    return position


def _verify_pyroboplan_fk(model: object, q: np.ndarray, expected_mujoco_pose: np.ndarray) -> None:
    import pinocchio  # type: ignore[import-not-found]

    data = model.createData()
    pinocchio.framesForwardKinematics(model, data, q)
    frame_id = model.getFrameId(TARGET_FRAME_NAME)
    if frame_id < 0:
        raise PlannerUnavailableError(f"URDF model does not contain frame {TARGET_FRAME_NAME!r}")
    urdf_pose = data.oMf[frame_id].homogeneous
    mujoco_pose = urdf_pose_to_mujoco_gripperframe(urdf_pose)
    error = float(np.linalg.norm(mujoco_pose[:3, 3] - expected_mujoco_pose[:3, 3]))
    if error > 0.03:
        raise PlanningError(f"pyroboplan FK differs from LeRobot/MuJoCo FK by {error:.4f} m")


def _plan_with_mujoco_rrt(
    env: SimEnv,
    start_position: dict[str, float],
    target_position: dict[str, float],
    config: MotionPlannerConfig,
    collision_context: CollisionPlanningContext,
) -> list[dict[str, float]]:
    validator = MujocoPathValidator(env, config, collision_context)
    start = _arm_array_degrees(start_position)
    goal = _arm_array_degrees(target_position)
    max_step = max(1e-6, math.degrees(config.step_size))

    if validator.edge_is_valid(start, goal, max_step):
        return interpolate_joint_path(start_position, target_position, config.step_size)

    lower, upper = _joint_bounds_degrees(env.model)
    rng = np.random.default_rng(config.rng_seed)
    start_tree = _Tree(start)
    goal_tree = _Tree(goal)
    deadline = time.monotonic() + max(0.1, config.timeout)
    swapped = False

    while time.monotonic() < deadline:
        sample = goal if rng.random() < config.goal_bias else rng.uniform(lower, upper)
        new_index = _extend_tree(start_tree, sample, validator, max_step)
        if new_index is not None:
            new_config = start_tree.nodes[new_index]
            goal_index = _connect_tree(goal_tree, new_config, validator, max_step)
            if goal_index is not None and np.allclose(goal_tree.nodes[goal_index], new_config, atol=1e-6):
                path = _combine_trees(start_tree, new_index, goal_tree, goal_index, swapped)
                return [_position_from_arm_array(q, start_position, target_position) for q in path]

        start_tree, goal_tree = goal_tree, start_tree
        swapped = not swapped

    raise PlanningError("MuJoCo RRT planner did not find a collision-free path before timeout")


class MujocoPathValidator:
    def __init__(
        self,
        env: SimEnv,
        config: MotionPlannerConfig,
        collision_context: CollisionPlanningContext,
    ) -> None:
        self.model = env.model
        self.start_qpos = env.data.qpos.copy()
        self.config = config
        self.collision_context = collision_context
        self.data = mujoco.MjData(env.model)
        self.gripper_body_ids = _body_ids(env.model, collision_context.gripper_body_names)
        self.allowed_support_body_ids = _body_ids(env.model, collision_context.allowed_support_body_names)
        self.attachment_anchor_body_id = _optional_body_id(
            env.model,
            collision_context.gripper_body_names[0] if collision_context.gripper_body_names else None,
        )
        self.robot_geom_ids, self.obstacle_geom_ids = _partition_collision_geoms(env.model, collision_context.object_body_names)
        self.attached_body_id = _optional_body_id(env.model, collision_context.attached_body_name)
        self.attached_freejoint_qpos_address = _optional_freejoint_qpos_address(env.model, collision_context.attached_freejoint_name)
        self.attachment_transform = self._attachment_transform(env.data)
        self.initial_contact_pairs = self._contact_pairs_for_current_state()

    def validate_path(self, path: list[dict[str, float]]) -> None:
        if len(path) < 2:
            raise PlanningError("Planner returned an empty path")
        max_step = max(1e-6, math.degrees(self.config.step_size))
        for first, second in zip(path, path[1:], strict=False):
            if not self.edge_is_valid(_arm_array_degrees(first), _arm_array_degrees(second), max_step):
                raise PlanningError("MuJoCo validation rejected planned path")

    def edge_is_valid(self, start: np.ndarray, goal: np.ndarray, max_step_degrees: float) -> bool:
        distance = _configuration_distance(start, goal)
        steps = max(1, math.ceil(distance / max_step_degrees))
        for step in range(steps + 1):
            alpha = step / steps
            if not self.state_is_valid((1.0 - alpha) * start + alpha * goal, allow_initial_contacts=step == 0):
                if self.config.debug:
                    print(f"planner rejected path sample {step}/{steps}")
                return False
        return True

    def state_is_valid(self, arm_degrees: np.ndarray, allow_initial_contacts: bool = False) -> bool:
        self._set_arm_state(arm_degrees)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = frozenset((int(contact.geom1), int(contact.geom2)))
            if allow_initial_contacts and pair in self.initial_contact_pairs:
                continue
            if self._contact_is_allowed(int(contact.geom1), int(contact.geom2)):
                continue
            if self._contact_is_relevant(int(contact.geom1), int(contact.geom2)):
                if self.config.debug:
                    print(f"planner rejected contact: {self._geom_name(contact.geom1)} <-> {self._geom_name(contact.geom2)}")
                return False

        if self.config.collision_padding > 0.0 and self._has_clearance_violation():
            return False
        return True

    def _set_arm_state(self, arm_degrees: np.ndarray) -> None:
        self.data.qpos[:] = self.start_qpos
        position = convert_to_dictionary(self.start_qpos)
        for joint_name, value in zip(ARM_JOINT_ORDER, arm_degrees, strict=True):
            position[joint_name] = float(value)
        self.data.qpos[: len(JOINT_ORDER)] = convert_to_list(position)
        mujoco.mj_forward(self.model, self.data)

        if self.attachment_transform is None or self.attached_freejoint_qpos_address is None:
            return
        if self.attachment_anchor_body_id is None:
            return
        gripper_pose = _body_pose(self.data, self.attachment_anchor_body_id)
        attached_pose = gripper_pose @ self.attachment_transform
        self.data.qpos[self.attached_freejoint_qpos_address : self.attached_freejoint_qpos_address + 7] = _freejoint_qpos(attached_pose)
        mujoco.mj_forward(self.model, self.data)

    def _attachment_transform(self, data: mujoco.MjData) -> np.ndarray | None:
        if self.attached_body_id is None or not self.gripper_body_ids:
            return None
        if self.attachment_anchor_body_id is None:
            return None
        return np.linalg.inv(_body_pose(data, self.attachment_anchor_body_id)) @ _body_pose(data, self.attached_body_id)

    def _contact_pairs_for_current_state(self) -> set[frozenset[int]]:
        self._set_arm_state(_arm_array_degrees(convert_to_dictionary(self.start_qpos)))
        return _contact_pairs(self.data)

    def _contact_is_allowed(self, geom1: int, geom2: int) -> bool:
        if geom1 in self.collision_context.allowed_gripper_contact_geom_ids and _body_in_subtree(
            self.model, _geom_body_id(self.model, geom2), self.gripper_body_ids
        ):
            return True
        if geom2 in self.collision_context.allowed_gripper_contact_geom_ids and _body_in_subtree(
            self.model, _geom_body_id(self.model, geom1), self.gripper_body_ids
        ):
            return True
        if self._is_allowed_support_contact(geom1, geom2):
            return True
        return False

    def _is_allowed_support_contact(self, geom1: int, geom2: int) -> bool:
        if self.attached_body_id is None or not self.allowed_support_body_ids:
            return False
        body1 = _geom_body_id(self.model, geom1)
        body2 = _geom_body_id(self.model, geom2)
        attached1 = _is_descendant_body(self.model, body1, self.attached_body_id)
        attached2 = _is_descendant_body(self.model, body2, self.attached_body_id)
        support1 = _body_in_subtree(self.model, body1, self.allowed_support_body_ids)
        support2 = _body_in_subtree(self.model, body2, self.allowed_support_body_ids)
        return (attached1 and support2) or (attached2 and support1)

    def _contact_is_relevant(self, geom1: int, geom2: int) -> bool:
        robot1, robot2 = geom1 in self.robot_geom_ids, geom2 in self.robot_geom_ids
        obstacle1, obstacle2 = geom1 in self.obstacle_geom_ids, geom2 in self.obstacle_geom_ids
        attached1 = self.attached_body_id is not None and _geom_body_id(self.model, geom1) == self.attached_body_id
        attached2 = self.attached_body_id is not None and _geom_body_id(self.model, geom2) == self.attached_body_id
        return (robot1 and obstacle2) or (robot2 and obstacle1) or (attached1 and obstacle2) or (attached2 and obstacle1)

    def _has_clearance_violation(self) -> bool:
        fromto = np.zeros(6, dtype=float)
        distmax = float(self.config.collision_padding)
        active_robot_geoms = self.robot_geom_ids
        if self.attached_body_id is not None:
            active_robot_geoms = active_robot_geoms | {
                geom_id for geom_id in range(self.model.ngeom) if _geom_body_id(self.model, geom_id) == self.attached_body_id
            }
        for geom1 in active_robot_geoms:
            for geom2 in self.obstacle_geom_ids:
                if geom1 == geom2 or _geom_body_id(self.model, geom1) == _geom_body_id(self.model, geom2):
                    continue
                distance = float(mujoco.mj_geomDistance(self.model, self.data, geom1, geom2, distmax, fromto))
                if distance < distmax:
                    if self.config.debug:
                        print(f"planner rejected clearance: {self._geom_name(geom1)} <-> {self._geom_name(geom2)} distance={distance:.4f}")
                    return True
        return False

    def _geom_name(self, geom_id: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or f"geom_{geom_id}"


class _Tree:
    def __init__(self, root: np.ndarray) -> None:
        self.nodes: list[np.ndarray] = [root.copy()]
        self.parents: list[int | None] = [None]

    def add(self, node: np.ndarray, parent: int) -> int:
        self.nodes.append(node.copy())
        self.parents.append(parent)
        return len(self.nodes) - 1

    def nearest_index(self, sample: np.ndarray) -> int:
        distances = [float(np.linalg.norm(node - sample)) for node in self.nodes]
        return int(np.argmin(distances))

    def path_to_root(self, index: int) -> list[np.ndarray]:
        path: list[np.ndarray] = []
        current: int | None = index
        while current is not None:
            path.append(self.nodes[current])
            current = self.parents[current]
        path.reverse()
        return path


def _extend_tree(tree: _Tree, sample: np.ndarray, validator: MujocoPathValidator, max_step: float) -> int | None:
    nearest_index = tree.nearest_index(sample)
    nearest = tree.nodes[nearest_index]
    direction = sample - nearest
    distance = float(np.linalg.norm(direction))
    if distance < 1e-9:
        return nearest_index
    candidate = nearest + direction / distance * min(max_step, distance)
    if not validator.edge_is_valid(nearest, candidate, max_step):
        return None
    return tree.add(candidate, nearest_index)


def _connect_tree(tree: _Tree, target: np.ndarray, validator: MujocoPathValidator, max_step: float) -> int | None:
    current_index = tree.nearest_index(target)
    while True:
        current = tree.nodes[current_index]
        direction = target - current
        distance = float(np.linalg.norm(direction))
        if distance < 1e-6:
            return current_index
        candidate = current + direction / distance * min(max_step, distance)
        if not validator.edge_is_valid(current, candidate, max_step):
            return None
        current_index = tree.add(candidate, current_index)


def _combine_trees(start_tree: _Tree, start_index: int, goal_tree: _Tree, goal_index: int, swapped: bool) -> list[np.ndarray]:
    start_path = start_tree.path_to_root(start_index)
    goal_path = goal_tree.path_to_root(goal_index)
    if swapped:
        return goal_path + list(reversed(start_path[:-1]))
    return start_path + list(reversed(goal_path[:-1]))


def _joint_bounds_degrees(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    lower: list[float] = []
    upper: list[float] = []
    for joint_name in ARM_JOINT_ORDER:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise PlanningError(f"Could not find joint {joint_name!r}")
        lower.append(math.degrees(float(model.jnt_range[joint_id, 0])))
        upper.append(math.degrees(float(model.jnt_range[joint_id, 1])))
    return np.array(lower, dtype=float), np.array(upper, dtype=float)


def _partition_collision_geoms(model: mujoco.MjModel, object_body_names: tuple[str, ...]) -> tuple[set[int], set[int]]:
    object_body_ids = {_optional_body_id(model, name) for name in object_body_names}
    object_body_ids.discard(None)
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, DEFAULT_FLOOR_GEOM_NAME)
    robot_geom_ids: set[int] = set()
    obstacle_geom_ids: set[int] = set()
    for geom_id in range(model.ngeom):
        if model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
            continue
        body_id = _geom_body_id(model, geom_id)
        if geom_id == floor_geom_id or any(_is_descendant_body(model, body_id, object_body_id) for object_body_id in object_body_ids):
            obstacle_geom_ids.add(geom_id)
        else:
            robot_geom_ids.add(geom_id)
    return robot_geom_ids, obstacle_geom_ids


def _body_ids(model: mujoco.MjModel, body_names: tuple[str, ...]) -> set[int]:
    ids = {_optional_body_id(model, name) for name in body_names}
    ids.discard(None)
    return {int(body_id) for body_id in ids}


def _body_in_subtree(model: mujoco.MjModel, body_id: int, ancestor_body_ids: set[int]) -> bool:
    return any(_is_descendant_body(model, body_id, ancestor_body_id) for ancestor_body_id in ancestor_body_ids)


def _contact_pairs(data: mujoco.MjData) -> set[frozenset[int]]:
    return {frozenset((int(data.contact[index].geom1), int(data.contact[index].geom2))) for index in range(data.ncon)}


def _optional_body_id(model: mujoco.MjModel, body_name: str | None) -> int | None:
    if body_name is None:
        return None
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return int(body_id) if body_id >= 0 else None


def _optional_freejoint_qpos_address(model: mujoco.MjModel, joint_name: str | None) -> int | None:
    if joint_name is None:
        return None
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return None
    return int(model.jnt_qposadr[joint_id])


def _geom_body_id(model: mujoco.MjModel, geom_id: int) -> int:
    return int(model.geom_bodyid[int(geom_id)])


def _is_descendant_body(model: mujoco.MjModel, body_id: int, ancestor_body_id: int | None) -> bool:
    if ancestor_body_id is None:
        return False
    current = int(body_id)
    while current >= 0:
        if current == ancestor_body_id:
            return True
        if current == 0:
            return False
        current = int(model.body_parentid[current])
    return False


def _body_pose(data: mujoco.MjData, body_id: int) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = data.xmat[body_id].reshape(3, 3)
    pose[:3, 3] = data.xpos[body_id]
    return pose


def _freejoint_qpos(pose: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)
    return np.array([pose[0, 3], pose[1, 3], pose[2, 3], *quat_wxyz], dtype=float)


def _arm_array_degrees(position: dict[str, float]) -> np.ndarray:
    return np.array([position[joint] for joint in ARM_JOINT_ORDER], dtype=float)


def _position_from_arm_array(
    arm_degrees: np.ndarray,
    start_position: dict[str, float],
    target_position: dict[str, float],
) -> dict[str, float]:
    position = dict(target_position)
    for joint_name, value in zip(ARM_JOINT_ORDER, arm_degrees, strict=True):
        position[joint_name] = float(value)
    position[GRIPPER_JOINT] = target_position.get(GRIPPER_JOINT, start_position[GRIPPER_JOINT])
    return position


def _interpolate_position(start_position: dict[str, float], target_position: dict[str, float], alpha: float) -> dict[str, float]:
    return {joint: (1.0 - alpha) * start_position[joint] + alpha * target_position[joint] for joint in JOINT_ORDER}


def _configuration_distance(start: np.ndarray, goal: np.ndarray) -> float:
    return float(np.linalg.norm(goal - start))
