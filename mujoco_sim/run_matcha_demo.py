from __future__ import annotations

"""Scripted SO101 matcha demo.

Run with the MuJoCo viewer on macOS:

    conda activate whisk-agent
    cd <repo root>
    mjpython mujoco_sim/run_matcha_demo.py

Run headless with regular Python:

    python mujoco_sim/run_matcha_demo.py --headless

This demo uses NVIDIA kitchen MJCF object meshes for the visual cups, ice, and
whisk, plus simple primitive collision proxies for stable interaction. The cup is
modeled as a hollow dodecagonal tube so the whisk head can descend INTO it, and
the whisk has a spool-shaped grip band that the gripper jaws can mechanically
lock onto.

The whisk uses assisted tracking by default after the gripper closes: the freejoint
tracks the real gripper pad midpoint so the visual grip stays stable while the arm
moves to the cup. Pass `--physics-only-whisk` to disable that assist and debug the
raw MuJoCo contact behavior.

The whisk starts vertical, with its local head pointing down toward the cup.
"""

import argparse
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import mujoco  # type: ignore[import-not-found]
import mujoco.viewer  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gripper import OPEN_GRIPPER
from motion import solve_ik
from sim_env import SimEnv, create_env
from so101_mujoco_utils import JOINT_ORDER, convert_to_dictionary, send_position_command


FIXED_JAW_BODY_NAME = "gripper"
MOVING_JAW_BODY_NAME = "moving_jaw_so101_v1"
VERTICAL_WHISK_QUAT = (0.7071068, -0.7071068, 0.0, 0.0)
MATCHA_CLOSED_GRIPPER = -5.0
WHISK_INITIAL_LOCK_NAME = "whisk_initial_lock"
# The whisk body's center is 35 mm BELOW its grip band in world Z when held vertically with the
# head pointing down (because the grip band geom is at body-local (0, -0.035, 0) and the body's
# quat rotates +Y to world -Z). For assisted-whisk mode, the gripper's IK target is at the grip
# band height, so the body pose must be shifted DOWN by this offset to align the grip band with
# the closed gripper jaws.
GRIPPER_TO_BODY_Z_OFFSET = np.array([0.0, 0.0, -0.035], dtype=float)


@dataclass(frozen=True)
class MatchaConfig:
    scene_path: Path = ROOT_DIR / "simulation_code" / "model" / "scene_matcha.xml"
    main_cup_position: tuple[float, float, float] = (0.32, 0.0, 0.045)
    ice_cup_position: tuple[float, float, float] = (0.42, -0.11, 0.045)
    # Whisk stands vertically on its head base. Body z = 0.077 puts the head bottom (10 mm thick
    # box centered 72 mm below body in world Z) on the table top (z=0).
    whisk_position: tuple[float, float, float] = (0.25, 0.12, 0.077)
    whisk_quat: tuple[float, float, float, float] = VERTICAL_WHISK_QUAT
    whisk_body_name: str = "matcha_whisk"
    whisk_freejoint_name: str = "matcha_whisk_freejoint"
    main_cup_body_name: str = "main_cup"
    ice_cup_body_name: str = "ice_cup"
    approach_height: float = 0.15
    grasp_target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # IK target z is the world Z of the SO101 claw frame, which is what the assisted-whisk script
    # uses as the reference for placing the whisk freejoint. The whisk body is offset DOWN from
    # this target by GRIPPER_TO_BODY_Z_OFFSET (-35 mm) so that the grip band aligns with the
    # gripper's claw frame (and visually with the closed pads when the strong arm actuators
    # successfully drive the gripper to the IK pose).
    grasp_height: float = 0.125
    # Cup-descend IK target: world Z of the gripper claw when the whisk is inside the cup. The
    # whisk body sits 35 mm below this. We want the whisk head 25 mm above the cup floor (head
    # world z = 0.030, body z = 0.102, claw z = body z + 0.035 = 0.137).
    whisk_tip_height: float = 0.065
    whisk_tip_to_body_height: float = 0.072
    # Vertical clearance between the gripper-at-cup-center pose and the
    # gripper-at-cup-approach pose. 0.080 keeps the whisk head ~20 mm above the
    # cup rim before descending so it cleanly enters the hollow cup.
    cup_approach_height: float = 0.150
    lift_height: float = 0.18
    whisk_stroke_length: float = 0.010
    whisk_strokes: int = 18
    approach_duration: float = 3.0
    preclose_duration: float = 0.8
    descend_duration: float = 1.6
    close_duration: float = 1.4
    squeeze_duration: float = 0.05
    lift_duration: float = 2.0
    whisk_duration: float = 4.0
    final_hold_duration: float = 1.0
    jaw_friction: tuple[float, float, float] = (12.0, 0.18, 0.018)
    object_friction: tuple[float, float, float] = (16.0, 0.24, 0.024)
    gripper_force: float = 50.0
    # Default sts3215 actuator forcerange (~2.94 N*m) is the realistic motor limit, but the
    # arm sags substantially under its own weight at full reach. Boost the per-arm-joint force
    # AND gain so the position controller can actually track the IK plan within ~0.1 deg.
    arm_joint_force: float = 80.0
    arm_joint_kp: float = 3000.0
    arm_joint_kv: float = 20.0
    # Approach the whisk with the gripper FULLY open (clears the whisk top), then narrow to
    # this preclose angle (~9 deg / ~36 mm separation) before descending so the open-jaw moving
    # body doesn't collide with the cup walls or table during the precise descent into the cup.
    preclose_gripper: float = 5.0
    # MuJoCo's rigid-body contact dynamics make long-thin-tool grasps brittle: the whisk
    # tends to slip through the parallel jaws under transport accelerations even with high
    # friction and large flanges. We default to the scripted-whisk path (assisted_whisk=True)
    # so the demo reliably picks up the whisk, descends INTO the cup, whisks, and lifts back
    # out. The strong arm actuators (above) plus the corrected GRIPPER_TO_BODY_Z_OFFSET keep
    # the whisk visually positioned cleanly between the gripper jaws -- no floating below the
    # claws. Pass --physics-only-whisk to disable the assist for physics-debugging mode.
    assisted_whisk: bool = True


@dataclass
class MatchaDiagnostics:
    model: mujoco.MjModel
    whisk_body_id: int
    main_cup_body_id: int
    fixed_jaw_body_id: int
    moving_jaw_body_id: int
    fixed_pad_geom_id: int
    moving_pad_geom_id: int
    whisk_geom_ids: set[int]
    steps: int = 0
    whisk_contact_steps: int = 0
    fixed_contact_steps: int = 0
    moving_contact_steps: int = 0
    max_contacts: int = 0
    contact_pairs: set[tuple[str, str]] = field(default_factory=set)

    def update(self, data: mujoco.MjData) -> None:
        self.steps += 1
        whisk_contacts = 0
        touching_fixed = False
        touching_moving = False

        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in self.whisk_geom_ids:
                whisk_geom_id, other_geom_id = geom1, geom2
            elif geom2 in self.whisk_geom_ids:
                whisk_geom_id, other_geom_id = geom2, geom1
            else:
                continue

            whisk_contacts += 1
            other_body_id = int(self.model.geom_bodyid[other_geom_id])
            touching_fixed = touching_fixed or other_body_id == self.fixed_jaw_body_id
            touching_moving = touching_moving or other_body_id == self.moving_jaw_body_id
            self.contact_pairs.add(
                (
                    _object_name(self.model, mujoco.mjtObj.mjOBJ_GEOM, whisk_geom_id),
                    _object_name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom_id),
                )
            )

        if whisk_contacts:
            self.whisk_contact_steps += 1
            self.max_contacts = max(self.max_contacts, whisk_contacts)
        if touching_fixed:
            self.fixed_contact_steps += 1
        if touching_moving:
            self.moving_contact_steps += 1


@dataclass(frozen=True)
class MatchaResult:
    completed: bool
    initial_whisk_position: tuple[float, float, float]
    final_whisk_position: tuple[float, float, float]
    diagnostics: MatchaDiagnostics


@dataclass(frozen=True)
class WhiskJawTracker:
    joint_name: str
    body_from_jaws: np.ndarray


def _object_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name or f"{obj_type.name.lower()}_{obj_id}"


def _require_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Could not find {obj_type.name.lower()} named {name!r}.")
    return obj_id


def _body_geom_ids(model: mujoco.MjModel, body_id: int, contact_only: bool = False) -> set[int]:
    geom_ids: set[int] = set()
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id:
            continue
        if contact_only and model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
            continue
        geom_ids.add(geom_id)
    return geom_ids


def _set_freejoint_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    position: tuple[float, float, float] | np.ndarray,
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
    joint_id = _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_address = int(model.jnt_qposadr[joint_id])
    qvel_address = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_address : qpos_address + 7] = [*np.asarray(position, dtype=float), *quat]
    data.qvel[qvel_address : qvel_address + 6] = 0.0
    mujoco.mj_forward(model, data)


def _set_body_position(model: mujoco.MjModel, body_name: str, position: tuple[float, float, float]) -> None:
    body_id = _require_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    model.body_pos[body_id] = np.asarray(position, dtype=float)


def _set_actuator_force(model: mujoco.MjModel, actuator_name: str, force: float) -> None:
    actuator_id = _require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    model.actuator_forcerange[actuator_id] = [-force, force]


def _set_actuator_gain(model: mujoco.MjModel, actuator_name: str, kp: float, kv: float | None = None) -> None:
    """Set the position-actuator gains. The MJCF position actuator computes
    force = kp*(ctrl - qpos) - kv*qvel via gainprm[0] = kp and biasprm[1] = -kp,
    biasprm[2] = -kv. Bumping kp lets the controller resist gravity torque with
    much smaller steady-state position error.
    """
    actuator_id = _require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    model.actuator_gainprm[actuator_id, 0] = kp
    model.actuator_biasprm[actuator_id, 1] = -kp
    if kv is not None:
        model.actuator_biasprm[actuator_id, 2] = -kv


def _set_geom_friction(model: mujoco.MjModel, geom_ids: set[int], friction: tuple[float, float, float]) -> None:
    for geom_id in geom_ids:
        model.geom_friction[geom_id] = friction


def _set_geom_collisions_enabled(model: mujoco.MjModel, geom_ids: set[int], enabled: bool) -> None:
    contype = 1 if enabled else 0
    conaffinity = 1 if enabled else 0
    for geom_id in geom_ids:
        model.geom_contype[geom_id] = contype
        model.geom_conaffinity[geom_id] = conaffinity


def _release_whisk_lock(env: SimEnv, equality_name: str = WHISK_INITIAL_LOCK_NAME) -> None:
    """Disable the weld that pins the whisk at its initial pose.

    The whisk is statically unstable when standing vertically on its narrow head, so the scene
    includes a weld equality constraint that holds it in place until the gripper has clamped onto
    the grip band. This helper deactivates the weld so the rest of the demo runs purely on physics.
    """
    eq_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name)
    if eq_id < 0:
        return
    env.data.eq_active[eq_id] = 0
    mujoco.mj_forward(env.model, env.data)


def _set_named_geom_friction(model: mujoco.MjModel, geom_name: str, friction: tuple[float, float, float]) -> None:
    geom_id = _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    model.geom_friction[geom_id] = friction


def _disable_anonymous_gripper_mesh_collisions(model: mujoco.MjModel, body_ids: set[int]) -> None:
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) not in body_ids:
            continue
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) is not None:
            continue
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0


def configure_matcha_env(env: SimEnv, config: MatchaConfig) -> MatchaDiagnostics:
    whisk_body_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, config.whisk_body_name)
    main_cup_body_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, config.main_cup_body_name)
    fixed_jaw_body_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, FIXED_JAW_BODY_NAME)
    moving_jaw_body_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, MOVING_JAW_BODY_NAME)

    _set_body_position(env.model, config.main_cup_body_name, config.main_cup_position)
    _set_body_position(env.model, config.ice_cup_body_name, config.ice_cup_position)
    _set_freejoint_pose(env.model, env.data, config.whisk_freejoint_name, config.whisk_position, config.whisk_quat)
    _set_actuator_force(env.model, "gripper", config.gripper_force)
    # The default sts3215 actuator (kp=998, forcerange=2.94 N*m) is realistic but too weak/soft
    # for a precision scripted demo: the arm sags several degrees under gravity, putting the
    # gripper several centimeters away from the IK target. Boost both forcerange (so the actuator
    # can apply enough torque) AND gain (so the steady-state position error is small).
    for arm_joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"):
        _set_actuator_force(env.model, arm_joint, config.arm_joint_force)
        _set_actuator_gain(env.model, arm_joint, kp=config.arm_joint_kp, kv=config.arm_joint_kv)
    _disable_anonymous_gripper_mesh_collisions(env.model, {fixed_jaw_body_id, moving_jaw_body_id})

    whisk_geom_ids = _body_geom_ids(env.model, whisk_body_id, contact_only=True)
    fixed_jaw_geom_ids = _body_geom_ids(env.model, fixed_jaw_body_id, contact_only=True)
    moving_jaw_geom_ids = _body_geom_ids(env.model, moving_jaw_body_id, contact_only=True)
    _set_geom_friction(env.model, whisk_geom_ids, (2.0, 0.03, 0.003))
    _set_named_geom_friction(env.model, "matcha_whisk_handle_proxy", config.object_friction)
    _set_named_geom_friction(env.model, "matcha_whisk_grip_band_proxy", config.object_friction)
    _set_named_geom_friction(env.model, "matcha_whisk_head_proxy", (0.25, 0.005, 0.0005))
    _set_geom_friction(env.model, fixed_jaw_geom_ids | moving_jaw_geom_ids, config.jaw_friction)
    mujoco.mj_forward(env.model, env.data)

    fixed_pad_geom_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_grip_pad")
    moving_pad_geom_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_grip_pad")

    return MatchaDiagnostics(
        model=env.model,
        whisk_body_id=whisk_body_id,
        main_cup_body_id=main_cup_body_id,
        fixed_jaw_body_id=fixed_jaw_body_id,
        moving_jaw_body_id=moving_jaw_body_id,
        fixed_pad_geom_id=fixed_pad_geom_id,
        moving_pad_geom_id=moving_pad_geom_id,
        whisk_geom_ids=whisk_geom_ids,
    )


def _interpolate_position(start_position: dict[str, float], target_position: dict[str, float], alpha: float) -> dict[str, float]:
    return {
        joint: (1.0 - alpha) * start_position[joint] + alpha * target_position[joint]
        for joint in JOINT_ORDER
    }


def _step_once(
    env: SimEnv,
    diagnostics: MatchaDiagnostics,
    viewer: mujoco.viewer.Handle | None,
    realtime: bool,
    post_step: Callable[[], None] | None = None,
) -> bool:
    step_start = time.time()
    mujoco.mj_step(env.model, env.data)
    if post_step is not None:
        post_step()
    diagnostics.update(env.data)

    if viewer is not None:
        viewer.sync()
        if not viewer.is_running():
            return False

    if realtime:
        sleep_time = env.model.opt.timestep - (time.time() - step_start)
        if sleep_time > 0:
            time.sleep(sleep_time)
    return True


def _jaws_midpoint(env: SimEnv, diagnostics: MatchaDiagnostics) -> np.ndarray:
    fixed_pad = env.data.geom_xpos[diagnostics.fixed_pad_geom_id]
    moving_pad = env.data.geom_xpos[diagnostics.moving_pad_geom_id]
    return 0.5 * (fixed_pad + moving_pad)


def _capture_whisk_tracker(env: SimEnv, diagnostics: MatchaDiagnostics, joint_name: str) -> WhiskJawTracker:
    body_xyz = np.array(env.data.xpos[diagnostics.whisk_body_id], dtype=float)
    return WhiskJawTracker(joint_name=joint_name, body_from_jaws=body_xyz - _jaws_midpoint(env, diagnostics))


def _track_jaws_for_whisk(env: SimEnv, diagnostics: MatchaDiagnostics, tracker: WhiskJawTracker) -> None:
    """Preserve the actual whisk-to-jaw transform captured when the gripper closes.

    Earlier versions hardcoded the grip band to the jaw midpoint, which made the whisk visibly
    teleport along the claw after pickup if the simulated contact settled somewhere else. Capturing
    the real post-close offset keeps the grasp location fixed, like a real gripper would.
    """
    body_xyz = _jaws_midpoint(env, diagnostics) + tracker.body_from_jaws
    _set_freejoint_pose(env.model, env.data, tracker.joint_name, body_xyz, VERTICAL_WHISK_QUAT)


def _target_for_tracked_body(body_xyz: np.ndarray, tracker: WhiskJawTracker | None) -> np.ndarray:
    if tracker is None:
        return body_xyz
    return body_xyz - tracker.body_from_jaws


def command_motion(
    env: SimEnv,
    target_position: dict[str, float],
    duration: float,
    diagnostics: MatchaDiagnostics,
    viewer: mujoco.viewer.Handle | None = None,
    realtime: bool = False,
    carried_whisk: tuple[str, np.ndarray, np.ndarray] | WhiskJawTracker | None = None,
) -> bool:
    start_position = convert_to_dictionary(env.data.qpos.copy())
    steps = max(1, math.ceil(duration / env.model.opt.timestep))

    for step_index in range(steps):
        alpha = (step_index + 1) / steps
        command = _interpolate_position(start_position, target_position, alpha)
        send_position_command(env.data, command)
        post_step: Callable[[], None] | None = None
        if isinstance(carried_whisk, WhiskJawTracker):
            _track_jaws_for_whisk(env, diagnostics, carried_whisk)
            post_step = lambda tracker=carried_whisk: _track_jaws_for_whisk(env, diagnostics, tracker)
        elif carried_whisk is not None:
            joint_name, start_xyz, end_xyz = carried_whisk
            _set_freejoint_pose(env.model, env.data, joint_name, (1.0 - alpha) * start_xyz + alpha * end_xyz, VERTICAL_WHISK_QUAT)
        if not _step_once(env, diagnostics, viewer, realtime, post_step=post_step):
            return False

    env.current_position = dict(target_position)
    return True


def hold_command(
    env: SimEnv,
    target_position: dict[str, float],
    duration: float,
    diagnostics: MatchaDiagnostics,
    viewer: mujoco.viewer.Handle | None = None,
    realtime: bool = False,
    carried_whisk: tuple[str, np.ndarray] | WhiskJawTracker | None = None,
) -> bool:
    steps = max(1, math.ceil(duration / env.model.opt.timestep))
    for _ in range(steps):
        send_position_command(env.data, target_position)
        post_step: Callable[[], None] | None = None
        if isinstance(carried_whisk, WhiskJawTracker):
            _track_jaws_for_whisk(env, diagnostics, carried_whisk)
            post_step = lambda tracker=carried_whisk: _track_jaws_for_whisk(env, diagnostics, tracker)
        elif carried_whisk is not None:
            joint_name, xyz = carried_whisk
            _set_freejoint_pose(env.model, env.data, joint_name, xyz, VERTICAL_WHISK_QUAT)
        if not _step_once(env, diagnostics, viewer, realtime, post_step=post_step):
            return False
    env.current_position = dict(target_position)
    return True


def solve_target(env: SimEnv, xyz: np.ndarray, gripper_position: float) -> dict[str, float]:
    plan = solve_ik(env, xyz, gripper_position=gripper_position)
    target = plan.target_pose[:3, 3]
    print(
        "target: "
        f"x={target[0]:.4f} y={target[1]:.4f} z={target[2]:.4f} m, "
        f"gripper={gripper_position:.1f}, IK error={plan.position_error:.6f} m"
    )
    return plan.target_position


def show_marker(viewer: mujoco.viewer.Handle | None, xyz: np.ndarray) -> None:
    if viewer is None:
        return
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.008, 0.0, 0.0],
        pos=xyz,
        mat=np.eye(3).flatten(),
        rgba=[0.0, 0.2, 1.0, 0.75],
    )
    viewer.user_scn.ngeom = 1
    viewer.sync()


def whisk_motion(
    env: SimEnv,
    config: MatchaConfig,
    center_xyz: np.ndarray,
    diagnostics: MatchaDiagnostics,
    viewer: mujoco.viewer.Handle | None = None,
    realtime: bool = False,
    tracker: WhiskJawTracker | None = None,
) -> bool:
    waypoint_count = max(2, config.whisk_strokes * 2)
    segment_duration = config.whisk_duration / waypoint_count
    half_stroke = 0.5 * config.whisk_stroke_length
    grasp_offset = np.asarray(config.grasp_target_offset, dtype=float)

    for waypoint_index in range(1, waypoint_count + 1):
        direction = 1.0 if waypoint_index % 2 else -1.0
        small_side_jitter = 0.003 * math.sin(math.pi * waypoint_index / 2.0)
        gripper_xyz = np.array(
            [
                center_xyz[0] + direction * half_stroke,
                center_xyz[1] + small_side_jitter,
                center_xyz[2],
            ],
            dtype=float,
        )
        target_position = solve_target(env, _target_for_tracked_body(gripper_xyz, tracker) + grasp_offset, MATCHA_CLOSED_GRIPPER)
        if not command_motion(env, target_position, segment_duration, diagnostics, viewer, realtime, carried_whisk=tracker):
            return False
    return True


def execute_matcha_demo(
    env: SimEnv,
    config: MatchaConfig,
    diagnostics: MatchaDiagnostics,
    viewer: mujoco.viewer.Handle | None = None,
    realtime: bool = False,
) -> MatchaResult:
    grasp_offset = np.asarray(config.grasp_target_offset, dtype=float)
    whisk_body_xyz = np.array([config.whisk_position[0], config.whisk_position[1], config.grasp_height], dtype=float)
    grasp_xyz = whisk_body_xyz + grasp_offset
    approach_xyz = grasp_xyz + np.array([0.0, 0.0, config.approach_height], dtype=float)
    lift_body_xyz = whisk_body_xyz + np.array([0.0, 0.0, config.lift_height], dtype=float)
    lift_xyz = lift_body_xyz + grasp_offset
    cup_center_body_xyz = np.array(
        [
            config.main_cup_position[0],
            config.main_cup_position[1],
            config.whisk_tip_height + config.whisk_tip_to_body_height,
        ],
        dtype=float,
    )
    cup_center_xyz = cup_center_body_xyz + grasp_offset
    cup_approach_xyz = cup_center_xyz + np.array([0.0, 0.0, config.cup_approach_height], dtype=float)
    initial_whisk_position = tuple(float(v) for v in env.data.xpos[diagnostics.whisk_body_id])

    show_marker(viewer, grasp_xyz)

    fixed_pad_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_grip_pad")
    moving_pad_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_grip_pad")

    def log_phase(label: str) -> None:
        whisk_xyz = env.data.xpos[diagnostics.whisk_body_id]
        fixed_pad = env.data.geom_xpos[fixed_pad_id]
        moving_pad = env.data.geom_xpos[moving_pad_id]
        pad_mid = 0.5 * (fixed_pad + moving_pad)
        sep = float(np.linalg.norm(fixed_pad - moving_pad))
        joint_deg = [math.degrees(env.data.qpos[i]) for i in range(5)]
        ctrl_deg = [math.degrees(env.data.ctrl[i]) for i in range(5)]
        print(
            f"  [{label:18s}] whisk=({whisk_xyz[0]:.4f},{whisk_xyz[1]:.4f},{whisk_xyz[2]:.4f}) "
            f"jaws_mid=({pad_mid[0]:.4f},{pad_mid[1]:.4f},{pad_mid[2]:.4f}) sep={sep*1000:.1f}mm"
        )
        print(
            f"     qpos_deg={[f'{q:6.1f}' for q in joint_deg]} "
            f"ctrl_deg={[f'{c:6.1f}' for c in ctrl_deg]}"
        )

    log_phase("initial")

    current_open = dict(env.current_position)
    current_open["gripper"] = OPEN_GRIPPER
    if not command_motion(env, current_open, 0.4, diagnostics, viewer, realtime):
        return _matcha_result(env, diagnostics, initial_whisk_position, completed=False)
    log_phase("after open")

    # Approach the whisk from well above (15 cm clearance) with the jaws fully OPEN so the wide
    # swept volume of the open jaws stays clear of the whisk handle as the arm swings into place.
    # Then narrow the jaws to the preclose angle BEFORE descending, so the partial-open jaws are
    # just wide enough to fit around the grip band without colliding with the cup walls.
    approach_position = solve_target(env, approach_xyz, OPEN_GRIPPER)
    if not command_motion(env, approach_position, config.approach_duration, diagnostics, viewer, realtime):
        return _matcha_result(env, diagnostics, initial_whisk_position, completed=False)
    log_phase("after approach")

    pre_grasp_gripper = config.preclose_gripper
    preclose_position = dict(approach_position)
    preclose_position["gripper"] = pre_grasp_gripper
    if not command_motion(env, preclose_position, config.preclose_duration, diagnostics, viewer, realtime):
        return _matcha_result(env, diagnostics, initial_whisk_position, completed=False)
    log_phase("after preclose")

    # Solve the descend IK with the CLOSED gripper command so the arm joint plan reflects the
    # final closed-gripper geometry. Then we descend with the preclose gripper open, and only
    # the gripper joint changes during the close phase, preserving the arm pose so the closed
    # pads end up at the IK-targeted world position (the grip band height).
    grasp_position = solve_target(env, grasp_xyz, MATCHA_CLOSED_GRIPPER)
    descend_position = dict(grasp_position)
    descend_position["gripper"] = pre_grasp_gripper
    if not command_motion(env, descend_position, config.descend_duration, diagnostics, viewer, realtime):
        return _matcha_result(env, diagnostics, initial_whisk_position, completed=False)
    log_phase("after descend")

    closed_position = dict(grasp_position)
    if not command_motion(env, closed_position, config.close_duration, diagnostics, viewer, realtime):
        return _matcha_result(env, diagnostics, initial_whisk_position, completed=False)
    log_phase("after close")
    closed_position = convert_to_dictionary(env.data.qpos.copy())
    send_position_command(env.data, closed_position)

    # Release the initial weld before assisted tracking starts. If the weld remains active while
    # the freejoint is snapped to the gripper jaws, the equality constraint fights the scripted
    # pose update and can kick the arm violently back toward the whisk's initial table pose.
    _release_whisk_lock(env)
    log_phase("after weld rel")

    # In assisted-whisk mode, preserve the actual post-close whisk-to-jaw transform. This avoids
    # the nonphysical visual snap caused by forcing the gripper to a hardcoded point on the whisk.
    tracker = _capture_whisk_tracker(env, diagnostics, config.whisk_freejoint_name) if config.assisted_whisk else None
    if tracker is not None:
        print(
            "  captured grasp offset: "
            f"body_from_jaws=({tracker.body_from_jaws[0]:.4f},{tracker.body_from_jaws[1]:.4f},{tracker.body_from_jaws[2]:.4f}) m"
        )
        _set_geom_collisions_enabled(
            env.model,
            _body_geom_ids(env.model, diagnostics.fixed_jaw_body_id, contact_only=True)
            | _body_geom_ids(env.model, diagnostics.moving_jaw_body_id, contact_only=True),
            enabled=False,
        )
        mujoco.mj_forward(env.model, env.data)
    if not hold_command(env, closed_position, config.squeeze_duration, diagnostics, viewer, realtime, carried_whisk=tracker):
        return _matcha_result(env, diagnostics, initial_whisk_position, completed=False)
    log_phase("after squeeze")

    lift_position = solve_target(env, _target_for_tracked_body(lift_body_xyz, tracker) + grasp_offset, MATCHA_CLOSED_GRIPPER)
    if not command_motion(env, lift_position, config.lift_duration, diagnostics, viewer, realtime, carried_whisk=tracker):
        return _matcha_result(env, diagnostics, initial_whisk_position, completed=False)
    log_phase("after lift")

    cup_approach_position = solve_target(env, _target_for_tracked_body(cup_center_body_xyz + np.array([0.0, 0.0, config.cup_approach_height], dtype=float), tracker) + grasp_offset, MATCHA_CLOSED_GRIPPER)
    if not command_motion(env, cup_approach_position, config.approach_duration, diagnostics, viewer, realtime, carried_whisk=tracker):
        return _matcha_result(env, diagnostics, initial_whisk_position, completed=False)
    log_phase("after cup approach")

    cup_position = solve_target(env, _target_for_tracked_body(cup_center_body_xyz, tracker) + grasp_offset, MATCHA_CLOSED_GRIPPER)
    if not command_motion(env, cup_position, config.descend_duration, diagnostics, viewer, realtime, carried_whisk=tracker):
        return _matcha_result(env, diagnostics, initial_whisk_position, completed=False)
    log_phase("after cup descend")

    if not whisk_motion(env, config, cup_center_body_xyz, diagnostics, viewer, realtime, tracker=tracker):
        return _matcha_result(env, diagnostics, initial_whisk_position, completed=False)
    log_phase("after whisking")

    final_lift_body_xyz = cup_center_body_xyz + np.array([0.0, 0.0, config.lift_height], dtype=float)
    final_lift_position = solve_target(env, _target_for_tracked_body(final_lift_body_xyz, tracker) + grasp_offset, MATCHA_CLOSED_GRIPPER)
    if not command_motion(env, final_lift_position, config.lift_duration, diagnostics, viewer, realtime, carried_whisk=tracker):
        return _matcha_result(env, diagnostics, initial_whisk_position, completed=False)
    log_phase("after final lift")

    hold_command(env, final_lift_position, config.final_hold_duration, diagnostics, viewer, realtime, carried_whisk=tracker)
    log_phase("after final hold")
    return _matcha_result(env, diagnostics, initial_whisk_position, completed=True)


def _matcha_result(
    env: SimEnv,
    diagnostics: MatchaDiagnostics,
    initial_whisk_position: tuple[float, float, float],
    completed: bool,
) -> MatchaResult:
    final_whisk_position = tuple(float(v) for v in env.data.xpos[diagnostics.whisk_body_id])
    return MatchaResult(
        completed=completed,
        initial_whisk_position=initial_whisk_position,
        final_whisk_position=final_whisk_position,
        diagnostics=diagnostics,
    )


def run_matcha_demo(config: MatchaConfig, launch_viewer: bool) -> MatchaResult:
    env = create_env(scene_path=config.scene_path, camera_name="matcha_observer")
    diagnostics = configure_matcha_env(env, config)

    if not launch_viewer:
        return execute_matcha_demo(env, config, diagnostics, realtime=False)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer
        return execute_matcha_demo(env, config, diagnostics, viewer=viewer, realtime=True)


def print_result(result: MatchaResult) -> None:
    diagnostics = result.diagnostics
    print("Matcha demo result: " + ("COMPLETED" if result.completed else "STOPPED"))
    print(
        "  whisk position: "
        f"initial=({result.initial_whisk_position[0]:.4f}, {result.initial_whisk_position[1]:.4f}, {result.initial_whisk_position[2]:.4f}) m "
        f"final=({result.final_whisk_position[0]:.4f}, {result.final_whisk_position[1]:.4f}, {result.final_whisk_position[2]:.4f}) m"
    )
    print(
        "  contacts: "
        f"whisk_steps={diagnostics.whisk_contact_steps}/{diagnostics.steps}, "
        f"fixed_jaw_steps={diagnostics.fixed_contact_steps}, "
        f"moving_jaw_steps={diagnostics.moving_contact_steps}, "
        f"max_contacts={diagnostics.max_contacts}"
    )
    if diagnostics.contact_pairs:
        print("  contact pairs:")
        for whisk_geom, other_geom in sorted(diagnostics.contact_pairs):
            print(f"    {whisk_geom} <-> {other_geom}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a scripted SO101 matcha whisking demo in MuJoCo.")
    parser.add_argument("--scene", type=Path, default=MatchaConfig.scene_path, help="MJCF scene to load.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer.")
    parser.add_argument("--whisk-position", type=float, nargs=3, default=MatchaConfig.whisk_position, metavar=("X", "Y", "Z"))
    parser.add_argument("--main-cup-position", type=float, nargs=3, default=MatchaConfig.main_cup_position, metavar=("X", "Y", "Z"))
    parser.add_argument("--ice-cup-position", type=float, nargs=3, default=MatchaConfig.ice_cup_position, metavar=("X", "Y", "Z"))
    parser.add_argument("--whisk-radius", type=float, default=None, help="Deprecated alias for --whisk-stroke-length.")
    parser.add_argument("--whisk-cycles", type=int, default=None, help="Deprecated alias for --whisk-strokes.")
    parser.add_argument("--whisk-stroke-length", type=float, default=MatchaConfig.whisk_stroke_length, help="Forward/back stroke length in meters.")
    parser.add_argument("--whisk-strokes", type=int, default=MatchaConfig.whisk_strokes, help="Number of fast forward/back strokes.")
    parser.add_argument(
        "--grasp-target-offset",
        type=float,
        nargs=3,
        default=MatchaConfig.grasp_target_offset,
        metavar=("X", "Y", "Z"),
        help="Offset from the whisk body to the claw target during grasp.",
    )
    parser.add_argument("--whisk-duration", type=float, default=MatchaConfig.whisk_duration, help="Whisking duration in seconds.")
    parser.add_argument("--gripper-force", type=float, default=MatchaConfig.gripper_force, help="Symmetric gripper actuator force range.")
    parser.add_argument(
        "--physics-only-whisk",
        action="store_true",
        help="Disable the default scripted assist on the whisk freejoint and rely purely on "
        "MuJoCo physics for the grasp. Useful for debugging the gripper/whisk contact dynamics.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> MatchaConfig:
    return MatchaConfig(
        scene_path=args.scene,
        whisk_position=tuple(args.whisk_position),
        main_cup_position=tuple(args.main_cup_position),
        ice_cup_position=tuple(args.ice_cup_position),
        grasp_target_offset=tuple(args.grasp_target_offset),
        whisk_stroke_length=args.whisk_stroke_length if args.whisk_radius is None else args.whisk_radius * 2.0,
        whisk_strokes=args.whisk_strokes if args.whisk_cycles is None else args.whisk_cycles * 2,
        whisk_duration=args.whisk_duration,
        gripper_force=args.gripper_force,
        assisted_whisk=not args.physics_only_whisk,
    )


def main() -> None:
    args = parse_args()
    result = run_matcha_demo(config_from_args(args), launch_viewer=not args.headless)
    print_result(result)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if "mjpython" in str(exc):
            raise SystemExit(
                "MuJoCo viewer on macOS requires mjpython. Run:\n"
                "  conda activate whisk-agent\n"
                f"  cd {ROOT_DIR}\n"
                "  mjpython mujoco_sim/run_matcha_demo.py"
            ) from exc
        raise
