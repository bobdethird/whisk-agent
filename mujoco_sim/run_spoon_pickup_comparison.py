from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco  # type: ignore[import-not-found]
import mujoco.viewer  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from motion import solve_ik
from pose_estimation import TagPoseEstimate, detect_apriltags
from sim_env import SimEnv, create_env
from so101_kinematics import CLAW_CENTER_TOOL_POINT
from so101_mujoco_utils import JOINT_ORDER, convert_to_dictionary, send_position_command

from mujoco_sim.spoon_scene_config import (
    SPOON,
    SPOON_HANDLE_TAG,
    SPOON_HANDLE_TAG_TO_GRASP_OFFSET,
    SPOON_SCENE_PATH,
    SPOON_TABLE_REF_TAG,
    SPOON_TABLE_TAG_TO_GRASP_OFFSET,
    SPOON_TAG_SIZE,
    TOP_DOWN_CAMERA_NAME,
)


FIXED_JAW_BODY_NAME = "gripper"
MOVING_JAW_BODY_NAME = "moving_jaw_so101_v1"
DEFAULT_GRIPPER_FORCE = 4.0
DEFAULT_JAW_FRICTION = (2.0, 0.02, 0.002)
DEFAULT_APPROACH_HEIGHT = 0.05
DEFAULT_LIFT_HEIGHT = 0.10
DEFAULT_SUCCESS_LIFT_DELTA = 0.015


@dataclass(frozen=True)
class SpoonPickupResult:
    mode: str
    success: bool
    tag_id: int
    camera_name: str
    grasp_error_m: float
    initial_spoon_z: float
    final_spoon_z: float
    max_spoon_z: float


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


def _set_geom_friction(model: mujoco.MjModel, geom_ids: set[int], friction: tuple[float, float, float]) -> None:
    for geom_id in geom_ids:
        model.geom_friction[geom_id] = friction


def _set_actuator_force(model: mujoco.MjModel, actuator_name: str, force: float) -> None:
    actuator_id = _require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    model.actuator_forcerange[actuator_id] = [-force, force]


def _set_freejoint_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    position: tuple[float, float, float],
    yaw_deg: float = 0.0,
) -> None:
    joint_id = _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_address = int(model.jnt_qposadr[joint_id])
    qvel_address = int(model.jnt_dofadr[joint_id])
    if abs(yaw_deg) < 1e-9:
        data.qpos[qpos_address : qpos_address + 7] = [*position, 1.0, 0.0, 0.0, 0.0]
    else:
        half = math.radians(yaw_deg) / 2.0
        quat = (math.cos(half), 0.0, 0.0, math.sin(half))
        data.qpos[qpos_address : qpos_address + 7] = [*position, *quat]
    data.qvel[qvel_address : qvel_address + 6] = 0.0


def _configure_scene(env: SimEnv, gripper_force: float, spoon_yaw_deg: float = 0.0) -> int:
    spoon_body_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, SPOON.body_name)
    fixed_jaw_body_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, FIXED_JAW_BODY_NAME)
    moving_jaw_body_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, MOVING_JAW_BODY_NAME)

    _set_freejoint_pose(env.model, env.data, SPOON.freejoint_name, SPOON.initial_position, yaw_deg=spoon_yaw_deg)
    _set_actuator_force(env.model, "gripper", gripper_force)

    spoon_geom_ids = _body_geom_ids(env.model, spoon_body_id, contact_only=True)
    fixed_jaw_geom_ids = _body_geom_ids(env.model, fixed_jaw_body_id, contact_only=True)
    moving_jaw_geom_ids = _body_geom_ids(env.model, moving_jaw_body_id, contact_only=True)
    _set_geom_friction(env.model, spoon_geom_ids, SPOON.friction)
    _set_geom_friction(env.model, fixed_jaw_geom_ids | moving_jaw_geom_ids, DEFAULT_JAW_FRICTION)

    mujoco.mj_forward(env.model, env.data)
    return spoon_body_id


def _set_tag_visibility_for_mode(env: SimEnv, mode: str) -> None:
    """Hide the non-active tag so the mode is visually unambiguous."""
    handle_body_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, SPOON_HANDLE_TAG.name)
    table_body_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, SPOON_TABLE_REF_TAG.name)

    if mode == "handle-tag":
        # Hide table reference tag far away.
        env.model.body_pos[table_body_id] = np.array([2.0, 2.0, -1.0], dtype=float)
    elif mode == "table-offset":
        # Hide spoon-mounted tag deep below the spoon body.
        env.model.body_pos[handle_body_id] = np.array([0.0, 0.0, -0.25], dtype=float)

    mujoco.mj_forward(env.model, env.data)


def _step_once(env: SimEnv, spoon_body_id: int, max_spoon_z: float, viewer: mujoco.viewer.Handle | None, realtime: bool) -> tuple[bool, float]:
    step_start = time.time()
    mujoco.mj_step(env.model, env.data)
    max_spoon_z = max(max_spoon_z, float(env.data.xpos[spoon_body_id, 2]))

    if viewer is not None:
        viewer.sync()
        if not viewer.is_running():
            return False, max_spoon_z

    if realtime:
        sleep_time = env.model.opt.timestep - (time.time() - step_start)
        if sleep_time > 0:
            time.sleep(sleep_time)
    return True, max_spoon_z


def _interpolate_position(start_position: dict[str, float], target_position: dict[str, float], alpha: float) -> dict[str, float]:
    return {joint: (1.0 - alpha) * start_position[joint] + alpha * target_position[joint] for joint in JOINT_ORDER}


def _command_motion(
    env: SimEnv,
    spoon_body_id: int,
    target_position: dict[str, float],
    duration: float,
    max_spoon_z: float,
    viewer: mujoco.viewer.Handle | None,
    realtime: bool,
) -> tuple[bool, float]:
    start_position = convert_to_dictionary(env.data.qpos.copy())
    steps = max(1, math.ceil(duration / env.model.opt.timestep))

    for step_index in range(steps):
        alpha = (step_index + 1) / steps
        command = _interpolate_position(start_position, target_position, alpha)
        send_position_command(env.data, command)
        ok, max_spoon_z = _step_once(env, spoon_body_id, max_spoon_z, viewer, realtime)
        if not ok:
            return False, max_spoon_z

    env.current_position = dict(target_position)
    return True, max_spoon_z


def _hold_command(
    env: SimEnv,
    spoon_body_id: int,
    target_position: dict[str, float],
    duration: float,
    max_spoon_z: float,
    viewer: mujoco.viewer.Handle | None,
    realtime: bool,
) -> tuple[bool, float]:
    steps = max(1, math.ceil(duration / env.model.opt.timestep))
    for _ in range(steps):
        send_position_command(env.data, target_position)
        ok, max_spoon_z = _step_once(env, spoon_body_id, max_spoon_z, viewer, realtime)
        if not ok:
            return False, max_spoon_z

    env.current_position = dict(target_position)
    return True, max_spoon_z


def _show_targets(viewer: mujoco.viewer.Handle | None, grasp_xyz: np.ndarray, pregrasp_xyz: np.ndarray, truth_xyz: np.ndarray) -> None:
    if viewer is None:
        return

    viewer.user_scn.ngeom = 0
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.006, 0.0, 0.0],
        pos=truth_xyz,
        mat=np.eye(3).flatten(),
        rgba=[0.0, 1.0, 0.0, 0.75],
    )
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[1],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.006, 0.0, 0.0],
        pos=grasp_xyz,
        mat=np.eye(3).flatten(),
        rgba=[0.0, 0.2, 1.0, 0.8],
    )
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[2],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.005, 0.0, 0.0],
        pos=pregrasp_xyz,
        mat=np.eye(3).flatten(),
        rgba=[1.0, 0.85, 0.0, 0.85],
    )
    viewer.user_scn.ngeom = 3
    viewer.sync()


def _detect_tag(
    env: SimEnv,
    tag_id: int,
    camera_names: tuple[str, ...],
    debug_frame_dir: Path | None,
) -> TagPoseEstimate:
    estimates = detect_apriltags(
        env,
        camera_name=camera_names[0],
        tag_sizes={tag_id: SPOON_TAG_SIZE},
        camera_names=camera_names,
        debug_frame_dir=debug_frame_dir,
    )
    if tag_id not in estimates:
        raise RuntimeError(f"Tag {tag_id} was not detected from cameras: {', '.join(camera_names)}")
    return estimates[tag_id]


def _plan_target(mode: str, estimate: TagPoseEstimate) -> np.ndarray:
    if mode == "handle-tag":
        offset = np.asarray(SPOON_HANDLE_TAG_TO_GRASP_OFFSET, dtype=float)
    elif mode == "table-offset":
        offset = np.asarray(SPOON_TABLE_TAG_TO_GRASP_OFFSET, dtype=float)
    else:
        raise ValueError(f"Unknown mode {mode!r}")
    return estimate.world_position + estimate.world_rotation @ offset


def _solve_target(
    env: SimEnv,
    xyz: np.ndarray,
    gripper_position: float,
    rotation: np.ndarray | None = None,
) -> dict[str, float]:
    plan = solve_ik(
        env,
        xyz,
        gripper_position=gripper_position,
        rotation=rotation,
        tool_point=CLAW_CENTER_TOOL_POINT,
    )
    return plan.target_position


def run_mode(
    mode: str,
    scene_path: Path,
    camera_names: tuple[str, ...],
    headless: bool,
    debug_frame_dir: Path | None,
    spoon_yaw_deg: float,
    gripper_force: float,
    approach_height: float,
    lift_height: float,
    success_lift_delta: float,
) -> SpoonPickupResult:
    env = create_env(scene_path=scene_path, camera_name=camera_names[0])
    spoon_body_id = _configure_scene(env, gripper_force=gripper_force, spoon_yaw_deg=spoon_yaw_deg)
    grasp_site_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_SITE, "spoon_grasp_site")

    if mode == "handle-tag":
        tag_id = SPOON_HANDLE_TAG.tag_id
    elif mode == "table-offset":
        tag_id = SPOON_TABLE_REF_TAG.tag_id
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    _set_tag_visibility_for_mode(env, mode)
    estimate = _detect_tag(env, tag_id=tag_id, camera_names=camera_names, debug_frame_dir=debug_frame_dir)
    print(f"active tag for mode {mode!r}: id={tag_id} from camera={estimate.camera_name}")
    pinch_xyz = _plan_target(mode, estimate)

    grasp_xyz = pinch_xyz + np.array([0.0, 0.0, 0.006], dtype=float)
    pregrasp_xyz = grasp_xyz + np.array([0.0, 0.0, approach_height], dtype=float)
    lift_xyz = grasp_xyz + np.array([0.0, 0.0, lift_height], dtype=float)

    truth_xyz = env.data.site_xpos[grasp_site_id].copy()
    grasp_error_m = float(np.linalg.norm(pinch_xyz - truth_xyz))
    initial_spoon_z = float(env.data.xpos[spoon_body_id, 2])
    max_spoon_z = initial_spoon_z

    open_gripper = 50.0
    closed_gripper = -25.0

    def execute(viewer: mujoco.viewer.Handle | None, realtime: bool) -> SpoonPickupResult:
        nonlocal max_spoon_z
        _show_targets(viewer, grasp_xyz, pregrasp_xyz, truth_xyz)

        pregrasp_position = _solve_target(env, pregrasp_xyz, open_gripper)
        ok, max_spoon_z = _command_motion(env, spoon_body_id, pregrasp_position, 1.8, max_spoon_z, viewer, realtime)
        if not ok:
            return _result(False)

        grasp_position = _solve_target(env, grasp_xyz, open_gripper)
        ok, max_spoon_z = _command_motion(env, spoon_body_id, grasp_position, 1.0, max_spoon_z, viewer, realtime)
        if not ok:
            return _result(False)

        closed_position = dict(grasp_position)
        closed_position["gripper"] = closed_gripper
        ok, max_spoon_z = _command_motion(env, spoon_body_id, closed_position, 0.8, max_spoon_z, viewer, realtime)
        if not ok:
            return _result(False)

        ok, max_spoon_z = _hold_command(env, spoon_body_id, closed_position, 0.4, max_spoon_z, viewer, realtime)
        if not ok:
            return _result(False)

        lift_position = _solve_target(env, lift_xyz, closed_gripper)
        ok, max_spoon_z = _command_motion(env, spoon_body_id, lift_position, 1.6, max_spoon_z, viewer, realtime)
        if not ok:
            return _result(False)

        ok, max_spoon_z = _hold_command(env, spoon_body_id, lift_position, 0.6, max_spoon_z, viewer, realtime)
        if not ok:
            return _result(False)

        final_spoon_z = float(env.data.xpos[spoon_body_id, 2])
        success = final_spoon_z >= initial_spoon_z + success_lift_delta
        return _result(success)

    def _result(success: bool) -> SpoonPickupResult:
        return SpoonPickupResult(
            mode=mode,
            success=success,
            tag_id=tag_id,
            camera_name=estimate.camera_name,
            grasp_error_m=grasp_error_m,
            initial_spoon_z=initial_spoon_z,
            final_spoon_z=float(env.data.xpos[spoon_body_id, 2]),
            max_spoon_z=max_spoon_z,
        )

    if headless:
        return execute(viewer=None, realtime=False)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer
        return execute(viewer=viewer, realtime=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare spoon pickup with (1) handle-mounted tag vs (2) table-tag offset strategy."
    )
    parser.add_argument("--scene", type=Path, default=SPOON_SCENE_PATH, help="MJCF scene path.")
    parser.add_argument(
        "--mode",
        choices=("handle-tag", "table-offset", "both"),
        default="both",
        help="Run one strategy or both back-to-back.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without opening MuJoCo viewer.")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=(TOP_DOWN_CAMERA_NAME, "spoon_observer"),
        metavar="CAMERA",
        help="Cameras to search for AprilTags.",
    )
    parser.add_argument("--spoon-yaw-deg", type=float, default=0.0, help="Initial spoon yaw about world Z in degrees.")
    parser.add_argument("--gripper-force", type=float, default=DEFAULT_GRIPPER_FORCE, help="Gripper actuator force range.")
    parser.add_argument("--approach-height", type=float, default=DEFAULT_APPROACH_HEIGHT, help="Approach height above grasp target.")
    parser.add_argument("--lift-height", type=float, default=DEFAULT_LIFT_HEIGHT, help="Lift height above grasp target.")
    parser.add_argument(
        "--success-lift-delta",
        type=float,
        default=DEFAULT_SUCCESS_LIFT_DELTA,
        help="Required spoon Z increase to mark success.",
    )
    parser.add_argument("--debug-camera-frame-dir", type=Path, default=None, help="Optional directory to save camera frames.")
    return parser.parse_args()


def print_result(result: SpoonPickupResult) -> None:
    verdict = "PASS" if result.success else "FAIL"
    print(f"[{result.mode}] {verdict}  tag={result.tag_id} camera={result.camera_name}")
    print(
        f"  spoon z: initial={result.initial_spoon_z:.4f} m "
        f"final={result.final_spoon_z:.4f} m max={result.max_spoon_z:.4f} m"
    )
    print(f"  estimated grasp error vs spoon_grasp_site: {result.grasp_error_m:.4f} m")


def main() -> None:
    args = parse_args()
    modes = ("handle-tag", "table-offset") if args.mode == "both" else (args.mode,)

    results: list[SpoonPickupResult] = []
    for mode in modes:
        print(f"\n=== Running mode: {mode} ===")
        result = run_mode(
            mode=mode,
            scene_path=args.scene,
            camera_names=tuple(args.cameras),
            headless=args.headless,
            debug_frame_dir=args.debug_camera_frame_dir,
            spoon_yaw_deg=args.spoon_yaw_deg,
            gripper_force=args.gripper_force,
            approach_height=args.approach_height,
            lift_height=args.lift_height,
            success_lift_delta=args.success_lift_delta,
        )
        print_result(result)
        results.append(result)

    if len(results) == 2:
        print("\n=== Comparison summary ===")
        a, b = results
        if a.success and not b.success:
            winner = a.mode
        elif b.success and not a.success:
            winner = b.mode
        elif a.success and b.success:
            winner = a.mode if a.max_spoon_z >= b.max_spoon_z else b.mode
        else:
            winner = a.mode if a.grasp_error_m <= b.grasp_error_m else b.mode
        print(f"better in this run: {winner}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if "mjpython" in str(exc):
            raise SystemExit(
                "MuJoCo viewer on macOS requires mjpython. Run:\n"
                "  conda activate whisk-agent\n"
                "  mjpython mujoco_sim/run_spoon_pickup_comparison.py"
            ) from exc
        raise
