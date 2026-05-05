"""Spoon pickup driven by `grasp_library` (tag-relative grasp planning).

Demonstrates that the same framework used for cup pickup generalizes to a new
rigid object: register a `TaggedObject` (one anchor + one grasp in object
frame) and reuse the detect -> fuse -> world_grasp_from_object -> oriented IK
pipeline. No spoon-specific tag-to-grasp offset is hardcoded; the grasp is
defined in the spoon's body frame.
"""

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

from grasp_library import SPOON, TagDetection, fuse_object_pose, world_grasp_from_object
from motion import solve_ik
from pose_estimation import detect_apriltags
from sim_env import create_env
from so101_mujoco_utils import JOINT_ORDER, convert_to_dictionary, send_position_command

from mujoco_sim.spoon_scene_config import SPOON as SPOON_CFG
from mujoco_sim.spoon_scene_config import SPOON_SCENE_PATH, TOP_DOWN_CAMERA_NAME

DEFAULT_GRIPPER_FORCE = 4.0
DEFAULT_JAW_FRICTION = (2.0, 0.02, 0.002)
DEFAULT_LIFT_HEIGHT = 0.10
DEFAULT_SUCCESS_LIFT_DELTA = 0.015


@dataclass(frozen=True)
class SpoonPickupResult:
    success: bool
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


def _set_actuator_force(model: mujoco.MjModel, name: str, force: float) -> None:
    actuator_id = _require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    model.actuator_forcerange[actuator_id] = [-force, force]


def _set_freejoint_pose(model, data, joint_name, position, yaw_deg=0.0):
    joint_id = _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_addr = int(model.jnt_qposadr[joint_id])
    qvel_addr = int(model.jnt_dofadr[joint_id])
    if abs(yaw_deg) < 1e-9:
        data.qpos[qpos_addr : qpos_addr + 7] = [*position, 1.0, 0.0, 0.0, 0.0]
    else:
        half = math.radians(yaw_deg) / 2.0
        data.qpos[qpos_addr : qpos_addr + 7] = [*position, math.cos(half), 0.0, 0.0, math.sin(half)]
    data.qvel[qvel_addr : qvel_addr + 6] = 0.0


def configure_scene(env, gripper_force: float, spoon_yaw_deg: float) -> int:
    spoon_body_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, SPOON_CFG.body_name)
    fixed_jaw_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    moving_jaw_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")

    _set_freejoint_pose(env.model, env.data, SPOON_CFG.freejoint_name, SPOON_CFG.initial_position, spoon_yaw_deg)
    _set_actuator_force(env.model, "gripper", gripper_force)

    _set_geom_friction(env.model, _body_geom_ids(env.model, spoon_body_id, contact_only=True), SPOON_CFG.friction)
    _set_geom_friction(
        env.model,
        _body_geom_ids(env.model, fixed_jaw_id, contact_only=True) | _body_geom_ids(env.model, moving_jaw_id, contact_only=True),
        DEFAULT_JAW_FRICTION,
    )

    mujoco.mj_forward(env.model, env.data)
    return spoon_body_id


def fuse_spoon_pose(env, camera_names: tuple[str, ...], debug_dir: Path | None) -> tuple[np.ndarray, np.ndarray]:
    estimates = detect_apriltags(
        env,
        camera_name=camera_names[0],
        tag_sizes=SPOON.tag_sizes,
        camera_names=camera_names,
        debug_frame_dir=debug_dir,
    )
    detections = [
        TagDetection(tag_id=e.tag_id, world_position=e.world_position, world_rotation=e.world_rotation)
        for e in estimates.values()
        if SPOON.anchor_for(e.tag_id) is not None
    ]
    if not detections:
        raise RuntimeError(
            f"No SPOON anchor tags detected. Looked for {sorted(SPOON.tag_sizes)} from cameras: "
            + ", ".join(camera_names)
        )
    obj_pos, obj_rot = fuse_object_pose(detections, SPOON)
    yaw_deg = math.degrees(math.atan2(obj_rot[1, 0], obj_rot[0, 0]))
    print(
        f"spoon fused from tags {[d.tag_id for d in detections]}: "
        f"pos=({obj_pos[0]:.4f},{obj_pos[1]:.4f},{obj_pos[2]:.4f}) yaw={yaw_deg:.2f} deg"
    )
    return obj_pos, obj_rot


def step_once(env, spoon_body_id, max_z, viewer, realtime) -> tuple[bool, float]:
    step_start = time.time()
    mujoco.mj_step(env.model, env.data)
    max_z = max(max_z, float(env.data.xpos[spoon_body_id, 2]))
    if viewer is not None:
        viewer.sync()
        if not viewer.is_running():
            return False, max_z
    if realtime:
        sleep_t = env.model.opt.timestep - (time.time() - step_start)
        if sleep_t > 0:
            time.sleep(sleep_t)
    return True, max_z


def command_motion(env, spoon_body_id, target_position, duration, max_z, viewer, realtime):
    start = convert_to_dictionary(env.data.qpos.copy())
    steps = max(1, math.ceil(duration / env.model.opt.timestep))
    for i in range(steps):
        alpha = (i + 1) / steps
        cmd = {j: (1 - alpha) * start[j] + alpha * target_position[j] for j in JOINT_ORDER}
        send_position_command(env.data, cmd)
        ok, max_z = step_once(env, spoon_body_id, max_z, viewer, realtime)
        if not ok:
            return False, max_z
    env.current_position = dict(target_position)
    return True, max_z


def hold(env, spoon_body_id, target_position, duration, max_z, viewer, realtime):
    steps = max(1, math.ceil(duration / env.model.opt.timestep))
    for _ in range(steps):
        send_position_command(env.data, target_position)
        ok, max_z = step_once(env, spoon_body_id, max_z, viewer, realtime)
        if not ok:
            return False, max_z
    env.current_position = dict(target_position)
    return True, max_z


def show_markers(viewer, obj_pos, grasp_pos, pregrasp_pos):
    if viewer is None:
        return
    viewer.user_scn.ngeom = 0
    for i, (pos, rgba, size) in enumerate([
        (obj_pos, [0.0, 1.0, 0.0, 0.55], 0.010),
        (grasp_pos, [0.0, 0.2, 1.0, 0.85], 0.008),
        (pregrasp_pos, [1.0, 0.85, 0.0, 0.85], 0.006),
    ]):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[size, 0.0, 0.0],
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=rgba,
        )
    viewer.user_scn.ngeom = 3
    viewer.sync()


def solve(env, xyz, gripper, rotation):
    plan = solve_ik(env, xyz, gripper_position=gripper, rotation=rotation)
    pos = plan.target_pose[:3, 3]
    print(
        f"  IK target xyz=({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}) gripper={gripper:.1f} "
        f"pos_err={plan.position_error:.5f} rot_err={plan.orientation_error:.4f}"
    )
    return plan.target_position


def _rot_y(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def show_scoop_marker(viewer, scoop_xyz: np.ndarray) -> None:
    """Add an orange marker sphere at the scoop target."""
    if viewer is None:
        return
    idx = viewer.user_scn.ngeom
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[idx],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.025, 0.0, 0.0],
        pos=scoop_xyz,
        mat=np.eye(3).flatten(),
        rgba=[1.0, 0.5, 0.0, 0.4],
    )
    viewer.user_scn.ngeom = idx + 1
    viewer.sync()


def do_scoop(
    env,
    spoon_body_id: int,
    base_rot: np.ndarray,
    gripper_closed: float,
    scoop_xyz: np.ndarray,
    max_z: float,
    viewer,
    realtime: bool,
) -> tuple[bool, float]:
    """
    Execute a scooping arc starting from the current arm pose.
    scoop_xyz is the center of the scooping zone at table level.

    Arc:
      1. Hover above scoop zone (level, bowl forward)
      2. Entry dip — bowl tilts down 25°, lower to scoop depth
      3. Sweep forward through scoop zone (still tilted)
      4. Pull up — bowl tilts back 15°, retain "contents"
      5. Return to carry height
    """
    show_scoop_marker(viewer, scoop_xyz)

    # Keep scoop zone close to pickup x so the arm stays in its reliable reach band.
    # Scoop at table level: dip down, sweep 4 cm forward, pull back up.
    scoop_depth_z = 0.014   # just above table surface
    sweep_xyz = scoop_xyz + np.array([0.04, 0.0, 0.002])

    # Tilt the gripper about world-y to angle the bowl down during the scoop.
    # Positive Ry tilts +x end (bowl) toward -z (down). -Ry tilts back up.
    rot_level     = base_rot
    rot_bowl_down = _rot_y(20.0) @ base_rot
    rot_bowl_back = _rot_y(-12.0) @ base_rot

    waypoints = [
        ("position", scoop_xyz + np.array([0.0, 0.0, 0.07]), rot_level,     1.2),
        ("entry",    scoop_xyz + np.array([0.0, 0.0, scoop_depth_z]),        rot_bowl_down, 1.0),
        ("sweep",    sweep_xyz,                                               rot_bowl_down, 0.8),
        ("pull-up",  sweep_xyz + np.array([0.0, 0.0, 0.04]),                 rot_bowl_back, 0.7),
        ("carry",    scoop_xyz + np.array([0.0, 0.0, 0.07]),                 rot_level,     1.0),
    ]

    for label, xyz, rot, dur in waypoints:
        print(f"\n[scoop:{label}]")
        q = solve(env, xyz, gripper_closed, rot)
        ok, max_z = command_motion(env, spoon_body_id, q, dur, max_z, viewer, realtime)
        if not ok:
            return False, max_z

    return True, max_z


def run_pickup(args) -> SpoonPickupResult:
    env = create_env(scene_path=args.scene, camera_name=args.cameras[0])
    spoon_body_id = configure_scene(env, args.gripper_force, args.spoon_yaw_deg)
    grasp_site_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_SITE, "spoon_grasp_site")

    obj_pos, obj_rot = fuse_spoon_pose(env, tuple(args.cameras), args.debug_camera_frame_dir)

    grasp = SPOON.grasps[args.grasp_index]
    print(f"using grasp index {args.grasp_index} of {len(SPOON.grasps)}")
    claw_target_world, claw_rot_world, pregrasp_world = world_grasp_from_object(grasp, obj_pos, obj_rot)
    lift_world = claw_target_world + np.array([0.0, 0.0, args.lift_height])

    truth = env.data.site_xpos[grasp_site_id].copy()
    pinch_pos_world = obj_pos + obj_rot @ grasp.pos_in_object
    grasp_error_m = float(np.linalg.norm(pinch_pos_world - truth))
    print(
        f"pinch_world=({pinch_pos_world[0]:.4f},{pinch_pos_world[1]:.4f},{pinch_pos_world[2]:.4f}) "
        f"site_truth=({truth[0]:.4f},{truth[1]:.4f},{truth[2]:.4f}) "
        f"err={grasp_error_m:.4f} m"
    )
    print(
        f"claw_target=({claw_target_world[0]:.4f},{claw_target_world[1]:.4f},{claw_target_world[2]:.4f}) "
        f"pregrasp=({pregrasp_world[0]:.4f},{pregrasp_world[1]:.4f},{pregrasp_world[2]:.4f})"
    )

    initial_spoon_z = float(env.data.xpos[spoon_body_id, 2])
    max_z = initial_spoon_z

    def execute(viewer, realtime) -> SpoonPickupResult:
        nonlocal max_z
        show_markers(viewer, obj_pos, claw_target_world, pregrasp_world)

        print("\n[pregrasp]")
        pre_q = solve(env, pregrasp_world, grasp.gripper_open, claw_rot_world)
        ok, max_z = command_motion(env, spoon_body_id, pre_q, 1.8, max_z, viewer, realtime)
        if not ok:
            return result(False)

        print("\n[grasp]")
        grasp_q = solve(env, claw_target_world, grasp.gripper_open, claw_rot_world)
        ok, max_z = command_motion(env, spoon_body_id, grasp_q, 1.0, max_z, viewer, realtime)
        if not ok:
            return result(False)

        print("\n[close]")
        closed_q = dict(grasp_q)
        closed_q["gripper"] = grasp.gripper_close
        ok, max_z = command_motion(env, spoon_body_id, closed_q, 0.8, max_z, viewer, realtime)
        if not ok:
            return result(False)
        ok, max_z = hold(env, spoon_body_id, closed_q, 0.4, max_z, viewer, realtime)
        if not ok:
            return result(False)

        print("\n[lift]")
        lift_q = solve(env, lift_world, grasp.gripper_close, claw_rot_world)
        ok, max_z = command_motion(env, spoon_body_id, lift_q, 1.6, max_z, viewer, realtime)
        if not ok:
            return result(False)
        ok, max_z = hold(env, spoon_body_id, lift_q, 0.6, max_z, viewer, realtime)
        if not ok:
            return result(False)

        if args.scoop:
            scoop_xyz = np.array([claw_target_world[0] + 0.04, claw_target_world[1], 0.0])
            ok, max_z = do_scoop(
                env, spoon_body_id, claw_rot_world, grasp.gripper_close,
                scoop_xyz, max_z, viewer, realtime,
            )
            if not ok:
                return result(False)

        final_z = float(env.data.xpos[spoon_body_id, 2])
        return result(final_z >= initial_spoon_z + args.success_lift_delta)

    def result(success: bool) -> SpoonPickupResult:
        return SpoonPickupResult(
            success=success,
            grasp_error_m=grasp_error_m,
            initial_spoon_z=initial_spoon_z,
            final_spoon_z=float(env.data.xpos[spoon_body_id, 2]),
            max_spoon_z=max_z,
        )

    if args.headless:
        return execute(None, False)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer
        return execute(viewer, True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spoon pickup using grasp_library (tag-relative grasp).")
    p.add_argument("--scene", type=Path, default=SPOON_SCENE_PATH)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--cameras", nargs="+", default=(TOP_DOWN_CAMERA_NAME, "spoon_observer"))
    p.add_argument("--spoon-yaw-deg", type=float, default=0.0)
    p.add_argument("--gripper-force", type=float, default=DEFAULT_GRIPPER_FORCE)
    p.add_argument("--lift-height", type=float, default=DEFAULT_LIFT_HEIGHT)
    p.add_argument("--grasp-index", type=int, default=1, help="0=top-down, 1=scooping-side")
    p.add_argument("--scoop", action="store_true", help="Execute scooping arc after pickup")
    p.add_argument("--success-lift-delta", type=float, default=DEFAULT_SUCCESS_LIFT_DELTA)
    p.add_argument("--debug-camera-frame-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    r = run_pickup(args)
    verdict = "PASS" if r.success else "FAIL"
    print(f"\n[{verdict}] grasp_library spoon pickup")
    print(f"  spoon z: initial={r.initial_spoon_z:.4f} final={r.final_spoon_z:.4f} max={r.max_spoon_z:.4f} m")
    print(f"  pinch_world vs spoon_grasp_site error: {r.grasp_error_m:.4f} m")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if "mjpython" in str(exc):
            raise SystemExit(
                "MuJoCo viewer on macOS requires mjpython. Run:\n"
                "  conda activate whisk-agent\n"
                "  mjpython mujoco_sim/run_spoon_pickup.py"
            ) from exc
        raise
