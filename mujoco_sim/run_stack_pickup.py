"""Sequential pickup for three separate stacked cups: top → mid → bottom.

Uses **one** AprilTag on the **bottom** cup (`grasp_library.THREE_CUP_STACK`, tag ID 8) so the tag
stays in view for every phase. Each phase runs AprilTag fusion on the bottom cup, then plans the
grasp on the **current** cup using that body's live pose from MuJoCo (`xpos` / `xmat`) so physics
after prior pickups stays consistent.

**Placement:** by default each cup is carried to a **fixed pad beside the tower** (tag ID 9) and set
down using that tag's fused pose (see `--place-mode pad`). The pad sits lateral to the stack (same
nominal X, offset Y) so the arm sidesteps instead of reaching farther +X toward the table edge or
obstacles. Tunable `--place-pad-offset-x/y`. Legacy fixed world deltas: `--place-mode offset`
(`--place-dx-m`, `--place-dy-m`, `--place-release-z-m`).

Moving to the pad does **not** affect detection of tag 8 on the stack: fusion reads both tags from
the same cameras each phase; the arm may briefly occlude one tag, in which case the sim falls back
to the corresponding body pose.

Scene: `simulation_code/model/scene_cup_stack.xml`

Examples:

  mjpython mujoco_sim/run_stack_pickup.py --no-debug-camera-frames
  mjpython mujoco_sim/run_stack_pickup.py --headless
  mjpython mujoco_sim/run_stack_pickup.py --place-mode offset --place-dy-m -0.28
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco  # type: ignore[import-not-found]
import mujoco.viewer  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from grasp_library import PLACEMENT_PAD, THREE_CUP_STACK, THREE_CUP_STACK_REMOVAL_ORDER, world_grasp_from_object
from gripper import CLOSED_GRIPPER, OPEN_GRIPPER
from mujoco_sim.run_cup_pickup import (
    ContactDiagnostics,
    PickupConfig,
    STACK_SCENE_PATH,
    _body_geom_ids,
    _require_id,
    _set_actuator_force,
    _set_geom_friction,
    _set_freejoint_pose,
    _scale_body_mass,
    command_motion,
    estimate_object_pose_from_tags,
    hold_command,
    show_cup_target_points,
    solve_target,
)
from sim_env import SimEnv, create_env

STACK_BODY_NAMES = ("cup_bottom", "cup_mid", "cup_top")

PLACEMENT_PAD_BODY_NAME = "placement_pad"
# placement_pad body origin at box center; top surface = center_z + half thickness.
PAD_HALF_THICKNESS_M = 0.003
CUP_CENTER_ABOVE_SURFACE_M = 0.045
STACK_SPACING_M = 0.09
# Pad-frame XY offset from placement_pad origin to set-down point (default: toward tower along +world Y).
DEFAULT_PLACE_PAD_OFFSET_X_M = 0.0
DEFAULT_PLACE_PAD_OFFSET_Y_M = 0.055

# After lifting, move this far in **world** XY (meters) before lowering — `--place-mode offset` only.
DEFAULT_PLACE_DX_M = 0.0
DEFAULT_PLACE_DY_M = -0.26
# Absolute world Z for the open-gripper pose (`--place-mode offset` only).
DEFAULT_PLACE_RELEASE_Z_M = 0.068
DEFAULT_TRANSFER_DURATION_S = 2.4
DEFAULT_LOWER_DURATION_S = 2.6


def fused_placement_pad_pose_with_fallback(
    env: SimEnv,
    config: PickupConfig,
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """AprilTag fusion on the placement pad (tag 9); fallback to MuJoCo body pose in simulation."""

    try:
        obj_pos, obj_rot, detections = estimate_object_pose_from_tags(env, PLACEMENT_PAD, config)
    except RuntimeError:
        detections = ()

    # `allow_config_position_fallback` can yield an empty detection list with the *cup* pose — invalid here.
    if not detections:
        bid = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, PLACEMENT_PAD_BODY_NAME)
        pos = np.array(env.data.xpos[bid, :3], dtype=float)
        rot = np.array(env.data.xmat[bid], dtype=float).reshape(3, 3)
        print(
            "  (placement tag 9 not fused — using placement_pad body pose from simulation; "
            "check camera coverage on hardware.)"
        )
        return pos, rot, ()

    return obj_pos, obj_rot, detections


def world_place_targets_from_pad(
    pad_pos: np.ndarray,
    pad_rot: np.ndarray,
    lift_xyz: np.ndarray,
    stack_index: int,
    offset_x_m: float,
    offset_y_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """World transfer (at lift z) and place (cup center z) from fused placement_pad pose."""

    offset_pad = np.array([offset_x_m, offset_y_m, 0.0], dtype=float)
    horizontal = pad_pos + pad_rot @ offset_pad
    surface_z = float(pad_pos[2]) + PAD_HALF_THICKNESS_M
    place_z = surface_z + CUP_CENTER_ABOVE_SURFACE_M + STACK_SPACING_M * stack_index
    transfer_xyz = np.array([horizontal[0], horizontal[1], float(lift_xyz[2])], dtype=float)
    place_xyz = np.array([horizontal[0], horizontal[1], place_z], dtype=float)
    return transfer_xyz, place_xyz


def settle_stack(env: SimEnv, steps: int) -> None:
    for _ in range(steps):
        mujoco.mj_step(env.model, env.data)


def fused_bottom_pose_with_fallback(
    env: SimEnv,
    config: PickupConfig,
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """AprilTag fusion on the bottom cup; if cameras lose the tag, use MuJoCo body pose (sim-only)."""

    try:
        return estimate_object_pose_from_tags(env, THREE_CUP_STACK, config)
    except RuntimeError:
        bid = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, "cup_bottom")
        pos = np.array(env.data.xpos[bid, :3], dtype=float)
        rot = np.array(env.data.xmat[bid], dtype=float).reshape(3, 3)
        print(
            "  (tag not detected — using cup_bottom body pose from simulation; "
            "fix camera coverage on hardware.)"
        )
        return pos, rot, ()


def configure_stack_scene(env: SimEnv, config: PickupConfig) -> None:
    """Tune friction/mass for all stack cups and spawn the bottom cup from config."""

    fixed_jaw_body_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    moving_jaw_body_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
    _set_actuator_force(env.model, "gripper", config.gripper_force)

    for name in STACK_BODY_NAMES:
        bid = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        _scale_body_mass(env.model, bid, config.cup_mass)

    all_cup_geoms: set[int] = set()
    for name in STACK_BODY_NAMES:
        bid = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        all_cup_geoms |= _body_geom_ids(env.model, bid, contact_only=True)

    fixed_jaw_geom_ids = _body_geom_ids(env.model, fixed_jaw_body_id, contact_only=True)
    moving_jaw_geom_ids = _body_geom_ids(env.model, moving_jaw_body_id, contact_only=True)
    _set_geom_friction(env.model, all_cup_geoms, config.cup_friction)
    _set_geom_friction(env.model, fixed_jaw_geom_ids | moving_jaw_geom_ids, config.jaw_friction)

    _set_freejoint_pose(
        env.model,
        env.data,
        "cup_bottom_freejoint",
        config.cup_position,
        yaw_deg=config.cup_yaw_deg,
    )
    mujoco.mj_forward(env.model, env.data)


def make_diagnostics_for_body(env: SimEnv, cup_body_name: str) -> ContactDiagnostics:
    cup_body_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, cup_body_name)
    fixed_jaw_body_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    moving_jaw_body_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
    return ContactDiagnostics(
        model=env.model,
        cup_body_id=cup_body_id,
        fixed_jaw_body_id=fixed_jaw_body_id,
        moving_jaw_body_id=moving_jaw_body_id,
        cup_geom_ids=_body_geom_ids(env.model, cup_body_id, contact_only=True),
    )


def execute_stack_phase(
    env: SimEnv,
    config: PickupConfig,
    diagnostics: ContactDiagnostics,
    grasp_template,
    member_body_name: str,
    bottom_pos: np.ndarray,
    viewer: mujoco.viewer.Handle | None,
    realtime: bool,
    *,
    place_mode: str,
    place_dx_m: float,
    place_dy_m: float,
    place_release_z_m: float,
    transfer_duration_s: float,
    lower_duration_s: float,
    pad_pos: np.ndarray | None,
    pad_rot: np.ndarray | None,
    pad_stack_index: int,
    place_pad_offset_x_m: float,
    place_pad_offset_y_m: float,
) -> bool:
    """Grasp → lift → transfer → lower → open gripper (pad or fixed-offset placement)."""

    member_body_id = _require_id(env.model, mujoco.mjtObj.mjOBJ_BODY, member_body_name)
    member_pos = np.array(env.data.xpos[member_body_id, :3], dtype=float)
    member_rot = np.array(env.data.xmat[member_body_id], dtype=float).reshape(3, 3)
    grasp_pos, grasp_rot, pregrasp_pos = world_grasp_from_object(
        grasp_template, member_pos, member_rot
    )
    lift_xyz = grasp_pos + np.array([0.0, 0.0, config.lift_height], dtype=float)

    show_cup_target_points(viewer, bottom_pos, grasp_pos, pregrasp_pos)

    pregrasp_position = solve_target(env, pregrasp_pos, OPEN_GRIPPER, rotation=grasp_rot)
    if not command_motion(env, pregrasp_position, config.approach_duration, diagnostics, viewer, realtime):
        return False

    grasp_position_dict = solve_target(env, grasp_pos, OPEN_GRIPPER, rotation=grasp_rot)
    if not command_motion(env, grasp_position_dict, config.descend_duration, diagnostics, viewer, realtime):
        return False

    closed_position = dict(grasp_position_dict)
    closed_position["gripper"] = CLOSED_GRIPPER
    if not command_motion(env, closed_position, config.close_duration, diagnostics, viewer, realtime):
        return False
    if not hold_command(env, closed_position, config.squeeze_duration, diagnostics, viewer, realtime):
        return False

    lift_position = solve_target(env, lift_xyz, CLOSED_GRIPPER, rotation=grasp_rot)
    if not command_motion(env, lift_position, config.lift_duration, diagnostics, viewer, realtime):
        return False
    hold_command(env, lift_position, config.final_hold_duration, diagnostics, viewer, realtime)

    if place_mode == "pad":
        if pad_pos is None or pad_rot is None:
            raise RuntimeError("place_mode='pad' requires pad_pos and pad_rot.")
        transfer_xyz, place_xyz = world_place_targets_from_pad(
            pad_pos,
            pad_rot,
            lift_xyz,
            pad_stack_index,
            place_pad_offset_x_m,
            place_pad_offset_y_m,
        )
        print(
            f"  placement pad target (cup center): "
            f"({place_xyz[0]:.4f}, {place_xyz[1]:.4f}, {place_xyz[2]:.4f}) m"
        )
    else:
        transfer_xyz = lift_xyz + np.array([place_dx_m, place_dy_m, 0.0], dtype=float)
        place_xyz = transfer_xyz.copy()
        place_xyz[2] = float(place_release_z_m)

    transfer_position = solve_target(env, transfer_xyz, CLOSED_GRIPPER, rotation=grasp_rot)
    if not command_motion(env, transfer_position, transfer_duration_s, diagnostics, viewer, realtime):
        return False

    place_position = solve_target(env, place_xyz, CLOSED_GRIPPER, rotation=grasp_rot)
    if not command_motion(env, place_position, lower_duration_s, diagnostics, viewer, realtime):
        return False

    release_position = dict(place_position)
    release_position["gripper"] = OPEN_GRIPPER
    hold_command(env, release_position, 0.45, diagnostics, viewer, realtime)
    return True


def run_stack_pickup(
    config: PickupConfig,
    launch_viewer: bool,
    *,
    place_mode: str = "pad",
    place_dx_m: float = DEFAULT_PLACE_DX_M,
    place_dy_m: float = DEFAULT_PLACE_DY_M,
    place_release_z_m: float = DEFAULT_PLACE_RELEASE_Z_M,
    transfer_duration_s: float = DEFAULT_TRANSFER_DURATION_S,
    lower_duration_s: float = DEFAULT_LOWER_DURATION_S,
    place_pad_offset_x_m: float = DEFAULT_PLACE_PAD_OFFSET_X_M,
    place_pad_offset_y_m: float = DEFAULT_PLACE_PAD_OFFSET_Y_M,
) -> bool:
    env = create_env(scene_path=config.scene_path)
    configure_stack_scene(env, config)

    grasp_template = THREE_CUP_STACK.grasps[config.grasp_index]

    def run_phases(viewer: mujoco.viewer.Handle | None) -> bool:
        for phase_index, body_name in enumerate(THREE_CUP_STACK_REMOVAL_ORDER):
            print(f"\n--- Phase {phase_index + 1}/3: remove {body_name} ---")
            if phase_index > 0:
                settle_stack(env, 180)
            bottom_pos, bottom_rot, _ = fused_bottom_pose_with_fallback(env, config)
            print(
                f"  fused bottom cup: pos=({bottom_pos[0]:.4f},{bottom_pos[1]:.4f},{bottom_pos[2]:.4f}) m "
                f"yaw={np.degrees(np.arctan2(bottom_rot[1, 0], bottom_rot[0, 0])):.2f}°"
            )
            pad_pos = None
            pad_rot = None
            if place_mode == "pad":
                pad_pos, pad_rot, _ = fused_placement_pad_pose_with_fallback(env, config)
                print(
                    f"  fused placement pad: pos=({pad_pos[0]:.4f},{pad_pos[1]:.4f},{pad_pos[2]:.4f}) m "
                    f"yaw={np.degrees(np.arctan2(pad_rot[1, 0], pad_rot[0, 0])):.2f}°"
                )
            diagnostics = make_diagnostics_for_body(env, body_name)
            if not execute_stack_phase(
                env,
                config,
                diagnostics,
                grasp_template,
                body_name,
                bottom_pos,
                viewer,
                realtime=launch_viewer,
                place_mode=place_mode,
                place_dx_m=place_dx_m,
                place_dy_m=place_dy_m,
                place_release_z_m=place_release_z_m,
                transfer_duration_s=transfer_duration_s,
                lower_duration_s=lower_duration_s,
                pad_pos=pad_pos,
                pad_rot=pad_rot,
                pad_stack_index=phase_index,
                place_pad_offset_x_m=place_pad_offset_x_m,
                place_pad_offset_y_m=place_pad_offset_y_m,
            ):
                print(f"Phase {phase_index + 1} motion aborted.")
                return False
            z_after = float(env.data.xpos[diagnostics.cup_body_id, 2])
            print(f"  {body_name} z ≈ {z_after:.3f} m after phase.")
        return True

    if not launch_viewer:
        return run_phases(None)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer
        return run_phases(viewer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer.")
    parser.add_argument("--scene", type=Path, default=STACK_SCENE_PATH, help="MJCF path (default: scene_cup_stack.xml).")
    parser.add_argument("--cup-position", type=float, nargs=3, default=(0.32, 0.0, 0.045), metavar=("X", "Y", "Z"))
    parser.add_argument("--cup-yaw-deg", type=float, default=0.0)
    parser.add_argument("--cup-mass", type=float, default=0.025, help="Per-cup mass (kg).")
    parser.add_argument("--gripper-force", type=float, default=2.2)
    parser.add_argument("--lift-height", type=float, default=0.12)
    parser.add_argument("--allow-config-cup-position-fallback", action="store_true")
    parser.add_argument("--no-debug-camera-frames", action="store_true")
    parser.add_argument("--debug-camera-frame-dir", type=Path, default=ROOT_DIR / "cup_camera_debug_frames")
    parser.add_argument(
        "--place-mode",
        choices=("pad", "offset"),
        default="pad",
        help="pad: AprilTag 9 on placement_pad (stack cups there); offset: fixed world deltas.",
    )
    parser.add_argument(
        "--place-pad-offset-x",
        type=float,
        default=DEFAULT_PLACE_PAD_OFFSET_X_M,
        metavar="M",
        help="Pad-frame X offset from placement_pad origin to set-down XY (meters).",
    )
    parser.add_argument(
        "--place-pad-offset-y",
        type=float,
        default=DEFAULT_PLACE_PAD_OFFSET_Y_M,
        metavar="M",
        help="Pad-frame Y offset toward cup stack when pad is beside the tower (meters).",
    )
    parser.add_argument(
        "--place-dx-m",
        type=float,
        default=DEFAULT_PLACE_DX_M,
        help="(--place-mode offset) world +X after lift before lowering (meters).",
    )
    parser.add_argument(
        "--place-dy-m",
        type=float,
        default=DEFAULT_PLACE_DY_M,
        help="(--place-mode offset) world +Y after lift before lowering (meters).",
    )
    parser.add_argument(
        "--place-release-z-m",
        type=float,
        default=DEFAULT_PLACE_RELEASE_Z_M,
        help="(--place-mode offset) absolute world Z when opening gripper (meters).",
    )
    parser.add_argument(
        "--transfer-duration-s",
        type=float,
        default=DEFAULT_TRANSFER_DURATION_S,
        help="Seconds for lateral move at lift height.",
    )
    parser.add_argument(
        "--lower-duration-s",
        type=float,
        default=DEFAULT_LOWER_DURATION_S,
        help="Seconds to lower to place-release height.",
    )
    return parser.parse_args()


def config_from_ns(args: argparse.Namespace) -> PickupConfig:
    return PickupConfig(
        scene_path=args.scene,
        cup_position=tuple(args.cup_position),
        cup_yaw_deg=args.cup_yaw_deg,
        cup_mass=args.cup_mass,
        gripper_force=args.gripper_force,
        lift_height=args.lift_height,
        primary_tag_id=8,
        grasp_index=0,
        skip_cup_geom_resize=True,
        manip_body_name="cup_bottom",
        freejoint_name="cup_bottom_freejoint",
        debug_camera_frame_dir=None if args.no_debug_camera_frames else args.debug_camera_frame_dir,
        allow_config_position_fallback=args.allow_config_cup_position_fallback,
    )


def main() -> None:
    args = parse_args()
    cfg = config_from_ns(args)
    ok = run_stack_pickup(
        cfg,
        launch_viewer=not args.headless,
        place_mode=args.place_mode,
        place_dx_m=args.place_dx_m,
        place_dy_m=args.place_dy_m,
        place_release_z_m=args.place_release_z_m,
        transfer_duration_s=args.transfer_duration_s,
        lower_duration_s=args.lower_duration_s,
        place_pad_offset_x_m=args.place_pad_offset_x,
        place_pad_offset_y_m=args.place_pad_offset_y,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
