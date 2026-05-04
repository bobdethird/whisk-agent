import math
import time

import mujoco


JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def convert_to_dictionary(qpos):
    return {
        "shoulder_pan": math.degrees(qpos[0]),
        "shoulder_lift": math.degrees(qpos[1]),
        "elbow_flex": math.degrees(qpos[2]),
        "wrist_flex": math.degrees(qpos[3]),
        "wrist_roll": math.degrees(qpos[4]),
        "gripper": qpos[5] * 100.0 / math.pi,
    }


def convert_to_list(position_dict):
    return [
        math.radians(position_dict["shoulder_pan"]),
        math.radians(position_dict["shoulder_lift"]),
        math.radians(position_dict["elbow_flex"]),
        math.radians(position_dict["wrist_flex"]),
        math.radians(position_dict["wrist_roll"]),
        position_dict["gripper"] * math.pi / 100.0,
    ]


def set_initial_pose(model, data, position_dict):
    data.qpos[: len(JOINT_ORDER)] = convert_to_list(position_dict)
    mujoco.mj_forward(model, data)


def send_position_command(data, position_dict):
    if data.ctrl.size < len(JOINT_ORDER):
        raise ValueError("SO101 scene.xml must define one position actuator per joint.")
    data.ctrl[: len(JOINT_ORDER)] = convert_to_list(position_dict)


def step_realtime(model, data, viewer, step_callback=None):
    step_start = time.time()
    mujoco.mj_step(model, data)
    if step_callback is not None:
        step_callback()
    viewer.sync()

    sleep_time = model.opt.timestep - (time.time() - step_start)
    if sleep_time > 0:
        time.sleep(sleep_time)


def move_to_pose(model, data, viewer, desired_position, duration, step_callback=None):
    start_time = time.time()
    starting_pose = convert_to_dictionary(data.qpos.copy())

    while viewer.is_running():
        elapsed = time.time() - start_time
        if elapsed > duration:
            break

        alpha = min(elapsed / duration, 1.0)
        position_dict = {}
        for joint in JOINT_ORDER:
            p0 = starting_pose[joint]
            pf = desired_position[joint]
            position_dict[joint] = (1.0 - alpha) * p0 + alpha * pf

        send_position_command(data, position_dict)
        step_realtime(model, data, viewer, step_callback=step_callback)

    send_position_command(data, desired_position)
    step_realtime(model, data, viewer, step_callback=step_callback)


def hold_position(model, data, viewer, duration, step_callback=None):
    position_dict = convert_to_dictionary(data.qpos.copy())
    start_time = time.time()

    while viewer.is_running() and time.time() - start_time < duration:
        send_position_command(data, position_dict)
        step_realtime(model, data, viewer, step_callback=step_callback)


def hold_position_until_closed(model, data, viewer, step_callback=None):
    position_dict = convert_to_dictionary(data.qpos.copy())

    while viewer.is_running():
        send_position_command(data, position_dict)
        step_realtime(model, data, viewer, step_callback=step_callback)
