# Whisk — Robotic Matcha-Making Agent

Software infrastructure for an autonomous matcha-making arm built on the **LeRobot SO-101** platform. An LLM agent loop (Claude) plans each step; AprilTag pose estimation locates objects; a trajectory engine moves the arm safely between waypoints.

---

## Architecture

```
whisk_agent/
├── perception/
│   ├── mock_poses.py     # Hardcoded pose map + Gaussian-noised get_mock_poses()
│   └── detector.py       # AprilTagDetector stub (production interface)
├── arm/
│   ├── grasp_configs.py  # Per-object gripper & height configs
│   ├── trajectory.py     # Y-ceiling waypoint planner
│   └── executor.py       # Executor Protocol + MockExecutor
├── agent/
│   ├── prompts.py        # System prompt, tool catalogue, state formatter
│   ├── tools.py          # validate_tool_call() + dispatch_tool()
│   └── loop.py           # run_agent_loop() — LLM → validate → dispatch → repeat
├── tests/
│   ├── test_trajectory.py
│   ├── test_perception.py
│   └── test_agent_loop.py
├── main.py               # Entry point (mock run, no hardware required)
└── requirements.txt
```

### How it works

```
┌────────────┐   poses    ┌──────────────┐  JSON tool call  ┌──────────┐
│ AprilTag   │──────────▶ │  Agent Loop  │◀────────────────▶│  Claude  │
│ Detector   │            │  (loop.py)   │                   │   LLM    │
└────────────┘            └──────┬───────┘                   └──────────┘
                                 │ dispatch
                          ┌──────▼───────┐
                          │ Tool Dispatch │  validate → trajectory → executor
                          │  (tools.py)  │
                          └──────┬───────┘
                                 │ waypoints
                          ┌──────▼───────┐
                          │   Executor   │  MockExecutor | LeRobotExecutor
                          └──────────────┘
```

**Each agent loop iteration:**
1. Fetch object poses (`get_poses_fn`)
2. Format world state into a prompt
3. Call Claude — expect a single raw-JSON tool call
4. Validate schema (`validate_tool_call`)
5. Dispatch to executor (`dispatch_tool`)
6. Record `{step, action, result}` in history
7. Stop on `done()`, 3 consecutive errors, or step budget

---

## Trajectory planning

All motion uses a **Y-ceiling** strategy — the arm lifts to a safe clearance height before any XZ translation, avoiding collisions with objects on the work surface.

| Function | Waypoints | Purpose |
|---|---|---|
| `compute_trajectory` | 3 | Simple hover / point-to-point |
| `compute_grasp_trajectory` | 4 | Adds pre-grasp hover above object |
| `compute_place_trajectory` | 3 | Stops just above surface for gentle set-down |

---

## Objects & poses

| Object | x | y | z | Notes |
|---|---|---|---|---|
| `matcha_cup` | 0.30 | 0.02 | 0.40 | Target vessel |
| `matcha_bowl` | 0.10 | 0.02 | 0.40 | Whisking bowl |
| `matcha_whisk` | 0.20 | 0.02 | 0.50 | Bamboo chasen |
| `matcha_scoop` | 0.35 | 0.02 | 0.45 | Bamboo chashaku |
| `cup_of_ice` | 0.00 | 0.10 | 0.35 | Ice source |
| `cup_of_water` | 0.05 | 0.10 | 0.35 | Hot water source |

Coordinates in metres; **y is vertical (up)**.

---

## Setup

```bash
git clone https://github.com/CalSaksham/Whisk--subtask.git
cd Whisk--subtask
pip install -r requirements.txt
```

Set your Anthropic API key — the agent reads it from the environment:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running

### Mock run (no hardware required)

```bash
python main.py
```

All arm calls are intercepted by `MockExecutor`, which logs every action with a `[MOCK]` prefix. No physical robot or camera needed.

### Real hardware

Implement the `Executor` Protocol in your own class:

```python
from arm.executor import Executor

class LeRobotExecutor:
    def move_to(self, waypoint: list[float]) -> dict: ...
    def set_gripper(self, width: float) -> dict: ...
    def get_end_effector_pose(self) -> list[float]: ...
```

Then in `main.py`, replace `MockExecutor()` with `LeRobotExecutor()`. Nothing else changes.

### Real perception

Implement `AprilTagDetector.get_poses()` in `perception/detector.py` and pass it as `get_poses_fn` to `run_agent_loop`. The interface is identical to `get_mock_poses`.

---

## Tests

```bash
pytest tests/ -v
```

32 tests covering trajectory geometry, pose noise bounds, and agent loop behaviour (done termination, JSON parse failures, retry logic, 3-consecutive-error abort). All LLM calls are mocked — no API key required for tests.

---

## Available tools (LLM-facing)

| Tool | Args | Description |
|---|---|---|
| `move_arm` | `target_object`, `action` | Move to object; action: `grasp` \| `place` \| `hover` |
| `open_gripper` | — | Open gripper fully (8 cm) |
| `close_gripper` | — | Close gripper fully |
| `wait` | `seconds` | Pause (e.g. let powder dissolve) |
| `done` | — | Signal task complete, end loop |

---

## Model

Default model: `claude-haiku-4-5-20251001`. Override via the `model` parameter of `run_agent_loop()`.
