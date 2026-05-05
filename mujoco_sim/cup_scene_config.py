from __future__ import annotations

from dataclasses import dataclass

try:
    from .apriltag_world_config import TABLE_SCENE
    from .scene_authoring import (
        MODEL_DIR,
        TAG_THICKNESS_M,
        AprilTagSpec,
        CameraSpec,
    )
except ImportError:
    from apriltag_world_config import TABLE_SCENE
    from scene_authoring import (
        MODEL_DIR,
        TAG_THICKNESS_M,
        AprilTagSpec,
        CameraSpec,
    )


CUP_SCENE_PATH = MODEL_DIR / "scene_cup.xml"
CUP_SCENE_METADATA_PATH = MODEL_DIR / "cup_scene_metadata.json"

DEFAULT_CUP_RADIUS = 0.023
DEFAULT_CUP_HALF_HEIGHT = 0.045
DEFAULT_CUP_RIM_OVERHANG = 0.0
DEFAULT_CUP_RIM_HALF_HEIGHT = 0.004
DEFAULT_CUP_MASS = 0.025
DEFAULT_CUP_FRICTION = (1.0, 0.02, 0.002)
DEFAULT_JAW_FRICTION = (1.2, 0.005, 0.0005)

DEFAULT_SPOON_MASS = 0.020
DEFAULT_SPOON_FRICTION = (0.9, 0.02, 0.002)
DEFAULT_SPOON_POSITION = (0.26, 0.12, 0.004)
DEFAULT_SPOON_HANDLE_LENGTH = 0.14
DEFAULT_SPOON_HANDLE_RADIUS = 0.004
DEFAULT_SPOON_BOWL_RADII = (0.018, 0.013, 0.004)

CUP_TAG_SIZE = 0.024
CUP_TAG_MOUNT_DISTANCE_FROM_CENTER = 0.026
CUP_TAG_TO_CUP_CENTER_OFFSET = (0.0, 0.0, 0.020)
SIDE_MOUNTED_TAG_QUAT = (0.7071067812, 0.0, -0.7071067812, 0.0)

WRIST_CAMERA_NAME = "wrist_cam"
TOP_DOWN_CAMERA_NAME = "table_observer"
CUP_TAG_CAMERA_NAMES = (WRIST_CAMERA_NAME, TOP_DOWN_CAMERA_NAME)
PLACE_TAG_CAMERA_NAMES = (TOP_DOWN_CAMERA_NAME,)


@dataclass(frozen=True)
class CupObjectSpec:
    label: str
    body_name: str
    initial_position: tuple[float, float, float]
    tag: AprilTagSpec
    rgba: tuple[float, float, float, float]
    tag_to_center_offset: tuple[float, float, float] = CUP_TAG_TO_CUP_CENTER_OFFSET

    @property
    def freejoint_name(self) -> str:
        return f"{self.body_name}_freejoint"

    @property
    def side_geom_name(self) -> str:
        return f"{self.body_name}_side_collision"

    @property
    def rim_geom_name(self) -> str:
        return f"{self.body_name}_rim_collision"

    @property
    def visual_geom_name(self) -> str:
        return f"{self.body_name}_visual"

    @property
    def site_name(self) -> str:
        return f"{self.body_name}_site"


@dataclass(frozen=True)
class SpoonObjectSpec:
    label: str
    body_name: str
    initial_position: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    mass: float = DEFAULT_SPOON_MASS
    friction: tuple[float, float, float] = DEFAULT_SPOON_FRICTION
    handle_length: float = DEFAULT_SPOON_HANDLE_LENGTH
    handle_radius: float = DEFAULT_SPOON_HANDLE_RADIUS
    bowl_radii: tuple[float, float, float] = DEFAULT_SPOON_BOWL_RADII

    @property
    def site_name(self) -> str:
        return f"{self.body_name}_site"


PLACE_TAG = AprilTagSpec(
    tag_id=0,
    size_m=CUP_TAG_SIZE,
    pos=(0.44, -0.08, TAG_THICKNESS_M),
    name_prefix="place_",
)

PRIMARY_CUP = CupObjectSpec(
    label="first cup",
    body_name="cup",
    initial_position=(0.32, 0.0, DEFAULT_CUP_HALF_HEIGHT),
    tag=AprilTagSpec(
        tag_id=6,
        size_m=CUP_TAG_SIZE,
        pos=(-CUP_TAG_MOUNT_DISTANCE_FROM_CENTER, 0.0, 0.0),
        quat=SIDE_MOUNTED_TAG_QUAT,
        name_prefix="cup_",
    ),
    rgba=(0.55, 0.75, 1.0, 0.35),
)

SECOND_CUP = CupObjectSpec(
    label="second cup",
    body_name="second_cup",
    initial_position=(0.32, -0.14, DEFAULT_CUP_HALF_HEIGHT),
    tag=AprilTagSpec(
        tag_id=1,
        size_m=CUP_TAG_SIZE,
        pos=(-CUP_TAG_MOUNT_DISTANCE_FROM_CENTER, 0.0, 0.0),
        quat=SIDE_MOUNTED_TAG_QUAT,
        name_prefix="second_cup_",
    ),
    rgba=(0.55, 0.95, 0.65, 0.35),
)

SPOON = SpoonObjectSpec(
    label="spoon",
    body_name="spoon",
    initial_position=DEFAULT_SPOON_POSITION,
    rgba=(0.86, 0.86, 0.90, 1.0),
)

CUPS = (PRIMARY_CUP, SECOND_CUP)
TABLE_TAGS = (PLACE_TAG,)
CUP_SCENE_TAGS = (*TABLE_TAGS, *(cup.tag for cup in CUPS))

CUP_SCENE_CAMERAS = (
    CameraSpec(
        name="cup_observer",
        pos=(0.25, -0.35, 0.35),
        xyaxes=(1.0, 0.0, 0.0, 0.0, 0.55, 0.4),
        fovy=55.0,
    ),
    CameraSpec(
        name=TOP_DOWN_CAMERA_NAME,
        pos=(0.08, -0.28, 0.18),
        xyaxes=(0.7945188897, -0.6072394371, 0.0, 0.1150475128, 0.1505294561, 0.9818884624),
        fovy=60.0,
    ),
)
