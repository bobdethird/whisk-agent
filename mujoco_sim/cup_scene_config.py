from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

NVIDIA_GLASS_CUP_ASSET_ROOT = MODEL_DIR / "assets" / "objects" / "nvidia_glass_cup" / "glass_cup"
NVIDIA_GLASS_CUP_MODEL_NAME = "GlassCup023"


@dataclass(frozen=True)
class MeshCupAssetSpec:
    model_name: str
    root: Path
    radius: float
    half_height: float
    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]

    @property
    def visual_mesh_path(self) -> Path:
        return self.root / "visual" / "Clear.obj"

    @property
    def texture_path(self) -> Path:
        return self.root / "visual" / "T_BC001.png"

    @property
    def collision_dir(self) -> Path:
        return self.root / "collision"

    @property
    def visual_mesh_name(self) -> str:
        return f"{self.model_name}_Clear_vis"

    @property
    def texture_name(self) -> str:
        return f"{self.model_name}_Clear_texture"

    @property
    def material_name(self) -> str:
        return f"{self.model_name}_Clear_material"

    def collision_mesh_name(self, index: int) -> str:
        return f"{self.model_name}_collision_mesh_{index}"


NVIDIA_GLASS_CUP_ASSET = MeshCupAssetSpec(
    model_name=NVIDIA_GLASS_CUP_MODEL_NAME,
    root=NVIDIA_GLASS_CUP_ASSET_ROOT / NVIDIA_GLASS_CUP_MODEL_NAME,
    radius=0.032809,
    half_height=0.0735655,
    min_xyz=(-0.032809, -0.032809, -0.0735655),
    max_xyz=(0.032809, 0.032809, 0.0735655),
)

DEFAULT_CUP_RADIUS = NVIDIA_GLASS_CUP_ASSET.radius
DEFAULT_CUP_HALF_HEIGHT = NVIDIA_GLASS_CUP_ASSET.half_height
DEFAULT_CUP_MASS = 0.025
DEFAULT_CUP_FRICTION = (1.0, 0.02, 0.002)
DEFAULT_JAW_FRICTION = (1.2, 0.005, 0.0005)

CUP_TAG_SIZE = 0.024
CUP_TAG_MOUNT_DISTANCE_FROM_CENTER = DEFAULT_CUP_RADIUS + TAG_THICKNESS_M
CUP_TAG_TO_CUP_CENTER_OFFSET = (0.0, 0.0, CUP_TAG_MOUNT_DISTANCE_FROM_CENTER)
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

    def collision_geom_name(self, index: int) -> str:
        return f"{self.body_name}_collision_{index:02d}"

    @property
    def visual_geom_name(self) -> str:
        return f"{self.body_name}_visual"

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
