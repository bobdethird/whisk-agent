from __future__ import annotations

from dataclasses import dataclass

try:
    from .apriltag_world_config import TABLE_SCENE
    from .scene_authoring import MODEL_DIR, TAG_THICKNESS_M, AprilTagSpec, CameraSpec
except ImportError:
    from apriltag_world_config import TABLE_SCENE
    from scene_authoring import MODEL_DIR, TAG_THICKNESS_M, AprilTagSpec, CameraSpec


SPOON_SCENE_PATH = MODEL_DIR / "scene_spoon.xml"
SPOON_SCENE_METADATA_PATH = MODEL_DIR / "spoon_scene_metadata.json"

SPOON_TAG_SIZE = 0.024
TOP_DOWN_CAMERA_NAME = "table_observer"


@dataclass(frozen=True)
class SpoonSpec:
    body_name: str = "spoon"
    freejoint_name: str = "spoon_freejoint"
    initial_position: tuple[float, float, float] = (0.30, 0.02, 0.012)
    mass: float = 0.010
    friction: tuple[float, float, float] = (2.0, 0.02, 0.002)
    handle_half_length: float = 0.075
    handle_radius: float = 0.009
    bowl_radii: tuple[float, float, float] = (0.018, 0.013, 0.004)
    # Site on the spoon shaft itself (not on a separate nearby feature)
    grasp_site_pos: tuple[float, float, float] = (-0.015, 0.0, 0.010)


SPOON = SpoonSpec()

# Mounted on the spoon handle (child body of spoon).
# Tag id 10 chosen because 6,7,8,9 are already registered in grasp_library.
SPOON_HANDLE_TAG = AprilTagSpec(
    tag_id=10,
    size_m=SPOON_TAG_SIZE,
    pos=(-0.020, 0.0, 0.020),
    name_prefix="spoon_handle_",
)

# Flat reference tag on the table next to the spoon (for fixed-offset comparison strategy only).
SPOON_TABLE_REF_TAG = AprilTagSpec(
    tag_id=9,
    size_m=SPOON_TAG_SIZE,
    pos=(0.23, 0.0, TAG_THICKNESS_M),
    name_prefix="spoon_ref_",
)

# Offsets are expressed in tag frame and transformed by estimate.world_rotation.
# These are calibrated to the authored spoon geometry in this scene.
SPOON_HANDLE_TAG_TO_GRASP_OFFSET = (0.0021, 0.0043, 0.0078)
SPOON_TABLE_TAG_TO_GRASP_OFFSET = (0.0540, -0.0180, -0.0151)

SPOON_SCENE_TAGS = (SPOON_HANDLE_TAG, SPOON_TABLE_REF_TAG)

SPOON_SCENE_CAMERAS = (
    CameraSpec(
        name="spoon_observer",
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
