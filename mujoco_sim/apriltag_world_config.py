from __future__ import annotations

try:
    from .scene_authoring import (
        APRILTAG_ASSET_DIR,
        DEFAULT_RENDER_HEIGHT,
        DEFAULT_RENDER_WIDTH,
        MODEL_DIR,
        ROBOT_MODEL_PATH,
        TAG_BLACK_SQUARE_FRACTION,
        TAG_FAMILY,
        TAG_THICKNESS_M,
        AprilTagSpec,
        CameraSpec,
        TableSceneSpec,
    )
except ImportError:
    from scene_authoring import (
        APRILTAG_ASSET_DIR,
        DEFAULT_RENDER_HEIGHT,
        DEFAULT_RENDER_WIDTH,
        MODEL_DIR,
        ROBOT_MODEL_PATH,
        TAG_BLACK_SQUARE_FRACTION,
        TAG_FAMILY,
        TAG_THICKNESS_M,
        AprilTagSpec,
        CameraSpec,
        TableSceneSpec,
    )


SCENE_PATH = MODEL_DIR / "scene.xml"
METADATA_PATH = MODEL_DIR / "apriltag_world_metadata.json"

TABLE_TOP_Z_M = 0.0


TABLE_SCENE = TableSceneSpec(
    name="work_table",
    top_pos_xy=(0.22, 0.0),
    top_half_size=(0.36, 0.26, 0.025),
    top_z=TABLE_TOP_Z_M,
    floor_z=-0.45,
    leg_half_size_xy=(0.018, 0.018),
    leg_margin=(0.045, 0.045),
    rgba=(0.55, 0.36, 0.20, 1.0),
)

TABLE_TAG_SIZE_M = 0.024
TABLE_TAG_Z_M = TABLE_TOP_Z_M + TAG_THICKNESS_M


APRILTAGS = (
    AprilTagSpec(tag_id=0, size_m=TABLE_TAG_SIZE_M, pos=(0.32, -0.08, TABLE_TAG_Z_M)),
    AprilTagSpec(tag_id=1, size_m=TABLE_TAG_SIZE_M, pos=(0.38, -0.08, TABLE_TAG_Z_M)),
    AprilTagSpec(tag_id=2, size_m=TABLE_TAG_SIZE_M, pos=(0.44, -0.08, TABLE_TAG_Z_M)),
    AprilTagSpec(tag_id=3, size_m=TABLE_TAG_SIZE_M, pos=(0.32, 0.04, TABLE_TAG_Z_M)),
    AprilTagSpec(tag_id=4, size_m=TABLE_TAG_SIZE_M, pos=(0.38, 0.04, TABLE_TAG_Z_M)),
    AprilTagSpec(tag_id=5, size_m=TABLE_TAG_SIZE_M, pos=(0.44, 0.04, TABLE_TAG_Z_M)),
)

CAMERAS = (
    CameraSpec(
        name="apriltag_overhead",
        pos=(0.25, -0.35, 0.45),
        xyaxes=(1.0, 0.0, 0.0, 0.0, 0.45, 0.35),
    ),
    CameraSpec(
        name="table_observer",
        pos=(0.25, 0.0, 0.5),
        xyaxes=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        fovy=60.0,
    ),
)
