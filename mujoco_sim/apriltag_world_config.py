from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "simulation_code" / "model"
SCENE_PATH = MODEL_DIR / "scene.xml"
METADATA_PATH = MODEL_DIR / "apriltag_world_metadata.json"
APRILTAG_ASSET_DIR = MODEL_DIR / "assets" / "apriltags"

TAG_FAMILY = "tag36h11"
TAG_THICKNESS_M = 0.002
TAG_BLACK_SQUARE_FRACTION = 0.80
DEFAULT_RENDER_WIDTH = 2560
DEFAULT_RENDER_HEIGHT = 1920

TABLE_TOP_Z_M = 0.0


@dataclass(frozen=True)
class AprilTagSpec:
    tag_id: int
    size_m: float
    pos: tuple[float, float, float]
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def name(self) -> str:
        return f"{TAG_FAMILY}_{self.tag_id:05d}"

    @property
    def asset_filename(self) -> str:
        return f"tag36_11_{self.tag_id:05d}.png"

    @property
    def source_svg_filename(self) -> str:
        return f"{TAG_FAMILY}-{self.tag_id}.svg"


@dataclass(frozen=True)
class CameraSpec:
    name: str
    pos: tuple[float, float, float]
    xyaxes: tuple[float, float, float, float, float, float]
    fovy: float = 45.0


@dataclass(frozen=True)
class TableSceneSpec:
    name: str
    top_pos_xy: tuple[float, float]
    top_half_size: tuple[float, float, float]
    top_z: float
    floor_z: float
    leg_half_size_xy: tuple[float, float]
    leg_margin: tuple[float, float]
    rgba: tuple[float, float, float, float]

    @property
    def top_center_pos(self) -> tuple[float, float, float]:
        return (self.top_pos_xy[0], self.top_pos_xy[1], self.top_z - self.top_half_size[2])

    @property
    def top_bottom_z(self) -> float:
        return self.top_z - 2.0 * self.top_half_size[2]

    @property
    def leg_half_size(self) -> tuple[float, float, float]:
        return (
            self.leg_half_size_xy[0],
            self.leg_half_size_xy[1],
            0.5 * (self.top_bottom_z - self.floor_z),
        )

    @property
    def leg_center_z(self) -> float:
        return 0.5 * (self.top_bottom_z + self.floor_z)

    @property
    def leg_positions(self) -> tuple[tuple[float, float, float], ...]:
        x_center, y_center = self.top_pos_xy
        x_offset = self.top_half_size[0] - self.leg_margin[0]
        y_offset = self.top_half_size[1] - self.leg_margin[1]
        z = self.leg_center_z
        return (
            (x_center - x_offset, y_center - y_offset, z),
            (x_center - x_offset, y_center + y_offset, z),
            (x_center + x_offset, y_center - y_offset, z),
            (x_center + x_offset, y_center + y_offset, z),
        )


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
