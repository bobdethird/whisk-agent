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


APRILTAGS = (
    AprilTagSpec(
        tag_id=0,
        size_m=0.01,
        pos=(0.25, 0.0, TAG_THICKNESS_M),
    ),
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
