import argparse
import json
import zlib
from pathlib import Path

import mujoco  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from apriltag_world_config import CAMERAS, DEFAULT_RENDER_HEIGHT, DEFAULT_RENDER_WIDTH, METADATA_PATH, SCENE_PATH, TAG_FAMILY


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = ROOT_DIR / "apriltag_camera_frame.png"


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    import struct

    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def write_rgb_png(path: Path, image: np.ndarray) -> None:
    import struct

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB uint8 image.")

    height, width, _ = image.shape
    raw_rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(raw_rows, 9))
        + png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def render_camera(scene_path: Path, camera_name: str, width: int, height: int) -> tuple[mujoco.MjModel, mujoco.MjData, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera_name)
        image = renderer.render()

    return model, data, image


def grayscale(image: np.ndarray) -> np.ndarray:
    return np.clip(
        0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2],
        0,
        255,
    ).astype(np.uint8)


def detect_apriltags(image: np.ndarray):
    try:
        from pupil_apriltags import Detector  # type: ignore[import-not-found]
    except ImportError:
        print("Skipping detection: install pupil-apriltags to detect rendered tags.")
        return []

    detector = Detector(
        families=TAG_FAMILY,
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )
    return detector.detect(grayscale(image))


def print_ground_truth(model: mujoco.MjModel, data: mujoco.MjData, metadata: dict) -> None:
    for tag in metadata.get("tags", []):
        site_name = tag["site"]
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            print(f"Ground truth unavailable: site {site_name!r} not found.")
            continue
        position = data.site_xpos[site_id]
        print(
            f"Ground truth {tag['name']} site position: "
            f"x={position[0]:.4f} y={position[1]:.4f} z={position[2]:.4f} m"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render and detect AprilTags from the generated MuJoCo world.")
    parser.add_argument("--scene", type=Path, default=SCENE_PATH, help="MJCF scene to render.")
    parser.add_argument(
        "--camera",
        default=CAMERAS[0].name,
        help="Named MuJoCo camera to render.",
    )
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH, help="World metadata JSON path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output PNG frame path.")
    parser.add_argument("--width", type=int, default=DEFAULT_RENDER_WIDTH, help="Render width in pixels.")
    parser.add_argument("--height", type=int, default=DEFAULT_RENDER_HEIGHT, help="Render height in pixels.")
    parser.add_argument("--skip-detection", action="store_true", help="Only render the camera frame.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    model, data, image = render_camera(args.scene, args.camera, args.width, args.height)

    write_rgb_png(args.output, image)
    print(f"Wrote camera frame: {args.output}")
    print_ground_truth(model, data, metadata)

    if args.skip_detection:
        return

    detections = detect_apriltags(image)
    if not detections:
        print("Detected 0 AprilTags.")
        return

    print(f"Detected {len(detections)} AprilTag(s):")
    for detection in detections:
        family = detection.tag_family.decode() if isinstance(detection.tag_family, bytes) else detection.tag_family
        corners = np.array2string(detection.corners, precision=1, suppress_small=True)
        print(f"  id={detection.tag_id} family={family} corners={corners}")


if __name__ == "__main__":
    main()
