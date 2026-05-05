import argparse
import json
import os
import re
import struct
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET

from apriltag_world_config import (
    APRILTAG_ASSET_DIR,
    APRILTAGS,
    CAMERAS,
    DEFAULT_RENDER_HEIGHT,
    DEFAULT_RENDER_WIDTH,
    METADATA_PATH,
    MODEL_DIR,
    ROBOT_MODEL_PATH,
    SCENE_PATH,
    TAG_BLACK_SQUARE_FRACTION,
    TAG_FAMILY,
    TAG_THICKNESS_M,
    TABLE_SCENE,
    AprilTagSpec,
    CameraSpec,
    TableSceneSpec,
)


TAG_TEXTURE_SIZE_PX = 640
TAG_CARD_MESH_FILE = "apriltags/tag_card.obj"


def format_floats(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:g}" for value in values)


def path_for_mjcf(path: Path, output_path: Path) -> str:
    return os.path.relpath(path, start=output_path.parent).replace(os.sep, "/")


def parse_svg_path(d: str) -> list[list[tuple[float, float]]]:
    tokens = re.findall(r"[MLZmlz]|-?\d+(?:\.\d+)?", d)
    polygons: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    command = ""
    i = 0

    while i < len(tokens):
        token = tokens[i]
        if token in {"M", "L", "Z", "m", "l", "z"}:
            command = token
            i += 1
            if command in {"Z", "z"}:
                if current:
                    polygons.append(current)
                    current = []
                continue

        if command in {"M", "L"}:
            x = float(tokens[i])
            y = float(tokens[i + 1])
            i += 2
            if command == "M" and current:
                polygons.append(current)
                current = []
            current.append((x, y))
            command = "L"
            continue

        raise ValueError(f"Unsupported SVG path command in {d!r}")

    if current:
        polygons.append(current)
    return polygons


def parse_svg_fill(style: str) -> tuple[int, int, int]:
    match = re.search(r"fill:\s*#([0-9a-fA-F]{6})", style)
    if match is None:
        raise ValueError(f"Unsupported SVG fill style: {style!r}")
    color = match.group(1)
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, point in enumerate(polygon):
        xi, yi = point
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def write_rgb_png(path: Path, width: int, height: int, rows: list[bytes]) -> None:
    raw_rows = b"".join(b"\x00" + row for row in rows)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(raw_rows, 9))
        + png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def rasterize_tag_svg(source_path: Path, output_path: Path, size_px: int = TAG_TEXTURE_SIZE_PX) -> None:
    if not source_path.exists():
        raise FileNotFoundError(f"Missing AprilTag SVG source: {source_path}")

    tree = ET.parse(source_path)
    root = tree.getroot()
    viewbox = root.attrib.get("viewBox")
    if viewbox is None:
        raise ValueError(f"SVG must define a viewBox: {source_path}")

    min_x, min_y, width, height = (float(value) for value in viewbox.split())

    shapes: list[tuple[tuple[int, int, int], list[list[tuple[float, float]]]]] = []
    for element in root.iter():
        if not element.tag.endswith("path"):
            continue
        shapes.append(
            (
                parse_svg_fill(element.attrib.get("style", "")),
                parse_svg_path(element.attrib["d"]),
            )
        )

    rows: list[bytes] = []
    for row in range(size_px):
        y = min_y + ((row + 0.5) / size_px) * height
        pixels = bytearray()
        for col in range(size_px):
            x = min_x + ((col + 0.5) / size_px) * width
            color = (255, 255, 255)
            for fill, polygons in shapes:
                if any(point_in_polygon(x, y, polygon) for polygon in polygons):
                    color = fill
            pixels.extend(color)
        rows.append(bytes(pixels))

    write_rgb_png(output_path, size_px, size_px, rows)


def ensure_tag_texture(tag: AprilTagSpec) -> Path:
    source_path = APRILTAG_ASSET_DIR / tag.source_svg_filename
    output_path = APRILTAG_ASSET_DIR / tag.asset_filename
    if output_path.exists() and output_path.stat().st_mtime >= source_path.stat().st_mtime:
        return output_path

    rasterize_tag_svg(source_path, output_path)
    print(f"Wrote AprilTag texture: {output_path}")
    return output_path


def add_visual(root: ET.Element) -> None:
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        diffuse="0.6 0.6 0.6",
        ambient="0.3 0.3 0.3",
        specular="0 0 0",
    )
    ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")
    ET.SubElement(
        visual,
        "global",
        azimuth="160",
        elevation="-20",
        offwidth=str(DEFAULT_RENDER_WIDTH),
        offheight=str(DEFAULT_RENDER_HEIGHT),
    )


def add_assets(
    root: ET.Element,
    tags: tuple[AprilTagSpec, ...],
    table_scene: TableSceneSpec,
    output_path: Path,
) -> None:
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1="0.3 0.5 0.7",
        rgb2="0 0 0",
        width="512",
        height="3072",
    )
    ET.SubElement(
        asset,
        "texture",
        type="2d",
        name="groundplane",
        builtin="checker",
        mark="edge",
        rgb1="0.18 0.19 0.20",
        rgb2="0.12 0.13 0.14",
        markrgb="0.35 0.35 0.35",
        width="300",
        height="300",
    )
    ET.SubElement(
        asset,
        "texture",
        type="2d",
        name=f"{table_scene.name}_top_texture",
        builtin="checker",
        mark="edge",
        rgb1="0.50 0.32 0.18",
        rgb2="0.42 0.26 0.14",
        markrgb="0.62 0.45 0.28",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "material",
        name="groundplane",
        texture="groundplane",
        texuniform="true",
        texrepeat="5 5",
        reflectance="0.04",
        specular="0.05",
        shininess="0.05",
    )
    ET.SubElement(
        asset,
        "material",
        name=f"{table_scene.name}_top_material",
        texture=f"{table_scene.name}_top_texture",
        rgba=format_floats(table_scene.rgba),
        texuniform="true",
        texrepeat="3 2",
        reflectance="0",
        specular="0.08",
        shininess="0.08",
    )
    ET.SubElement(
        asset,
        "material",
        name=f"{table_scene.name}_leg_material",
        rgba="0.36 0.22 0.12 1",
        reflectance="0",
        specular="0.05",
        shininess="0.05",
    )
    for tag in tags:
        texture_name = f"{tag.name}_texture"
        asset_path = APRILTAG_ASSET_DIR / tag.asset_filename
        if not asset_path.exists():
            raise FileNotFoundError(f"Missing AprilTag asset: {asset_path}")
        card_size_m = tag.size_m / TAG_BLACK_SQUARE_FRACTION

        ET.SubElement(
            asset,
            "mesh",
            name=f"{tag.name}_mesh",
            file=TAG_CARD_MESH_FILE,
            scale=format_floats((card_size_m, card_size_m, TAG_THICKNESS_M)),
        )
        ET.SubElement(
            asset,
            "texture",
            type="2d",
            name=texture_name,
            file=path_for_mjcf(asset_path, output_path),
        )
        ET.SubElement(
            asset,
            "material",
            name=tag.name,
            texture=texture_name,
            rgba="1 1 1 1",
            texuniform="true",
            texrepeat="1 1",
            emission="0.5",
            specular="0",
            shininess="0",
            reflectance="0",
        )


def add_apriltag_body(parent: ET.Element, tag: AprilTagSpec, pos: tuple[float, float, float]) -> None:
    body = ET.SubElement(
        parent,
        "body",
        name=tag.name,
        pos=format_floats(pos),
        euler=format_floats(tag.euler),
    )
    ET.SubElement(
        body,
        "geom",
        name=f"{tag.name}_visual",
        type="mesh",
        mesh=f"{tag.name}_mesh",
        material=tag.name,
        contype="0",
        conaffinity="0",
        density="0",
    )
    ET.SubElement(
        body,
        "site",
        name=f"{tag.name}_site",
        pos="0 0 0",
        size="0.005",
        rgba="0 1 0 0",
    )


def add_table_scene(worldbody: ET.Element, table_scene: TableSceneSpec) -> None:
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        size="0 0 0.05",
        pos=format_floats((0.0, 0.0, table_scene.floor_z)),
        type="plane",
        material="groundplane",
    )
    table_body = ET.SubElement(worldbody, "body", name=table_scene.name)
    ET.SubElement(
        table_body,
        "geom",
        name=f"{table_scene.name}_top",
        type="box",
        pos=format_floats(table_scene.top_center_pos),
        size=format_floats(table_scene.top_half_size),
        material=f"{table_scene.name}_top_material",
        friction="1.3 0.01 0.001",
    )
    for index, leg_pos in enumerate(table_scene.leg_positions):
        ET.SubElement(
            table_body,
            "geom",
            name=f"{table_scene.name}_leg_{index}",
            type="box",
            pos=format_floats(leg_pos),
            size=format_floats(table_scene.leg_half_size),
            material=f"{table_scene.name}_leg_material",
            friction="1.0 0.01 0.001",
        )


def add_worldbody(
    root: ET.Element,
    tags: tuple[AprilTagSpec, ...],
    table_scene: TableSceneSpec,
    cameras: tuple[CameraSpec, ...],
) -> None:
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", pos="0 0 3.5", dir="0 0 -1", directional="true")
    ET.SubElement(worldbody, "light", pos="0.15 -0.45 0.75", dir="0.1 0.35 -1", directional="true", diffuse="0.45 0.4 0.35")
    add_table_scene(worldbody, table_scene)

    for tag in tags:
        add_apriltag_body(worldbody, tag, tag.pos)

    for camera in cameras:
        ET.SubElement(
            worldbody,
            "camera",
            name=camera.name,
            pos=format_floats(camera.pos),
            xyaxes=format_floats(camera.xyaxes),
            fovy=f"{camera.fovy:g}",
        )


def build_scene(
    tags: tuple[AprilTagSpec, ...] = APRILTAGS,
    table_scene: TableSceneSpec = TABLE_SCENE,
    cameras: tuple[CameraSpec, ...] = CAMERAS,
    output_path: Path = SCENE_PATH,
) -> ET.ElementTree:
    root = ET.Element("mujoco", model="scene")
    root.append(
        ET.Comment(
            "Generated by mujoco_sim/generate_apriltag_world.py; edit apriltag_world_config.py instead."
        )
    )
    ET.SubElement(root, "include", file=path_for_mjcf(ROBOT_MODEL_PATH, output_path))
    add_visual(root)
    add_assets(root, tags, table_scene, output_path)
    add_worldbody(root, tags, table_scene, cameras)
    ET.indent(root, space="    ")
    return ET.ElementTree(root)


def write_metadata(
    metadata_path: Path,
    scene_path: Path,
    tags: tuple[AprilTagSpec, ...],
    table_scene: TableSceneSpec,
    cameras: tuple[CameraSpec, ...],
) -> None:
    metadata = {
        "scene": str(scene_path.relative_to(MODEL_DIR.parent.parent)),
        "tag_family": TAG_FAMILY,
        "tag_thickness_m": TAG_THICKNESS_M,
        "table": {
            "name": table_scene.name,
            "top_z": table_scene.top_z,
            "floor_z": table_scene.floor_z,
            "top_center_pos": table_scene.top_center_pos,
            "top_half_size": table_scene.top_half_size,
        },
        "tags": [
            {
                "id": tag.tag_id,
                "name": tag.name,
                "body": tag.name,
                "site": f"{tag.name}_site",
                "source_svg": str((APRILTAG_ASSET_DIR / tag.source_svg_filename).relative_to(MODEL_DIR.parent.parent)),
                "asset": str((APRILTAG_ASSET_DIR / tag.asset_filename).relative_to(MODEL_DIR.parent.parent)),
                "size_m": tag.size_m,
                "pos": tag.pos,
                "euler": tag.euler,
            }
            for tag in tags
        ],
        "cameras": [
            {
                "name": camera.name,
                "pos": camera.pos,
                "xyaxes": camera.xyaxes,
                "fovy": camera.fovy,
            }
            for camera in cameras
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def generate_world(scene_path: Path = SCENE_PATH, metadata_path: Path = METADATA_PATH) -> None:
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    APRILTAG_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for tag in APRILTAGS:
        ensure_tag_texture(tag)

    tree = build_scene(output_path=scene_path)
    tree.write(scene_path, encoding="unicode", xml_declaration=False)
    scene_path.write_text(scene_path.read_text() + "\n")
    write_metadata(metadata_path, scene_path, APRILTAGS, TABLE_SCENE, CAMERAS)
    print(f"Wrote MuJoCo scene: {scene_path}")
    print(f"Wrote world metadata: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a MuJoCo world with 36h11 AprilTags.")
    parser.add_argument(
        "--scene",
        type=Path,
        default=SCENE_PATH,
        help="Output MJCF scene path.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=METADATA_PATH,
        help="Output world metadata JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_world(scene_path=args.scene, metadata_path=args.metadata)


if __name__ == "__main__":
    main()
