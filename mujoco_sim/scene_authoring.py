from __future__ import annotations

import os
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "simulation_code" / "model"
APRILTAG_ASSET_DIR = MODEL_DIR / "assets" / "apriltags"
ROBOT_MODEL_PATH = MODEL_DIR / "so101.xml"

TAG_FAMILY = "tag36h11"
TAG_THICKNESS_M = 0.002
TAG_BLACK_SQUARE_FRACTION = 0.80
TAG_TEXTURE_SIZE_PX = 640
TAG_CARD_MESH_FILE = "apriltags/tag_card.obj"

DEFAULT_RENDER_WIDTH = 2560
DEFAULT_RENDER_HEIGHT = 1920


@dataclass(frozen=True)
class AprilTagSpec:
    tag_id: int
    size_m: float
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] | None = None
    name_prefix: str = ""

    @property
    def name(self) -> str:
        return f"{self.name_prefix}{TAG_FAMILY}_{self.tag_id:05d}"

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


def format_floats(values: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


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


def ensure_tag_textures(tags: tuple[AprilTagSpec, ...]) -> None:
    APRILTAG_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        ensure_tag_texture(tag)


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


def add_base_assets(root: ET.Element, table_scene: TableSceneSpec) -> ET.Element:
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
        rgba=format_floats(list(table_scene.rgba)),
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
    return asset


def add_apriltag_assets(asset: ET.Element, tags: tuple[AprilTagSpec, ...], output_path: Path) -> None:
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
            scale=format_floats([card_size_m, card_size_m, TAG_THICKNESS_M]),
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


def add_lights(worldbody: ET.Element) -> None:
    ET.SubElement(worldbody, "light", pos="0 0 3.5", dir="0 0 -1", directional="true")
    ET.SubElement(
        worldbody,
        "light",
        pos="0.15 -0.45 0.75",
        dir="0.1 0.35 -1",
        directional="true",
        diffuse="0.45 0.4 0.35",
    )


def add_apriltag_body(parent: ET.Element, tag: AprilTagSpec, pos: tuple[float, float, float] | None = None) -> ET.Element:
    attributes = {
        "name": tag.name,
        "pos": format_floats(list(tag.pos if pos is None else pos)),
    }
    if tag.quat is None:
        attributes["euler"] = format_floats(list(tag.euler))
    else:
        attributes["quat"] = format_floats(list(tag.quat))

    body = ET.SubElement(parent, "body", **attributes)
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
    return body


def add_table_scene(worldbody: ET.Element, table_scene: TableSceneSpec) -> None:
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        size="0 0 0.05",
        pos=format_floats([0.0, 0.0, table_scene.floor_z]),
        type="plane",
        material="groundplane",
    )
    table_body = ET.SubElement(worldbody, "body", name=table_scene.name)
    ET.SubElement(
        table_body,
        "geom",
        name=f"{table_scene.name}_top",
        type="box",
        pos=format_floats(list(table_scene.top_center_pos)),
        size=format_floats(list(table_scene.top_half_size)),
        material=f"{table_scene.name}_top_material",
        friction="1.3 0.01 0.001",
    )
    for index, leg_pos in enumerate(table_scene.leg_positions):
        ET.SubElement(
            table_body,
            "geom",
            name=f"{table_scene.name}_leg_{index}",
            type="box",
            pos=format_floats(list(leg_pos)),
            size=format_floats(list(table_scene.leg_half_size)),
            material=f"{table_scene.name}_leg_material",
            friction="1.0 0.01 0.001",
        )


def add_camera(worldbody: ET.Element, camera: CameraSpec) -> None:
    ET.SubElement(
        worldbody,
        "camera",
        name=camera.name,
        pos=format_floats(list(camera.pos)),
        xyaxes=format_floats(list(camera.xyaxes)),
        fovy=f"{camera.fovy:g}",
    )


def add_cameras(worldbody: ET.Element, cameras: tuple[CameraSpec, ...]) -> None:
    for camera in cameras:
        add_camera(worldbody, camera)


def make_scene_root(model_name: str, output_path: Path, generated_by: str) -> ET.Element:
    root = ET.Element("mujoco", model=model_name)
    root.append(ET.Comment(f"Generated by {generated_by}; edit the matching *_config.py file instead."))
    ET.SubElement(root, "include", file=path_for_mjcf(ROBOT_MODEL_PATH, output_path))
    return root
