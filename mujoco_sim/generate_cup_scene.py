from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    from .cup_scene_config import (
        CUP_SCENE_CAMERAS,
        CUP_SCENE_METADATA_PATH,
        CUP_SCENE_PATH,
        CUP_SCENE_TAGS,
        CUPS,
        DEFAULT_CUP_HALF_HEIGHT,
        DEFAULT_CUP_MASS,
        DEFAULT_CUP_RADIUS,
        DEFAULT_CUP_RIM_HALF_HEIGHT,
        DEFAULT_CUP_RIM_OVERHANG,
        PLACE_TAG,
        SPOON,
        TABLE_TAGS,
        TABLE_SCENE,
        CupObjectSpec,
        SpoonObjectSpec,
    )
    from .scene_authoring import (
        MODEL_DIR,
        add_apriltag_assets,
        add_apriltag_body,
        add_base_assets,
        add_cameras,
        add_lights,
        add_table_scene,
        add_visual,
        ensure_tag_textures,
        format_floats,
        make_scene_root,
    )
except ImportError:
    from cup_scene_config import (
        CUP_SCENE_CAMERAS,
        CUP_SCENE_METADATA_PATH,
        CUP_SCENE_PATH,
        CUP_SCENE_TAGS,
        CUPS,
        DEFAULT_CUP_HALF_HEIGHT,
        DEFAULT_CUP_MASS,
        DEFAULT_CUP_RADIUS,
        DEFAULT_CUP_RIM_HALF_HEIGHT,
        DEFAULT_CUP_RIM_OVERHANG,
        PLACE_TAG,
        SPOON,
        TABLE_TAGS,
        TABLE_SCENE,
        CupObjectSpec,
        SpoonObjectSpec,
    )
    from scene_authoring import (
        MODEL_DIR,
        add_apriltag_assets,
        add_apriltag_body,
        add_base_assets,
        add_cameras,
        add_lights,
        add_table_scene,
        add_visual,
        ensure_tag_textures,
        format_floats,
        make_scene_root,
    )


def add_cup_material(asset: ET.Element) -> None:
    ET.SubElement(
        asset,
        "material",
        name="cup_material",
        rgba="0.85 0.92 1.0 0.65",
        specular="0.2",
        shininess="0.25",
    )
    ET.SubElement(
        asset,
        "material",
        name="spoon_material",
        rgba="0.86 0.86 0.90 1.0",
        specular="0.35",
        shininess="0.5",
    )


def add_cup_body(worldbody: ET.Element, cup: CupObjectSpec) -> None:
    body = ET.SubElement(worldbody, "body", name=cup.body_name, pos=format_floats(list(cup.initial_position)))
    ET.SubElement(body, "freejoint", name=cup.freejoint_name)
    ET.SubElement(
        body,
        "geom",
        name=cup.side_geom_name,
        type="cylinder",
        size=format_floats([DEFAULT_CUP_RADIUS, DEFAULT_CUP_HALF_HEIGHT]),
        mass=f"{DEFAULT_CUP_MASS:g}",
        condim="6",
        friction="1.0 0.02 0.002",
        solref="0.01 1",
        rgba=format_floats(list(cup.rgba)),
    )
    ET.SubElement(
        body,
        "geom",
        name=cup.rim_geom_name,
        type="cylinder",
        size=format_floats([DEFAULT_CUP_RADIUS + DEFAULT_CUP_RIM_OVERHANG, DEFAULT_CUP_RIM_HALF_HEIGHT]),
        pos=format_floats([0.0, 0.0, DEFAULT_CUP_HALF_HEIGHT - DEFAULT_CUP_RIM_HALF_HEIGHT]),
        density="0",
        condim="6",
        friction="1.0 0.02 0.002",
        solref="0.01 1",
        rgba=format_floats(list(cup.rgba)),
    )
    ET.SubElement(
        body,
        "geom",
        name=cup.visual_geom_name,
        type="cylinder",
        size=format_floats([DEFAULT_CUP_RADIUS + 0.0005, DEFAULT_CUP_HALF_HEIGHT + 0.0005]),
        material="cup_material",
        contype="0",
        conaffinity="0",
        density="0",
    )
    ET.SubElement(body, "site", name=cup.site_name, pos="0 0 0", size="0.006", rgba="0 1 0 0.5")
    add_apriltag_body(body, cup.tag)


def add_spoon_body(worldbody: ET.Element, spoon: SpoonObjectSpec) -> None:
    body = ET.SubElement(worldbody, "body", name=spoon.body_name, pos=format_floats(list(spoon.initial_position)))
    half_handle = 0.5 * spoon.handle_length
    ET.SubElement(
        body,
        "geom",
        name=f"{spoon.body_name}_handle_collision",
        type="capsule",
        fromto=format_floats([-half_handle, 0.0, 0.0, half_handle, 0.0, 0.0]),
        size=f"{spoon.handle_radius:g}",
        mass=f"{0.5 * spoon.mass:g}",
        condim="6",
        friction=format_floats(list(spoon.friction)),
        solref="0.01 1",
        rgba=format_floats(list(spoon.rgba)),
    )
    ET.SubElement(
        body,
        "geom",
        name=f"{spoon.body_name}_bowl_collision",
        type="ellipsoid",
        pos=format_floats([half_handle + spoon.bowl_radii[0] * 0.6, 0.0, 0.0]),
        size=format_floats(list(spoon.bowl_radii)),
        mass=f"{0.5 * spoon.mass:g}",
        condim="6",
        friction=format_floats(list(spoon.friction)),
        solref="0.01 1",
        rgba=format_floats(list(spoon.rgba)),
    )
    ET.SubElement(
        body,
        "geom",
        name=f"{spoon.body_name}_visual",
        type="capsule",
        fromto=format_floats([-half_handle, 0.0, 0.0, half_handle + spoon.bowl_radii[0] * 1.4, 0.0, 0.0]),
        size=f"{max(spoon.handle_radius * 0.9, 0.003):g}",
        material="spoon_material",
        contype="0",
        conaffinity="0",
        density="0",
    )
    ET.SubElement(body, "site", name=spoon.site_name, pos="0 0 0", size="0.004", rgba="1 1 0 0.5")


def build_scene(output_path: Path = CUP_SCENE_PATH) -> ET.ElementTree:
    root = make_scene_root("cup_pickup_scene", output_path, "mujoco_sim/generate_cup_scene.py")
    add_visual(root)
    asset = add_base_assets(root, TABLE_SCENE)
    add_cup_material(asset)
    add_apriltag_assets(asset, CUP_SCENE_TAGS, output_path)

    worldbody = ET.SubElement(root, "worldbody")
    add_lights(worldbody)
    add_table_scene(worldbody, TABLE_SCENE)
    for tag in TABLE_TAGS:
        add_apriltag_body(worldbody, tag)
    for cup in CUPS:
        add_cup_body(worldbody, cup)
    add_spoon_body(worldbody, SPOON)
    add_cameras(worldbody, CUP_SCENE_CAMERAS)

    ET.indent(root, space="    ")
    return ET.ElementTree(root)


def tag_metadata(tag) -> dict:
    metadata = {
        "id": tag.tag_id,
        "name": tag.name,
        "body": tag.name,
        "site": f"{tag.name}_site",
        "asset": str((MODEL_DIR / "assets" / "apriltags" / tag.asset_filename).relative_to(MODEL_DIR.parent.parent)),
        "size_m": tag.size_m,
        "pos": tag.pos,
    }
    if tag.quat is None:
        metadata["euler"] = tag.euler
    else:
        metadata["quat"] = tag.quat
    return metadata


def write_metadata(metadata_path: Path, scene_path: Path) -> None:
    metadata = {
        "scene": str(scene_path.relative_to(MODEL_DIR.parent.parent)),
        "place_tag": tag_metadata(PLACE_TAG),
        "table_tags": [tag_metadata(tag) for tag in TABLE_TAGS],
        "cups": [
            {
                "label": cup.label,
                "body": cup.body_name,
                "freejoint": cup.freejoint_name,
                "side_geom": cup.side_geom_name,
                "rim_geom": cup.rim_geom_name,
                "visual_geom": cup.visual_geom_name,
                "initial_position": cup.initial_position,
                "tag_to_center_offset": cup.tag_to_center_offset,
                "tag": tag_metadata(cup.tag),
            }
            for cup in CUPS
        ],
        "spoon": {
            "label": SPOON.label,
            "body": SPOON.body_name,
            "initial_position": SPOON.initial_position,
            "mass": SPOON.mass,
            "friction": SPOON.friction,
            "handle_length": SPOON.handle_length,
            "handle_radius": SPOON.handle_radius,
            "bowl_radii": SPOON.bowl_radii,
            "site": SPOON.site_name,
        },
        "cameras": [
            {
                "name": camera.name,
                "pos": camera.pos,
                "xyaxes": camera.xyaxes,
                "fovy": camera.fovy,
            }
            for camera in CUP_SCENE_CAMERAS
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def generate_cup_scene(scene_path: Path = CUP_SCENE_PATH, metadata_path: Path = CUP_SCENE_METADATA_PATH) -> None:
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_tag_textures(CUP_SCENE_TAGS)
    tree = build_scene(output_path=scene_path)
    tree.write(scene_path, encoding="unicode", xml_declaration=False)
    scene_path.write_text(scene_path.read_text() + "\n")
    write_metadata(metadata_path, scene_path)
    print(f"Wrote MuJoCo cup scene: {scene_path}")
    print(f"Wrote cup scene metadata: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the MuJoCo cup pickup scene.")
    parser.add_argument("--scene", type=Path, default=CUP_SCENE_PATH, help="Output MJCF scene path.")
    parser.add_argument("--metadata", type=Path, default=CUP_SCENE_METADATA_PATH, help="Output scene metadata JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_cup_scene(scene_path=args.scene, metadata_path=args.metadata)


if __name__ == "__main__":
    main()
