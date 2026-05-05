from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET

try:
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
    from .spoon_scene_config import (
        SPOON,
        SPOON_HANDLE_TAG,
        SPOON_SCENE_CAMERAS,
        SPOON_SCENE_METADATA_PATH,
        SPOON_SCENE_PATH,
        SPOON_SCENE_TAGS,
        SPOON_TABLE_REF_TAG,
        SPOON_TABLE_TAG_TO_GRASP_OFFSET,
        SPOON_HANDLE_TAG_TO_GRASP_OFFSET,
        TABLE_SCENE,
    )
except ImportError:
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
    from spoon_scene_config import (
        SPOON,
        SPOON_HANDLE_TAG,
        SPOON_SCENE_CAMERAS,
        SPOON_SCENE_METADATA_PATH,
        SPOON_SCENE_PATH,
        SPOON_SCENE_TAGS,
        SPOON_TABLE_REF_TAG,
        SPOON_TABLE_TAG_TO_GRASP_OFFSET,
        SPOON_HANDLE_TAG_TO_GRASP_OFFSET,
        TABLE_SCENE,
    )


def add_spoon_material(asset: ET.Element) -> None:
    ET.SubElement(
        asset,
        "material",
        name="spoon_material",
        rgba="0.86 0.86 0.90 1.0",
        specular="0.35",
        shininess="0.5",
    )


def add_spoon_body(worldbody: ET.Element) -> None:
    body = ET.SubElement(worldbody, "body", name=SPOON.body_name, pos=format_floats(list(SPOON.initial_position)))
    ET.SubElement(body, "freejoint", name=SPOON.freejoint_name)

    ET.SubElement(
        body,
        "geom",
        name="spoon_handle_collision",
        type="capsule",
        fromto=format_floats(
            [
                -SPOON.handle_half_length,
                0.0,
                0.0,
                SPOON.handle_half_length,
                0.0,
                0.0,
            ]
        ),
        size=f"{SPOON.handle_radius:g}",
        mass=f"{0.5 * SPOON.mass:g}",
        condim="6",
        friction=format_floats(list(SPOON.friction)),
        solref="0.01 1",
        rgba="0.86 0.86 0.90 1",
    )
    ET.SubElement(
        body,
        "geom",
        name="spoon_bowl_collision",
        type="ellipsoid",
        pos=format_floats([SPOON.handle_half_length + SPOON.bowl_radii[0] * 0.6, 0.0, 0.0]),
        size=format_floats(list(SPOON.bowl_radii)),
        mass=f"{0.5 * SPOON.mass:g}",
        condim="6",
        friction=format_floats(list(SPOON.friction)),
        solref="0.01 1",
        rgba="0.86 0.86 0.90 1",
    )
    ET.SubElement(
        body,
        "geom",
        name="spoon_visual",
        type="capsule",
        fromto=format_floats(
            [
                -SPOON.handle_half_length,
                0.0,
                0.0,
                SPOON.handle_half_length + SPOON.bowl_radii[0] * 1.4,
                0.0,
                0.0,
            ]
        ),
        size=f"{max(SPOON.handle_radius * 0.9, 0.003):g}",
        material="spoon_material",
        contype="0",
        conaffinity="0",
        density="0",
    )
    ET.SubElement(
        body,
        "site",
        name="spoon_grasp_site",
        pos=format_floats(list(SPOON.grasp_site_pos)),
        size="0.004",
        rgba="1 1 0 0.6",
    )

    add_apriltag_body(body, SPOON_HANDLE_TAG)


def build_scene(output_path: Path = SPOON_SCENE_PATH) -> ET.ElementTree:
    root = make_scene_root("spoon_pickup_scene", output_path, "mujoco_sim/generate_spoon_scene.py")
    add_visual(root)
    asset = add_base_assets(root, TABLE_SCENE)
    add_spoon_material(asset)
    add_apriltag_assets(asset, SPOON_SCENE_TAGS, output_path)

    worldbody = ET.SubElement(root, "worldbody")
    add_lights(worldbody)
    add_table_scene(worldbody, TABLE_SCENE)
    add_apriltag_body(worldbody, SPOON_TABLE_REF_TAG)
    add_spoon_body(worldbody)
    add_cameras(worldbody, SPOON_SCENE_CAMERAS)

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
        "spoon": {
            "body": SPOON.body_name,
            "freejoint": SPOON.freejoint_name,
            "initial_position": SPOON.initial_position,
            "mass": SPOON.mass,
            "friction": SPOON.friction,
            "handle_half_length": SPOON.handle_half_length,
            "handle_radius": SPOON.handle_radius,
            "bowl_radii": SPOON.bowl_radii,
            "grasp_site": "spoon_grasp_site",
            "grasp_site_pos": SPOON.grasp_site_pos,
        },
        "spoon_handle_tag": tag_metadata(SPOON_HANDLE_TAG),
        "table_reference_tag": tag_metadata(SPOON_TABLE_REF_TAG),
        "spoon_handle_tag_to_grasp_offset": SPOON_HANDLE_TAG_TO_GRASP_OFFSET,
        "table_tag_to_grasp_offset": SPOON_TABLE_TAG_TO_GRASP_OFFSET,
        "cameras": [
            {
                "name": camera.name,
                "pos": camera.pos,
                "xyaxes": camera.xyaxes,
                "fovy": camera.fovy,
            }
            for camera in SPOON_SCENE_CAMERAS
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def generate_spoon_scene(scene_path: Path = SPOON_SCENE_PATH, metadata_path: Path = SPOON_SCENE_METADATA_PATH) -> None:
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_tag_textures(SPOON_SCENE_TAGS)
    tree = build_scene(output_path=scene_path)
    tree.write(scene_path, encoding="unicode", xml_declaration=False)
    scene_path.write_text(scene_path.read_text() + "\n")
    write_metadata(metadata_path, scene_path)
    print(f"Wrote MuJoCo spoon scene: {scene_path}")
    print(f"Wrote spoon scene metadata: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the MuJoCo spoon pickup scene.")
    parser.add_argument("--scene", type=Path, default=SPOON_SCENE_PATH, help="Output MJCF scene path.")
    parser.add_argument("--metadata", type=Path, default=SPOON_SCENE_METADATA_PATH, help="Output metadata JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_spoon_scene(scene_path=args.scene, metadata_path=args.metadata)


if __name__ == "__main__":
    main()
