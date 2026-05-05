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
        DEFAULT_CUP_FRICTION,
        DEFAULT_CUP_HALF_HEIGHT,
        DEFAULT_CUP_MASS,
        DEFAULT_CUP_RADIUS,
        NVIDIA_GLASS_CUP_ASSET,
        PLACE_TAG,
        TABLE_TAGS,
        TABLE_SCENE,
        CupObjectSpec,
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
        path_for_mjcf,
    )
except ImportError:
    from cup_scene_config import (
        CUP_SCENE_CAMERAS,
        CUP_SCENE_METADATA_PATH,
        CUP_SCENE_PATH,
        CUP_SCENE_TAGS,
        CUPS,
        DEFAULT_CUP_FRICTION,
        DEFAULT_CUP_HALF_HEIGHT,
        DEFAULT_CUP_MASS,
        DEFAULT_CUP_RADIUS,
        NVIDIA_GLASS_CUP_ASSET,
        PLACE_TAG,
        TABLE_TAGS,
        TABLE_SCENE,
        CupObjectSpec,
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
        path_for_mjcf,
    )


def _collision_mesh_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def mesh_asset_path(path: Path) -> str:
    return path.relative_to(MODEL_DIR / "assets").as_posix()


def cup_collision_mesh_paths() -> tuple[Path, ...]:
    paths = tuple(sorted(NVIDIA_GLASS_CUP_ASSET.collision_dir.glob("*.obj"), key=_collision_mesh_index))
    if not paths:
        raise FileNotFoundError(f"Missing cup collision meshes: {NVIDIA_GLASS_CUP_ASSET.collision_dir}")
    return paths


def validate_cup_asset() -> None:
    required_paths = (
        NVIDIA_GLASS_CUP_ASSET.visual_mesh_path,
        NVIDIA_GLASS_CUP_ASSET.texture_path,
        NVIDIA_GLASS_CUP_ASSET.root / "model.xml",
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing NVIDIA glass cup asset: {path}")
    cup_collision_mesh_paths()


def add_cup_assets(asset: ET.Element, output_path: Path) -> None:
    validate_cup_asset()
    ET.SubElement(
        asset,
        "mesh",
        name=NVIDIA_GLASS_CUP_ASSET.visual_mesh_name,
        file=mesh_asset_path(NVIDIA_GLASS_CUP_ASSET.visual_mesh_path),
    )
    ET.SubElement(
        asset,
        "texture",
        type="2d",
        name=NVIDIA_GLASS_CUP_ASSET.texture_name,
        file=path_for_mjcf(NVIDIA_GLASS_CUP_ASSET.texture_path, output_path),
    )
    ET.SubElement(
        asset,
        "material",
        name=NVIDIA_GLASS_CUP_ASSET.material_name,
        texture=NVIDIA_GLASS_CUP_ASSET.texture_name,
        rgba="1 1 1 0.2",
        specular="0.310344850266",
        shininess="0.447213590145",
    )
    for index, collision_mesh_path in enumerate(cup_collision_mesh_paths()):
        ET.SubElement(
            asset,
            "mesh",
            name=NVIDIA_GLASS_CUP_ASSET.collision_mesh_name(index),
            file=mesh_asset_path(collision_mesh_path),
        )


def add_cup_body(worldbody: ET.Element, cup: CupObjectSpec) -> None:
    body = ET.SubElement(worldbody, "body", name=cup.body_name, pos=format_floats(list(cup.initial_position)))
    ET.SubElement(body, "freejoint", name=cup.freejoint_name)
    ET.SubElement(
        body,
        "geom",
        name=cup.visual_geom_name,
        type="mesh",
        mesh=NVIDIA_GLASS_CUP_ASSET.visual_mesh_name,
        material=NVIDIA_GLASS_CUP_ASSET.material_name,
        contype="0",
        conaffinity="0",
        density="0",
        group="1",
    )
    collision_mesh_paths = cup_collision_mesh_paths()
    collision_mass = DEFAULT_CUP_MASS / len(collision_mesh_paths)
    for index, _ in enumerate(collision_mesh_paths):
        ET.SubElement(
            body,
            "geom",
            name=cup.collision_geom_name(index),
            type="mesh",
            mesh=NVIDIA_GLASS_CUP_ASSET.collision_mesh_name(index),
            mass=f"{collision_mass:.12g}",
            condim="6",
            friction=format_floats(list(DEFAULT_CUP_FRICTION)),
            solref="0.01 1",
            rgba=format_floats(list(cup.rgba)),
            group="0",
        )
    ET.SubElement(body, "site", name=cup.site_name, pos="0 0 0", size="0.006", rgba="0 1 0 0.5")
    add_apriltag_body(body, cup.tag)


def build_scene(output_path: Path = CUP_SCENE_PATH) -> ET.ElementTree:
    root = make_scene_root("cup_pickup_scene", output_path, "mujoco_sim/generate_cup_scene.py")
    add_visual(root)
    asset = add_base_assets(root, TABLE_SCENE)
    add_cup_assets(asset, output_path)
    add_apriltag_assets(asset, CUP_SCENE_TAGS, output_path)

    worldbody = ET.SubElement(root, "worldbody")
    add_lights(worldbody)
    add_table_scene(worldbody, TABLE_SCENE)
    for tag in TABLE_TAGS:
        add_apriltag_body(worldbody, tag)
    for cup in CUPS:
        add_cup_body(worldbody, cup)
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
    collision_mesh_paths = cup_collision_mesh_paths()
    metadata = {
        "scene": str(scene_path.relative_to(MODEL_DIR.parent.parent)),
        "cup_asset": {
            "model_name": NVIDIA_GLASS_CUP_ASSET.model_name,
            "root": str(NVIDIA_GLASS_CUP_ASSET.root.relative_to(MODEL_DIR.parent.parent)),
            "visual_mesh": str(NVIDIA_GLASS_CUP_ASSET.visual_mesh_path.relative_to(MODEL_DIR.parent.parent)),
            "texture": str(NVIDIA_GLASS_CUP_ASSET.texture_path.relative_to(MODEL_DIR.parent.parent)),
            "collision_meshes": [
                str(path.relative_to(MODEL_DIR.parent.parent))
                for path in collision_mesh_paths
            ],
            "radius_m": DEFAULT_CUP_RADIUS,
            "half_height_m": DEFAULT_CUP_HALF_HEIGHT,
            "bounds_m": {
                "min": NVIDIA_GLASS_CUP_ASSET.min_xyz,
                "max": NVIDIA_GLASS_CUP_ASSET.max_xyz,
            },
        },
        "place_tag": tag_metadata(PLACE_TAG),
        "table_tags": [tag_metadata(tag) for tag in TABLE_TAGS],
        "cups": [
            {
                "label": cup.label,
                "body": cup.body_name,
                "freejoint": cup.freejoint_name,
                "collision_geoms": [
                    cup.collision_geom_name(index)
                    for index, _ in enumerate(collision_mesh_paths)
                ],
                "visual_geom": cup.visual_geom_name,
                "initial_position": cup.initial_position,
                "tag_to_center_offset": cup.tag_to_center_offset,
                "tag": tag_metadata(cup.tag),
            }
            for cup in CUPS
        ],
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
