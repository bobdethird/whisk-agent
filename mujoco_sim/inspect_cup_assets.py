from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "simulation_code" / "model"
DEFAULT_ASSET_ROOT = MODEL_DIR / "assets" / "objects" / "_downloads" / "glass_cup_nvidia" / "glass_cup"


@dataclass(frozen=True)
class ObjBounds:
    path: Path
    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]

    @property
    def size_xyz(self) -> tuple[float, float, float]:
        return tuple(max_value - min_value for min_value, max_value in zip(self.min_xyz, self.max_xyz))

    @property
    def max_xy_diameter(self) -> float:
        size = self.size_xyz
        return max(size[0], size[1])

    @property
    def height(self) -> float:
        return self.size_xyz[2]


@dataclass(frozen=True)
class CupCandidate:
    name: str
    root: Path
    visual_bounds: ObjBounds
    collision_mesh_count: int

    @property
    def diameter(self) -> float:
        return self.visual_bounds.max_xy_diameter

    @property
    def height(self) -> float:
        return self.visual_bounds.height


def read_obj_bounds(path: Path) -> ObjBounds:
    vertices: list[tuple[float, float, float]] = []
    for line in path.read_text().splitlines():
        if not line.startswith("v "):
            continue
        _, x, y, z, *_ = line.split()
        vertices.append((float(x), float(y), float(z)))
    if not vertices:
        raise ValueError(f"No vertices found in {path}")
    min_xyz = tuple(min(vertex[index] for vertex in vertices) for index in range(3))
    max_xyz = tuple(max(vertex[index] for vertex in vertices) for index in range(3))
    return ObjBounds(path=path, min_xyz=min_xyz, max_xyz=max_xyz)


def cup_candidates(asset_root: Path) -> list[CupCandidate]:
    candidates: list[CupCandidate] = []
    for model_xml in sorted(asset_root.glob("GlassCup*/model.xml")):
        root = model_xml.parent
        visual_mesh = root / "visual" / "Clear.obj"
        collision_meshes = sorted((root / "collision").glob("*.obj"))
        if not visual_mesh.exists() or not collision_meshes:
            continue
        candidates.append(
            CupCandidate(
                name=root.name,
                root=root,
                visual_bounds=read_obj_bounds(visual_mesh),
                collision_mesh_count=len(collision_meshes),
            )
        )
    return sorted(candidates, key=lambda candidate: (candidate.diameter, candidate.height))


def print_report(candidates: list[CupCandidate]) -> None:
    print("name,diameter_m,height_m,collision_meshes,root")
    for candidate in candidates:
        print(
            f"{candidate.name},"
            f"{candidate.diameter:.4f},"
            f"{candidate.height:.4f},"
            f"{candidate.collision_mesh_count},"
            f"{candidate.root}"
        )
    if candidates:
        selected = candidates[0]
        print(
            "\nsmallest_candidate="
            f"{selected.name} diameter={selected.diameter:.4f}m "
            f"height={selected.height:.4f}m "
            f"collision_meshes={selected.collision_mesh_count}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure local glass cup OBJ candidates.")
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = cup_candidates(args.asset_root)
    if not candidates:
        raise SystemExit(f"No cup candidates found under {args.asset_root}")
    print_report(candidates)


if __name__ == "__main__":
    main()