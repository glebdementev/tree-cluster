from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal

import laspy
from pydantic import BaseModel


class PointGeometry(BaseModel):
    type: Literal["Point"]
    coordinates: List[float]


class TreeProperties(BaseModel):
    id: int
    x: float
    y: float
    radius: float
    species: str
    species_eng: str | None = None


class Feature(BaseModel):
    type: Literal["Feature"]
    properties: TreeProperties
    geometry: PointGeometry


class FeatureCollection(BaseModel):
    type: Literal["FeatureCollection"]
    features: List[Feature]


def canonicalize_species(species_eng: str | None, species_local: str | None) -> str:
    source = species_eng if species_eng else (species_local if species_local else "unknown")
    key = source.strip().lower()
    mapping = {
        "birch": "birch",
        "spruce": "spruce",
        "pine": "pine",
        "listvennitsa": "larch",
        "larch": "larch",
        "cedar": "cedar",
        "kedr": "cedar",
        "aspen": "aspen",
        "fir": "fir",
        "lipa": "linden",
        "tilted birch": "birch",
        "linden": "linden",
        "unknown": "unknown",
    }
    return mapping[key] if key in mapping else key


def sanitize_filename_component(text: str) -> str:
    forbidden = "\\/:*?\"<>|"
    sanitized = text
    for ch in forbidden:
        sanitized = sanitized.replace(ch, "_")
    return sanitized.strip().replace(" ", "_")


def extract_cylinders(
    las_path: Path,
    geojson_path: Path,
    output_dir: Path,
) -> int:
    """Extract cylinders from a LAS file based on GeoJSON features.
    
    Args:
        las_path: Path to the input LAS/LAZ file
        geojson_path: Path to the GeoJSON file containing tree features
        output_dir: Directory where extracted cylinder LAS files will be written
        
    Returns:
        Number of cylinders extracted
    """
    with geojson_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    fc = FeatureCollection.parse_obj(raw)

    source_las = laspy.read(str(las_path))
    xs = source_las.x
    ys = source_las.y

    output_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    for feature in fc.features:
        x0 = feature.properties.x
        y0 = feature.properties.y
        r = feature.properties.radius
        radius_sq = r * r

        dx = xs - x0
        dy = ys - y0
        mask = (dx * dx + dy * dy) <= radius_sq

        header = laspy.LasHeader(
            point_format=source_las.header.point_format,
            version=source_las.header.version,
        )
        header.scales = source_las.header.scales
        header.offsets = source_las.header.offsets
        subset = laspy.LasData(header)
        subset.points = source_las.points[mask]

        mapped = canonicalize_species(feature.properties.species_eng, feature.properties.species)
        species_name = sanitize_filename_component(mapped)
        out_path = output_dir / f"id_{feature.properties.id}_{species_name}.las"
        subset.write(str(out_path))
        processed_count += 1

    return processed_count

def main():
    las_path = Path("dataset/sample/117.las")
    geojson_path = Path("dataset/sample/117.geojson")
    output_dir = Path("dataset/sample/117_treeiso")
    extract_cylinders(las_path, geojson_path, output_dir)

if __name__ == "__main__":
    main()