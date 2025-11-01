from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import List, Optional, Literal, Tuple

from pydantic import BaseModel
import matplotlib.pyplot as plt


def canonicalize_species(species_eng: Optional[str], species_local: Optional[str]) -> str:
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


class CrsProperties(BaseModel):
    name: str


class Crs(BaseModel):
    type: Literal["name"]
    properties: CrsProperties


class PointGeometry(BaseModel):
    type: Literal["Point"]
    coordinates: List[float]


class TreeProperties(BaseModel):
    id: int
    cloud_id: int
    chunk_id: int
    is_dead: bool
    height: Optional[float] = None
    x: float
    y: float
    z: float
    radius: Optional[float] = None
    comments: Optional[str] = None
    source: Optional[str] = None
    species_eng: Optional[str] = None
    species: Optional[str] = None


class Feature(BaseModel):
    type: Literal["Feature"]
    properties: TreeProperties
    geometry: PointGeometry


class FeatureCollection(BaseModel):
    type: Literal["FeatureCollection"]
    name: Optional[str] = None
    crs: Optional[Crs] = None
    features: List[Feature]


def compute_basic_stats(values: List[float]) -> Tuple[int, float, float, float, float, float, float]:
    count = len(values)
    sorted_values = sorted(values)
    minimum = sorted_values[0]
    maximum = sorted_values[-1]
    mean = statistics.fmean(sorted_values)
    median = statistics.median(sorted_values)
    # Percentiles using nearest-rank method
    def percentile(p: float) -> float:
        if count == 1:
            return sorted_values[0]
        index = p * (count - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return sorted_values[lower]
        weight = index - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    p25 = percentile(0.25)
    p75 = percentile(0.75)
    # Population standard deviation for stability on small samples
    std = statistics.pstdev(sorted_values) if count > 1 else 0.0
    return count, minimum, p25, median, p75, maximum, std


def compute_histogram(values: List[float], bins: int = 10) -> List[Tuple[float, float, int]]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        return [(vmin, vmax, len(values))]
    width = (vmax - vmin) / bins
    counts = [0] * bins
    for v in values:
        idx = int((v - vmin) / width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    histogram: List[Tuple[float, float, int]] = []
    for i in range(bins):
        start = vmin + i * width
        end = vmin + (i + 1) * width
        histogram.append((start, end, counts[i]))
    return histogram


def pretty_float(value: float) -> str:
    return f"{value:.4f}"


def print_numeric_distribution(title: str, values: List[float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not values:
        print("No data.")
        return
    count, minimum, p25, median, p75, maximum, std = compute_basic_stats(values)
    mean = statistics.fmean(values)
    print(
        "count=", count,
        "min=", pretty_float(minimum),
        "p25=", pretty_float(p25),
        "median=", pretty_float(median),
        "p75=", pretty_float(p75),
        "max=", pretty_float(maximum),
        "mean=", pretty_float(mean),
        "std=", pretty_float(std),
    )
    print("bins:")
    for start, end, c in compute_histogram(values, bins=10):
        print(f"  [{pretty_float(start)}, {pretty_float(end)}) -> {c}")


def save_png(species_counts: Counter[str], radii: List[float], heights: List[float], output_path: Path) -> None:
    fig = plt.figure(figsize=(16, 9), dpi=200)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    ax_species = fig.add_subplot(gs[0, :])
    ax_radius = fig.add_subplot(gs[1, 0])
    ax_height = fig.add_subplot(gs[1, 1])

    # Species counts bar chart
    species_items = species_counts.most_common()
    species_names = [name for name, _ in species_items]
    species_values = [count for _, count in species_items]
    ax_species.bar(species_names, species_values, color="tab:green")
    ax_species.set_title("Species counts")
    ax_species.set_ylabel("Count")
    ax_species.tick_params(axis="x", rotation=45, labelsize=8)

    # Radius histogram
    if radii:
        ax_radius.hist(radii, bins=30, color="tab:blue", edgecolor="white")
    else:
        ax_radius.text(0.5, 0.5, "No data", ha="center", va="center")
    ax_radius.set_title("Radius distribution")
    ax_radius.set_xlabel("Radius")
    ax_radius.set_ylabel("Frequency")

    # Height (z) histogram
    if heights:
        ax_height.hist(heights, bins=30, color="tab:orange", edgecolor="white")
    else:
        ax_height.text(0.5, 0.5, "No data", ha="center", va="center")
    ax_height.set_title("Height (z) distribution")
    ax_height.set_xlabel("Height (z)")
    ax_height.set_ylabel("Frequency")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_species_by_file_stacked_bar(
    species_counts_by_file: List[Tuple[str, Counter[str]]],
    top_n: int,
    output_path: Path,
) -> None:
    if not species_counts_by_file:
        return
    # Determine global top species across all files
    global_counts: Counter[str] = Counter()
    for _, cnt in species_counts_by_file:
        global_counts.update(cnt)

    if top_n and top_n > 0:
        categories = [name for name, _ in global_counts.most_common(top_n)]
    else:
        categories = [name for name, _ in global_counts.most_common()]

    num_files = len(species_counts_by_file)
    fig_width = max(12.0, min(30.0, 0.5 * num_files + 6.0))
    fig = plt.figure(figsize=(fig_width, 8), dpi=200)
    ax = fig.add_subplot(1, 1, 1)

    indices = list(range(num_files))
    bottoms = [0.0] * num_files

    from matplotlib.cm import get_cmap

    cmap = get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(len(categories))]

    totals_per_file = [sum(cnt.values()) for _, cnt in species_counts_by_file]
    for ci, category in enumerate(categories):
        raw_counts_for_category: List[int] = []
        for _, cnt in species_counts_by_file:
            raw_counts_for_category.append(cnt[category] if category in cnt else 0)
        heights_for_category = [
            (value / total * 100.0) if total > 0 else 0.0
            for value, total in zip(raw_counts_for_category, totals_per_file)
        ]
        ax.bar(
            indices,
            heights_for_category,
            bottom=bottoms,
            color=colors[ci],
            edgecolor="white",
            label=category,
        )
        bottoms = [b + h for b, h in zip(bottoms, heights_for_category)]

    ax.set_xticks(indices)
    labels = [
        (label[:-8] if label.lower().endswith('.geojson') else label)
        for label, _ in species_counts_by_file
    ]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Share (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Species distribution per file (100% stacked)")
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1), fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    # Resolve dataset directory relative to repo root
    repo_root = Path(__file__).resolve().parent.parent
    dataset_root = repo_root / "dataset"

    if not dataset_root.exists():
        print(f"Dataset directory not found: {dataset_root}")
        return

    files: List[Path] = []
    for path in dataset_root.rglob("*.geojson"):
        full_lower = str(path).lower()
        # Skip files whose filename or any parent directory contains "partial"
        if "partial" in full_lower:
            continue
        files.append(path)

    species_counts: Counter[str] = Counter()
    per_file_species: List[Tuple[str, Counter[str]]] = []
    radii: List[float] = []
    heights: List[float] = []
    total_features = 0

    for fpath in files:
        with fpath.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        fc = FeatureCollection.parse_obj(raw)
        total_features += len(fc.features)
        file_species_counts: Counter[str] = Counter()
        for feature in fc.features:
            props = feature.properties
            name = canonicalize_species(props.species_eng, props.species)
            species_counts[name] += 1
            file_species_counts[name] += 1
            if props.radius is not None:
                radii.append(props.radius)
            # Heights are z
            heights.append(props.z)
        per_file_species.append((fpath.name, file_species_counts))

    print("Processed files:", len(files))
    print("Total features:", total_features)

    print("\nSpecies counts (descending):")
    for species_name, count in species_counts.most_common():
        print(f"  {species_name}: {count}")

    print_numeric_distribution("Radius (units)", radii)
    print_numeric_distribution("Height (z)", heights)

    # Save combined matplotlib PNG
    output_path = repo_root / "geojson_stats.png"
    save_png(species_counts, radii, heights, output_path)
    print(f"\nSaved figure to: {output_path}")

    # Save per-file stacked bar chart of species distribution
    output_path_per_file = repo_root / "geojson_species_by_file.png"
    save_species_by_file_stacked_bar(per_file_species, top_n=0, output_path=output_path_per_file)
    print(f"Saved figure to: {output_path_per_file}")


if __name__ == "__main__":
    main()


