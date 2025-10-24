from __future__ import annotations
from pathlib import Path
from typing import List, Sequence
from enum import Enum
from collections import Counter
from pydantic import BaseModel
import laspy
from statistics import median, quantiles
from matplotlib import pyplot as plt


# Set your dataset root here. You can change this as needed.
DATASET_ROOT = Path("/home/gleb/dev/tree-cluster/dataset")
assert DATASET_ROOT.exists() and DATASET_ROOT.is_dir(), "Dataset root must exist and be a directory"

# Only consider .las files
LAS_SUFFIX = ".las"
MIN_CLASS_COUNT = 5
EXCLUDE_UNKNOWN = True


class Species(str, Enum):
    birch = "birch"
    cedar = "cedar"
    fir = "fir"
    larch = "larch"
    pine = "pine"
    spruce = "spruce"
    unknown = "unknown"


class LasFileRecord(BaseModel):
    path: str
    species: Species
    dataset_folder: str  # the immediate higher-level folder above the species folder
    point_count: int
    max_z: float

    @property
    def path_obj(self) -> Path:
        return Path(self.path)


SPECIES_NAMES = {s.value for s in Species}


def infer_species(path: Path) -> Species:
    parent = path.parent.name.lower()
    if parent in SPECIES_NAMES:
        return Species(parent)  # type: ignore[arg-type]
    return Species.unknown


def infer_dataset_folder(path: Path) -> str:
    species_folder = path.parent
    higher = species_folder.parent
    return higher.name


def read_point_count(file_path: Path) -> int:
    with laspy.open(file_path) as las_reader:
        return int(las_reader.header.point_count)


def read_max_z(file_path: Path) -> float:
    with laspy.open(file_path) as las_reader:
        return float(las_reader.header.maxs[2])


class FiveNumberSummary(BaseModel):
    count: int
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float


def compute_five_number_summary(values: Sequence[float]) -> FiveNumberSummary:
    values_list = list(values)
    values_sorted = sorted(values_list)
    n = len(values_sorted)
    assert n > 0, "values must be non-empty"
    if n == 1:
        v = float(values_sorted[0])
        return FiveNumberSummary(
            count=1,
            minimum=v,
            q1=v,
            median=v,
            q3=v,
            maximum=v,
        )
    q = quantiles(values_sorted, n=4, method="inclusive")
    return FiveNumberSummary(
        count=n,
        minimum=float(values_sorted[0]),
        q1=float(q[0]),
        median=float(median(values_sorted)),
        q3=float(q[2]),
        maximum=float(values_sorted[-1]),
    )


def plot_species_boxplots(records: List[LasFileRecord], output_path: Path) -> None:
    species_present = sorted({r.species for r in records}, key=lambda s: s.value)
    labels = [sp.value for sp in species_present]
    data_counts = [[float(r.point_count) for r in records if r.species == sp] for sp in species_present]
    data_zmax = [[r.max_z for r in records if r.species == sp] for sp in species_present]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].boxplot(data_counts, labels=labels, showfliers=False)
    axes[0].set_title("Point count by species")
    axes[0].set_ylabel("Points")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].boxplot(data_zmax, labels=labels, showfliers=False)
    axes[1].set_title("Max Z by species")
    axes[1].set_ylabel("Max Z (units)")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle("Species-wise distributions")
    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)


def scan_las_files(root: Path) -> List[LasFileRecord]:
    results: List[LasFileRecord] = []
    for p in root.rglob(f"*{LAS_SUFFIX}"):
        if not p.is_file():
            continue
        species = infer_species(p)
        dataset_folder = infer_dataset_folder(p)
        count = read_point_count(p)
        zmax = read_max_z(p)
        results.append(
            LasFileRecord(
                path=str(p),
                species=species,
                dataset_folder=dataset_folder,
                point_count=count,
                max_z=zmax,
            )
        )
    return results


def filter_records_by_class(records: List[LasFileRecord], min_per_class: int, exclude_unknown: bool) -> List[LasFileRecord]:
    counts = Counter(r.species for r in records)
    allowed_species = {
        sp
        for sp, cnt in counts.items()
        if cnt >= min_per_class and (sp != Species.unknown if exclude_unknown else True)
    }
    return [r for r in records if r.species in allowed_species]


def main() -> None:
    records_all = scan_las_files(DATASET_ROOT)
    records = filter_records_by_class(records_all, MIN_CLASS_COUNT, EXCLUDE_UNKNOWN)

    # Summary by species (file counts)
    species_counts = Counter(r.species for r in records)
    print("Counts by species:")
    for sp in sorted([s for s in Species], key=lambda s: s.value):
        count = species_counts[sp] if sp in species_counts else 0
        print(f"  {sp.value}: {count}")

    # Summary by dataset folder (file counts)
    folder_counts = Counter(r.dataset_folder for r in records)
    print("\nCounts by dataset folder:")
    for folder, count in sorted(folder_counts.items()):
        print(f"  {folder}: {count}")

    # Cross-tab: folder x species (file counts)
    print("\nFolder x species:")
    folders_sorted = sorted(folder_counts.keys())
    all_species_sorted = sorted([s for s in Species], key=lambda s: s.value)
    for folder in folders_sorted:
        subset = [r for r in records if r.dataset_folder == folder]
        c = Counter(r.species for r in subset)
        row = ", ".join(
            f"{sp.value}={(c[sp] if sp in c else 0)}" for sp in all_species_sorted
        )
        print(f"  {folder}: {row}")

    # Per-file point counts
    print("\nPer-file point counts:")
    for r in records:
        print(f"  {r.path}: {r.point_count} points")

    # Species-wise five-number summaries
    print("\nSpecies-wise five-number summaries (point_count):")
    species_present = sorted({r.species for r in records}, key=lambda s: s.value)
    for sp in species_present:
        vals = [float(r.point_count) for r in records if r.species == sp]
        if len(vals) == 0:
            continue
        s = compute_five_number_summary(vals)
        print(
            f"  {sp.value}: min={s.minimum:.0f}, q1={s.q1:.0f}, median={s.median:.0f}, q3={s.q3:.0f}, max={s.maximum:.0f} (n={s.count})"
        )

    print("\nSpecies-wise five-number summaries (max_z):")
    for sp in species_present:
        vals = [r.max_z for r in records if r.species == sp]
        if len(vals) == 0:
            continue
        s = compute_five_number_summary(vals)
        print(
            f"  {sp.value}: min={s.minimum:.3f}, q1={s.q1:.3f}, median={s.median:.3f}, q3={s.q3:.3f}, max={s.maximum:.3f} (n={s.count})"
        )

    # Plot boxplots to file
    output_image = Path(__file__).with_name("species_boxplots.png")
    plot_species_boxplots(records, output_image)
    print(f"\nSaved boxplots to {output_image}")


if __name__ == "__main__":
    main()


