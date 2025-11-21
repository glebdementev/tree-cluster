from __future__ import annotations

from pathlib import Path
from typing import Optional

# Support running both as a module and as a script
try:
    from correspondence.mapping import FIELD_NAME, _repo_root, process_pair
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from correspondence.mapping import FIELD_NAME, _repo_root, process_pair


def _find_dataset_root() -> Path:
    """
    Locate the dataset root.

    Preference order:
    1. /home/gleb/dev/dataset on Ubuntu
    2. ../dataset (sibling of the repo root), which matches
       C:\\Users\\Gleb\\Work\\OpenForest\\dev\\dataset
    3. ./dataset inside the repo, as a fallback.
    """
    # Explicit Ubuntu path if present
    ubuntu_dataset = Path("/home/gleb/dev/dataset")
    if ubuntu_dataset.is_dir():
        return ubuntu_dataset

    repo_root = _repo_root()
    parent_dataset = repo_root.parent / "dataset"
    if parent_dataset.is_dir():
        return parent_dataset

    in_repo_dataset = repo_root / "dataset"
    if in_repo_dataset.is_dir():
        return in_repo_dataset

    raise FileNotFoundError(
        f"Could not find dataset directory. Checked:\n"
        f"  {parent_dataset}\n"
        f"  {in_repo_dataset}"
    )


def _find_geojson_for_las(las_path: Path) -> Optional[Path]:
    """
    Find a GeoJSON file in the same directory as the LAS file.

    Rules:
    - Only consider .geojson files in the same folder.
    - Prefer one with the same stem as the LAS file, if it exists.
    - Otherwise, fall back to the first .geojson file (sorted by name).
    """
    folder = las_path.parent
    candidates = sorted(folder.glob("*.geojson"))
    if not candidates:
        return None

    # Prefer exact stem match if available.
    las_stem = las_path.stem
    for gj in candidates:
        if gj.stem == las_stem:
            return gj

    # Fall back to the first candidate.
    return candidates[0]


def _iter_iso_laz_files(dataset_root: Path):
    """Yield all *_iso.laz files under dataset_root recursively."""
    yield from dataset_root.rglob("*_iso.laz")


def main() -> None:
    dataset_root = _find_dataset_root()
    print(f"Scanning dataset recursively for *_iso.laz under: {dataset_root}")

    total_files = 0
    processed_files = 0

    for las_path in _iter_iso_laz_files(dataset_root):
        total_files += 1
        geojson_path = _find_geojson_for_las(las_path)
        if geojson_path is None:
            print(f"Skipping {las_path} (no .geojson in same folder)")
            continue

        out_las_path = las_path.with_name(
            f"{las_path.stem}_classified{las_path.suffix}"
        )

        print()
        print(f"Processing LAS: {las_path}")
        print(f"  GeoJSON: {geojson_path}")
        print(f"  Output : {out_las_path}")

        summary = process_pair(
            las_path,
            geojson_path,
            field_name=FIELD_NAME,
            out_las_path=out_las_path,
            write_json_summary=False,  # do not create JSONs for batch processing
        )

        if summary is None:
            print(f"  No candidates for {las_path}, no output written.")
            continue

        processed_files += 1

    print()
    print(
        f"Finished batch mapping. Processed {processed_files} / {total_files} "
        f"*_iso.laz files with neighboring GeoJSON."
    )


if __name__ == "__main__":
    main()


