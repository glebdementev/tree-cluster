import argparse
from dataclasses import dataclass
from typing import List, Optional

import laspy
import numpy as np
import pyvista as pv


@dataclass
class ViewerState:
    las: laspy.LasData
    field: str
    unique_values: np.ndarray
    current_index: int
    max_points: int
    use_rgb: bool
    point_size: float


def list_fields(las: laspy.LasData) -> List[str]:
    """Return list of candidate fields (excluding coordinates, rgb)."""
    dims = list(las.point_format.dimension_names)
    for name in ("X", "Y", "Z", "red", "green", "blue"):
        if name in dims:
            dims.remove(name)
    return sorted(dims)


def build_polydata_for_value(state: ViewerState, value) -> pv.PolyData:
    """Create a PyVista PolyData for the given field value."""
    las = state.las
    field_data = np.asarray(las[state.field])

    mask = field_data == value
    indices = np.nonzero(mask)[0]
    total_points = int(indices.size)

    if total_points == 0:
        return pv.PolyData()

    if total_points > state.max_points:
        chosen = np.random.choice(indices, state.max_points, replace=False)
        indices = np.sort(chosen)

    x = np.asarray(las.x, dtype=np.float64)[indices]
    y = np.asarray(las.y, dtype=np.float64)[indices]
    z = np.asarray(las.z, dtype=np.float64)[indices]

    xyz = np.stack([x, y, z], axis=1)
    poly = pv.PolyData(xyz)

    dims = set(las.point_format.dimension_names)
    if state.use_rgb and {"red", "green", "blue"}.issubset(dims):
        r = np.asarray(las.red, dtype=np.float32)[indices]
        g = np.asarray(las.green, dtype=np.float32)[indices]
        b = np.asarray(las.blue, dtype=np.float32)[indices]

        max_rgb = max(float(r.max()), float(g.max()), float(b.max()), 1.0)
        if max_rgb <= 255.0:
            denom = 255.0
        else:
            denom = 65535.0

        colors = np.stack([r, g, b], axis=1) / denom
        poly["colors"] = colors.astype(np.float32)
    else:
        # Fallback: grayscale by height
        z_min = float(z.min())
        z_max = float(z.max())
        if z_max > z_min:
            zn = (z - z_min) / (z_max - z_min)
        else:
            zn = np.zeros_like(z)
        colors = np.stack([zn, zn, zn], axis=1)
        poly["colors"] = colors.astype(np.float32)

    return poly


def run_viewer(state: ViewerState) -> None:
    """Run PyVista viewer with keyboard controls."""
    pv.set_plot_theme("dark")
    plotter = pv.Plotter(window_size=(1280, 720), title="LAS/LAZ Viewer (PyVista)")

    # Rotate around the points, not translate the camera
    plotter.enable_trackball_style()

    initial_value = state.unique_values[state.current_index]
    mesh = build_polydata_for_value(state, initial_value)
    actor = plotter.add_mesh(
        mesh,
        scalars="colors",
        rgb=True,
        render_points_as_spheres=False,
        point_size=state.point_size,
    )

    plotter.show_axes()

    # Ensure Z is world-up and look at the centroid
    center = np.array(mesh.center)
    plotter.camera.up = (0.0, 0.0, 1.0)
    # Use keyword for 'bounds' to avoid PyVista deprecation warning about positional args.
    plotter.reset_camera(bounds=mesh.bounds, render=True)
    # Re-center camera on centroid to avoid initial panning offset
    cam = plotter.camera
    direction = np.array(cam.position) - np.array(cam.focal_point)
    cam.focal_point = center
    cam.position = tuple(center + direction)

    print("\nControls:")
    print("  [Left / A]  previous field value")
    print("  [Right / D] next field value")
    print("  [-]         decrease point size")
    print("  [+]         increase point size")
    print("  Mouse:      rotate, pan, scroll to zoom (PyVista/VTK defaults)")
    print("  Q / Esc:    close viewer\n")

    def update_for_index(idx: int) -> None:
        nonlocal actor
        state.current_index = idx
        value = state.unique_values[idx]
        new_mesh = build_polydata_for_value(state, value)

        plotter.remove_actor(actor)
        actor = plotter.add_mesh(
            new_mesh,
            scalars="colors",
            rgb=True,
            render_points_as_spheres=False,
            point_size=state.point_size,
        )

        # Keep rotating around the new centroid and eliminate any accumulated pan:
        # move camera so the focal point is the centroid, preserving the view direction.
        center_new = np.array(new_mesh.center)
        cam = plotter.camera
        direction = np.array(cam.position) - np.array(cam.focal_point)
        cam.focal_point = center_new
        cam.position = tuple(center_new + direction)
        cam.up = (0.0, 0.0, 1.0)

        print(
            f"{state.field} = {value} "
            f"[{state.current_index + 1}/{state.unique_values.size}] "
            f"points: {new_mesh.n_points}"
        )

    def on_prev() -> None:
        if state.current_index > 0:
            update_for_index(state.current_index - 1)

    def on_next() -> None:
        if state.current_index < state.unique_values.size - 1:
            update_for_index(state.current_index + 1)

    def on_decrease_size() -> None:
        prop = actor.GetProperty()
        size = max(1.0, prop.GetPointSize() - 1.0)
        prop.SetPointSize(size)
        state.point_size = size
        plotter.render()

    def on_increase_size() -> None:
        prop = actor.GetProperty()
        size = min(100.0, prop.GetPointSize() + 1.0)
        prop.SetPointSize(size)
        state.point_size = size
        plotter.render()

    # Key bindings
    plotter.add_key_event("Left", on_prev)
    plotter.add_key_event("Right", on_next)
    plotter.add_key_event("a", on_prev)
    plotter.add_key_event("d", on_next)
    plotter.add_key_event("-", on_decrease_size)
    plotter.add_key_event("equal", on_increase_size)  # '=' key (Shift+'+' for many layouts)
    plotter.add_key_event("plus", on_increase_size)

    plotter.show(auto_close=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PyVista-based LAS/LAZ point cloud viewer with field-based filtering, "
            "RGB coloring, and keyboard navigation."
        )
    )
    parser.add_argument(
        "file",
        help="Path to a .las or .laz file.",
    )
    parser.add_argument(
        "--field",
        help="Name of point attribute field to use for filtering. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=500_000,
        help="Maximum number of points to show per view (for performance).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=5.0,
        help="Initial point size in pixels (adjustable in viewer with +/- keys).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    las = laspy.read(args.file)

    fields = list_fields(las)
    if not fields:
        raise SystemExit("No non-coordinate fields found in LAS/LAZ file.")

    field: Optional[str] = args.field
    if field is None:
        print("Available fields:")
        for i, name in enumerate(fields):
            print(f"  [{i}] {name}")
        while True:
            choice = input("Select field index: ").strip()
            try:
                idx = int(choice)
            except ValueError:
                print("Please enter a valid integer index.")
                continue
            if 0 <= idx < len(fields):
                field = fields[idx]
                break
            print("Index out of range, try again.")

    if field not in las.point_format.dimension_names:
        raise SystemExit(f"Field '{field}' not found in LAS/LAZ file.")

    data = las[field]
    unique_values = np.unique(data)
    if unique_values.size == 0:
        raise SystemExit(f"Field '{field}' has no values.")

    dims = set(las.point_format.dimension_names)
    use_rgb = {"red", "green", "blue"}.issubset(dims)

    state = ViewerState(
        las=las,
        field=field,
        unique_values=unique_values,
        current_index=0,
        max_points=args.max_points,
        use_rgb=use_rgb,
        point_size=args.point_size,
    )

    print(f"Loaded file: {args.file}")
    print(f"Using field: {field}, unique values: {unique_values.size}")
    run_viewer(state)


if __name__ == "__main__":
    main()

