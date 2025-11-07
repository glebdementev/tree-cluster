import os
from glob import glob

import laspy
import numpy as np

from .treeiso import process_point_cloud


def process_las_file_largest(path_to_las: str, output_path: str | None = None) -> None:
    print('*******Processing LAS/LAZ (largest segment only)******* ' + path_to_las)
    las = laspy.read(path_to_las)

    pcd = np.transpose([las.x, las.y, las.z])
    _, _, final_labels, dec_inverse_idx, _ = process_point_cloud(pcd)

    per_point_final = final_labels[dec_inverse_idx]
    labels, counts = np.unique(per_point_final, return_counts=True)
    largest_label = labels[np.argmax(counts)]
    mask = per_point_final == largest_label

    las.points = las.points[mask]

    if output_path is None:
        output_base = path_to_las[:-4] + "_treeiso_largest"
        available_backends = list(laspy.LazBackend.detect_available())
        if available_backends:
            output_path = output_base + ".laz"
        else:
            output_path = output_base + ".las"
    else:
        available_backends = list(laspy.LazBackend.detect_available())

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if output_path.lower().endswith('.laz') and available_backends:
        las.write(output_path, do_compress=True, laz_backend=available_backends[0])
    else:
        if output_path.lower().endswith('.laz'):
            output_path = output_path[:-4] + ".las"
        las.write(output_path)
    print('*******End processing*******')


def run_treeiso(path_input: str, output_path: str | None = None) -> None:
    if os.path.isfile(path_input) and path_input.lower().endswith(('.las', '.laz')):
        process_las_file_largest(path_input, output_path)
        return
    if os.path.isdir(path_input):
        pathes_to_las = glob(os.path.join(path_input, "*.la[sz]"))
        for path_to_las in pathes_to_las:
            process_las_file_largest(path_to_las)
        if len(pathes_to_las) == 0:
            print('Failed to find the las/laz files from your input directory')
        return
    print(f'PATH_INPUT "{path_input}" is not a valid file or directory')

