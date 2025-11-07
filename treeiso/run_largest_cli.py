"""CLI to run tree isolation and save only the largest final segment."""
import importlib.util
import os

spec = importlib.util.spec_from_file_location("treeiso_module", os.path.join(os.path.dirname(__file__), "treeiso.py"))
treeiso_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(treeiso_module)
run_treeiso = treeiso_module.run_treeiso

PATH = "dataset/sample/117_treeiso/id_666_pine.las"
OUTPUT_PATH = "dataset/treeiso/id_666_pine_treeiso_largest.las"

def main():
    run_treeiso(PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()


