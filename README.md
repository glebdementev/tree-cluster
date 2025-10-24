Basically doing a bunch of clustering

## Setup

Create virtual environment and install dependencies:

```bash
python3 -m venv /home/gleb/dev/tree-cluster/.venv && /home/gleb/dev/tree-cluster/.venv/bin/python -m pip install --upgrade pip && /home/gleb/dev/tree-cluster/.venv/bin/python -m pip install ipykernel pydantic && /home/gleb/dev/tree-cluster/.venv/bin/python -m ipykernel install --user --name python312-tree-cluster --display-name "Python 3.12 (tree-cluster)"
```

## Activate Virtual Environment

```bash
source .venv/bin/activate
```

To deactivate when you're done:

```bash
deactivate
```

torch
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Dataset Summary

```bash
python dataset_summary.py
```

## Train PointNetTiny Classifier

```bash
python -m training.train
```