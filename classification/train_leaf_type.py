from __future__ import annotations
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from classification.config import TrainingConfig, PointNetParams
from classification.data import (
    scan_records,
    filter_records,
    train_val_split,
    read_points_xyz,
    sample_points,
    center_points_xy,
    PrecomputedPointCloudDataset,
)
from classification.schemas import Species, SampleRecord
from classification.model import PointNetTiny


class LeafType(str, Enum):
    deciduous = "deciduous"
    conifer = "conifer"
    unknown = "unknown"


# User-configurable parameters at the top
CONFIG = TrainingConfig(
    dataset_root=Path("/home/gleb/dev/tree-cluster/dataset"),
    points_per_cloud=1024,
    batch_size=16,
    num_epochs=20,
    learning_rate=1e-3,
    train_fraction=0.85,
    exclude_unknown=True,
    min_class_count=5,
    num_workers=2,
    seed=42,
    device="auto",
)

POINTNET = PointNetParams(
    conv_channels=(64, 128, 256),
    fc_dims=(128, 64),
    dropout=0.2,
)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if spec == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def species_to_leaf_type(species: Species) -> LeafType:
    if species == Species.birch:
        return LeafType.deciduous
    if species in (Species.cedar, Species.fir, Species.larch, Species.pine, Species.spruce):
        return LeafType.conifer
    return LeafType.unknown


def build_leaf_label_mapping(records: List[SampleRecord]) -> Dict[LeafType, int]:
    present_leaf_types = sorted({species_to_leaf_type(r.species) for r in records}, key=lambda t: t.value)
    mapping: Dict[LeafType, int] = {lt: i for i, lt in enumerate(present_leaf_types) if lt != LeafType.unknown}
    return mapping


def visualize_one_per_leaf_type_plotly(
    records: List[SampleRecord],
    leaf_to_index: Dict[LeafType, int],
    points_per_cloud: int,
    seed: int,
) -> None:
    index_to_leaf = {v: k for k, v in leaf_to_index.items()}
    chosen_paths: Dict[int, Path] = {}
    for r in records:
        lt = species_to_leaf_type(r.species)
        if lt == LeafType.unknown:
            continue
        idx = leaf_to_index[lt]
        if idx not in chosen_paths:
            chosen_paths[idx] = Path(r.path)
        if len(chosen_paths) == len(leaf_to_index):
            break

    rng = np.random.RandomState(seed)
    samples: List[Tuple[int, np.ndarray]] = []
    for idx, p in sorted(chosen_paths.items()):
        pts = read_points_xyz(p)
        pts = center_points_xy(pts)
        pts = sample_points(pts, points_per_cloud, rng)
        samples.append((idx, pts))

    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e"]
    for i, (idx, pts) in enumerate(samples):
        leaf = index_to_leaf[idx]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=2, color=colors[i % len(colors)]),
                name=str(leaf.value),
            )
        )

    fig.update_layout(
        title="One sample per leaf type",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        width=1000,
        height=650,
        legend=dict(orientation="h"),
    )
    out_dir = Path("./artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "one_sample_per_leaf_type.html"
    fig.write_html(str(html_path), auto_open=True, include_plotlyjs="cdn")
    print(f"Saved visualization to {html_path}")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * xb.size(0)
        preds = logits.argmax(dim=1)
        total_correct += int((preds == yb).sum().item())
        total_examples += xb.size(0)
    avg_loss = total_loss / float(total_examples)
    acc = float(total_correct) / float(total_examples)
    return avg_loss, acc


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            total_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=1)
            total_correct += int((preds == yb).sum().item())
            total_examples += xb.size(0)
    avg_loss = total_loss / float(total_examples)
    acc = float(total_correct) / float(total_examples)
    return avg_loss, acc


def precompute_leaf_dataset(
    records: List[SampleRecord],
    points_per_cloud: int,
    leaf_to_index: Dict[LeafType, int],
    seed: int,
    out_path: Path,
) -> Path:
    n = len(records)
    assert n > 0, "no records to precompute"
    rng = np.random.RandomState(seed)
    X = np.empty((n, points_per_cloud, 3), dtype=np.float32)
    y = np.empty((n,), dtype=np.int64)
    for i, rec in enumerate(records):
        pts = read_points_xyz(Path(rec.path))
        pts_centered_full = center_points_xy(pts)
        lt = species_to_leaf_type(rec.species)
        assert lt != LeafType.unknown, "encountered unknown leaf type in precompute"
        idx_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
        pts_sampled = sample_points(pts_centered_full, points_per_cloud, idx_rng)
        X[i] = pts_sampled.astype(np.float32)
        y[i] = int(leaf_to_index[lt])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), X=X, y=y)
    return out_path


def compute_and_save_confusion_matrix(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    leaf_to_index: Dict[LeafType, int],
    out_dir: Path,
) -> None:
    model.eval()
    all_preds: List[int] = []
    all_true: List[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(yb.cpu().numpy())

    all_preds_np = np.array(all_preds)
    all_true_np = np.array(all_true)
    cm = confusion_matrix(all_true_np, all_preds_np)

    np.save(out_dir / "confusion_matrix_leaf_type.npy", cm)

    index_to_leaf = {v: k for k, v in leaf_to_index.items()}
    labels = [index_to_leaf[i].value for i in range(len(leaf_to_index))]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix (Leaf Type)')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_leaf_type.png", dpi=100, bbox_inches='tight')
    plt.close()


def main() -> None:
    seed_everything(CONFIG.seed)
    device = resolve_device(CONFIG.device)

    records = scan_records(CONFIG.dataset_root)
    records = filter_records(records, CONFIG.min_class_count, CONFIG.exclude_unknown)
    leaf_to_index = build_leaf_label_mapping(records)
    # Ensure only deciduous and conifer present
    assert set(leaf_to_index.keys()).issubset({LeafType.deciduous, LeafType.conifer}), "unexpected leaf types present"
    num_classes = len(leaf_to_index)
    assert num_classes == 2, "expected exactly 2 classes for leaf type classification"

    visualize_one_per_leaf_type_plotly(records, leaf_to_index, CONFIG.points_per_cloud, CONFIG.seed)

    train_records, val_records = train_val_split(records, CONFIG.train_fraction, CONFIG.seed)

    out_dir = Path("./artifacts")
    train_npz = out_dir / "train_points_leaf_type.npz"
    val_npz = out_dir / "val_points_leaf_type.npz"
    precompute_leaf_dataset(train_records, CONFIG.points_per_cloud, leaf_to_index, CONFIG.seed, train_npz)
    precompute_leaf_dataset(val_records, CONFIG.points_per_cloud, leaf_to_index, CONFIG.seed + 1, val_npz)

    train_ds = PrecomputedPointCloudDataset(train_npz)
    val_ds = PrecomputedPointCloudDataset(val_npz)

    train_loader = DataLoader(train_ds, batch_size=CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_workers)
    val_loader = DataLoader(val_ds, batch_size=CONFIG.batch_size, shuffle=False, num_workers=CONFIG.num_workers)

    model = PointNetTiny(
        num_classes=num_classes,
        conv_channels=POINTNET.conv_channels,
        fc_dims=POINTNET.fc_dims,
        dropout=POINTNET.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.learning_rate)

    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    plt.ion()

    for epoch in range(1, CONFIG.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        epochs = list(range(1, epoch + 1))

        ax_loss.clear()
        ax_loss.plot(epochs, train_losses, marker='o', label='Train Loss', color='blue')
        ax_loss.plot(epochs, val_losses, marker='s', label='Val Loss', color='red')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title('Training and Validation Loss (Leaf Type)')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        ax_acc.clear()
        ax_acc.plot(epochs, train_accs, marker='o', label='Train Acc', color='blue')
        ax_acc.plot(epochs, val_accs, marker='s', label='Val Acc', color='red')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_title('Training and Validation Accuracy (Leaf Type)')
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.pause(0.1)

    plt.ioff()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "training_metrics_leaf_type.png", dpi=100, bbox_inches='tight')
    plt.close(fig)

    compute_and_save_confusion_matrix(model, val_loader, device, leaf_to_index, out_dir)

    torch.save({
        "model_state": model.state_dict(),
        "leaf_type_to_index": {k.value: v for k, v in leaf_to_index.items()},
        "config": CONFIG.model_dump(),
        "pointnet": POINTNET.model_dump(),
    }, out_dir / "pointnet_tiny_leaf_type.pt")
    print(f"Saved model to {out_dir / 'pointnet_tiny_leaf_type.pt'}")


if __name__ == "__main__":
    main()


