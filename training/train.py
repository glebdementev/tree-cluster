from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import TrainingConfig, PointNetParams
from training.data import (
    scan_records,
    filter_records,
    build_label_mapping,
    train_val_split,
    LasPointCloudDataset,
    read_points_xyz,
    sample_points,
    center_points,
)
from training.model import PointNetTiny


# User-configurable parameters at the top
CONFIG = TrainingConfig(
    dataset_root=Path("/home/gleb/dev/tree-cluster/dataset"),
    points_per_cloud=1024,
    batch_size=16,
    num_epochs=10,
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


def visualize_one_per_class(
    records: list,
    species_to_index: dict,
    points_per_cloud: int,
    seed: int,
) -> None:
    # Pick first record per species
    index_to_species = {v: k for k, v in species_to_index.items()}
    chosen_paths: dict[int, Path] = {}
    for r in records:
        idx = species_to_index[r.species]
        if idx not in chosen_paths:
            chosen_paths[idx] = Path(r.path)
        if len(chosen_paths) == len(species_to_index):
            break

    rng = np.random.RandomState(seed)
    samples: list[tuple[int, np.ndarray]] = []
    for idx, p in sorted(chosen_paths.items()):
        pts = read_points_xyz(p)
        pts = sample_points(pts, points_per_cloud, rng)
        pts = center_points(pts)
        samples.append((idx, pts))

    # Arrange clouds side-by-side along X for a clean comparison
    gap = 4.0
    geometries: list[o3d.geometry.Geometry] = []
    colors = np.array([
        [0.89, 0.10, 0.11],
        [0.12, 0.47, 0.71],
        [0.17, 0.63, 0.17],
        [0.84, 0.15, 0.16],
        [0.58, 0.40, 0.74],
        [1.00, 0.50, 0.00],
        [0.65, 0.81, 0.89],
        [0.70, 0.87, 0.54],
    ], dtype=np.float64)
    for i, (idx, pts) in enumerate(samples):
        offset = np.array([i * gap, 0.0, 0.0], dtype=np.float32)
        shifted = pts + offset
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(shifted.astype(np.float64))
        color = colors[i % len(colors)]
        pcd.colors = o3d.utility.Vector3dVector(np.repeat(color[None, :], shifted.shape[0], axis=0))
        geometries.append(pcd)

        # Add a small text label with species name using a 3D coordinate frame + geometry
        # Open3D lacks native 3D text; we add a coordinate frame as a marker near each cloud
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        frame.translate((float(offset[0]), -2.0, 0.0))
        geometries.append(frame)

    o3d.visualization.draw_geometries(
        geometries,
        window_name="One sample per class (close to start training)",
        width=1280,
        height=720,
        left=100,
        top=100,
        point_show_normal=False,
    )


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


def main() -> None:
    seed_everything(CONFIG.seed)
    device = resolve_device(CONFIG.device)

    records = scan_records(CONFIG.dataset_root)
    records = filter_records(records, CONFIG.min_class_count, CONFIG.exclude_unknown)
    species_to_index = build_label_mapping(records)
    num_classes = len(species_to_index)

    # Block and visualize one centered sample per class before training
    visualize_one_per_class(records, species_to_index, CONFIG.points_per_cloud, CONFIG.seed)

    train_records, val_records = train_val_split(records, CONFIG.train_fraction, CONFIG.seed)
    train_ds = LasPointCloudDataset(train_records, CONFIG.points_per_cloud, species_to_index, CONFIG.seed)
    val_ds = LasPointCloudDataset(val_records, CONFIG.points_per_cloud, species_to_index, CONFIG.seed + 1)

    train_loader = DataLoader(train_ds, batch_size=CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_workers)
    val_loader = DataLoader(val_ds, batch_size=CONFIG.batch_size, shuffle=False, num_workers=CONFIG.num_workers)

    model = PointNetTiny(
        num_classes=num_classes,
        conv_channels=POINTNET.conv_channels,
        fc_dims=POINTNET.fc_dims,
        dropout=POINTNET.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.learning_rate)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

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
        ax_loss.set_title('Training and Validation Loss')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        ax_acc.clear()
        ax_acc.plot(epochs, train_accs, marker='o', label='Train Acc', color='blue')
        ax_acc.plot(epochs, val_accs, marker='s', label='Val Acc', color='red')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_title('Training and Validation Accuracy')
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.pause(0.1)

    plt.ioff()
    out_dir = Path("./artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "training_metrics.png", dpi=100, bbox_inches='tight')
    plt.close(fig)

    torch.save({
        "model_state": model.state_dict(),
        "species_to_index": {k.value: v for k, v in species_to_index.items()},
        "config": CONFIG.model_dump(),
        "pointnet": POINTNET.model_dump(),
    }, out_dir / "pointnet_tiny.pt")
    print(f"Saved model to {out_dir / 'pointnet_tiny.pt'}")


if __name__ == "__main__":
    main()


