from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .config import TrainingConfig, PointNetParams
from .data import (
    scan_records,
    filter_records,
    build_label_mapping,
    train_val_split,
    LasPointCloudDataset,
)
from .model import PointNetTiny


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

    for epoch in range(1, CONFIG.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    out_dir = Path("./artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "species_to_index": {k.value: v for k, v in species_to_index.items()},
        "config": CONFIG.model_dump(),
        "pointnet": POINTNET.model_dump(),
    }, out_dir / "pointnet_tiny.pt")
    print(f"Saved model to {out_dir / 'pointnet_tiny.pt'}")


if __name__ == "__main__":
    main()


