from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import laspy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


@dataclass(frozen=True)
class TrainConfig:
    repo_root: Path
    data_root: Path
    output_dir: Path
    points_per_sample: int
    batch_size: int
    epochs: int
    learning_rate: float
    train_split: float
    seed: int
    num_workers: int


def find_all_las_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.las")]


def extract_species_from_filename(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        return "unknown"
    return parts[2]


def compute_point_counts(files: List[Path]) -> List[int]:
    counts: List[int] = []
    for p in files:
        las = laspy.read(p)
        counts.append(len(las.points))
    return counts


def print_stats_point_counts(counts: List[int]) -> None:
    if not counts:
        print("No LAS files found for stats.")
        return
    total = len(counts)
    sorted_counts = sorted(counts)
    minimum = sorted_counts[0]
    maximum = sorted_counts[-1]
    mean = float(np.mean(sorted_counts))
    median = float(np.median(sorted_counts))
    p25 = float(np.percentile(sorted_counts, 25))
    p75 = float(np.percentile(sorted_counts, 75))
    print("Point counts per cylinder (files):")
    print(
        "  count=", total,
        "min=", minimum,
        "p25=", f"{p25:.1f}",
        "median=", f"{median:.1f}",
        "p75=", f"{p75:.1f}",
        "max=", maximum,
        "mean=", f"{mean:.1f}",
    )


class CylinderDataset(Dataset):
    def __init__(
        self,
        files: List[Tuple[Path, int]],
        class_count: int,
        points_per_sample: int,
        seed: int,
    ) -> None:
        self.files = files
        self.class_count = class_count
        self.points_per_sample = points_per_sample
        self.random = random.Random(seed)
        self._cache: Dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.files)

    def _load_points(self, path: Path) -> np.ndarray:
        if path in self._cache:
            return self._cache[path]
        las = laspy.read(path)
        # Use x, y, z in a contiguous float32 array
        pts = np.stack([las.x, las.y, las.z], axis=1).astype(np.float32)
        self._cache[path] = pts
        return pts

    def _sample_points(self, pts: np.ndarray) -> np.ndarray:
        n = pts.shape[0]
        if n >= self.points_per_sample:
            idx = np.array(self.random.sample(range(n), self.points_per_sample))
            return pts[idx]
        # If fewer points than required, sample with replacement
        idx = np.array([self.random.randrange(n) for _ in range(self.points_per_sample)])
        return pts[idx]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.files[index]
        pts = self._load_points(path)
        sampled = self._sample_points(pts)
        # Normalize per-sample: subtract centroid, scale to unit sphere
        centroid = sampled.mean(axis=0, keepdims=True)
        centered = sampled - centroid
        scale = np.linalg.norm(centered, axis=1).max()
        if scale > 0:
            centered = centered / scale
        # Shape to (3, N) for Conv1d in PyTorch
        tensor = torch.from_numpy(centered.T)  # (3, points_per_sample)
        return tensor, label


class PointNetTiny(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # Shared MLP via Conv1d with BatchNorm
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2)[0]  # global max pool -> (B, 256)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.drop2(x)
        logits = self.fc3(x)
        return logits


def split_train_val(files_by_class: Dict[int, List[Path]], split: float, seed: int) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    rng = random.Random(seed)
    train: List[Tuple[Path, int]] = []
    val: List[Tuple[Path, int]] = []
    for cls, files in files_by_class.items():
        shuffled = files[:]
        rng.shuffle(shuffled)
        n_train = int(len(shuffled) * split)
        train.extend((p, cls) for p in shuffled[:n_train])
        val.extend((p, cls) for p in shuffled[n_train:])
    return train, val


def build_balanced_index(samples: List[Tuple[Path, int]], num_classes: int, seed: int) -> List[Tuple[Path, int]]:
    # Balance by oversampling scarce classes using multiple distinct subsets per file
    rng = random.Random(seed)
    per_class: Dict[int, List[Path]] = {c: [] for c in range(num_classes)}
    for p, c in samples:
        per_class[c].append(p)
    class_sizes = {c: len(v) for c, v in per_class.items()}
    target = max(class_sizes.values()) if class_sizes else 0
    balanced: List[Tuple[Path, int]] = []
    for c, files in per_class.items():
        if not files:
            continue
        k = math.ceil(target / len(files))
        # Create k entries per file to force different random subsets downstream
        for _ in range(k):
            shuffled = files[:]
            rng.shuffle(shuffled)
            for p in shuffled:
                balanced.append((p, c))
    return balanced


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            pts, labels = batch
            pts = pts.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pts)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * pts.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total += int(pts.size(0))
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for pts, labels in val_loader:
                pts = pts.to(device)
                labels = labels.to(device)
                logits = model(pts)
                preds = torch.argmax(logits, dim=1)
                val_correct += int((preds == labels).sum().item())
                val_total += int(pts.size(0))
        val_acc = val_correct / max(1, val_total)
        print(f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root / "las_cylinders_5m"
    output_dir = repo_root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        repo_root=repo_root,
        data_root=data_root,
        output_dir=output_dir,
        points_per_sample=512,
        batch_size=16,
        epochs=25,
        learning_rate=1e-3,
        train_split=0.85,
        seed=42,
        num_workers=0,
    )

    print("Step 1: Scanning LAS cylinders recursively...")
    files = find_all_las_files(cfg.data_root)
    print("  Found files:", len(files))
    if not files:
        print("No LAS files found in", cfg.data_root)
        return

    print("Step 2: Deriving species labels from filenames, selecting target classes, and computing stats...")
    species_names_all: List[str] = [extract_species_from_filename(p) for p in files]
    target_classes_order = ["aspen", "birch", "spruce", "pine"]
    selected: List[Tuple[Path, str]] = [
        (p, s) for p, s in zip(files, species_names_all) if s in set(target_classes_order)
    ]
    files = [p for p, _ in selected]
    species_names: List[str] = [s for _, s in selected]
    unique_species = [s for s in target_classes_order if s in set(species_names)]
    species_to_index: Dict[str, int] = {s: i for i, s in enumerate(unique_species)}
    print("  Target classes (species):", unique_species)

    files_by_class: Dict[int, List[Path]] = {i: [] for i in range(len(unique_species))}
    for p, s in zip(files, species_names):
        files_by_class[species_to_index[s]].append(p)

    print("  Samples per class (by file count):")
    for s in unique_species:
        c = species_to_index[s]
        print("   ", s, "->", len(files_by_class[c]))

    counts = compute_point_counts(files)
    print_stats_point_counts(counts)

    print("Step 3: Splitting train/val...")
    train_samples, val_samples = split_train_val(files_by_class, cfg.train_split, cfg.seed)
    print("  Train files:", len(train_samples), " Val files:", len(val_samples))

    print("Step 4: Balancing classes by oversampling rarer classes with different point subsets...")
    balanced_train = build_balanced_index(train_samples, len(unique_species), cfg.seed)
    print("  Balanced train samples:", len(balanced_train))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Step 5: Building datasets and model (device:", device.type, ") ...")
    train_ds = CylinderDataset(balanced_train, len(unique_species), cfg.points_per_sample, cfg.seed)
    val_ds = CylinderDataset(val_samples, len(unique_species), cfg.points_per_sample, cfg.seed)
    pin = True if device.type == "cuda" else False
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)

    model = PointNetTiny(num_classes=len(unique_species)).to(device)

    print("Step 6: Training PointNet...")
    train_loop(model, train_loader, val_loader, cfg.epochs, device, cfg.learning_rate)

    model_path = cfg.output_dir / "pointnet_tiny.pth"
    torch.save(model.state_dict(), model_path)
    with (cfg.output_dir / "species_index.json").open("w", encoding="utf-8") as fh:
        json.dump({"classes": unique_species}, fh, ensure_ascii=False, indent=2)
    print("Saved model to:", model_path)
    print("Saved class mapping to:", cfg.output_dir / "species_index.json")


if __name__ == "__main__":
    main()


