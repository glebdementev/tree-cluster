from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from classification.config import TrainingConfig
from classification.data import (
    scan_records,
    filter_records,
    build_label_mapping,
    train_val_split,
    LasPointCloudDataset,
    read_points_xyz,
    sample_points,
    center_points,
)
from classification.schemas import Species
from clustering.config import ClusteringConfig, HeadParams
from clustering.model import PointNetEncoder


CONFIG = ClusteringConfig(
    dataset_root=Path("/home/gleb/dev/tree-cluster/dataset"),
    points_per_cloud=1024,
    batch_size=64,
    num_epochs=10,
    learning_rate=1e-3,
    exclude_unknown=True,
    min_class_count=5,
    num_workers=2,
    seed=42,
    device="auto",
    num_clusters=6,
    embedding_dim=256,
)

HEAD = HeadParams(dropout=0.1)


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


class UnlabeledDataset(Dataset):
    def __init__(self, records, points_per_cloud: int, seed: int) -> None:
        self.records = records
        self.points_per_cloud = points_per_cloud
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        pts = read_points_xyz(Path(rec.path))
        pts_sampled = sample_points(pts, self.points_per_cloud, self.rng)
        pts_centered = center_points(pts_sampled)
        x = torch.from_numpy(pts_centered.astype(np.float32))  # (P,3)
        return x


@torch.no_grad()
def extract_embeddings(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    feats: list[np.ndarray] = []
    for xb in loader:
        xb = xb.to(device)
        z = model(xb)
        feats.append(z.cpu().numpy())
    return np.concatenate(feats, axis=0)


class KMeansState:
    def __init__(self, num_clusters: int, dim: int, seed: int) -> None:
        rng = np.random.RandomState(seed)
        self.centroids = rng.normal(size=(num_clusters, dim)).astype(np.float32)

    def assign(self, embeddings: np.ndarray) -> np.ndarray:
        # squared euclidean
        d2 = (
            np.sum(embeddings**2, axis=1, keepdims=True)
            - 2.0 * embeddings @ self.centroids.T
            + np.sum(self.centroids**2, axis=1, keepdims=True).T
        )
        return d2.argmin(axis=1).astype(np.int64)

    def update(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        k = self.centroids.shape[0]
        dim = self.centroids.shape[1]
        new_centroids = np.zeros_like(self.centroids)
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new_centroids[j] = embeddings[mask].mean(axis=0)
            else:
                new_centroids[j] = self.centroids[j]
        self.centroids = new_centroids


def train_one_epoch(model: torch.nn.Module, loader: DataLoader, pseudo_labels: np.ndarray, device: torch.device, num_clusters: int) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    offset = 0
    for xb in loader:
        bsz = xb.size(0)
        yb_np = pseudo_labels[offset:offset+bsz]
        offset += bsz
        yb = torch.from_numpy(yb_np).to(device)
        xb = xb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        total_loss += float(loss.item()) * bsz
        preds = logits.argmax(dim=1)
        total_correct += int((preds == yb).sum().item())
        total_examples += bsz
    return total_loss / float(total_examples), float(total_correct) / float(total_examples)


def main() -> None:
    seed_everything(CONFIG.seed)
    device = resolve_device(CONFIG.device)

    records = scan_records(CONFIG.dataset_root)
    records = filter_records(records, CONFIG.min_class_count, CONFIG.exclude_unknown)

    dataset = UnlabeledDataset(records, CONFIG.points_per_cloud, CONFIG.seed)
    loader = DataLoader(dataset, batch_size=CONFIG.batch_size, shuffle=False, num_workers=CONFIG.num_workers)

    model = PointNetEncoder(embedding_dim=CONFIG.embedding_dim, dropout=HEAD.dropout).to(device)
    # classification head for pseudo-label training
    classifier = torch.nn.Linear(CONFIG.embedding_dim, CONFIG.num_clusters).to(device)
    full_model = torch.nn.Sequential(model, classifier)
    optimizer = torch.optim.Adam(full_model.parameters(), lr=CONFIG.learning_rate)

    out_dir = Path("./artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    plt.ion()

    kmeans = KMeansState(CONFIG.num_clusters, CONFIG.embedding_dim, CONFIG.seed)

    train_losses: list[float] = []
    train_accs: list[float] = []

    for epoch in range(1, CONFIG.num_epochs + 1):
        embeddings = extract_embeddings(model, loader, device)
        assignments = kmeans.assign(embeddings)
        kmeans.update(embeddings, assignments)
        align_loader = DataLoader(dataset, batch_size=CONFIG.batch_size, shuffle=False, num_workers=CONFIG.num_workers)
        loss, acc = train_one_epoch(full_model, align_loader, assignments, device, CONFIG.num_clusters)

        train_losses.append(loss)
        train_accs.append(acc)

        epochs = list(range(1, epoch + 1))
        ax_loss.clear()
        ax_loss.plot(epochs, train_losses, marker='o', label='Train Loss', color='blue')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title('Clustering Pseudo-Label Loss')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        ax_acc.clear()
        ax_acc.plot(epochs, train_accs, marker='o', label='Train Acc', color='blue')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_title('Pseudo-Label Accuracy')
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.pause(0.1)

    plt.ioff()
    fig.savefig(out_dir / "clustering_metrics.png", dpi=100, bbox_inches='tight')
    plt.close(fig)

    final_loader = DataLoader(dataset, batch_size=CONFIG.batch_size, shuffle=False, num_workers=CONFIG.num_workers)
    final_embeddings = extract_embeddings(model, final_loader, device)
    final_assignments = kmeans.assign(final_embeddings)

    np.save(out_dir / "embeddings.npy", final_embeddings)
    np.save(out_dir / "assignments.npy", final_assignments)

    torch.save({
        "encoder_state": model.state_dict(),
        "classifier_state": classifier.state_dict(),
        "config": CONFIG.model_dump(),
    }, out_dir / "pointnet_encoder_clustering.pt")
    print(f"Saved clustering artifacts to {out_dir}")


if __name__ == "__main__":
    main()
