from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pydantic import BaseModel, Field
import laspy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

from clustering.model import PointNetEncoder


class DatasetConfig(BaseModel):
    root_dir: Path
    points_per_tree: int = Field(default=1024, ge=1)
    num_workers: int = 4


class TrainConfig(BaseModel):
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 50
    trees_per_batch: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    embedding_dim: int = 128
    seed: int = 42


def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CrownSample(BaseModel):
    file_path: Path
    species: str


def list_las_files(root: Path) -> List[CrownSample]:
    paths = sorted(root.rglob("*.las"))
    return [CrownSample(file_path=p, species=parse_species_from_filename(p)) for p in paths]


def parse_species_from_filename(path: Path) -> str:
    name = path.stem
    if name.startswith("id_"):
        parts = name.split("_")
        if len(parts) >= 3:
            return "_".join(parts[2:])
    return "unknown"


def robust_xy_radius(xy: np.ndarray) -> float:
    centroid = xy.mean(axis=0)
    diffs = xy - centroid
    r = np.sqrt((diffs * diffs).sum(axis=1))
    return float(np.percentile(r, 95.0))


def robust_z_scale(z: np.ndarray) -> float:
    z_min = float(z.min())
    z_95 = float(np.percentile(z, 95.0))
    scale = z_95 - z_min
    return scale if scale > 0.0 else 1.0


def sample_points(points: np.ndarray, num: int) -> np.ndarray:
    n = points.shape[0]
    if n >= num:
        idx = np.random.choice(n, num, replace=False)
    else:
        idx = np.random.choice(n, num, replace=True)
    return points[idx]


class CrownDataset(Dataset):
    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg
        self.samples = list_las_files(cfg.root_dir)
        self.species_vocab = sorted({s.species for s in self.samples})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        sample = self.samples[idx]
        data = laspy.read(sample.file_path)
        x = np.asarray(data.x, dtype=np.float32)
        y = np.asarray(data.y, dtype=np.float32)
        z = np.asarray(data.z, dtype=np.float32)
        coords = np.stack([x, y, z], axis=1)

        coords = sample_points(coords, self.cfg.points_per_tree)

        xyz = coords[:, :3]
        centroid = xyz.mean(axis=0)
        xyz_centered = xyz - centroid
        r_xy = robust_xy_radius(xyz_centered[:, :2])
        if r_xy > 0.0:
            xyz_centered[:, 0:2] = xyz_centered[:, 0:2] / r_xy
        z_scale = robust_z_scale(xyz)
        xyz_centered[:, 2] = (xyz[:, 2] - float(xyz[:, 2].min())) / z_scale
        xyz_centered[:, 2] = xyz_centered[:, 2] - float(xyz_centered[:, 2].mean())

        tensor = torch.from_numpy(xyz_centered.astype(np.float32))
        label = int(self.species_vocab.index(sample.species))
        return tensor, label, str(sample.file_path)


def collate_crowns(batch: List[Tuple[torch.Tensor, int, str]]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    pts = [b[0] for b in batch]
    y = [b[1] for b in batch]
    files = [b[2] for b in batch]
    x = torch.stack(pts, dim=0)
    y_t = torch.as_tensor(y, dtype=torch.long)
    return x, y_t, files


class SupPointNet(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = PointNetEncoder(embedding_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)  # (B, D)
        logits = self.classifier(F.normalize(z, dim=1))
        return z, logits


def hungarian_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_true: int) -> Tuple[float, np.ndarray, np.ndarray]:
    K = int(y_pred.max()) + 1 if y_pred.size > 0 else 0
    cm = np.zeros((num_true, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    if K == 0:
        return 0.0, cm, cm
    row_ind, col_ind = linear_sum_assignment(-cm)
    # Build a column permutation so that, for each true class row r,
    # its matched predicted column is placed at column index r.
    matched_cols: List[int | None] = [None] * num_true
    used_cols: List[int] = []
    for r, c in zip(row_ind, col_ind):
        matched_cols[int(r)] = int(c)
        used_cols.append(int(c))
    remaining_cols = [c for c in range(K) if c not in used_cols]
    ordered_cols: List[int] = []
    rem_idx = 0
    for r in range(num_true):
        c_match = matched_cols[r]
        if c_match is not None:
            ordered_cols.append(c_match)
        else:
            if rem_idx < len(remaining_cols):
                ordered_cols.append(remaining_cols[rem_idx])
                rem_idx += 1
            else:
                ordered_cols.append(0)
    # Append any still remaining columns (if K > num_true)
    while rem_idx < len(remaining_cols):
        ordered_cols.append(remaining_cols[rem_idx])
        rem_idx += 1
    aligned = cm[:, ordered_cols]
    hits = int(cm[row_ind, col_ind].sum())
    total = int(cm.sum())
    acc = float(hits) / float(total) if total > 0 else 0.0
    return acc, cm, aligned


def train_and_cluster(
    data_cfg: DatasetConfig,
    train_cfg: TrainConfig,
    out_dir: Path,
) -> None:
    set_all_seeds(train_cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = CrownDataset(data_cfg)
    if len(ds) == 0:
        print(f"No .las files found under: {data_cfg.root_dir}")
        return

    num_classes = len(ds.species_vocab)

    loader = DataLoader(
        ds,
        batch_size=train_cfg.trees_per_batch,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        collate_fn=collate_crowns,
        drop_last=True,
        pin_memory=True,
    )

    model = SupPointNet(embedding_dim=train_cfg.embedding_dim, num_classes=num_classes).to(train_cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    ce = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(train_cfg.epochs):
        epoch_loss_sum = 0.0
        epoch_batches = 0
        for batch_pts, batch_y, _ in loader:
            batch_pts = batch_pts.to(train_cfg.device)
            batch_y = batch_y.to(train_cfg.device)
            z, logits = model(batch_pts)
            loss = ce(logits, batch_y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss_sum += float(loss.item())
            epoch_batches += 1
        epoch_loss = epoch_loss_sum / float(max(1, epoch_batches))
        print(f"epoch={epoch+1}/{train_cfg.epochs} loss={epoch_loss:.4f}")

    # Embed all crowns
    model.eval()
    embed_loader = DataLoader(
        ds,
        batch_size=train_cfg.trees_per_batch,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        collate_fn=collate_crowns,
        pin_memory=True,
    )
    all_embeddings: List[np.ndarray] = []
    all_files: List[str] = []
    all_species_idx: List[int] = []
    with torch.no_grad():
        for batch_pts, batch_y, files in embed_loader:
            batch_pts = batch_pts.to(train_cfg.device)
            z, _ = model(batch_pts)
            all_embeddings.append(z.cpu().numpy())
            all_files.extend(files)
            all_species_idx.extend([int(v) for v in batch_y.numpy().tolist()])
    E = np.concatenate(all_embeddings, axis=0)
    y_true_all = np.asarray(all_species_idx, dtype=np.int64)

    # Group by plot (parent folder relative to data root)
    data_root_abs = data_cfg.root_dir.resolve()
    plot_keys: List[str] = []
    for f in all_files:
        p = Path(f).resolve()
        rel = p.relative_to(data_root_abs)
        plot_keys.append(str(rel.parent).replace("\\", "/"))

    unique_plots = sorted(set(plot_keys))

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "clusters.csv").open("w", encoding="utf-8") as fh:
        fh.write("plot,file,true_species,cluster\n")
        overall_hits = 0
        overall_total = 0

        for plot in unique_plots:
            idxs = [i for i, k in enumerate(plot_keys) if k == plot]
            if len(idxs) == 0:
                continue
            E_plot = E[idxs]
            y_true_plot = y_true_all[idxs]

            # Species present in this plot define K
            present_species = sorted(set(int(v) for v in y_true_plot.tolist()))
            K = len(present_species)
            if K == 0:
                continue

            kmeans = KMeans(n_clusters=K, n_init=10, random_state=train_cfg.seed)
            plot_labels = kmeans.fit_predict(E_plot)

            # Evaluate via Hungarian matching
            acc, cm_raw, cm_aligned = hungarian_accuracy(y_true_plot, plot_labels, num_true=num_classes)

            # Track overall accuracy weighted by sample count
            overall_hits += int(np.trace(cm_aligned))
            overall_total += int(np.sum(cm_aligned))

            # Plot confusion (aligned) and normalized
            safe_plot = plot.replace("/", "_").replace("\\", "_")
            C_true = num_classes
            K_pred = cm_aligned.shape[1]

            fig1, ax1 = plt.subplots(figsize=(max(6, 0.6 * C_true), max(5, 0.6 * C_true)))
            im1 = ax1.imshow(cm_aligned, cmap="Blues")
            ax1.set_title(f"{safe_plot} — Confusion (aligned), acc={acc:.3f}")
            ax1.set_xlabel("Predicted clusters (Hungarian-aligned)")
            ax1.set_ylabel("True species")
            ax1.set_xticks(np.arange(K_pred))
            ax1.set_yticks(np.arange(C_true))
            ax1.set_xticklabels([str(i) for i in range(K_pred)], rotation=45, ha="right")
            ax1.set_yticklabels(ds.species_vocab)
            fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            fig1.tight_layout()
            fig1.savefig(plots_dir / f"{safe_plot}_confusion.png", dpi=200)
            plt.close(fig1)

            row_sums = cm_aligned.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_norm = cm_aligned / row_sums

            fig2, ax2 = plt.subplots(figsize=(max(6, 0.6 * C_true), max(5, 0.6 * C_true)))
            im2 = ax2.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
            ax2.set_title(f"{safe_plot} — Confusion (row-normalized)")
            ax2.set_xlabel("Predicted clusters (Hungarian-aligned)")
            ax2.set_ylabel("True species")
            ax2.set_xticks(np.arange(K_pred))
            ax2.set_yticks(np.arange(C_true))
            ax2.set_xticklabels([str(i) for i in range(K_pred)], rotation=45, ha="right")
            ax2.set_yticklabels(ds.species_vocab)
            fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            fig2.tight_layout()
            fig2.savefig(plots_dir / f"{safe_plot}_confusion_norm.png", dpi=200)
            plt.close(fig2)

            for local_pos, i_global in enumerate(idxs):
                fh.write(f"{plot},{all_files[i_global]},{ds.species_vocab[int(y_true_plot[local_pos])]}," f"{int(plot_labels[local_pos])}\n")

        overall_acc = float(overall_hits) / float(overall_total) if overall_total > 0 else 0.0
        print(f"Overall clustering accuracy (Hungarian-aligned): {overall_acc:.4f}")

    # Save embeddings and encoder
    np.save(out_dir / "embeddings.npy", E)
    torch.save(model.encoder.state_dict(), out_dir / "encoder.pt")
    print(f"Wrote: {out_dir / 'encoder.pt'}, {out_dir / 'embeddings.npy'}, {out_dir / 'clusters.csv'}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root / "las_cylinders_5m"
    out_dir = repo_root / "clustering" / "outputs"

    data_cfg = DatasetConfig(root_dir=data_root, points_per_tree=1024, num_workers=4)
    train_cfg = TrainConfig()
    train_and_cluster(data_cfg, train_cfg, out_dir)


if __name__ == "__main__":
    main()


