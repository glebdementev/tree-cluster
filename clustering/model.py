from __future__ import annotations
import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    def __init__(
        self,
        conv_channels: tuple[int, int, int] = (64, 128, 256),
        embedding_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        c1, c2, c3 = conv_channels
        self.mlp = nn.Sequential(
            nn.Conv1d(3, c1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c2, c3, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(c3, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, 3)
        x = x.transpose(1, 2)  # (B, 3, P)
        x = self.mlp(x)  # (B, C, P)
        x = torch.amax(x, dim=2)  # (B, C)
        z = self.head(x)  # (B, D)
        return z
