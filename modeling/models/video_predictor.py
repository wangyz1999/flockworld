from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlockWM(nn.Module):
    """Small baseline model for wiring data and training.

    It summarizes context frames and partial observations, embeds future action
    features per step, and predicts each future global frame independently.
    Replace this with a diffusion, transformer, or recurrent generator when the
    data pipeline is validated.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        action_dim: int,
        action_hidden_dim: int,
        out_channels: int,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, hidden_channels),
            ConvBlock(hidden_channels, hidden_channels),
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, action_hidden_dim),
            nn.SiLU(),
            nn.Linear(action_hidden_dim, hidden_channels),
        )
        self.decoder = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        context_video: torch.Tensor,
        partial_video: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        context = context_video.mean(dim=1)
        partial = partial_video.mean(dim=1)
        features = self.encoder(torch.cat([context, partial], dim=1))

        batch_size, horizon, _ = actions.shape
        action_features = self.action_mlp(actions).view(batch_size, horizon, -1, 1, 1)
        features = features[:, None] + action_features

        frames = [
            self.decoder(features[:, t])
            for t in range(horizon)
        ]
        return torch.stack(frames, dim=1)
