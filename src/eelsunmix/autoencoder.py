"""Constrained linear-unmixing autoencoder.

The decoder IS the linear mixing model: reconstruction = abundances @ endmembers
with endmembers kept non-negative through a softplus parametrization, and the
encoder outputs pass through a softmax so abundances are non-negative and sum
to one per pixel. The network therefore cannot fit anything but a simplex
mixture of k non-negative spectra; everything it learns is interpretable as
endmembers and abundance maps by construction.

Training minimizes mean squared error on dose-normalized spectra. Like NMF,
the loss surface is non-convex and the result depends on the seed; the
benchmark reports that spread instead of hiding it, and model selection
across seeds uses only the reconstruction loss (available on real data),
never ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .methods import Decomposition


@dataclass(frozen=True)
class AETrainConfig:
    """Autoencoder architecture and training settings.

    Attributes:
        k: Number of endmembers.
        hidden: Sizes of the two encoder hidden layers.
        epochs: Full-batch epochs (the whole spectrum image fits in memory).
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        seed: Torch seed for initialization.
    """

    k: int = 3
    hidden: tuple[int, int] = (96, 48)
    epochs: int = 1200
    lr: float = 3e-3
    weight_decay: float = 0.0
    seed: int = 0


class UnmixingAE(nn.Module):
    """Encoder MLP -> softmax abundances -> non-negative linear decoder."""

    def __init__(self, n_channels: int, config: AETrainConfig):
        super().__init__()
        h1, h2 = config.hidden
        self.encoder = nn.Sequential(
            nn.Linear(n_channels, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, config.k),
        )
        self.raw_endmembers = nn.Parameter(torch.randn(config.k, n_channels) * 0.1)
        self.config = config

    def endmembers(self) -> torch.Tensor:
        """Non-negative endmember spectra, shape (k, n_channels)."""
        return nn.functional.softplus(self.raw_endmembers)

    def abundances(self, x: torch.Tensor) -> torch.Tensor:
        """Simplex-constrained abundances, shape (n_pixels, k)."""
        return torch.softmax(self.encoder(x), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.abundances(x) @ self.endmembers()


def train_autoencoder(
    x: np.ndarray, config: AETrainConfig, verbose: bool = False
) -> tuple[UnmixingAE, list[float]]:
    """Train the unmixing autoencoder on one spectrum image.

    Spectra are normalized by the global mean count so the loss scale is
    dose-independent. Training is full batch (a 64 x 64 x 200 cube is only
    a few MB) with Adam.

    Args:
        x: Counts, shape (n_pixels, n_channels).
        config: Architecture and training settings.
        verbose: Print loss every 200 epochs.

    Returns:
        Tuple of (trained model, per-epoch loss history).
    """
    torch.manual_seed(config.seed)
    scale = float(np.mean(x)) + 1e-12
    xt = torch.tensor(np.asarray(x, dtype=np.float32) / scale)
    model = UnmixingAE(xt.shape[1], config)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    losses: list[float] = []
    for epoch in range(config.epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(xt), xt)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if verbose and (epoch + 1) % 200 == 0:
            print(f"epoch {epoch + 1:5d}  mse {losses[-1]:.6f}")
    model.eval()
    return model, losses


def ae_unmix(x: np.ndarray, config: AETrainConfig | None = None, n_seeds: int = 1) -> Decomposition:
    """Train the autoencoder (optionally over several seeds) and extract factors.

    With n_seeds > 1, one model is trained per seed and the one with the
    lowest final reconstruction loss is kept, mirroring the NMF restart
    protocol; selection never sees ground truth.

    Args:
        x: Counts, shape (n_pixels, n_channels).
        config: Settings; defaults to AETrainConfig() with k=3.
        n_seeds: Number of independently seeded trainings to take the best of.

    Returns:
        Decomposition; meta holds each seed's final loss and the loss history
        of the selected run.
    """
    config = config or AETrainConfig()
    best: tuple[float, UnmixingAE, list[float]] | None = None
    finals = []
    for s in range(n_seeds):
        cfg = AETrainConfig(**{**asdict(config), "seed": config.seed + s})
        model, losses = train_autoencoder(x, cfg)
        finals.append(losses[-1])
        if best is None or losses[-1] < best[0]:
            best = (losses[-1], model, losses)
    assert best is not None
    _, model, losses = best
    return decomposition_from_model(model, x, extra={"seed_losses": finals, "loss_curve": losses})


def decomposition_from_model(
    model: UnmixingAE, x: np.ndarray, extra: dict | None = None
) -> Decomposition:
    """Extract endmembers and abundances from a trained model.

    Args:
        model: Trained UnmixingAE.
        x: Counts, shape (n_pixels, n_channels).
        extra: Extra entries for the meta dict.

    Returns:
        Decomposition with unit-mean endmembers and simplex abundances.
    """
    scale = float(np.mean(x)) + 1e-12
    xt = torch.tensor(np.asarray(x, dtype=np.float32) / scale)
    with torch.no_grad():
        spectra = model.endmembers().numpy().astype(np.float64)
        abund = model.abundances(xt).numpy().astype(np.float64)
        recon = (abund @ spectra) * scale
    norm = spectra.mean(axis=1, keepdims=True) + 1e-12
    err = float(np.linalg.norm(np.asarray(x, dtype=np.float64) - recon))
    meta = {"train_config": asdict(model.config)}
    meta.update(extra or {})
    return Decomposition(spectra / norm, abund, err, meta)


def save_model(model: UnmixingAE, path: str, n_channels: int) -> None:
    """Save weights plus the architecture needed to rebuild the model.

    Args:
        model: Trained model.
        path: Output .pt path (parent directories are created).
        n_channels: Number of energy channels the model was trained on.
    """
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_channels": n_channels,
            "config": asdict(model.config),
        },
        path,
    )


def load_model(path: str) -> UnmixingAE:
    """Load a model saved by save_model.

    Args:
        path: Path to the .pt file.

    Returns:
        UnmixingAE in eval mode.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cfg_dict = dict(payload["config"])
    cfg_dict["hidden"] = tuple(cfg_dict["hidden"])
    config = AETrainConfig(**cfg_dict)
    model = UnmixingAE(payload["n_channels"], config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
