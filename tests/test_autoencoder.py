"""Autoencoder: constraints hold by construction, training works, IO round-trips."""

import numpy as np
import pytest
import torch

from eelsunmix.autoencoder import (
    AETrainConfig,
    UnmixingAE,
    ae_unmix,
    decomposition_from_model,
    load_model,
    save_model,
    train_autoencoder,
)
from eelsunmix.sim import SimConfig, simulate


@pytest.fixture(scope="module")
def small_scene():
    return simulate(SimConfig(nx=16, ny=16, n_channels=100, dose=2000.0, seed=0))


@pytest.fixture(scope="module")
def trained(small_scene):
    cfg = AETrainConfig(k=3, epochs=300, seed=0)
    return train_autoencoder(small_scene.flat, cfg)


def test_abundances_on_simplex_by_construction():
    model = UnmixingAE(50, AETrainConfig(k=3, seed=0))
    x = torch.rand(17, 50)
    a = model.abundances(x)
    assert a.shape == (17, 3)
    assert float(a.min()) >= 0.0
    torch.testing.assert_close(a.sum(dim=1), torch.ones(17))


def test_endmembers_nonnegative_by_construction():
    model = UnmixingAE(50, AETrainConfig(k=4, seed=1))
    assert float(model.endmembers().min()) >= 0.0


def test_training_reduces_loss(trained):
    _, losses = trained
    assert losses[-1] < 0.5 * losses[0]


def test_trained_model_recovers_scene(small_scene, trained):
    model, _ = trained
    dec = decomposition_from_model(model, small_scene.flat)
    from eelsunmix.metrics import match_endmembers

    match = match_endmembers(dec.spectra, small_scene.endmembers)
    assert match.mean_sad_deg < 25.0  # loose smoke bound at only 300 epochs


def test_decomposition_shapes(small_scene, trained):
    model, _ = trained
    dec = decomposition_from_model(model, small_scene.flat)
    assert dec.spectra.shape == (3, 100)
    assert dec.abundances.shape == (256, 3)
    np.testing.assert_allclose(dec.abundances.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(dec.spectra.mean(axis=1), 1.0, atol=1e-6)


def test_seed_determinism(small_scene):
    cfg = AETrainConfig(k=3, epochs=20, seed=5)
    _, la = train_autoencoder(small_scene.flat, cfg)
    _, lb = train_autoencoder(small_scene.flat, cfg)
    assert la[-1] == pytest.approx(lb[-1], rel=1e-6)


def test_ae_unmix_best_of_seeds_selects_lowest_loss(small_scene):
    dec = ae_unmix(small_scene.flat, AETrainConfig(k=3, epochs=40, seed=0), n_seeds=3)
    finals = dec.meta["seed_losses"]
    assert len(finals) == 3
    assert dec.meta["loss_curve"][-1] == pytest.approx(min(finals))


def test_save_load_roundtrip(tmp_path, small_scene, trained):
    model, _ = trained
    path = str(tmp_path / "model.pt")
    save_model(model, path, n_channels=100)
    loaded = load_model(path)
    x = torch.rand(5, 100)
    with torch.no_grad():
        torch.testing.assert_close(model(x), loaded(x))
