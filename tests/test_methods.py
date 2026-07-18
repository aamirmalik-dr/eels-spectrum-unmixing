"""Classical methods: recovery in controlled cases, constraint satisfaction."""

import numpy as np
import pytest

from eelsunmix.methods import nmf_unmix, pca_decompose, pca_scree, solve_abundances, vca, vca_unmix
from eelsunmix.metrics import match_endmembers
from eelsunmix.sim import SimConfig, simulate


@pytest.fixture(scope="module")
def easy_scene():
    # High dose, no drift: close to an exact linear mixture.
    return simulate(
        SimConfig(nx=24, ny=24, n_channels=120, dose=20000.0, drift_channels=0.0, seed=0)
    )


def _pure_pixel_data(rng, n_px=300, n_ch=60, k=3):
    s = rng.uniform(0.1, 1.0, (k, n_ch))
    a = rng.dirichlet(np.ones(k) * 0.5, size=n_px)
    a[:k] = np.eye(k)  # guarantee pure pixels
    return a @ s, s, a


def test_vca_exact_on_noiseless_pure_pixels():
    rng = np.random.default_rng(0)
    x, s_true, _ = _pure_pixel_data(rng)
    s_est = vca(x, 3, seed=0)
    match = match_endmembers(s_est, s_true)
    assert match.mean_sad_deg < 0.5


def test_solve_abundances_exact_on_noiseless_mixture():
    rng = np.random.default_rng(1)
    x, s_true, a_true = _pure_pixel_data(rng)
    a_est = solve_abundances(x, s_true, sum_to_one=True)
    np.testing.assert_allclose(a_est, a_true, atol=1e-4)


def test_solve_abundances_constraints():
    rng = np.random.default_rng(2)
    x, s_true, _ = _pure_pixel_data(rng)
    a = solve_abundances(x + rng.poisson(5, x.shape), s_true, sum_to_one=True)
    assert a.min() >= 0.0
    np.testing.assert_allclose(a.sum(axis=1), 1.0, atol=1e-9)


def test_nmf_factors_are_nonnegative(easy_scene):
    dec = nmf_unmix(easy_scene.flat, 3, seed=0)
    assert dec.spectra.min() >= 0.0
    assert dec.abundances.min() >= 0.0


def test_nmf_recovers_easy_scene(easy_scene):
    # Even at high dose the shared O K edge and background make the factors
    # correlated, so NMF lands near but not on the truth; this bounds sanity
    # (well below the ~40+ deg of an arbitrary factorization), not perfection.
    dec = nmf_unmix(easy_scene.flat, 3, seed=0)
    match = match_endmembers(dec.spectra, easy_scene.endmembers)
    assert match.mean_sad_deg < 20.0


def test_nmf_restarts_never_worse_than_single():
    rng = np.random.default_rng(3)
    x, _, _ = _pure_pixel_data(rng)
    single = nmf_unmix(x, 3, seed=0, init="random")
    best = nmf_unmix(x, 3, seed=0, n_restarts=5)
    assert best.reconstruction_error <= single.reconstruction_error + 1e-9
    assert len(best.meta["restart_errors"]) == 5


def test_vca_unmix_full_pipeline(easy_scene):
    dec = vca_unmix(easy_scene.flat, 3, seed=0)
    assert dec.spectra.shape == (3, 120)
    assert dec.abundances.shape == (easy_scene.flat.shape[0], 3)
    np.testing.assert_allclose(dec.abundances.sum(axis=1), 1.0, atol=1e-9)


def test_pca_scree_finds_rank_three():
    # Noiseless mixture has exactly rank 3; centered PCA sees rank 2 signal
    # above machine noise plus the mean, so components 4+ are negligible.
    rng = np.random.default_rng(4)
    x, _, _ = _pure_pixel_data(rng, n_px=500)
    evr = pca_scree(x, kmax=6)
    assert evr[:2].sum() > 0.999
    assert evr[3] < 1e-10


def test_pca_reconstruction_improves_with_k(easy_scene):
    errs = [pca_decompose(easy_scene.flat, k).reconstruction_error for k in (1, 2, 3)]
    assert errs[0] > errs[1] > errs[2]
