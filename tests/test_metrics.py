"""Metric properties: scale invariance, optimal matching, subspace angles."""

import numpy as np
import pytest

from eelsunmix.metrics import (
    abundance_rmse,
    match_endmembers,
    reconstruction_r2,
    spectral_angle_deg,
    subspace_error_deg,
)


def test_spectral_angle_zero_for_scaled_copy():
    s = np.random.default_rng(0).uniform(0.1, 1.0, 50)
    assert spectral_angle_deg(s, 7.3 * s) == pytest.approx(0.0, abs=1e-6)


def test_spectral_angle_orthogonal_is_90():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert spectral_angle_deg(a, b) == pytest.approx(90.0)


def test_spectral_angle_zero_vector_returns_90():
    assert spectral_angle_deg(np.zeros(5), np.ones(5)) == pytest.approx(90.0)


def test_matching_undoes_permutation():
    rng = np.random.default_rng(1)
    s_true = rng.uniform(0.1, 1.0, (3, 40))
    perm = [2, 0, 1]
    match = match_endmembers(s_true[perm], s_true)
    assert match.mean_sad_deg == pytest.approx(0.0, abs=1e-6)
    # est row match.est_index[i] must equal true row match.true_index[i]
    for e, t in zip(match.est_index, match.true_index):
        np.testing.assert_allclose(s_true[perm][e], s_true[t])


def test_matching_handles_more_estimates_than_truth():
    rng = np.random.default_rng(2)
    s_true = rng.uniform(0.1, 1.0, (3, 40))
    s_est = np.vstack([s_true, rng.uniform(0.1, 1.0, (2, 40))])
    match = match_endmembers(s_est, s_true)
    assert len(match.sad_deg) == 3
    assert match.mean_sad_deg == pytest.approx(0.0, abs=1e-6)


def test_abundance_rmse_zero_for_perfect_recovery():
    rng = np.random.default_rng(3)
    s_true = rng.uniform(0.1, 1.0, (3, 40))
    a = rng.dirichlet(np.ones(3), size=100).T  # (3, 100)
    match = match_endmembers(s_true, s_true)
    assert abundance_rmse(a, a, match) == pytest.approx(0.0, abs=1e-9)


def test_abundance_rmse_scale_invariant_with_renormalize():
    rng = np.random.default_rng(4)
    s_true = rng.uniform(0.1, 1.0, (3, 40))
    a = rng.dirichlet(np.ones(3), size=100).T
    match = match_endmembers(s_true, s_true)
    assert abundance_rmse(5.0 * a, a, match, renormalize=True) == pytest.approx(0.0, abs=1e-9)


def test_reconstruction_r2_perfect_and_mean():
    x = np.random.default_rng(5).normal(size=(20, 10))
    assert reconstruction_r2(x, x) == pytest.approx(1.0)
    assert reconstruction_r2(x, np.full_like(x, x.mean())) == pytest.approx(0.0, abs=1e-9)


def test_subspace_error_zero_when_span_captured():
    rng = np.random.default_rng(6)
    s_true = rng.uniform(0.1, 1.0, (3, 40))
    mixing = rng.normal(size=(3, 3)) + 3 * np.eye(3)
    basis = mixing @ s_true  # same span, different vectors
    assert subspace_error_deg(s_true, basis) == pytest.approx(0.0, abs=1e-4)


def test_subspace_error_large_for_disjoint_spans():
    s_true = np.eye(3, 40)
    basis = np.eye(40)[10:13]
    assert subspace_error_deg(s_true, basis) == pytest.approx(90.0)
