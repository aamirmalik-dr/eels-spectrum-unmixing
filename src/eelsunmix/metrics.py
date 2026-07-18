"""Scoring: spectral angles, Hungarian matching, abundance error, subspace error.

All endmember scores are computed after an optimal one-to-one assignment
between estimated and true components (Hungarian algorithm on the spectral
angle matrix), so no method is penalized for returning components in a
different order. Spectral angle is scale-invariant, which removes the
intensity ambiguity every factorization has; abundance maps are renormalized
to sum to one per pixel before comparison for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def spectral_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Spectral angle between two spectra in degrees.

    Args:
        a: Spectrum, shape (n_channels,).
        b: Spectrum, shape (n_channels,).

    Returns:
        Angle in degrees, in [0, 180]. Zero means identical up to scale.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 90.0
    cosang = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def sad_matrix(s_est: np.ndarray, s_true: np.ndarray) -> np.ndarray:
    """Pairwise spectral-angle matrix.

    Args:
        s_est: Estimated endmembers, shape (m, n_channels).
        s_true: True endmembers, shape (k, n_channels).

    Returns:
        Matrix of angles in degrees, shape (m, k).
    """
    return np.array(
        [[spectral_angle_deg(e, t) for t in np.atleast_2d(s_true)] for e in np.atleast_2d(s_est)]
    )


@dataclass
class MatchResult:
    """Optimal assignment of estimated components to true endmembers.

    Attributes:
        est_index: Row indices into the estimated components.
        true_index: Column indices into the true endmembers, aligned with
            est_index.
        sad_deg: Spectral angle of each matched pair, in degrees, ordered by
            true_index.
        mean_sad_deg: Mean matched spectral angle.
    """

    est_index: np.ndarray
    true_index: np.ndarray
    sad_deg: np.ndarray
    mean_sad_deg: float


def match_endmembers(s_est: np.ndarray, s_true: np.ndarray) -> MatchResult:
    """Optimally assign estimated components to true endmembers by spectral angle.

    When the estimated set is larger than the true set, the extra components
    are left unmatched (each true endmember gets its best partner under a
    one-to-one constraint); when smaller, some true endmembers go unmatched
    and only the matched ones are scored. This is stated wherever it applies.

    Args:
        s_est: Estimated endmembers, shape (m, n_channels).
        s_true: True endmembers, shape (k, n_channels).

    Returns:
        MatchResult ordered by true-endmember index.
    """
    cost = sad_matrix(s_est, s_true)
    rows, cols = linear_sum_assignment(cost)
    order = np.argsort(cols)
    rows, cols = rows[order], cols[order]
    sads = cost[rows, cols]
    return MatchResult(rows, cols, sads, float(sads.mean()))


def abundance_rmse(
    a_est: np.ndarray, a_true: np.ndarray, match: MatchResult, renormalize: bool = True
) -> float:
    """RMSE between matched abundance maps.

    Args:
        a_est: Estimated abundances, shape (m, ny, nx) or (m, n_pixels).
        a_true: True abundances, shape (k, ny, nx) or (k, n_pixels).
        match: Assignment from match_endmembers.
        renormalize: If True, rescale the matched estimated maps to sum to one
            per pixel first (removes the global scale ambiguity of NMF/VCA).

    Returns:
        Root-mean-square abundance error over matched components and pixels.
    """
    est = np.asarray(a_est, dtype=np.float64).reshape(a_est.shape[0], -1)
    true = np.asarray(a_true, dtype=np.float64).reshape(a_true.shape[0], -1)
    est_m = est[match.est_index]
    true_m = true[match.true_index]
    if renormalize:
        est_m = est_m / (est_m.sum(axis=0, keepdims=True) + 1e-12)
    return float(np.sqrt(np.mean((est_m - true_m) ** 2)))


def reconstruction_r2(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Coefficient of determination of a reconstruction, pooled over all entries.

    Args:
        x: Data, any shape.
        x_hat: Reconstruction, same shape.

    Returns:
        R squared; 1 is perfect, 0 matches predicting the global mean.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    x_hat = np.asarray(x_hat, dtype=np.float64).ravel()
    ss_res = np.sum((x - x_hat) ** 2)
    ss_tot = np.sum((x - x.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def subspace_error_deg(s_true: np.ndarray, basis: np.ndarray) -> float:
    """Largest principal angle between the true endmember span and a basis.

    This is the fair score for PCA, whose components are an orthogonal basis
    of the signal subspace rather than physical endmembers: it asks whether
    the true spectra live inside the recovered subspace at all.

    Args:
        s_true: True endmembers, shape (k, n_channels).
        basis: Basis vectors, shape (m, n_channels), m >= k.

    Returns:
        Largest principal angle in degrees (0 means the span is captured).
    """
    qt, _ = np.linalg.qr(np.asarray(s_true, dtype=np.float64).T)
    qb, _ = np.linalg.qr(np.asarray(basis, dtype=np.float64).T)
    sv = np.linalg.svd(qt.T @ qb, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(sv.min(), -1.0, 1.0))))
