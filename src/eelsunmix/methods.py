"""Classical decomposition baselines: PCA, NMF, and VCA with constrained abundances.

Conventions: the data matrix X has shape (n_pixels, n_channels), raw counts.
Every method returns endmember spectra S of shape (k, n_channels) and
abundances A of shape (n_pixels, k) so that X is approximately A @ S.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls
from sklearn.decomposition import NMF


@dataclass
class Decomposition:
    """Result of one unmixing run.

    Attributes:
        spectra: Component spectra, shape (k, n_channels).
        abundances: Per-pixel loadings, shape (n_pixels, k).
        reconstruction_error: Frobenius norm of X - A @ S.
        meta: Method-specific extras (restart losses, chosen indices, ...).
    """

    spectra: np.ndarray
    abundances: np.ndarray
    reconstruction_error: float
    meta: dict


def pca_decompose(x: np.ndarray, k: int) -> Decomposition:
    """PCA via SVD of the mean-centered data.

    PCA components are an orthogonal basis with negative entries, not
    physical endmembers; this baseline exists for subspace and rank analysis
    and is scored as such (see metrics.subspace_error_deg).

    Args:
        x: Data, shape (n_pixels, n_channels).
        k: Number of components.

    Returns:
        Decomposition; meta holds the full explained-variance ratio curve.
    """
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    xc = x - mean
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    evr = (s**2) / np.sum(s**2)
    spectra = vt[:k]
    scores = u[:, :k] * s[:k]
    recon = scores @ spectra + mean
    err = float(np.linalg.norm(x - recon))
    return Decomposition(spectra, scores, err, {"explained_variance_ratio": evr, "mean": mean})


def pca_scree(x: np.ndarray, kmax: int = 10) -> np.ndarray:
    """Explained-variance ratios of the first kmax principal components.

    Args:
        x: Data, shape (n_pixels, n_channels).
        kmax: Number of leading components to report.

    Returns:
        Explained-variance ratios, shape (kmax,).
    """
    return pca_decompose(x, 1).meta["explained_variance_ratio"][:kmax]


def nmf_unmix(
    x: np.ndarray,
    k: int,
    seed: int = 0,
    n_restarts: int = 1,
    init: str = "nndsvda",
    max_iter: int = 1000,
) -> Decomposition:
    """Non-negative matrix factorization with optional random restarts.

    With n_restarts == 1 the deterministic nndsvda initialization is used.
    With more, all restarts use random initialization and the run with the
    lowest reconstruction error is kept; selection uses only the fit error,
    never ground truth, so it is a procedure available on real data.

    Args:
        x: Non-negative data, shape (n_pixels, n_channels).
        k: Number of components.
        seed: Base random seed.
        n_restarts: Number of runs to take the best of.
        init: Initialization for the single-run case.
        max_iter: Maximum NMF iterations per run.

    Returns:
        Decomposition; meta holds every restart's reconstruction error.
    """
    x = np.asarray(x, dtype=np.float64)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    errors = []
    for r in range(n_restarts):
        run_init = init if n_restarts == 1 else "random"
        model = NMF(
            n_components=k,
            init=run_init,
            random_state=seed + r,
            max_iter=max_iter,
            tol=1e-5,
        )
        w = model.fit_transform(x)
        h = model.components_
        err = float(np.linalg.norm(x - w @ h))
        errors.append(err)
        if best is None or err < best[0]:
            best = (err, h, w)
    assert best is not None
    err, h, w = best
    scale = h.mean(axis=1, keepdims=True) + 1e-12
    return Decomposition(h / scale, w * scale.T, err, {"restart_errors": errors})


def vca(x: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Vertex component analysis: extract endmembers as extreme pixels.

    Implements the standard VCA scheme: estimate the signal SNR, project the
    data onto a k (or k-1) dimensional subspace accordingly, then iteratively
    pick the pixel with the largest projection onto a direction orthogonal to
    the simplex spanned by the endmembers found so far. Endmembers are
    returned as actual pixel spectra, so VCA needs at least one near-pure
    pixel per phase to succeed; on diffuse scenes it degrades honestly.

    Args:
        x: Data, shape (n_pixels, n_channels).
        k: Number of endmembers.
        seed: Seed for the random direction draws.

    Returns:
        Endmember spectra, shape (k, n_channels).
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(x, dtype=np.float64).T  # (channels, pixels)
    n_ch, n_px = r.shape

    mean = r.mean(axis=1, keepdims=True)
    r0 = r - mean
    ud_c = np.linalg.svd(r0 @ r0.T / n_px, hermitian=True)[0][:, :k]
    xp = ud_c.T @ r0
    p_y = float(np.sum(r**2)) / n_px
    p_x = float(np.sum(xp**2)) / n_px + float((mean.T @ mean).item())
    snr = 10.0 * np.log10(max(p_x - k / n_ch * p_y, 1e-12) / max(p_y - p_x, 1e-12))
    snr_threshold = 15.0 + 10.0 * np.log10(k)

    if snr > snr_threshold:
        ud = np.linalg.svd(r @ r.T / n_px, hermitian=True)[0][:, :k]
        xproj = ud.T @ r
        u = xproj.mean(axis=1, keepdims=True)
        denom = np.sum(xproj * u, axis=0)
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        y = xproj / denom
    else:
        d = max(k - 1, 1)
        xproj = ud_c[:, :d].T @ r0
        c = float(np.max(np.sqrt(np.sum(xproj**2, axis=0))))
        y = np.vstack([xproj, c * np.ones((1, n_px))])

    dim = y.shape[0]
    indices = np.zeros(k, dtype=int)
    a = np.zeros((dim, k))
    a[-1, 0] = 1.0
    for i in range(k):
        w = rng.normal(size=(dim, 1))
        f = w - a @ np.linalg.pinv(a) @ w
        f = f / (np.linalg.norm(f) + 1e-12)
        v = f.T @ y
        indices[i] = int(np.argmax(np.abs(v)))
        a[:, i] = y[:, indices[i]]
    return r[:, indices].T.copy()


def solve_abundances(
    x: np.ndarray, spectra: np.ndarray, sum_to_one: bool = True, delta: float = 30.0
) -> np.ndarray:
    """Per-pixel non-negative least-squares abundances for fixed endmembers.

    With sum_to_one, the sum constraint is enforced softly by augmenting the
    system with a heavily weighted row of ones (the standard SCLS device),
    then renormalizing exactly.

    Args:
        x: Data, shape (n_pixels, n_channels).
        spectra: Endmember spectra, shape (k, n_channels).
        sum_to_one: Enforce the abundance sum-to-one constraint.
        delta: Weight of the sum-to-one row relative to the mean data scale.

    Returns:
        Abundances, shape (n_pixels, k), non-negative (and summing to one per
        pixel when sum_to_one is set).
    """
    x = np.asarray(x, dtype=np.float64)
    s = np.asarray(spectra, dtype=np.float64)
    k = s.shape[0]
    design = s.T
    if sum_to_one:
        w = delta * float(np.abs(x).mean() + 1e-12)
        design = np.vstack([design, w * np.ones((1, k))])
    out = np.empty((x.shape[0], k))
    for i in range(x.shape[0]):
        target = x[i] if not sum_to_one else np.concatenate([x[i], [w]])
        out[i], _ = nnls(design, target)
    if sum_to_one:
        out = out / (out.sum(axis=1, keepdims=True) + 1e-12)
    return out


def vca_unmix(x: np.ndarray, k: int, seed: int = 0) -> Decomposition:
    """VCA endmember extraction followed by constrained least-squares abundances.

    Args:
        x: Data, shape (n_pixels, n_channels).
        k: Number of endmembers.
        seed: Seed for VCA's random directions.

    Returns:
        Decomposition with pixel-spectrum endmembers (normalized to unit mean)
        and simplex-constrained abundances.
    """
    spectra = vca(x, k, seed)
    scale = spectra.mean(axis=1, keepdims=True) + 1e-12
    spectra_n = spectra / scale
    abund = solve_abundances(x, spectra_n, sum_to_one=True)
    recon = abund @ spectra_n * (np.asarray(x).sum(axis=1, keepdims=True) / spectra_n.shape[1])
    err = float(np.linalg.norm(np.asarray(x, dtype=np.float64) - recon))
    return Decomposition(spectra_n, abund, err, {"pixel_indices": None})
