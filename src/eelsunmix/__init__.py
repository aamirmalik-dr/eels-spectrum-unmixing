"""eelsunmix: spectral unmixing benchmark for simulated STEM-EELS spectrum images.

Simulate a three-phase oxide scene with exact ground truth, decompose it with
PCA, NMF, VCA, or a constrained linear-unmixing autoencoder, and score the
recovered endmember spectra and abundance maps against truth.
"""

from .autoencoder import AETrainConfig, UnmixingAE, ae_unmix, load_model, train_autoencoder
from .benchmark import run_config, run_method, score
from .io import load_cube, load_scene, save_scene
from .methods import (
    Decomposition,
    nmf_unmix,
    pca_decompose,
    pca_scree,
    solve_abundances,
    vca,
    vca_unmix,
)
from .metrics import (
    MatchResult,
    abundance_rmse,
    match_endmembers,
    reconstruction_r2,
    spectral_angle_deg,
    subspace_error_deg,
)
from .sim import EELSScene, SimConfig, oxide_phase_specs, render_endmember, simulate

__version__ = "0.1.0"

__all__ = [
    "AETrainConfig",
    "Decomposition",
    "EELSScene",
    "MatchResult",
    "SimConfig",
    "UnmixingAE",
    "abundance_rmse",
    "ae_unmix",
    "load_cube",
    "load_model",
    "load_scene",
    "match_endmembers",
    "nmf_unmix",
    "oxide_phase_specs",
    "pca_decompose",
    "pca_scree",
    "reconstruction_r2",
    "render_endmember",
    "run_config",
    "run_method",
    "save_scene",
    "score",
    "simulate",
    "solve_abundances",
    "spectral_angle_deg",
    "subspace_error_deg",
    "train_autoencoder",
    "vca",
    "vca_unmix",
    "__version__",
]
