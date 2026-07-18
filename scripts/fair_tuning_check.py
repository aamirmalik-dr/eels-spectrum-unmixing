"""Fair-tuning check for the classical NMF baseline.

Before crediting the autoencoder with a win, verify the baseline is not
crippled. The checks, all scored against ground truth on the same scenes
the benchmark uses:

- NMF converges before its iteration cap on the headline nndsvda runs, and
  raising the cap 4x changes nothing.
- The Poisson-motivated alternative, KL-divergence NMF with multiplicative
  updates, does not beat the Frobenius default.
- Poisson noise-whitened NMF (each channel scaled by 1/sqrt of its mean
  count, which preserves the linear mixing model exactly; endmembers are
  un-scaled afterwards) does not beat the plain fit, so the gap is not an
  unweighted-loss artifact.
- Sum-to-one NMF via the standard augmented-column device does not close
  the gap at any useful constraint weight, so the autoencoder's win is not
  just "NMF denied a constraint".

Writes results/fair_tuning.json.

    python scripts/fair_tuning_check.py
"""

from __future__ import annotations

import json
import warnings

import numpy as np
from sklearn.decomposition import NMF

from eelsunmix.benchmark import score
from eelsunmix.methods import Decomposition
from eelsunmix.sim import SimConfig, simulate

VARIANTS = [
    {"label": "frobenius_cd_1000", "kwargs": {"max_iter": 1000}},
    {"label": "frobenius_cd_4000", "kwargs": {"max_iter": 4000}},
    {
        "label": "kl_mu_3000",
        "kwargs": {"beta_loss": "kullback-leibler", "solver": "mu", "max_iter": 3000},
    },
    {"label": "poisson_whitened_4000", "kwargs": {"max_iter": 4000}, "whiten": True},
    {"label": "sum_to_one_delta20_4000", "kwargs": {"max_iter": 4000}, "sum_to_one_delta": 20.0},
    {"label": "sum_to_one_delta100_4000", "kwargs": {"max_iter": 4000}, "sum_to_one_delta": 100.0},
]


def run_variant(x: np.ndarray, variant: dict) -> tuple[Decomposition, int]:
    """Fit one NMF variant and map its factors back to raw-count space.

    Args:
        x: Data, shape (n_pixels, n_channels), raw counts.
        variant: Entry from VARIANTS.

    Returns:
        Tuple of (Decomposition in raw-count space, iterations used).
    """
    n_channels = x.shape[1]
    fit_x = x
    channel_scale = None
    if variant.get("whiten"):
        channel_scale = 1.0 / np.sqrt(x.mean(axis=0) + 1e-12)
        fit_x = x * channel_scale[None, :]
    if variant.get("sum_to_one_delta"):
        delta = float(variant["sum_to_one_delta"]) * float(x.mean())
        fit_x = np.hstack([x, np.full((x.shape[0], 1), delta)])
    model = NMF(n_components=3, init="nndsvda", random_state=0, tol=1e-5, **variant["kwargs"])
    w = model.fit_transform(fit_x)
    h = model.components_[:, :n_channels]
    if channel_scale is not None:
        h = h / channel_scale[None, :]
    scale = h.mean(axis=1, keepdims=True) + 1e-12
    return Decomposition(h / scale, w * scale.T, 0.0, {}), int(model.n_iter_)


def main() -> None:
    warnings.filterwarnings("ignore")
    out = []
    for dose in (200, 1000):
        scene = simulate(SimConfig(dose=dose, seed=0))
        x = scene.flat
        for variant in VARIANTS:
            dec, n_iter = run_variant(x, variant)
            rec = {"dose": dose, "variant": variant["label"], "n_iter": n_iter}
            rec.update(score(scene, dec))
            del rec["sad_per_component_deg"]
            out.append(rec)
            print(
                f"dose {dose:5g} {variant['label']:24s} SAD {rec['sad_mean_deg']:6.2f} "
                f"RMSE {rec['abundance_rmse']:.4f}  stopped at iter {rec['n_iter']}"
            )
    with open("results/fair_tuning.json", "w", encoding="utf-8") as fh:
        json.dump({"name": "fair_tuning", "records": out}, fh, indent=1)
    print("wrote results/fair_tuning.json")


if __name__ == "__main__":
    main()
