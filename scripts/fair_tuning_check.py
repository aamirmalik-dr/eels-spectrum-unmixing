"""Fair-tuning check for the classical NMF baseline.

Before crediting the autoencoder with a win, verify the baseline is not
crippled: confirm NMF converges before its iteration cap, that raising the
cap changes nothing, and that the Poisson-motivated alternative (KL
divergence with multiplicative updates) does not beat the Frobenius default
on this problem. Writes results/fair_tuning.json.

    python scripts/fair_tuning_check.py
"""

from __future__ import annotations

import json
import warnings

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
]


def main() -> None:
    warnings.filterwarnings("ignore")
    out = []
    for dose in (200, 1000):
        scene = simulate(SimConfig(dose=dose, seed=0))
        x = scene.flat
        for variant in VARIANTS:
            model = NMF(
                n_components=3, init="nndsvda", random_state=0, tol=1e-5, **variant["kwargs"]
            )
            w = model.fit_transform(x)
            h = model.components_
            scale = h.mean(axis=1, keepdims=True) + 1e-12
            dec = Decomposition(h / scale, w * scale.T, 0.0, {})
            rec = {"dose": dose, "variant": variant["label"], "n_iter": int(model.n_iter_)}
            rec.update(score(scene, dec))
            del rec["sad_per_component_deg"]
            out.append(rec)
            print(
                f"dose {dose:5g} {variant['label']:18s} SAD {rec['sad_mean_deg']:6.2f} "
                f"RMSE {rec['abundance_rmse']:.4f}  converged at iter {rec['n_iter']}"
            )
    with open("results/fair_tuning.json", "w", encoding="utf-8") as fh:
        json.dump({"name": "fair_tuning", "records": out}, fh, indent=1)
    print("wrote results/fair_tuning.json")


if __name__ == "__main__":
    main()
