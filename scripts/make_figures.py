"""Regenerate every committed figure from the committed configs and results.

Run from the repository root after `eelsunmix benchmark configs/<name>.yaml`
has produced the JSONs in results/:

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

from eelsunmix.autoencoder import decomposition_from_model, load_model
from eelsunmix.benchmark import run_method
from eelsunmix.io import load_scene
from eelsunmix.plots import (
    plot_ae_epochs,
    plot_components,
    plot_hero,
    plot_scene,
    plot_stability,
    plot_sweep,
)
from eelsunmix.sim import SimConfig, simulate

RESULTS = Path("results")
FIGURES = Path("figures")
SAMPLE = Path("data/sample/oxide_interface_d1000.npz")
MODEL = Path("models/autoencoder.pt")


def _load(name: str) -> dict:
    with open(RESULTS / f"{name}.json", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)

    for name in ("dose_sweep", "drift_sweep", "overlap_sweep"):
        plot_sweep(_load(name), FIGURES / f"{name}.png")
        print(f"wrote figures/{name}.png")
    plot_components(_load("components"), FIGURES / "components.png")
    print("wrote figures/components.png")
    plot_stability(_load("stability"), FIGURES / "stability.png")
    print("wrote figures/stability.png")
    plot_ae_epochs(_load("ae_epochs"), FIGURES / "ae_epochs.png")
    print("wrote figures/ae_epochs.png")

    # Scene overview at the headline operating point (dose 200).
    plot_scene(simulate(SimConfig(dose=200, seed=0)), FIGURES / "scene_overview.png")
    print("wrote figures/scene_overview.png")

    # Hero: all methods on the committed dose-1000 sample; the committed
    # autoencoder is used as-is so the hero matches the shipped artifact.
    scene = load_scene(SAMPLE)
    decs = {
        "nmf": run_method(scene.flat, {"kind": "nmf", "name": "nmf"}, 0),
        "vca": run_method(scene.flat, {"kind": "vca", "name": "vca"}, 0),
        "ae": decomposition_from_model(load_model(str(MODEL)), scene.flat),
    }
    plot_hero(scene, decs, FIGURES / "hero_unmixing.png")
    print("wrote figures/hero_unmixing.png")
    # figures/training_loss.png is written at training time:
    # eelsunmix train --loss-figure figures/training_loss.png


if __name__ == "__main__":
    main()
