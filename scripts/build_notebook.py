"""Build and execute notebooks/tutorial.ipynb from source cells.

The notebook is generated programmatically so it stays reproducible, then
executed in place with nbclient (every committed output cell is real):

    python scripts/build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

MD = "markdown"
CODE = "code"

CELLS: list[tuple[str, str]] = [
    (
        MD,
        """# Unmixing a simulated STEM-EELS spectrum image

A spectrum image is a datacube: at every probe position of a STEM scan, a full
energy-loss spectrum. When the specimen contains a few phases, each pixel
spectrum is, to first order, a non-negative mixture of a few phase spectra
(endmembers), weighted by how much of each phase the beam crossed. Recovering
the endmembers and their abundance maps from the cube alone is the unmixing
problem, and because real maps never come with ground truth, the only way to
know how well an algorithm does is to simulate.

This notebook walks the full path: simulate a three-phase oxide scene with
exact ground truth, look at the data, choose the number of components, unmix
with NMF, VCA, and a constrained autoencoder, and score everything against
the known answer. The condition used here is the repository's headline
operating point: dose 200 counts per pixel spectrum, a photon-starved but
realistic budget for core-loss mapping.""",
    ),
    (
        CODE,
        """import numpy as np
from IPython.display import Image

from eelsunmix import SimConfig, simulate

scene = simulate(SimConfig(dose=200, seed=0))
print("cube:", scene.cube.shape, "counts, mean total per px:", round(scene.cube.sum(2).mean(), 1))
print("phases:", scene.names)
print("energy axis:", scene.energy_ev[0], "to", scene.energy_ev[-1], "eV")""",
    ),
    (
        MD,
        """## Step 1: the scene

The specimen is a lamella with a titanium oxide grain and a manganese oxide
grain meeting at a diffuse, slightly wavy interface, plus an iron oxide
precipitate straddling it. Each phase spectrum is built from a power-law
background and its ionization edges at tabulated energies: Ti L2,3 at 456 eV,
O K at 532 eV, Mn L2,3 at 640 eV, Fe L2,3 at 708 eV, with Gaussian white
lines and per-element L3/L2 ratios. All three phases share the O K edge, so
the endmembers are correlated the way real oxide spectra are. The cube is
finished with a smooth per-pixel energy drift (1.5 channels peak-to-peak)
and Poisson counting noise set by the dose.""",
    ),
    (
        CODE,
        """from eelsunmix.plots import plot_scene

plot_scene(scene, "nb_figs/scene.png")
Image("nb_figs/scene.png")""",
    ),
    (
        MD,
        """## Step 2: how many components?

On real data this is the first decision, and it is made without ground truth.
The PCA eigenvalue scree is the standard tool: signal components carry large
explained variance, noise components form a flat floor. Three phases plus
Poisson noise should give three components above the floor (drift blurs this
slightly, since a shifted spectrum is not exactly in the three-dimensional
span).""",
    ),
    (
        CODE,
        """import matplotlib.pyplot as plt

from eelsunmix import pca_scree

evr = pca_scree(scene.flat, kmax=10)
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(range(1, 11), evr, marker="o", color="#333333")
ax.axvline(3, color="#d55e00", ls="--", lw=1, label="true k = 3")
ax.set_xlabel("principal component")
ax.set_ylabel("explained variance ratio")
ax.set_title("PCA scree at dose 200")
ax.legend()
fig.savefig("nb_figs/scree.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("explained variance ratios:", np.round(evr[:6], 5))
Image("nb_figs/scree.png")""",
    ),
    (
        MD,
        """The mean spectrum dominates PC1; the scree drops after the third component
and flattens into the Poisson floor, so k = 3 is the defensible choice, and
it is what the rest of the notebook uses. The benchmark's `components` config
quantifies what happens when k is set wrong.

## Step 3: NMF

Non-negative matrix factorization is the workhorse: factor the cube into
non-negative spectra and non-negative loadings. Two honest caveats come with
it: the answer depends on initialization (the benchmark's `stability` config
measures that spread), and plain NMF does not know abundances should sum to
one. Here it runs with the deterministic nndsvda initialization.""",
    ),
    (
        CODE,
        """from eelsunmix import match_endmembers, nmf_unmix

nmf_dec = nmf_unmix(scene.flat, k=3, seed=0)
nmf_match = match_endmembers(nmf_dec.spectra, scene.endmembers)
for t, sad in zip(nmf_match.true_index, nmf_match.sad_deg):
    print(f"{scene.names[t]:>9s}: spectral angle {sad:5.2f} deg")
print(f"mean: {nmf_match.mean_sad_deg:.2f} deg")""",
    ),
    (
        MD,
        """## Step 4: VCA

Vertex component analysis takes the geometric view: under the linear mixing
model, pixel spectra live in a simplex whose vertices are the endmembers, so
it hunts for the most extreme pixels. That works exactly when near-pure
pixels exist. In this scene the grain interiors are nearly pure but carry a
deliberate 8% mixing field, so VCA's endmembers are real pixel spectra,
Poisson noise included; abundances then come from non-negative least squares
with the sum-to-one constraint.""",
    ),
    (
        CODE,
        """from eelsunmix import vca_unmix

vca_dec = vca_unmix(scene.flat, k=3, seed=0)
vca_match = match_endmembers(vca_dec.spectra, scene.endmembers)
for t, sad in zip(vca_match.true_index, vca_match.sad_deg):
    print(f"{scene.names[t]:>9s}: spectral angle {sad:5.2f} deg")
print(f"mean: {vca_match.mean_sad_deg:.2f} deg")""",
    ),
    (
        MD,
        """## Step 5: the constrained autoencoder

The autoencoder's decoder is the linear mixing model itself: the encoder MLP
maps each spectrum to k numbers, a softmax turns them into abundances
(non-negative, sum to one), and reconstruction multiplies them by a softplus
parametrized endmember matrix (non-negative). The network physically cannot
represent anything but a simplex mixture of non-negative spectra. Compared
with NMF it adds the sum-to-one constraint and denoises through the encoder;
like NMF it is non-convex, so its seed sensitivity is part of the benchmark,
not hidden.""",
    ),
    (
        CODE,
        """from eelsunmix import AETrainConfig, ae_unmix
from eelsunmix.plots import plot_loss_curve

ae_dec = ae_unmix(scene.flat, AETrainConfig(k=3, epochs=1200, seed=0))
plot_loss_curve(ae_dec.meta["loss_curve"], "nb_figs/loss.png")
ae_match = match_endmembers(ae_dec.spectra, scene.endmembers)
for t, sad in zip(ae_match.true_index, ae_match.sad_deg):
    print(f"{scene.names[t]:>9s}: spectral angle {sad:5.2f} deg")
print(f"mean: {ae_match.mean_sad_deg:.2f} deg")
Image("nb_figs/loss.png")""",
    ),
    (
        MD,
        """## Step 6: recovered endmembers and abundance maps against truth

The figure below is the whole story in one place: each recovered endmember
overlaid on the true spectrum (with its spectral angle), then the true
abundance maps and each method's matched maps.""",
    ),
    (
        CODE,
        """from eelsunmix.plots import plot_hero

plot_hero(scene, {"nmf": nmf_dec, "vca": vca_dec, "ae": ae_dec}, "nb_figs/hero.png")
Image("nb_figs/hero.png")""",
    ),
    (
        MD,
        """## Step 7: the final score

Spectral angle (degrees, after optimal matching; lower is better) for the
endmembers, RMSE for the sum-to-one abundance maps, and the largest principal
angle between the true endmember span and each method's component span.""",
    ),
    (
        CODE,
        """from eelsunmix import abundance_rmse, subspace_error_deg

rows = []
for name, dec, match in (
    ("nmf", nmf_dec, nmf_match),
    ("vca", vca_dec, vca_match),
    ("ae", ae_dec, ae_match),
):
    rows.append(
        (
            name,
            match.mean_sad_deg,
            abundance_rmse(dec.abundances.T, scene.abundances, match),
            subspace_error_deg(scene.endmembers, dec.spectra),
        )
    )
print(f"{'method':>8s} {'SAD (deg)':>10s} {'abund RMSE':>11s} {'subspace (deg)':>15s}")
for name, sad, rmse, sub in rows:
    print(f"{name:>8s} {sad:10.2f} {rmse:11.4f} {sub:15.2f}")""",
    ),
    (
        MD,
        """## What to take away, and what not to

One seed at one dose is a demonstration, not a result. The committed
benchmark configs in `configs/` run the same comparison across doses (10 to
5000 counts), drift amplitudes, spectral-overlap levels, component counts,
and 10 seeds, with best-of-N variants for the initialization-sensitive
methods; `RESULTS.md` reads those numbers out and is the citable summary.

Everything here is simulation with a deliberately simple physical model: no
plural scattering, no spectrometer point-spread function, no channel gain
variation. The value of the synthetic route is exact ground truth, which
real spectrum images never have; absolute numbers will not transfer to any
particular instrument, but the method ranking and its failure modes
(initialization spread, overlap collapse, drift sensitivity) are the
transferable part.""",
    ),
]


def main() -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    for kind, source in CELLS:
        if kind == MD:
            nb.cells.append(nbformat.v4.new_markdown_cell(source))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(source))

    out = Path("notebooks/tutorial.ipynb")
    out.parent.mkdir(exist_ok=True)
    (out.parent / "nb_figs").mkdir(exist_ok=True)

    client = NotebookClient(nb, timeout=1800, resources={"metadata": {"path": str(out.parent)}})
    client.execute()
    nbformat.write(nb, out)
    n_err = sum(
        1
        for c in nb.cells
        if c.cell_type == "code"
        for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    )
    print(f"wrote {out} ({len(nb.cells)} cells, {n_err} error outputs)")


if __name__ == "__main__":
    main()
