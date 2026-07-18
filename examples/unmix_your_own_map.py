"""Unmix your own STEM-EELS or EDX spectrum image (bring-your-own-data recipe).

The library never needs ground truth to run; only the benchmark does. Point
this script at a 3D array saved as .npy or .npz and it writes recovered
endmember spectra and abundance maps next to the input.

Converting vendor formats first (HyperSpy is not a dependency of this
package; install it separately if you need it):

    import hyperspy.api as hs
    import numpy as np
    signal = hs.load("map.dm4")
    np.savez("map.npz", cube=signal.data,
             energy_ev=signal.axes_manager[-1].axis)

Preprocessing that matters before unmixing:
    - Counts must be non-negative; clip gain-correction artifacts at zero.
    - Crop the energy axis to the region containing the edges you care
      about; a huge zero-loss tail swamps the factorization.
    - Do not pre-normalize per pixel; the methods handle scale.

Usage:
    python examples/unmix_your_own_map.py map.npz --k 3 --method nmf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eelsunmix import ae_unmix, load_cube, nmf_unmix, vca_unmix
from eelsunmix.autoencoder import AETrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="your .npy or .npz spectrum image, (ny, nx, channels)")
    parser.add_argument("--k", type=int, default=3, help="number of components to extract")
    parser.add_argument("--method", choices=("nmf", "vca", "ae"), default="nmf")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cube, energy = load_cube(args.path)
    ny, nx, nc = cube.shape
    x = cube.reshape(-1, nc)
    if energy is None:
        energy = np.arange(nc, dtype=float)
        energy_label = "channel"
    else:
        energy_label = "energy loss (eV)"

    if args.method == "nmf":
        dec = nmf_unmix(x, args.k, seed=args.seed, n_restarts=10)
    elif args.method == "vca":
        dec = vca_unmix(x, args.k, seed=args.seed)
    else:
        dec = ae_unmix(x, AETrainConfig(k=args.k, seed=args.seed), n_seeds=3)

    maps = dec.abundances.T.reshape(args.k, ny, nx)
    stem = Path(args.path).with_suffix("")
    out_npz = f"{stem}_unmixed.npz"
    np.savez_compressed(out_npz, spectra=dec.spectra, abundances=maps, energy=energy)

    fig, axes = plt.subplots(1, args.k + 1, figsize=(3.2 * (args.k + 1), 3.0))
    for i in range(args.k):
        axes[0].plot(energy, dec.spectra[i], label=f"component {i}")
        im = axes[i + 1].imshow(maps[i], cmap="magma")
        axes[i + 1].set_title(f"component {i}")
        axes[i + 1].set_xticks([])
        axes[i + 1].set_yticks([])
        fig.colorbar(im, ax=axes[i + 1], fraction=0.046)
    axes[0].set_xlabel(energy_label)
    axes[0].set_ylabel("intensity")
    axes[0].legend(fontsize=8)
    out_png = f"{stem}_unmixed.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_npz} and {out_png}")
    print("no ground truth available for real data: inspect the spectra for")
    print("physically sensible edges and the maps for spatial coherence")


if __name__ == "__main__":
    main()
