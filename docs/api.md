# eelsunmix Python API

Every example below is runnable as-is from the repository root after
installation. The library works on plain NumPy arrays: a spectrum image is a
cube shaped `(ny, nx, n_channels)`, and every unmixing method consumes the
flattened `(n_pixels, n_channels)` matrix.

## Simulate a scene with ground truth

```python
from eelsunmix import SimConfig, simulate

scene = simulate(SimConfig(dose=200, drift_channels=1.5, seed=0))
print(scene.cube.shape)        # (64, 64, 200) noisy counts
print(scene.names)             # ['Ti oxide', 'Mn oxide', 'Fe oxide']
print(scene.endmembers.shape)  # (3, 200) exact endmember spectra
print(scene.abundances.shape)  # (3, 64, 64) exact abundance maps, simplex
```

`SimConfig` exposes the physically meaningful knobs: `dose` (mean counts per
pixel spectrum before Poisson noise), `drift_channels` (peak-to-peak energy
drift), `edge_separation_ev` (Mn L3 to Fe L3 onset distance; shrink it to make
the endmembers collinear), `interface_width_px`, and `n_phases` (2 or 3).

## Unmix with the classical baselines

```python
from eelsunmix import SimConfig, nmf_unmix, simulate, vca_unmix

scene = simulate(SimConfig(dose=1000, seed=0))
x = scene.flat                          # (4096, 200)

nmf = nmf_unmix(x, k=3, seed=0)         # deterministic nndsvda init
nmf10 = nmf_unmix(x, k=3, seed=0, n_restarts=10)  # best of 10 random inits
vca = vca_unmix(x, k=3, seed=0)         # VCA + constrained least squares

print(nmf.spectra.shape)      # (3, 200), unit-mean normalized
print(nmf.abundances.shape)   # (4096, 3)
```

Restart selection uses only the reconstruction error, never ground truth, so
the same procedure applies to real data.

## Unmix with the constrained autoencoder

```python
from eelsunmix import AETrainConfig, SimConfig, ae_unmix, simulate

scene = simulate(SimConfig(dose=1000, seed=0))
dec = ae_unmix(scene.flat, AETrainConfig(k=3, epochs=1200, seed=0), n_seeds=1)
print(dec.spectra.shape, dec.abundances.shape)
print(dec.meta["train_config"])
```

The decoder is the linear mixing model itself: abundances pass through a
softmax (non-negative, sum to one) and endmembers through a softplus
(non-negative), so the factors are physical by construction.

## Score against ground truth

```python
from eelsunmix import SimConfig, abundance_rmse, match_endmembers, nmf_unmix, simulate

scene = simulate(SimConfig(dose=1000, seed=0))
dec = nmf_unmix(scene.flat, k=3, seed=0)
match = match_endmembers(dec.spectra, scene.endmembers)   # Hungarian on SAD
print(match.mean_sad_deg)                                  # degrees
print(abundance_rmse(dec.abundances.T, scene.abundances, match))
```

Matching is an optimal one-to-one assignment on the spectral-angle matrix, so
no method is penalized for component order. `subspace_error_deg` is the fair
score for PCA, whose components are a basis rather than physical endmembers.

## Run a benchmark config

```python
from eelsunmix import run_config

payload = run_config("configs/operating_point.yaml", "results")
print(payload["summary"])
```

## Bring your own data

```python
import numpy as np
from eelsunmix import load_cube, nmf_unmix

cube, energy = load_cube("your_map.npy")   # (ny, nx, n_channels), counts
x = cube.reshape(-1, cube.shape[2])
dec = nmf_unmix(x, k=3, seed=0, n_restarts=10)
maps = dec.abundances.T.reshape(3, cube.shape[0], cube.shape[1])
np.savez("unmixed.npz", spectra=dec.spectra, abundances=maps)
```

See `examples/unmix_your_own_map.py` for the full recipe, including how to
convert vendor formats with HyperSpy and what preprocessing matters
(non-negative counts, no gain-corrected negative values, energy axis cropped
to the core-loss region of interest).
