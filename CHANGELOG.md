# Changelog

## 0.1.1 (2026-09-26)

- Package metadata completed: keywords, classifiers, and project URLs in `pyproject.toml`.
- Citation file (`CITATION.cff`) and this changelog added.
- CI badge added to the README.
- Related repositories section in the README linking the six sibling electron-microscopy repositories.
- Test guarding the package `__version__` against the installed distribution metadata.

## 0.1.0 (2026-07-18)

- STEM-EELS spectrum-image simulator: endmembers from power-law backgrounds and tabulated ionization edges with white lines, a three-phase oxide scene with a diffuse interface, smooth per-pixel energy drift, and Poisson noise set by one dose parameter; exact endmembers, abundances, and drift field returned with every cube.
- Four unmixing methods: PCA, NMF with a restart protocol, a from-scratch VCA with simplex-constrained NNLS abundances, and a PyTorch autoencoder whose decoder is the linear mixing model.
- Scoring by Hungarian-matched spectral angle, sum-to-one abundance RMSE, and principal-angle subspace error; benchmarks over dose, drift, spectral overlap, component count, autoencoder training length, and initialization stability.
- Fair-tuning audit of the NMF baseline (convergence, iteration cap, KL versus Frobenius, whitened and sum-to-one probes) committed as its own script and JSON.
- The `eelsunmix` CLI, a bring-your-own-map example, committed autoencoder weights and ground-truth sample cube, results JSON, figures, model card, API docs, executed tutorial, and a CI workflow.
- Maintenance after publication: stored scene config parsed with `ast.literal_eval`, figure layout defects fixed, README expanded and restructured, hero re-exported as RGB.
