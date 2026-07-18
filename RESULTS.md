# Results

Every number on this page was measured this session in a fresh virtual
environment by running the committed configs; raw values are in
`results/*.json` and every figure and table regenerates with
`python scripts/run_all.py`. Scenes are 64 x 64 x 200 simulated STEM-EELS
cubes (three oxide phases, shared O K edge, Poisson noise, 1.5 channels of
energy drift unless the sweep varies it), and all methods fit k = 3 unless
stated. Metrics: mean spectral angle (SAD, degrees, after optimal Hungarian
matching, lower is better), abundance RMSE (on per-pixel sum-to-one maps),
and subspace error (largest principal angle between the true endmember span
and the recovered span). PCA is scored only on subspace error and SAD: its
scores are signed coordinates, not abundances, so an abundance RMSE for PCA
would be a large but meaningless number (see `benchmark.score`).

## Headline operating point (dose 200 counts per spectrum, 3 scene seeds)

`configs/operating_point.yaml`, `results/operating_point.json`.

| Method | SAD (deg) | Abundance RMSE | Subspace error (deg) | Time (s) |
|---|---|---|---|---|
| PCA (k = 3) | 83.73 +/- 0.62 | n/a | 27.9 | 0.5 |
| NMF (nndsvda) | 17.22 +/- 0.59 | 0.259 | 30.6 | 1.3 |
| NMF, best of 10 restarts | 17.29 +/- 0.52 | 0.260 | 30.6 | 7.7 |
| VCA + constrained NNLS | 44.26 +/- 0.92 | 0.354 | 64.4 | 0.1 |
| Autoencoder (1200 epochs) | **8.51 +/- 0.67** | **0.166** | 28.5 | 11.9 |
| Autoencoder, best of 5 seeds | 8.67 +/- 0.56 | 0.168 | 27.5 | 49.6 |

Readings, in order of importance:

1. **The constrained autoencoder halves NMF's endmember error at this dose**
   (8.5 vs 17.2 deg) and improves abundance RMSE by a third. The gap is not
   a subspace effect: all three factorization methods sit at a similar
   subspace error (28 to 31 deg), so they find roughly the same signal
   subspace, and the autoencoder's win is in placing the endmembers inside
   it, which is exactly what its simplex and non-negativity constraints
   are for.
2. **Extra selection effort buys nothing here.** Best-of-10 NMF restarts and
   best-of-5 autoencoder seeds, both selected by reconstruction error only
   (the signal available on real data), reproduce the single-run scores to
   within 0.2 deg. Fit error cannot rank solutions that fit almost equally
   well; see the stability section for the one failure mode it does catch.
3. **VCA is the wrong tool at this dose,** and predictably so: it returns
   actual pixel spectra as endmembers, and at 200 counts per spectrum every
   pixel spectrum is noise-dominated (44 deg is mostly the angle of the
   Poisson noise itself). Its abundance maps still show the phase geometry
   (figures/hero_unmixing.png) because constrained least squares averages
   over channels.
4. **PCA's 83.7 deg SAD is the documented expectation, not a failure:**
   orthogonal components cannot be one-sided physical spectra. Its subspace
   error (27.9 deg, mean-augmented basis) matches the other methods, which
   is the sense in which PCA "works" and why it remains the right rank
   diagnostic.

## Dose sweep (10 to 5000 counts per spectrum, 3 seeds per point)

`configs/dose_sweep.yaml`, figures/dose_sweep.png.

SAD (deg):

| Dose | NMF | VCA | AE |
|---|---|---|---|
| 10 | 48.5 | 75.1 | **36.7** |
| 50 | 30.8 | 60.6 | **19.6** |
| 200 | 17.2 | 44.3 | **8.5** |
| 1000 | 13.7 | 23.4 | **6.9** |
| 5000 | 13.4 | 13.1 | **7.2** |

- The autoencoder leads at every dose; the margin is largest exactly where
  EELS mapping actually operates (tens to hundreds of counts).
- VCA improves fastest with dose, from useless (75 deg at dose 10) to
  matching NMF at dose 5000 (13.1 vs 13.4), because its pure-pixel
  assumption becomes progressively true as pixel spectra denoise. At high
  dose the interesting competition is VCA vs NMF, not either vs the
  autoencoder.
- NMF and the autoencoder both saturate above dose 1000 (13.4 and ~7 deg).
  That floor is not photon statistics; the drift sweep shows what it is.

## Energy-drift sweep (0 to 8 channels peak-to-peak, dose 1000, 3 seeds)

`configs/drift_sweep.yaml`, figures/drift_sweep.png.

| Drift (channels) | NMF | VCA | AE |
|---|---|---|---|
| 0 | 10.4 | 22.7 | **2.7** |
| 1 | 12.1 | 23.0 | **5.2** |
| 2 | 14.8 | 24.7 | **8.3** |
| 4 | 18.6 | 26.1 | **12.2** |
| 8 | 20.7 | 28.0 | **14.7** |

With drift removed the autoencoder recovers endmembers to 2.7 deg, so the
~7 deg high-dose floor above is almost entirely model mismatch: a drifted
spectrum is not a linear mixture of the three endmembers, and every method
built on the linear mixing model pays. Drift costs the autoencoder roughly
2.5 to 3 deg per channel over the first two channels, flattening beyond,
and NMF a similar slope from a worse starting point. For a spectroscopist the practical
reading is the usual one: align your spectra before unmixing; no
factorization will do it for you.

## Spectral-overlap sweep (Fe L3 moved toward Mn L3, dose 1000, 3 seeds)

`configs/overlap_sweep.yaml`, figures/overlap_sweep.png. The physical
Mn L3 to Fe L3 separation is 68 eV; the sweep compresses it to 8 eV, making
the two metal-oxide endmembers progressively collinear (all phases already
share the O K edge).

| Separation (eV) | NMF SAD | AE SAD | NMF abund. RMSE | AE abund. RMSE |
|---|---|---|---|---|
| 68 | 13.7 | 6.9 | 0.185 | 0.074 |
| 40 | 13.9 | 8.4 | 0.191 | 0.123 |
| 25 | 13.6 | 7.4 | 0.193 | 0.088 |
| 15 | 13.9 | 9.5 | 0.201 | 0.156 |
| 8 | 14.0 | 10.3 | 0.209 | 0.201 |

The surprise is how mild the spectral collapse is: even at 8 eV separation
the endmember SAD degrades only a few degrees, because the O K fine
structure and background exponents still separate the phases. The damage
shows up in the abundance maps instead; the autoencoder's abundance RMSE
almost triples (0.074 to 0.201) as the mixing matrix becomes
ill-conditioned. Recovering spectra and recovering maps are different
problems, and overlap hurts the maps first.

## How many components? (dose 1000, seed 0)

`configs/components.yaml`, figures/components.png.

PCA scree (explained variance ratio): 0.149, 0.047, 0.0094, then a flat
noise floor at 0.0078, 0.0073, 0.0067, ... The k = 3 elbow is present but
modest, a factor of 1.2 above the floor, not a cliff; energy drift spreads
signal variance into what would otherwise be noise components. Anyone who
has stared at a real scree plot will recognize this ambiguity, and it is
the honest state of affairs, not a defect of the plot.

| k fitted | NMF SAD | AE SAD | AE recon. error |
|---|---|---|---|
| 2 | 9.5 (2 matched) | 15.3 (2 matched) | 2254 |
| 3 | 14.3 | **7.5** | 2026 |
| 4 | 16.9 | 16.9 | 2254 |
| 5 | 20.2 | 7.6 | 2026 |
| 6 | 23.2 | 16.9 | 2254 |

Two different failure styles:

- NMF degrades gracefully and monotonically as k grows past the truth: it
  splits real components to spend the extra factors.
- The autoencoder is bimodal. At k = 4 and 6 (seed 0) it collapses onto an
  effectively two-component solution: reconstruction error 2254, identical
  to the k = 2 fit, with the extra components dead. At k = 5 it lands in
  the good three-component basin (7.6 deg, error 2026). Which basin it
  finds is initialization luck, and this is the same collapse mode the
  stability section quantifies. The saving grace: the two basins are 11%
  apart in reconstruction error, so the collapse is detectable without
  ground truth.

## Seed stability (dose 1000, one fixed scene, 10 method seeds)

`configs/stability.yaml`, figures/stability.png.

| Method | SAD mean +/- std | min | max |
|---|---|---|---|
| NMF (random init) | 14.7 +/- 0.6 | 13.5 | 15.7 |
| VCA | 25.7 +/- 1.2 | 23.5 | 26.9 |
| Autoencoder | 10.3 +/- 4.3 | 7.5 | 17.0 |

The autoencoder's spread is the important number. Seven of ten seeds land
in the good basin (7.5 to 7.6 deg); three collapse to the two-component
solution (16.6 to 17.0 deg, abundance RMSE 0.57). The collapsed runs are
cleanly identifiable by fit: reconstruction error 2254 vs 2026, an 11% gap
with nothing in between. So the honest protocol for the autoencoder is
best-of-N with selection by reconstruction error, which reliably rejects
collapse, and the honest claim is conditional: with that protocol its
expected SAD is ~7.5 deg here; a single unlucky seed gives 17. NMF never
collapses on this scene; its 2 deg spread is the cost it never pays in
catastrophic form.

## Autoencoder training length (dose 200, 3 seeds per point)

`configs/ae_epochs.yaml`, figures/ae_epochs.png.

| Epochs | SAD (deg) | Recon. error |
|---|---|---|
| 300 | 13.3 | 918.3 |
| 600 | 11.2 | 909.8 |
| 1200 | **9.3** | 902.2 |
| 2400 | 12.7 | 901.2 |
| 4800 | 15.4 | 901.1 |

Reconstruction error decreases monotonically; endmember recovery peaks at
1200 epochs and then degrades by 6 deg as the network starts spending its
capacity fitting Poisson noise through the abundance maps. There is no
ground-truth-free signal that finds the optimum, because fit keeps
improving past it. The shipped default of 1200 epochs is the best of the
sampled settings on this scene family, and that is a disclosed,
benchmark-derived choice, not a neutral one; on different data the optimum
will move and nothing inside the method will tell you where it went. This is the
autoencoder's most important honest caveat.

## Was the classical baseline fairly tuned?

`scripts/fair_tuning_check.py`, `results/fair_tuning.json`. Checks run
before crediting the autoencoder's win, all on the seed-0 scenes:

- The headline nndsvda NMF runs converge before the iteration cap (818
  iterations at dose 200, 444 at dose 1000, cap 1000); raising the cap to
  4000 changes nothing. Random-init restart runs inside the best-of-10
  variant can hit the cap (sklearn warns about it), but rerunning all 10
  restarts with a 4x cap leaves the selected run's score unchanged to two
  decimals on every scene seed, so nothing was left unconverged that
  selection could have used.
- The Poisson-motivated alternative, KL-divergence NMF with multiplicative
  updates, is worse on both doses (19.9 vs 17.8 deg at dose 200, 16.8 vs
  14.3 at dose 1000), so the Frobenius default is the stronger baseline
  and is the one benchmarked.
- Poisson noise-whitening (scale each channel by 1/sqrt of its mean count,
  fit, un-scale the endmembers; this preserves the linear mixing model
  exactly) does not close the gap either: 18.9 vs 17.8 deg at dose 200,
  14.8 vs 14.3 at dose 1000, abundance RMSE within 0.01 both ways. The
  autoencoder's margin is not an artifact of NMF's unweighted loss.
- Handing NMF the autoencoder's sum-to-one constraint through the standard
  augmented-column device never helps: neutral at a weak constraint weight
  (17.9 deg at 20x the mean count) and harmful at a strong one (20.3 deg,
  RMSE 0.33, still unconverged at 4000 iterations, at 100x). The win is
  not "NMF denied a constraint".
- Random restarts do not help NMF (operating-point table), so the
  deterministic nndsvda run is not a lucky draw.
- VCA is given the standard SNR-dependent projection and simplex-constrained
  NNLS abundances; its low-dose failure is a property of the pure-pixel
  assumption, not of a broken implementation (at dose 5000 it matches NMF,
  and in tests it recovers noiseless pure-pixel mixtures to under 0.5 deg).

## Reproducing

```
python scripts/run_all.py            # all 7 configs + all figures, ~40 min CPU
python scripts/fair_tuning_check.py  # the baseline-tuning audit
eelsunmix benchmark configs/<name>.yaml   # any single table above
```
