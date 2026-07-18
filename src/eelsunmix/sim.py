"""Synthetic STEM-EELS spectrum-image simulator with exact ground truth.

The simulated scene is a three-phase oxide lamella observed in the core-loss
region (380 to 780 eV by default): a titanium oxide grain, a manganese oxide
grain, and an iron oxide precipitate straddling their interface. Each phase is
one endmember spectrum built from a power-law background, ionization edges with
a saturating onset and a post-edge power-law decay, and Gaussian white lines
for the transition-metal L2,3 edges (the L3/L2 intensity ratio is set per
element). All three phases share the O K edge near 532 eV, which makes the
endmembers genuinely correlated, as real oxide spectra are.

The datacube is built with the linear mixing model, cube[y, x, :] =
sum_k A[k, y, x] * S[k, :], scaled to a target dose (mean counts per
spectrum), distorted by a smooth per-pixel energy drift, and finished with
Poisson counting noise. The exact endmembers, abundance maps, and drift field
are all returned, so every benchmark scores against known ground truth.

The model is deliberately kinematic and instrument-free: no plural scattering,
no point-spread function of the spectrometer, no channel-to-channel gain
variation. It exists to give unmixing algorithms a controlled, honest target,
not to reproduce any particular instrument.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class WhiteLine:
    """A single Gaussian white line (or ELNES peak) on top of an edge onset.

    Attributes:
        center_ev: Peak center in eV.
        height: Peak height relative to the edge continuum step.
        width_ev: Gaussian sigma in eV.
    """

    center_ev: float
    height: float
    width_ev: float


@dataclass(frozen=True)
class Edge:
    """An ionization edge: saturating onset times power-law decay, plus white lines.

    Attributes:
        onset_ev: Edge onset energy in eV.
        step: Continuum step height at onset (arbitrary intensity units).
        onset_width_ev: Width of the sigmoidal onset in eV.
        decay: Post-edge power-law exponent, intensity ~ (E / onset) ** -decay.
        white_lines: Gaussian peaks riding on the edge (L3/L2 white lines, ELNES).
    """

    onset_ev: float
    step: float
    onset_width_ev: float
    decay: float
    white_lines: tuple[WhiteLine, ...] = ()


@dataclass(frozen=True)
class EndmemberSpec:
    """One phase: a power-law background plus a set of ionization edges.

    Attributes:
        name: Human-readable phase name.
        background_amp: Background amplitude at the first energy channel.
        background_exponent: Power-law exponent r in A * (E / E0) ** -r.
        edges: Ionization edges present in this phase.
    """

    name: str
    background_amp: float
    background_exponent: float
    edges: tuple[Edge, ...]


@dataclass(frozen=True)
class SimConfig:
    """Full configuration of one simulated spectrum image.

    Attributes:
        nx: Scan width in pixels (fast scan direction).
        ny: Scan height in pixels (slow scan direction).
        n_channels: Number of energy channels.
        e_min_ev: First channel energy in eV.
        e_max_ev: Last channel energy in eV.
        dose: Target mean total counts per pixel spectrum before Poisson noise.
        drift_channels: Peak-to-peak amplitude of the energy-drift field,
            in channels. Zero disables drift.
        edge_separation_ev: Separation between the Mn L3 and Fe L3 onsets.
            The Fe edge is moved toward the Mn edge as this shrinks, which
            makes the two metal-oxide endmembers progressively collinear.
            The physical value is about 68 eV.
        interface_width_px: Width of the diffuse interface between the two
            grains, in pixels.
        precipitate_radius_px: Radius of the third-phase precipitate.
        n_phases: 2 or 3. With 2, the precipitate phase is omitted.
        seed: Seed for all randomness (geometry, drift, Poisson noise).
    """

    nx: int = 64
    ny: int = 64
    n_channels: int = 200
    e_min_ev: float = 380.0
    e_max_ev: float = 778.0
    dose: float = 1000.0
    drift_channels: float = 1.5
    edge_separation_ev: float = 68.0
    interface_width_px: float = 4.0
    precipitate_radius_px: float = 11.0
    n_phases: int = 3
    seed: int = 0


@dataclass
class EELSScene:
    """A simulated spectrum image together with its exact ground truth.

    Attributes:
        cube: Noisy counts, shape (ny, nx, n_channels).
        energy_ev: Energy axis, shape (n_channels,).
        endmembers: Ground-truth endmember spectra, shape (k, n_channels),
            each normalized to unit mean.
        abundances: Ground-truth abundance maps, shape (k, ny, nx),
            non-negative and summing to one at every pixel.
        drift_channels: Per-pixel energy drift in channels, shape (ny, nx).
        names: Phase names, length k.
        config: The SimConfig that produced the scene.
    """

    cube: np.ndarray
    energy_ev: np.ndarray
    endmembers: np.ndarray
    abundances: np.ndarray
    drift_channels: np.ndarray
    names: list[str] = field(default_factory=list)
    config: SimConfig = field(default_factory=SimConfig)

    @property
    def flat(self) -> np.ndarray:
        """Cube reshaped to (n_pixels, n_channels)."""
        ny, nx, nc = self.cube.shape
        return self.cube.reshape(ny * nx, nc)

    @property
    def flat_abundances(self) -> np.ndarray:
        """Abundances reshaped to (n_pixels, k)."""
        k = self.abundances.shape[0]
        return self.abundances.reshape(k, -1).T


def _l32_ratio(white_lines: list[WhiteLine], ratio: float) -> tuple[WhiteLine, WhiteLine]:
    """Scale an (L3, L2) pair so their height ratio equals `ratio`."""
    l3, l2 = white_lines
    return l3, WhiteLine(l2.center_ev, l3.height / ratio, l2.width_ev)


def oxide_phase_specs(edge_separation_ev: float = 68.0) -> list[EndmemberSpec]:
    """Return the three oxide phase specs used by the default scene.

    The Ti, Mn, and Fe L2,3 edge onsets and the O K edge sit at their
    tabulated energies (456, 640, 708, 532 eV). White-line L3/L2 ratios
    are set per element (Ti near 1, Mn and Fe well above 2), and each
    phase carries a slightly different O K fine structure, the way ELNES
    differs between oxides. Shrinking `edge_separation_ev` moves the Fe
    L3 onset toward the Mn L3 onset to create controlled spectral overlap.

    Args:
        edge_separation_ev: Separation between Mn L3 and Fe L3 onsets in eV.

    Returns:
        List of three EndmemberSpec, order [Ti oxide, Mn oxide, Fe oxide].
    """
    fe_l3 = 640.0 + edge_separation_ev
    fe_l2 = fe_l3 + 13.0
    ti = EndmemberSpec(
        name="Ti oxide",
        background_amp=1.0,
        background_exponent=2.8,
        edges=(
            Edge(
                onset_ev=456.0,
                step=0.55,
                onset_width_ev=2.0,
                decay=2.2,
                white_lines=(
                    WhiteLine(458.0, 1.9, 1.4),
                    WhiteLine(463.5, 1.7, 1.6),
                ),
            ),
            Edge(
                onset_ev=532.0,
                step=0.50,
                onset_width_ev=2.5,
                decay=2.4,
                white_lines=(WhiteLine(534.0, 0.75, 2.2), WhiteLine(543.0, 0.35, 4.0)),
            ),
        ),
    )
    mn = EndmemberSpec(
        name="Mn oxide",
        background_amp=1.1,
        background_exponent=2.9,
        edges=(
            Edge(
                onset_ev=532.0,
                step=0.45,
                onset_width_ev=2.5,
                decay=2.4,
                white_lines=(WhiteLine(533.0, 0.55, 2.0), WhiteLine(541.0, 0.30, 4.5)),
            ),
            Edge(
                onset_ev=640.0,
                step=0.60,
                onset_width_ev=1.8,
                decay=2.1,
                white_lines=(
                    WhiteLine(641.5, 3.0, 1.5),
                    WhiteLine(652.5, 3.0 / 3.2, 1.8),
                ),
            ),
        ),
    )
    fe = EndmemberSpec(
        name="Fe oxide",
        background_amp=0.9,
        background_exponent=2.7,
        edges=(
            Edge(
                onset_ev=532.0,
                step=0.55,
                onset_width_ev=2.5,
                decay=2.4,
                white_lines=(WhiteLine(532.5, 0.85, 1.8), WhiteLine(544.5, 0.30, 4.0)),
            ),
            Edge(
                onset_ev=fe_l3,
                step=0.65,
                onset_width_ev=1.8,
                decay=2.1,
                white_lines=(
                    WhiteLine(fe_l3 + 1.5, 2.6, 1.6),
                    WhiteLine(fe_l2 + 1.5, 2.6 / 4.0, 2.0),
                ),
            ),
        ),
    )
    return [ti, mn, fe]


def render_endmember(spec: EndmemberSpec, energy_ev: np.ndarray) -> np.ndarray:
    """Render one endmember spectrum on an energy axis, normalized to unit mean.

    Args:
        spec: Phase specification.
        energy_ev: Energy axis in eV, shape (n_channels,).

    Returns:
        Spectrum, shape (n_channels,), non-negative, mean 1.
    """
    e = np.asarray(energy_ev, dtype=np.float64)
    s = spec.background_amp * (e / e[0]) ** (-spec.background_exponent)
    for edge in spec.edges:
        onset = 1.0 / (1.0 + np.exp(-(e - edge.onset_ev) / edge.onset_width_ev))
        decay = np.where(
            e > edge.onset_ev, (np.maximum(e, edge.onset_ev) / edge.onset_ev) ** (-edge.decay), 1.0
        )
        s = s + edge.step * onset * decay
        for wl in edge.white_lines:
            s = s + edge.step * wl.height * np.exp(-0.5 * ((e - wl.center_ev) / wl.width_ev) ** 2)
    return s / s.mean()


def make_endmembers(
    energy_ev: np.ndarray, edge_separation_ev: float = 68.0, n_phases: int = 3
) -> tuple[list[str], np.ndarray]:
    """Build the ground-truth endmember matrix for the oxide scene.

    Args:
        energy_ev: Energy axis in eV.
        edge_separation_ev: Mn L3 to Fe L3 onset separation in eV.
        n_phases: 2 or 3 phases.

    Returns:
        Tuple of (names, S) with S of shape (n_phases, n_channels).
    """
    specs = oxide_phase_specs(edge_separation_ev)[:n_phases]
    spectra = np.stack([render_endmember(sp, energy_ev) for sp in specs])
    return [sp.name for sp in specs], spectra


def make_abundance_maps(config: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Build ground-truth abundance maps: two grains, a wavy diffuse interface,
    and (for three phases) a precipitate straddling the interface.

    A smooth low-amplitude mixing field is added so pixels away from the
    interface are dominated by, but not purely, one phase; every map is
    non-negative and the maps sum to one at each pixel.

    Args:
        config: Scene configuration.
        rng: Random generator (controls boundary waviness, precipitate
            position, and the smooth mixing field).

    Returns:
        Abundances, shape (n_phases, ny, nx).
    """
    ny, nx = config.ny, config.nx
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)

    boundary_x = nx / 2.0 + 3.0 * np.sin(2.0 * np.pi * yy[:, 0] / ny * 1.7 + rng.uniform(0, 6.28))
    t = (xx - boundary_x[:, None]) / max(config.interface_width_px, 1e-6)
    right = 1.0 / (1.0 + np.exp(-t))
    a_ti = 1.0 - right
    a_mn = right

    maps = [a_ti, a_mn]
    if config.n_phases >= 3:
        cy = ny / 2.0 + rng.uniform(-ny / 8.0, ny / 8.0)
        cx = float(np.interp(cy, np.arange(ny), boundary_x))
        r = np.hypot(yy - cy, xx - cx)
        a_fe = 1.0 / (1.0 + np.exp((r - config.precipitate_radius_px) / 1.5))
        maps = [a_ti * (1.0 - a_fe), a_mn * (1.0 - a_fe), a_fe]

    a = np.stack(maps)
    mix = np.stack([gaussian_filter(rng.normal(size=(ny, nx)), sigma=6.0) for _ in maps])
    mix = 0.08 * mix / (np.abs(mix).max() + 1e-12)
    a = np.clip(a + mix, 1e-4, None)
    return a / a.sum(axis=0, keepdims=True)


def make_drift_field(config: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Build a smooth per-pixel energy-drift field in channels.

    The field is a slow ramp down the slow-scan direction (monochromator or
    high-tension drift over acquisition time) plus a smooth random component,
    scaled so its peak-to-peak span equals `config.drift_channels`.

    Args:
        config: Scene configuration.
        rng: Random generator.

    Returns:
        Drift in channels, shape (ny, nx). All zeros if drift is disabled.
    """
    ny, nx = config.ny, config.nx
    if config.drift_channels <= 0:
        return np.zeros((ny, nx))
    yy = np.mgrid[0:ny, 0:nx][0].astype(np.float64)
    ramp = yy / max(ny - 1, 1)
    smooth = gaussian_filter(rng.normal(size=(ny, nx)), sigma=8.0)
    fieldd = ramp + 0.6 * smooth / (np.abs(smooth).max() + 1e-12)
    fieldd = fieldd - fieldd.min()
    span = fieldd.max() + 1e-12
    return fieldd / span * config.drift_channels


def apply_drift(clean: np.ndarray, drift: np.ndarray) -> np.ndarray:
    """Shift every pixel spectrum along the energy axis by its drift value.

    Linear interpolation with edge clamping; positive drift moves spectral
    features to higher channel indices.

    Args:
        clean: Noiseless cube, shape (ny, nx, n_channels).
        drift: Per-pixel shift in channels, shape (ny, nx).

    Returns:
        Drifted cube, same shape.
    """
    ny, nx, nc = clean.shape
    idx = np.arange(nc, dtype=np.float64)
    flat = clean.reshape(-1, nc)
    shifts = drift.reshape(-1)
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        out[i] = np.interp(idx - shifts[i], idx, flat[i])
    return out.reshape(ny, nx, nc)


def simulate(config: SimConfig) -> EELSScene:
    """Simulate one spectrum image with exact ground truth.

    Pipeline: endmembers -> abundance maps -> linear mixing -> dose scaling
    -> per-pixel energy drift -> Poisson noise.

    Args:
        config: Scene configuration.

    Returns:
        EELSScene with noisy counts and full ground truth.
    """
    rng = np.random.default_rng(config.seed)
    energy = np.linspace(config.e_min_ev, config.e_max_ev, config.n_channels)
    names, endmembers = make_endmembers(energy, config.edge_separation_ev, config.n_phases)
    abundances = make_abundance_maps(config, rng)
    drift = make_drift_field(config, rng)

    clean = np.tensordot(abundances, endmembers, axes=(0, 0))
    clean = clean / clean.sum(axis=2, keepdims=True).mean() * config.dose
    drifted = apply_drift(clean, drift)
    cube = rng.poisson(np.clip(drifted, 0.0, None)).astype(np.float64)

    return EELSScene(
        cube=cube,
        energy_ev=energy,
        endmembers=endmembers,
        abundances=abundances,
        drift_channels=drift,
        names=names,
        config=config,
    )
