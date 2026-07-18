"""Simulator invariants: shapes, ground-truth constraints, physics knobs."""

import numpy as np
import pytest

from eelsunmix.sim import (
    SimConfig,
    make_endmembers,
    oxide_phase_specs,
    render_endmember,
    simulate,
)


@pytest.fixture(scope="module")
def scene():
    return simulate(SimConfig(nx=24, ny=20, n_channels=120, dose=500.0, seed=3))


def test_shapes(scene):
    assert scene.cube.shape == (20, 24, 120)
    assert scene.energy_ev.shape == (120,)
    assert scene.endmembers.shape == (3, 120)
    assert scene.abundances.shape == (3, 20, 24)
    assert scene.drift_channels.shape == (20, 24)
    assert scene.flat.shape == (480, 120)
    assert scene.flat_abundances.shape == (480, 3)


def test_abundances_form_a_simplex(scene):
    assert scene.abundances.min() >= 0.0
    np.testing.assert_allclose(scene.abundances.sum(axis=0), 1.0, atol=1e-9)


def test_endmembers_nonnegative_unit_mean(scene):
    assert scene.endmembers.min() >= 0.0
    np.testing.assert_allclose(scene.endmembers.mean(axis=1), 1.0, atol=1e-9)


def test_counts_are_nonnegative_integers(scene):
    assert scene.cube.min() >= 0
    np.testing.assert_allclose(scene.cube, np.round(scene.cube))


def test_dose_sets_mean_total_counts(scene):
    # Poisson preserves the mean; the pre-noise cube is scaled to dose exactly.
    assert scene.cube.sum(axis=2).mean() == pytest.approx(500.0, rel=0.05)


def test_seed_reproducibility():
    config = SimConfig(nx=12, ny=12, n_channels=64, seed=7)
    a = simulate(config)
    b = simulate(config)
    np.testing.assert_array_equal(a.cube, b.cube)
    np.testing.assert_array_equal(a.abundances, b.abundances)


def test_different_seeds_differ():
    a = simulate(SimConfig(nx=12, ny=12, n_channels=64, seed=0))
    b = simulate(SimConfig(nx=12, ny=12, n_channels=64, seed=1))
    assert not np.array_equal(a.cube, b.cube)


def test_drift_field_span_matches_config():
    scene = simulate(SimConfig(nx=16, ny=16, drift_channels=3.0, seed=0))
    assert scene.drift_channels.min() == pytest.approx(0.0, abs=1e-9)
    assert scene.drift_channels.max() == pytest.approx(3.0, rel=1e-6)


def test_zero_drift_is_all_zeros():
    scene = simulate(SimConfig(nx=8, ny=8, drift_channels=0.0, seed=0))
    assert np.all(scene.drift_channels == 0.0)


def test_two_phase_scene():
    scene = simulate(SimConfig(nx=16, ny=16, n_phases=2, seed=0))
    assert scene.endmembers.shape[0] == 2
    assert scene.abundances.shape[0] == 2
    np.testing.assert_allclose(scene.abundances.sum(axis=0), 1.0, atol=1e-9)


def test_edge_separation_controls_collinearity():
    energy = np.linspace(380.0, 778.0, 200)
    _, far = make_endmembers(energy, edge_separation_ev=68.0)
    _, near = make_endmembers(energy, edge_separation_ev=8.0)

    def corr(s):
        mn, fe = s[1], s[2]
        return float(np.corrcoef(mn, fe)[0, 1])

    assert corr(near) > corr(far)


def test_white_line_appears_at_tabulated_energy():
    energy = np.linspace(380.0, 778.0, 400)
    spec = oxide_phase_specs()[1]  # Mn oxide
    s = render_endmember(spec, energy)
    peak_ev = energy[int(np.argmax(s))]
    assert peak_ev == pytest.approx(641.5, abs=2.0)  # Mn L3 white line


def test_background_falls_as_power_law():
    energy = np.linspace(380.0, 778.0, 200)
    spec = oxide_phase_specs()[0]
    s = render_endmember(spec, energy)
    assert s[0] > s[30]  # pre-edge region (380 to 440 eV) decays before Ti L3 at 456 eV
