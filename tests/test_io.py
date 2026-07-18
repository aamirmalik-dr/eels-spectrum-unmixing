"""Scene persistence and the bring-your-own-data loader."""

import numpy as np
import pytest

from eelsunmix.io import load_cube, load_scene, save_scene
from eelsunmix.sim import SimConfig, simulate


def test_scene_roundtrip(tmp_path):
    scene = simulate(SimConfig(nx=12, ny=10, n_channels=64, dose=800.0, seed=2))
    path = tmp_path / "scene.npz"
    save_scene(scene, path)
    loaded = load_scene(path)
    np.testing.assert_array_equal(loaded.cube, scene.cube)
    np.testing.assert_allclose(loaded.endmembers, scene.endmembers, atol=1e-6)
    np.testing.assert_allclose(loaded.abundances, scene.abundances, atol=1e-7)
    assert loaded.names == scene.names
    assert loaded.config == scene.config


def test_load_cube_npy(tmp_path):
    cube = np.random.default_rng(0).poisson(5.0, (8, 9, 30)).astype(float)
    path = tmp_path / "cube.npy"
    np.save(path, cube)
    loaded, energy = load_cube(path)
    np.testing.assert_array_equal(loaded, cube)
    assert energy is None


def test_load_cube_npz_with_energy(tmp_path):
    cube = np.random.default_rng(1).poisson(5.0, (8, 9, 30)).astype(float)
    energy = np.linspace(400.0, 700.0, 30)
    path = tmp_path / "cube.npz"
    np.savez(path, cube=cube, energy_ev=energy)
    loaded, e = load_cube(path)
    np.testing.assert_array_equal(loaded, cube)
    np.testing.assert_allclose(e, energy)


def test_load_cube_rejects_wrong_dimensionality(tmp_path):
    path = tmp_path / "flat.npy"
    np.save(path, np.zeros((10, 20)))
    with pytest.raises(ValueError, match="3D"):
        load_cube(path)


def test_load_cube_rejects_negative_counts(tmp_path):
    path = tmp_path / "neg.npy"
    np.save(path, -np.ones((4, 4, 10)))
    with pytest.raises(ValueError, match="non-negative"):
        load_cube(path)


def test_load_cube_rejects_unknown_format(tmp_path):
    path = tmp_path / "cube.txt"
    path.write_text("nope")
    with pytest.raises(ValueError, match="Unsupported"):
        load_cube(path)


def test_load_scene_never_executes_stored_config(tmp_path):
    # The config string inside a scene file is parsed with ast.literal_eval,
    # so a tampered file raises instead of running code.
    scene = simulate(SimConfig(nx=8, ny=8, n_channels=32, seed=0))
    path = tmp_path / "scene.npz"
    save_scene(scene, path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["config"] = np.array(["__import__('os').getcwd()"])
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        load_scene(path)
