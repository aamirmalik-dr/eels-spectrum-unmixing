"""Benchmark harness: every mode end-to-end on tiny scenes, JSON contract."""

import json

import pytest
import yaml

from eelsunmix.benchmark import run_config

TINY_SCENE = {"nx": 14, "ny": 14, "n_channels": 80}


def _write_config(tmp_path, name, payload):
    path = tmp_path / f"{name}.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh)
    return path


def test_sweep_mode(tmp_path):
    config = {
        "name": "tiny_sweep",
        "mode": "sweep",
        "sweep": {"parameter": "dose", "values": [100, 1000]},
        "seeds": [0],
        "scene": TINY_SCENE,
        "methods": [
            {"name": "nmf", "kind": "nmf"},
            {"name": "vca", "kind": "vca"},
        ],
    }
    payload = run_config(_write_config(tmp_path, "s", config), tmp_path / "out")
    assert len(payload["records"]) == 4
    doses = {r["dose"] for r in payload["records"]}
    assert doses == {100, 1000}
    with open(tmp_path / "out" / "tiny_sweep.json", encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["records"] == payload["records"]


def test_components_mode(tmp_path):
    config = {
        "name": "tiny_components",
        "mode": "components",
        "seeds": [0],
        "k_values": [2, 3, 4],
        "scene": TINY_SCENE,
        "methods": [{"name": "nmf", "kind": "nmf"}],
    }
    payload = run_config(_write_config(tmp_path, "c", config), tmp_path / "out")
    assert len(payload["records"]) == 3
    assert len(payload["pca_scree"]) == 10
    ks = [r["k"] for r in payload["records"]]
    assert ks == [2, 3, 4]
    # under-specified k matches fewer components
    assert payload["records"][0]["n_matched"] == 2
    assert payload["records"][1]["n_matched"] == 3


def test_stability_mode(tmp_path):
    config = {
        "name": "tiny_stability",
        "mode": "stability",
        "scene_seed": 0,
        "method_seeds": [0, 1, 2],
        "scene": TINY_SCENE,
        "methods": [{"name": "nmf", "kind": "nmf", "init": "random"}],
    }
    payload = run_config(_write_config(tmp_path, "st", config), tmp_path / "out")
    assert len(payload["records"]) == 3
    stats = payload["summary"]["nmf"]
    assert stats["sad_mean_deg_min"] <= stats["sad_mean_deg_mean"] <= stats["sad_mean_deg_max"]


def test_operating_point_mode(tmp_path):
    config = {
        "name": "tiny_op",
        "mode": "operating_point",
        "seeds": [0, 1],
        "scene": TINY_SCENE,
        "methods": [
            {"name": "pca", "kind": "pca"},
            {"name": "nmf_best3", "kind": "nmf", "n_restarts": 3},
        ],
    }
    payload = run_config(_write_config(tmp_path, "op", config), tmp_path / "out")
    assert len(payload["records"]) == 4
    assert set(payload["summary"]) == {"pca", "nmf_best3"}
    for stats in payload["summary"].values():
        assert set(stats) >= {"sad_mean_deg", "abundance_rmse", "subspace_error_deg"}


def test_pca_scored_on_subspace_not_abundances(tmp_path):
    # PCA scores are signed coordinates, not abundances: RMSE must be null,
    # and the subspace error must use the mean-augmented basis, which captures
    # the true endmember span almost exactly on a near-noiseless scene.
    config = {
        "name": "tiny_pca",
        "mode": "operating_point",
        "seeds": [0],
        "scene": {
            # geometry scaled to the small grid so all three phases occupy
            # a substantial pixel fraction (the default 11 px precipitate
            # would swallow a 14 px wide scene)
            "nx": 32,
            "ny": 32,
            "n_channels": 100,
            "precipitate_radius_px": 6.0,
            "dose": 50000,
            "drift_channels": 0.0,
        },
        "methods": [{"name": "pca", "kind": "pca"}],
    }
    payload = run_config(_write_config(tmp_path, "pca", config), tmp_path / "out")
    rec = payload["records"][0]
    assert rec["abundance_rmse"] is None
    assert payload["summary"]["pca"]["abundance_rmse"] is None
    assert rec["subspace_error_deg"] < 10.0


def test_unknown_mode_raises(tmp_path):
    config = {"name": "bad", "mode": "nope", "methods": []}
    with pytest.raises(ValueError, match="Unknown mode"):
        run_config(_write_config(tmp_path, "bad", config), tmp_path / "out")


def test_ae_method_in_harness(tmp_path):
    config = {
        "name": "tiny_ae",
        "mode": "operating_point",
        "seeds": [0],
        "scene": TINY_SCENE,
        "methods": [{"name": "ae", "kind": "ae", "epochs": 30}],
    }
    payload = run_config(_write_config(tmp_path, "ae", config), tmp_path / "out")
    assert payload["records"][0]["sad_mean_deg"] > 0
