"""CLI smoke tests: every subcommand runs end-to-end on tiny inputs."""

import numpy as np
import yaml

from eelsunmix.cli import main


def test_version(capsys):
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "eelsunmix" in capsys.readouterr().out


def test_simulate_then_unmix(tmp_path, capsys):
    scene_path = str(tmp_path / "scene.npz")
    assert main(["simulate", "--dose", "800", "--seed", "1", "--output", scene_path]) == 0
    assert (
        main(["unmix", scene_path, "--method", "nmf", "--figure", str(tmp_path / "hero.png")]) == 0
    )
    out = capsys.readouterr().out
    assert "sad_mean_deg" in out
    assert (tmp_path / "hero.png").exists()


def test_unmix_bring_your_own_cube(tmp_path, capsys):
    cube = np.random.default_rng(0).poisson(20.0, (10, 10, 40)).astype(float)
    path = str(tmp_path / "cube.npy")
    np.save(path, cube)
    abund_path = str(tmp_path / "abund.npz")
    assert main(["unmix", path, "--method", "nmf", "--save-abundances", abund_path]) == 0
    assert "bring-your-own-data" in capsys.readouterr().out
    with np.load(abund_path) as data:
        assert data["abundances"].shape == (3, 10, 10)


def test_benchmark_command(tmp_path, capsys):
    config = {
        "name": "cli_tiny",
        "mode": "operating_point",
        "seeds": [0],
        "scene": {"nx": 12, "ny": 12, "n_channels": 60},
        "methods": [{"name": "nmf", "kind": "nmf"}],
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)
    assert main(["benchmark", str(cfg_path), "--output-dir", str(tmp_path / "results")]) == 0
    assert (tmp_path / "results" / "cli_tiny.json").exists()


def test_train_then_demo(tmp_path, capsys):
    model_path = str(tmp_path / "model.pt")
    sample_path = str(tmp_path / "sample.npz")
    assert main(["simulate", "--dose", "1000", "--seed", "0", "--output", sample_path]) == 0
    assert main(["train", "--epochs", "30", "--seed", "0", "--output", model_path]) == 0
    assert main(["demo", "--sample", sample_path, "--model", model_path]) == 0
    assert "spectral angle" in capsys.readouterr().out
