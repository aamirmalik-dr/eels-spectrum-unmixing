"""Saving and loading scenes, plus a bring-your-own loader for real datacubes."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .sim import EELSScene, SimConfig


def save_scene(scene: EELSScene, path: str | Path) -> None:
    """Save a simulated scene with full ground truth to a compressed .npz.

    Counts are stored as uint16 when they fit (they do at any dose this
    package uses), which keeps committed samples small.

    Args:
        scene: Scene to save.
        path: Output .npz path.
    """
    cube = scene.cube
    if cube.max() < np.iinfo(np.uint16).max:
        cube = cube.astype(np.uint16)
    np.savez_compressed(
        path,
        cube=cube,
        energy_ev=scene.energy_ev.astype(np.float32),
        endmembers=scene.endmembers.astype(np.float32),
        abundances=scene.abundances.astype(np.float32),
        drift_channels=scene.drift_channels.astype(np.float32),
        names=np.array(scene.names),
        config=np.array([repr(asdict(scene.config))]),
    )


def load_scene(path: str | Path) -> EELSScene:
    """Load a scene saved by save_scene.

    Args:
        path: Path to the .npz file.

    Returns:
        EELSScene with float64 counts and the original SimConfig.
    """
    with np.load(path, allow_pickle=False) as data:
        config_repr = str(data["config"][0])
        config = SimConfig(**eval(config_repr, {"__builtins__": {}}, {}))
        return EELSScene(
            cube=data["cube"].astype(np.float64),
            energy_ev=data["energy_ev"].astype(np.float64),
            endmembers=data["endmembers"].astype(np.float64),
            abundances=data["abundances"].astype(np.float64),
            drift_channels=data["drift_channels"].astype(np.float64),
            names=[str(n) for n in data["names"]],
            config=config,
        )


def load_cube(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a user-provided spectrum image for unmixing (bring your own data).

    Accepted formats:
        - .npy: a 3D array shaped (ny, nx, n_channels).
        - .npz: an archive with a 3D array under the key 'cube' (or a single
          3D array under any key), optionally an 'energy_ev' 1D array.

    Vendor formats (.dm3/.dm4, .hspy) are not read directly; convert them
    first, for example with HyperSpy:
    ``hs.load('map.dm4').data`` then ``np.save('map.npy', data)``.

    Args:
        path: Path to the file.

    Returns:
        Tuple of (cube shaped (ny, nx, n_channels), energy axis or None).

    Raises:
        ValueError: If the file cannot be interpreted as a 3D spectrum image.
    """
    path = Path(path)
    if path.suffix == ".npy":
        cube = np.load(path)
        energy = None
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "cube" in data:
                cube = data["cube"]
            else:
                arrays_3d = [k for k in data.files if data[k].ndim == 3]
                if len(arrays_3d) != 1:
                    raise ValueError(
                        f"{path} must contain exactly one 3D array "
                        f"(found {len(arrays_3d)}); use the key 'cube'."
                    )
                cube = data[arrays_3d[0]]
            energy = data["energy_ev"] if "energy_ev" in data else None
    else:
        raise ValueError(f"Unsupported format {path.suffix}; use .npy or .npz (see docstring).")
    cube = np.asarray(cube, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D (ny, nx, n_channels) array, got shape {cube.shape}.")
    if cube.min() < 0:
        raise ValueError("Counts must be non-negative for NMF and Poisson-noise assumptions.")
    return cube, None if energy is None else np.asarray(energy, dtype=np.float64)
