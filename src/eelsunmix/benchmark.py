"""Config-driven benchmark harness with fixed seeds.

Four modes, all driven by small YAML files (see configs/):

- ``sweep``: vary one SimConfig parameter (dose, drift_channels,
  edge_separation_ev, ...) over a grid, score every method at every point,
  several scene seeds per point.
- ``components``: fix the scene, fit k = 2..6 components with the true k
  known to be 3; report reconstruction error for selection and matched
  spectral angles for diagnosis, plus the PCA scree.
- ``stability``: fix the scene, rerun the initialization-sensitive methods
  (NMF from random inits, VCA, the autoencoder) across many seeds and report
  the spread, not just the best run.
- ``operating_point``: one headline condition, every method including the
  best-of-N variants, so the comparison at matched selection effort is
  explicit.

Every mode writes one JSON file with the config echoed back, so each
committed figure and table regenerates from its config alone.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .autoencoder import AETrainConfig, ae_unmix
from .methods import Decomposition, nmf_unmix, pca_decompose, pca_scree, vca_unmix
from .metrics import abundance_rmse, match_endmembers, subspace_error_deg
from .sim import EELSScene, SimConfig, simulate

TRUE_K = 3


def run_method(x: np.ndarray, spec: dict[str, Any], seed: int) -> Decomposition:
    """Run one method spec on a flat data matrix.

    Args:
        x: Data, shape (n_pixels, n_channels).
        spec: Method spec with 'kind' in {pca, nmf, vca, ae} and optional
            parameters (k, n_restarts, n_seeds, epochs, init).
        seed: Seed offset for this run (combined with any spec seed).

    Returns:
        Decomposition from the method.
    """
    kind = spec["kind"]
    k = int(spec.get("k", TRUE_K))
    if kind == "pca":
        return pca_decompose(x, k)
    if kind == "nmf":
        return nmf_unmix(
            x,
            k,
            seed=seed,
            n_restarts=int(spec.get("n_restarts", 1)),
            init=spec.get("init", "nndsvda"),
        )
    if kind == "vca":
        return vca_unmix(x, k, seed=seed)
    if kind == "ae":
        cfg = AETrainConfig(k=k, epochs=int(spec.get("epochs", 1200)), seed=seed)
        return ae_unmix(x, cfg, n_seeds=int(spec.get("n_seeds", 1)))
    raise ValueError(f"Unknown method kind: {kind}")


def score(scene: EELSScene, dec: Decomposition) -> dict[str, Any]:
    """Score a decomposition against the scene's ground truth.

    Args:
        scene: Scene with ground truth.
        dec: Decomposition to score.

    Returns:
        Dict with mean/per-component spectral angle (degrees), abundance RMSE
        (matched, renormalized), and the largest principal angle between the
        true endmember span and the recovered component span.

        PCA is a special case, scored for what it actually estimates: its
        subspace error uses the centered components augmented with the mean
        spectrum (together they span PCA's signal-subspace estimate), and its
        abundance RMSE is null, because principal-component scores are signed
        coordinates, not abundances, and forcing them onto a simplex would
        produce a large but meaningless number.
    """
    match = match_endmembers(dec.spectra, scene.endmembers)
    per_component = {
        scene.names[t]: round(float(s), 4)
        for t, s in zip(match.true_index.tolist(), match.sad_deg.tolist())
    }
    is_pca = "explained_variance_ratio" in dec.meta
    if is_pca:
        basis = np.vstack([dec.meta["mean"][None, :], dec.spectra])
        ab_rmse = None
    else:
        basis = dec.spectra
        ab_rmse = round(abundance_rmse(dec.abundances.T, scene.flat_abundances.T, match), 5)
    return {
        "sad_mean_deg": round(match.mean_sad_deg, 4),
        "sad_per_component_deg": per_component,
        "abundance_rmse": ab_rmse,
        "subspace_error_deg": round(subspace_error_deg(scene.endmembers, basis), 4),
        "n_matched": int(len(match.sad_deg)),
    }


def _scene_config(base: dict[str, Any], **overrides: Any) -> SimConfig:
    merged = {**base, **overrides}
    return SimConfig(**merged)


def run_sweep(config: dict[str, Any]) -> dict[str, Any]:
    """Sweep one scene parameter over a grid, scoring every method."""
    records = []
    param = config["sweep"]["parameter"]
    values = config["sweep"]["values"]
    seeds = config.get("seeds", [0])
    for value in values:
        for seed in seeds:
            scene = simulate(_scene_config(config.get("scene", {}), **{param: value}, seed=seed))
            for spec in config["methods"]:
                t0 = time.perf_counter()
                dec = run_method(scene.flat, spec, seed)
                rec = {
                    "method": spec["name"],
                    param: value,
                    "seed": seed,
                    "seconds": round(time.perf_counter() - t0, 2),
                }
                rec.update(score(scene, dec))
                records.append(rec)
    return {"records": records}


def run_components(config: dict[str, Any]) -> dict[str, Any]:
    """Fit a range of component counts to one scene, plus the PCA scree."""
    seed = int(config.get("seeds", [0])[0])
    scene = simulate(_scene_config(config.get("scene", {}), seed=seed))
    records = []
    for k in config["k_values"]:
        for spec in config["methods"]:
            spec_k = {**spec, "k": k}
            t0 = time.perf_counter()
            dec = run_method(scene.flat, spec_k, seed)
            rec = {
                "method": spec["name"],
                "k": k,
                "seed": seed,
                "reconstruction_error": round(dec.reconstruction_error, 2),
                "seconds": round(time.perf_counter() - t0, 2),
            }
            rec.update(score(scene, dec))
            records.append(rec)
    scree = pca_scree(scene.flat, kmax=int(config.get("scree_kmax", 10)))
    return {"records": records, "pca_scree": [round(float(v), 6) for v in scree]}


def run_stability(config: dict[str, Any]) -> dict[str, Any]:
    """Rerun initialization-sensitive methods across seeds on one fixed scene."""
    scene_seed = int(config.get("scene_seed", 0))
    scene = simulate(_scene_config(config.get("scene", {}), seed=scene_seed))
    records = []
    for spec in config["methods"]:
        for seed in config["method_seeds"]:
            t0 = time.perf_counter()
            dec = run_method(scene.flat, spec, seed)
            rec = {
                "method": spec["name"],
                "seed": seed,
                "reconstruction_error": round(dec.reconstruction_error, 2),
                "seconds": round(time.perf_counter() - t0, 2),
            }
            rec.update(score(scene, dec))
            records.append(rec)
    summary = {}
    for spec in config["methods"]:
        sads = [r["sad_mean_deg"] for r in records if r["method"] == spec["name"]]
        summary[spec["name"]] = {
            "sad_mean_deg_mean": round(float(np.mean(sads)), 4),
            "sad_mean_deg_std": round(float(np.std(sads)), 4),
            "sad_mean_deg_min": round(float(np.min(sads)), 4),
            "sad_mean_deg_max": round(float(np.max(sads)), 4),
        }
    return {"records": records, "summary": summary}


def run_operating_point(config: dict[str, Any]) -> dict[str, Any]:
    """Score every method (including best-of-N variants) at one condition."""
    seeds = config.get("seeds", [0])
    records = []
    for seed in seeds:
        scene = simulate(_scene_config(config.get("scene", {}), seed=seed))
        for spec in config["methods"]:
            t0 = time.perf_counter()
            dec = run_method(scene.flat, spec, seed)
            rec = {
                "method": spec["name"],
                "seed": seed,
                "seconds": round(time.perf_counter() - t0, 2),
            }
            rec.update(score(scene, dec))
            records.append(rec)
    summary = {}
    for spec in config["methods"]:
        rows = [r for r in records if r["method"] == spec["name"]]
        rmses = [r["abundance_rmse"] for r in rows if r["abundance_rmse"] is not None]
        summary[spec["name"]] = {
            "sad_mean_deg": round(float(np.mean([r["sad_mean_deg"] for r in rows])), 4),
            "sad_std_deg": round(float(np.std([r["sad_mean_deg"] for r in rows])), 4),
            "abundance_rmse": round(float(np.mean(rmses)), 5) if rmses else None,
            "subspace_error_deg": round(float(np.mean([r["subspace_error_deg"] for r in rows])), 4),
            "mean_seconds": round(float(np.mean([r["seconds"] for r in rows])), 2),
        }
    return {"records": records, "summary": summary}


MODES = {
    "sweep": run_sweep,
    "components": run_components,
    "stability": run_stability,
    "operating_point": run_operating_point,
}


def run_config(path: str | Path, output_dir: str | Path = "results") -> dict[str, Any]:
    """Run one YAML benchmark config and write its JSON result.

    Args:
        path: Path to the YAML config.
        output_dir: Directory for the JSON output (created if needed).

    Returns:
        The result dict that was written.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    mode = config["mode"]
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {sorted(MODES)}")
    result = MODES[mode](config)
    scene_defaults = asdict(SimConfig())
    payload = {
        "name": config.get("name", path.stem),
        "mode": mode,
        "config": config,
        "scene_defaults": scene_defaults,
        **result,
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{payload['name']}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    return payload
