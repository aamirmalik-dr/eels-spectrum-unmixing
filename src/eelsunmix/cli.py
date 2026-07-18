"""Command-line interface: simulate, unmix, benchmark, train, demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import __version__
from .autoencoder import AETrainConfig, decomposition_from_model, load_model, train_autoencoder
from .benchmark import TRUE_K, run_config, run_method, score
from .io import load_cube, load_scene, save_scene
from .metrics import match_endmembers
from .plots import plot_hero, plot_loss_curve, plot_scene
from .sim import SimConfig, simulate


def _cmd_simulate(args: argparse.Namespace) -> int:
    config = SimConfig(
        dose=args.dose,
        drift_channels=args.drift,
        edge_separation_ev=args.separation,
        n_phases=args.phases,
        seed=args.seed,
    )
    scene = simulate(config)
    save_scene(scene, args.output)
    print(f"wrote {args.output}: cube {scene.cube.shape}, dose {config.dose:g}")
    if args.figure:
        plot_scene(scene, args.figure)
        print(f"wrote {args.figure}")
    return 0


def _cmd_unmix(args: argparse.Namespace) -> int:
    path = Path(args.cube)
    scene = None
    try:
        scene = load_scene(path)
        x = scene.flat
        shape = scene.cube.shape
    except (KeyError, ValueError, TypeError):
        cube, _ = load_cube(path)
        x = cube.reshape(-1, cube.shape[2])
        shape = cube.shape
    spec = {"kind": args.method, "name": args.method, "k": args.k}
    if args.method == "nmf" and args.restarts > 1:
        spec["n_restarts"] = args.restarts
    if args.method == "ae":
        spec["epochs"] = args.epochs
    dec = run_method(x, spec, args.seed)
    print(f"{args.method}: k={args.k}, reconstruction error {dec.reconstruction_error:.1f}")
    if scene is not None:
        result = score(scene, dec)
        print(json.dumps(result, indent=1))
    else:
        print("no ground truth in file: reporting fit only (bring-your-own-data mode)")
    if args.figure and scene is not None:
        plot_hero(scene, {args.method: dec}, args.figure)
        print(f"wrote {args.figure}")
    elif args.figure:
        print("figure skipped: hero overlay needs ground truth")
    if args.save_abundances:
        ny, nx = shape[0], shape[1]
        np.savez_compressed(
            args.save_abundances,
            spectra=dec.spectra,
            abundances=dec.abundances.T.reshape(-1, ny, nx),
        )
        print(f"wrote {args.save_abundances}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    payload = run_config(args.config, args.output_dir)
    print(f"{payload['name']}: {len(payload['records'])} records -> {args.output_dir}")
    if "summary" in payload:
        print(json.dumps(payload["summary"], indent=1))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .autoencoder import save_model

    scene = simulate(SimConfig(dose=args.dose, seed=args.seed))
    cfg = AETrainConfig(k=TRUE_K, epochs=args.epochs, seed=args.seed)
    model, losses = train_autoencoder(scene.flat, cfg, verbose=True)
    save_model(model, args.output, scene.cube.shape[2])
    print(f"wrote {args.output}, final loss {losses[-1]:.6f}")
    if args.loss_figure:
        plot_loss_curve(losses, args.loss_figure)
        print(f"wrote {args.loss_figure}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    scene = load_scene(args.sample)
    model = load_model(args.model)
    dec = decomposition_from_model(model, scene.flat)
    match = match_endmembers(dec.spectra, scene.endmembers)
    print(f"committed sample: {Path(args.sample).name}, cube {scene.cube.shape}")
    print(f"committed autoencoder: {Path(args.model).name}")
    for t, sad in zip(match.true_index.tolist(), match.sad_deg.tolist()):
        print(f"  {scene.names[t]:>9s}: spectral angle {sad:5.2f} deg")
    print(json.dumps(score(scene, dec), indent=1))
    if args.figure:
        plot_hero(scene, {"ae": dec}, args.figure)
        print(f"wrote {args.figure}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the eelsunmix console command."""
    parser = argparse.ArgumentParser(
        prog="eelsunmix",
        description="Spectral unmixing benchmark for simulated STEM-EELS spectrum images.",
    )
    parser.add_argument("--version", action="version", version=f"eelsunmix {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="simulate a scene and save it with ground truth")
    p.add_argument("--dose", type=float, default=1000.0)
    p.add_argument("--drift", type=float, default=1.5)
    p.add_argument("--separation", type=float, default=68.0)
    p.add_argument("--phases", type=int, default=3, choices=(2, 3))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="scene.npz")
    p.add_argument("--figure", default=None)
    p.set_defaults(func=_cmd_simulate)

    p = sub.add_parser("unmix", help="unmix a saved scene or your own .npy/.npz cube")
    p.add_argument("cube")
    p.add_argument("--method", choices=("pca", "nmf", "vca", "ae"), default="nmf")
    p.add_argument("--k", type=int, default=TRUE_K)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--figure", default=None)
    p.add_argument("--save-abundances", default=None)
    p.set_defaults(func=_cmd_unmix)

    p = sub.add_parser("benchmark", help="run a YAML benchmark config")
    p.add_argument("config")
    p.add_argument("--output-dir", default="results")
    p.set_defaults(func=_cmd_benchmark)

    p = sub.add_parser("train", help="train the unmixing autoencoder and save weights")
    p.add_argument("--dose", type=float, default=1000.0)
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="models/autoencoder.pt")
    p.add_argument("--loss-figure", default=None)
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("demo", help="run the committed model on the committed sample")
    p.add_argument("--sample", default="data/sample/oxide_interface_d1000.npz")
    p.add_argument("--model", default="models/autoencoder.pt")
    p.add_argument("--figure", default=None)
    p.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
