"""Figure generation for scenes, decompositions, and benchmark payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .methods import Decomposition
from .metrics import match_endmembers
from .sim import EELSScene

METHOD_COLORS = {
    "pca": "#999999",
    "nmf": "#0072b2",
    "nmf_best10": "#56b4e9",
    "vca": "#e69f00",
    "ae": "#d55e00",
    "ae_best5": "#cc79a7",
}
ABUNDANCE_CMAP = "magma"


def _color(method: str) -> str:
    return METHOD_COLORS.get(method, "#333333")


def _save(fig: plt.Figure, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scene(scene: EELSScene, path: str | Path) -> None:
    """Overview of one scene: total-counts image, ground-truth maps and spectra.

    Args:
        scene: Simulated scene.
        path: Output image path.
    """
    k = scene.abundances.shape[0]
    fig, axes = plt.subplots(1, k + 2, figsize=(3.4 * (k + 2), 3.0), layout="constrained")
    im = axes[0].imshow(scene.cube.sum(axis=2), cmap="gray")
    axes[0].set_title("total counts")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    for i in range(k):
        im = axes[i + 1].imshow(scene.abundances[i], cmap=ABUNDANCE_CMAP, vmin=0, vmax=1)
        axes[i + 1].set_title(f"true {scene.names[i]}")
        fig.colorbar(im, ax=axes[i + 1], fraction=0.046)
    for ax in axes[:-1]:
        ax.set_xticks([])
        ax.set_yticks([])
    for i in range(k):
        axes[-1].plot(scene.energy_ev, scene.endmembers[i], label=scene.names[i])
    axes[-1].set_xlabel("energy loss (eV)")
    axes[-1].set_ylabel("intensity (unit mean)")
    axes[-1].set_title("true endmembers")
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"dose {scene.config.dose:g} counts/px, drift {scene.config.drift_channels:g} ch")
    _save(fig, path)


def plot_hero(scene: EELSScene, decompositions: dict[str, Decomposition], path: str | Path) -> None:
    """Hero figure: recovered endmembers overlaid on truth, plus abundance maps.

    Top row: one panel per true endmember, ground truth in black with each
    method's matched, rescaled estimate overlaid. Following rows: ground-truth
    abundance maps, then one row of matched abundance maps per method.

    Args:
        scene: Scene with ground truth.
        decompositions: Mapping of method name to its Decomposition.
        path: Output image path.
    """
    k = scene.endmembers.shape[0]
    n_rows = 2 + len(decompositions)
    fig = plt.figure(figsize=(3.4 * k, 2.7 * n_rows))
    gs = fig.add_gridspec(n_rows, k, hspace=0.45, wspace=0.25)

    matches = {
        name: match_endmembers(dec.spectra, scene.endmembers)
        for name, dec in decompositions.items()
    }

    for j in range(k):
        ax = fig.add_subplot(gs[0, j])
        ax.plot(scene.energy_ev, scene.endmembers[j], color="black", lw=2.0, label="truth")
        for name, dec in decompositions.items():
            match = matches[name]
            pos = np.where(match.true_index == j)[0]
            if len(pos) == 0:
                continue
            est = dec.spectra[match.est_index[pos[0]]]
            scale = np.dot(est, scene.endmembers[j]) / (np.dot(est, est) + 1e-12)
            ax.plot(
                scene.energy_ev,
                est * scale,
                color=_color(name),
                lw=1.2,
                alpha=0.9,
                label=f"{name} ({match.sad_deg[pos[0]]:.1f} deg)",
            )
        ax.set_title(f"{scene.names[j]}")
        ax.set_xlabel("energy loss (eV)")
        if j == 0:
            ax.set_ylabel("intensity (rescaled)")
        ax.legend(fontsize=7)

    def _map_row(row: int, maps: np.ndarray, label: str, order: np.ndarray | None = None) -> None:
        for j in range(k):
            ax = fig.add_subplot(gs[row, j])
            if order is None:
                data = maps[j]
            else:
                pos = np.where(order == j)[0]
                data = maps[pos[0]] if len(pos) else np.zeros_like(maps[0])
            ax.imshow(data, cmap=ABUNDANCE_CMAP, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(label, fontsize=10)
            ax.set_title(scene.names[j], fontsize=8)

    ny, nx = scene.abundances.shape[1:]
    _map_row(1, scene.abundances, "truth")
    for i, (name, dec) in enumerate(decompositions.items()):
        match = matches[name]
        maps = dec.abundances.T.reshape(-1, ny, nx)
        maps = maps / (maps.sum(axis=0, keepdims=True) + 1e-12)
        _map_row(2 + i, maps[match.est_index], name, order=match.true_index)
    _save(fig, path)


def plot_sweep(payload: dict[str, Any], path: str | Path) -> None:
    """Sweep curves: mean spectral angle and abundance RMSE versus the parameter.

    Args:
        payload: Result dict from benchmark.run_config (mode 'sweep').
        path: Output image path.
    """
    param = payload["config"]["sweep"]["parameter"]
    records = payload["records"]
    methods = list(dict.fromkeys(r["method"] for r in records))
    log_x = param == "dose"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric, ax, label in (
        ("sad_mean_deg", axes[0], "mean spectral angle (deg)"),
        ("abundance_rmse", axes[1], "abundance RMSE"),
    ):
        for m in methods:
            rows = [r for r in records if r["method"] == m and r.get(metric) is not None]
            if not rows:
                continue  # PCA has no abundance RMSE (see benchmark.score)
            values = sorted(set(r[param] for r in rows))
            means = [np.mean([r[metric] for r in rows if r[param] == v]) for v in values]
            stds = [np.std([r[metric] for r in rows if r[param] == v]) for v in values]
            ax.errorbar(
                values, means, yerr=stds, marker="o", ms=4, capsize=3, color=_color(m), label=m
            )
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(param)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(payload["name"])
    _save(fig, path)


def plot_components(payload: dict[str, Any], path: str | Path) -> None:
    """Component-number analysis: PCA scree, fit error versus k, angle versus k.

    Args:
        payload: Result dict from benchmark.run_config (mode 'components').
        path: Output image path.
    """
    records = payload["records"]
    methods = list(dict.fromkeys(r["method"] for r in records))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    scree = payload["pca_scree"]
    axes[0].semilogy(range(1, len(scree) + 1), scree, marker="o", color="#333333")
    axes[0].axvline(3, color="#d55e00", ls="--", lw=1, label="true k = 3")
    axes[0].set_xlabel("principal component")
    axes[0].set_ylabel("explained variance ratio")
    axes[0].set_title("PCA scree")
    axes[0].legend(fontsize=8)

    for m in methods:
        rows = sorted((r for r in records if r["method"] == m), key=lambda r: r["k"])
        ks = [r["k"] for r in rows]
        axes[1].plot(
            ks, [r["reconstruction_error"] for r in rows], marker="o", color=_color(m), label=m
        )
        axes[2].plot(ks, [r["sad_mean_deg"] for r in rows], marker="o", color=_color(m), label=m)
    axes[1].set_xlabel("fitted components k")
    axes[1].set_ylabel("reconstruction error (Frobenius)")
    axes[1].set_title("fit error vs k")
    axes[2].axvline(3, color="#999999", ls="--", lw=1)
    axes[2].set_xlabel("fitted components k")
    axes[2].set_ylabel("mean matched spectral angle (deg)")
    axes[2].set_title("endmember recovery vs k")
    for ax in axes[1:]:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(payload["name"])
    _save(fig, path)


def plot_stability(payload: dict[str, Any], path: str | Path) -> None:
    """Seed-stability strip plot: per-seed spectral angle for each method.

    Args:
        payload: Result dict from benchmark.run_config (mode 'stability').
        path: Output image path.
    """
    records = payload["records"]
    methods = list(dict.fromkeys(r["method"] for r in records))
    fig, ax = plt.subplots(figsize=(1.6 * len(methods) + 2, 4))
    rng = np.random.default_rng(0)
    for i, m in enumerate(methods):
        sads = [r["sad_mean_deg"] for r in records if r["method"] == m]
        jitter = rng.uniform(-0.12, 0.12, size=len(sads))
        ax.scatter(np.full(len(sads), i) + jitter, sads, color=_color(m), alpha=0.75, s=30)
        ax.hlines(np.median(sads), i - 0.25, i + 0.25, color="black", lw=2)
    ax.set_xticks(range(len(methods)), methods)
    ax.set_ylabel("mean spectral angle (deg)")
    ax.set_title(f"{payload['name']}: spread over {len(payload['config']['method_seeds'])} seeds")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, path)


def plot_ae_epochs(payload: dict[str, Any], path: str | Path) -> None:
    """Training-length ablation: recovery and fit error versus epochs.

    Expects a 'stability'-mode payload whose method specs each carry an
    'epochs' field. Left axis: mean matched spectral angle (mean and range
    over seeds). Right axis: reconstruction error. The point of the figure
    is that the two need not move together.

    Args:
        payload: Result dict from benchmark.run_config (mode 'stability').
        path: Output image path.
    """
    records = payload["records"]
    specs = payload["config"]["methods"]
    epochs = [int(s["epochs"]) for s in specs]
    names = [s["name"] for s in specs]
    sad_mean, sad_min, sad_max, recon = [], [], [], []
    for name in names:
        rows = [r for r in records if r["method"] == name]
        sads = [r["sad_mean_deg"] for r in rows]
        sad_mean.append(float(np.mean(sads)))
        sad_min.append(float(np.min(sads)))
        sad_max.append(float(np.max(sads)))
        recon.append(float(np.mean([r["reconstruction_error"] for r in rows])))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(epochs, sad_mean, marker="o", color="#d55e00", label="spectral angle (mean)")
    ax.fill_between(epochs, sad_min, sad_max, color="#d55e00", alpha=0.2, label="seed range")
    ax.set_ylim(min(sad_min) - 0.5, max(sad_max) + 0.5)
    ax.set_xscale("log")
    ax.set_xticks(epochs, [str(e) for e in epochs])
    ax.minorticks_off()
    ax.set_xlabel("training epochs")
    ax.set_ylabel("mean matched spectral angle (deg)", color="#d55e00")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(epochs, recon, marker="s", color="#0072b2", ls="--", label="reconstruction error")
    ax2.set_ylabel("reconstruction error (Frobenius)", color="#0072b2")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labels, fontsize=8, loc="upper right")
    ax.set_title("autoencoder: fit keeps improving, recovery does not")
    _save(fig, path)


def plot_loss_curve(losses: list[float], path: str | Path) -> None:
    """Autoencoder training loss on a log axis.

    Args:
        losses: Per-epoch MSE values.
        path: Output image path.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(losses, color="#d55e00", lw=1.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (dose-normalized)")
    ax.set_title("autoencoder training loss")
    ax.grid(alpha=0.3)
    _save(fig, path)
