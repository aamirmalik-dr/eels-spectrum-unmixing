"""Run every committed benchmark config, then regenerate every figure.

Reproduces the full results/ and figures/ trees from scratch (the committed
sample and model are reproduced by the two CLI commands in data/README.md
and models/MODEL_CARD.md). Expect roughly 30 to 60 minutes on CPU; progress
is printed per config.

    python scripts/run_all.py
"""

from __future__ import annotations

import time

from eelsunmix.benchmark import run_config

CONFIGS = [
    "operating_point",
    "dose_sweep",
    "drift_sweep",
    "overlap_sweep",
    "components",
    "stability",
    "ae_epochs",
]


def main() -> None:
    for name in CONFIGS:
        t0 = time.perf_counter()
        payload = run_config(f"configs/{name}.yaml", "results")
        print(
            f"[done] {name}: {len(payload['records'])} records "
            f"in {time.perf_counter() - t0:.0f}s",
            flush=True,
        )
    import make_figures

    make_figures.main()
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
