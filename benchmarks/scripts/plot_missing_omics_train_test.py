#!/usr/bin/env python3
"""
Generate plots for the missing-omics train/test benchmark.

Inputs:
  - benchmarks/missing_omics_train_test_results.csv

Outputs:
  - benchmarks/missing_omics_train_test_plot.png
"""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
import atexit
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    results_csv = here / "missing_omics_train_test_results.csv"
    if not results_csv.exists():
        raise SystemExit(
            f"Missing input CSV: {results_csv}\n"
            "Run the benchmark first to generate it."
        )

    # Ensure Matplotlib/fontconfig can write caches within the sandbox.
    mpl_config_dir = Path(tempfile.mkdtemp(prefix="mplconfig-"))
    xdg_cache_home = Path(tempfile.mkdtemp(prefix="xdg-cache-"))
    atexit.register(shutil.rmtree, mpl_config_dir, ignore_errors=True)
    atexit.register(shutil.rmtree, xdg_cache_home, ignore_errors=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_home))

    import matplotlib  # noqa: WPS433

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    rows: list[dict[str, str]] = []
    with results_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for r in rows:
        # keys used for plotting
        for k in (
            "train_missing_pct",
            "test_missing_pct",
            "ebv_test_total",
            "ebv_test_indirect",
            "epv_test_total",
            "epv_test_trait",
            "time_seconds",
        ):
            r[k] = float(r[k])  # type: ignore[assignment]

    # Group by test_missing_pct
    groups: dict[float, list[dict[str, float]]] = {}
    for r0 in rows:
        r = {k: float(v) if isinstance(v, (int, float)) else v for k, v in r0.items()}  # type: ignore[arg-type]
        test_missing = float(r["test_missing_pct"])
        groups.setdefault(test_missing, []).append(r)  # type: ignore[arg-type]

    for test_missing, rs in groups.items():
        rs.sort(key=lambda x: float(x["train_missing_pct"]))  # type: ignore[index]
        groups[test_missing] = rs

    # Metadata for title
    seed = int(float(rows[0]["seed"]))  # type: ignore[arg-type]
    chain_length = int(float(rows[0]["chain_length"]))  # type: ignore[arg-type]
    burnin = int(float(rows[0]["burnin"]))  # type: ignore[arg-type]
    missing_mode = str(rows[0].get("missing_mode", ""))

    def series(test_missing: float, key: str) -> tuple[list[float], list[float]]:
        rs = groups.get(test_missing, [])
        xs = [float(r["train_missing_pct"]) * 100.0 for r in rs]  # type: ignore[index]
        ys = [float(r[key]) for r in rs]  # type: ignore[index]
        return xs, ys

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8])
    ax_ebv_total = fig.add_subplot(gs[0, 0])
    ax_ebv_indir = fig.add_subplot(gs[0, 1], sharex=ax_ebv_total)
    ax_epv_total = fig.add_subplot(gs[1, 0], sharex=ax_ebv_total)
    ax_epv_trait = fig.add_subplot(gs[1, 1], sharex=ax_ebv_total)
    ax_time = fig.add_subplot(gs[2, :], sharex=ax_ebv_total)

    fig.suptitle(
        f"Missing Omics Train/Test Benchmark (seed={seed}, chain={chain_length}, burnin={burnin}, mode={missing_mode})"
    )

    colors = {0.0: "#1f77b4", 1.0: "#d62728"}  # blue, red
    labels = {0.0: "Test omics 0% missing", 1.0: "Test omics 100% missing"}

    def plot_metric(ax, key: str, title: str) -> None:
        for test_missing in sorted(groups.keys()):
            xs, ys = series(test_missing, key)
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                markersize=4,
                color=colors.get(test_missing),
                label=labels.get(test_missing, f"test_missing={test_missing}"),
            )
        ax.set_title(title)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

    plot_metric(ax_ebv_total, "ebv_test_total", "EBV (test) vs genetic_total")
    plot_metric(ax_ebv_indir, "ebv_test_indirect", "EBV (test) vs genetic_indirect")
    plot_metric(ax_epv_total, "epv_test_total", "EPV (test) vs genetic_total")
    plot_metric(ax_epv_trait, "epv_test_trait", "EPV (test) vs trait1")

    # Runtime plot
    for test_missing in sorted(groups.keys()):
        xs, ys = series(test_missing, "time_seconds")
        ax_time.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            markersize=4,
            color=colors.get(test_missing),
        )
    ax_time.set_title("Runtime (seconds) per configuration")
    ax_time.set_ylabel("Seconds")
    ax_time.grid(True, alpha=0.3)

    ax_time.set_xlabel("Training omics missing (%)")
    ax_ebv_total.legend(loc="lower left", fontsize=9)

    out_png = here / "missing_omics_train_test_plot.png"
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(out_png, dpi=200)
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
