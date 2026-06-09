#!/usr/bin/env python3
"""Plot PCA benchmark results."""

from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict
from typing import Iterable

try:
    import matplotlib.pyplot as plt
except Exception:
    print("matplotlib not available; please install matplotlib to generate plots")
    raise


FULL_METHOD_STYLES = {
    "eig_ser": {"color": "#1f77b4", "marker": "o", "linestyle": "-", "label": "eig_ser"},
    "eig_par[root_serial_jacobi]": {
        "color": "#1f77b4",
        "marker": "^",
        "linestyle": "--",
        "label": "eig_par[root_serial_jacobi]",
    },
    "eig_par[root_omp_jacobi]": {
        "color": "#2ca02c",
        "marker": "P",
        "linestyle": "-.",
        "label": "eig_par[root_omp_jacobi]",
    },
    "eig_par[mpi_jacobi]": {
        "color": "#17becf",
        "marker": "v",
        "linestyle": ":",
        "label": "eig_par[mpi_jacobi]",
    },
    "svd_ser": {"color": "#ff7f0e", "marker": "s", "linestyle": "-", "label": "svd_ser"},
    "svd_par": {"color": "#ff7f0e", "marker": "D", "linestyle": "--", "label": "svd_par"},
}

BASE_METHOD_STYLES = {
    "eig_par": {"color": "#1f77b4", "marker": "^", "linestyle": "--", "label": "eig_par"},
    "svd_par": {"color": "#ff7f0e", "marker": "D", "linestyle": "--", "label": "svd_par"},
}


def method_base(method: str) -> str:
    if "[" in method:
        return method.split("[", 1)[0]
    return method


def style_for_method(method: str) -> dict[str, str]:
    if method in FULL_METHOD_STYLES:
        return FULL_METHOD_STYLES[method]
    base = method_base(method)
    if base in BASE_METHOD_STYLES:
        return BASE_METHOD_STYLES[base]
    return {"color": "#555555", "marker": "o", "linestyle": "-", "label": method}


def read_results(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        header = reader.fieldnames or []
        has_explained_ratio = "explained_ratio" in header

        for row in reader:
            if not row:
                continue
            try:
                row["n"] = int(row["n"])
                row["d"] = int(row["d"])
                row["k"] = int(row["k"])
                row["num_procs"] = int(row["num_procs"])
                row["t_read"] = float(row["t_read"])
                row["t_center"] = float(row["t_center"])
                row["t_pca"] = float(row["t_pca"])
                row["t_total"] = float(row["t_total"])
                row["explained_ratio"] = (
                    float(row["explained_ratio"])
                    if has_explained_ratio and row["explained_ratio"]
                    else None
                )
            except (KeyError, TypeError, ValueError):
                continue

            if (
                row["t_read"] < 0.0
                or row["t_center"] < 0.0
                or row["t_pca"] < 0.0
                or row["t_total"] < 0.0
            ):
                continue
            if row["t_total"] + 1e-12 < max(row["t_read"], row["t_center"], row["t_pca"]):
                continue

            rows.append(row)
    return rows


def best_row(rows: Iterable[dict], *, procs: int | None = None) -> dict | None:
    candidates = [row for row in rows if procs is None or row["num_procs"] == procs]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row["t_total"])


def per_k_grid(ks: list[int]) -> tuple[int, int]:
    if len(ks) <= 2:
        return 1, len(ks)
    ncols = 2
    nrows = math.ceil(len(ks) / ncols)
    return nrows, ncols


def route_series(
    rows: list[dict], serial_method: str, parallel_method: str, k: int
) -> tuple[list[int], list[float], float | None]:
    serial_row = best_row(
        (row for row in rows if row["method"] == serial_method and row["k"] == k),
        procs=1,
    )
    parallel_rows = sorted(
        (
            row
            for row in rows
            if row["method"] == parallel_method and row["k"] == k and row["num_procs"] > 1
        ),
        key=lambda row: row["num_procs"],
    )

    xs: list[int] = []
    ys: list[float] = []
    serial_total = None

    if serial_row is not None:
        xs.append(1)
        ys.append(serial_row["t_total"])
        serial_total = serial_row["t_total"]
    else:
        parallel_one = best_row(
            (row for row in rows if row["method"] == parallel_method and row["k"] == k),
            procs=1,
        )
        if parallel_one is not None:
            xs.append(1)
            ys.append(parallel_one["t_total"])
            serial_total = parallel_one["t_total"]

    for row in parallel_rows:
        xs.append(row["num_procs"])
        ys.append(row["t_total"])

    return xs, ys, serial_total


def explained_ratio_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for method in sorted({row["method"] for row in rows}):
        method_rows = [
            row
            for row in rows
            if row["method"] == method and row["explained_ratio"] is not None
        ]
        by_k: dict[int, dict] = {}
        for row in method_rows:
            k = row["k"]
            if k not in by_k or row["num_procs"] < by_k[k]["num_procs"]:
                by_k[k] = row
        grouped[method] = [by_k[k] for k in sorted(by_k)]
    return grouped


def overlap_notes(grouped: dict[str, list[dict]], tol: float = 1e-8) -> list[str]:
    by_k: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for method, method_rows in grouped.items():
        for row in method_rows:
            by_k[row["k"]].append((method, row["explained_ratio"]))

    notes: list[str] = []
    for k, values in sorted(by_k.items()):
        if len(values) < 2:
            continue
        numbers = [value for _, value in values]
        if max(numbers) - min(numbers) <= tol:
            notes.append(f"k={k}: curves overlap")
    return notes


def build_summary_figure(rows: list[dict], out_png: str) -> None:
    ks = sorted({row["k"] for row in rows})
    nrows_k, ncols_k = per_k_grid(ks)
    plot_cols = max(2, ncols_k)

    fig = plt.figure(figsize=(6.4 * plot_cols, 4.8 * (nrows_k + 1)))
    grid = fig.add_gridspec(nrows_k + 1, plot_cols)

    ax_serial = fig.add_subplot(grid[0, 0])
    ax_explained = fig.add_subplot(grid[0, 1])

    for method in ("eig_ser", "svd_ser"):
        method_rows = sorted(
            (
                row
                for row in rows
                if row["num_procs"] == 1 and row["method"] == method
            ),
            key=lambda row: row["k"],
        )
        if not method_rows:
            continue
        style = style_for_method(method)
        ax_serial.plot(
            [row["k"] for row in method_rows],
            [row["t_total"] for row in method_rows],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            label=style["label"],
        )

    ax_serial.set_title("Serial total time vs k")
    ax_serial.set_xlabel("k")
    ax_serial.set_ylabel("Total time (s)")
    ax_serial.grid(True, alpha=0.3)
    ax_serial.legend()

    grouped_ratios = explained_ratio_rows(rows)
    has_ratio = any(grouped_ratios[method] for method in grouped_ratios)
    if has_ratio:
        for method in sorted(grouped_ratios):
            method_rows = grouped_ratios[method]
            if not method_rows:
                continue
            style = style_for_method(method)
            ax_explained.plot(
                [row["k"] for row in method_rows],
                [row["explained_ratio"] for row in method_rows],
                color=style["color"],
                marker=style["marker"],
                markerfacecolor="none",
                linestyle=style["linestyle"],
                linewidth=1.8,
                label=style["label"],
            )
        if overlap_notes(grouped_ratios):
            ax_explained.text(
                0.02,
                0.02,
                "Most curves overlap because all methods solve the same PCA target.",
                transform=ax_explained.transAxes,
                fontsize=9,
                color="dimgray",
                ha="left",
                va="bottom",
            )
        ax_explained.set_ylim(0.0, 1.05)
        ax_explained.legend(fontsize=9)
    else:
        ax_explained.text(
            0.5,
            0.5,
            "No explained_ratio column found.",
            ha="center",
            va="center",
            transform=ax_explained.transAxes,
        )
    ax_explained.set_title("Explained variance ratio vs k")
    ax_explained.set_xlabel("k")
    ax_explained.set_ylabel("Explained ratio")
    ax_explained.grid(True, alpha=0.3)

    eig_parallel_methods = sorted(
        {row["method"] for row in rows if method_base(row["method"]) == "eig_par"}
    )

    for index, k in enumerate(ks):
        row_idx = 1 + index // plot_cols
        col_idx = index % plot_cols
        ax = fig.add_subplot(grid[row_idx, col_idx])

        for serial_method, parallel_method in (("svd_ser", "svd_par"),):
            xs, ys, _ = route_series(rows, serial_method, parallel_method, k)
            if not xs:
                continue
            style = style_for_method(parallel_method)
            ax.plot(
                xs,
                ys,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                label=style["label"],
            )

        for eig_method in eig_parallel_methods:
            xs, ys, _ = route_series(rows, "eig_ser", eig_method, k)
            if not xs:
                continue
            style = style_for_method(eig_method)
            ax.plot(
                xs,
                ys,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                label=style["label"],
            )

        x_ticks = sorted({1, *[row["num_procs"] for row in rows if row["k"] == k]})
        ax.set_title(f"Total time vs processes (k={k})")
        ax.set_xlabel("Number of processes")
        ax.set_ylabel("Total time (s)")
        ax.set_xticks(x_ticks)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print("Wrote summary plot to", out_png)


def build_speedup_figure(rows: list[dict], out_png: str) -> None:
    ks = sorted({row["k"] for row in rows})
    nrows_k, ncols_k = per_k_grid(ks)
    fig, axes = plt.subplots(
        nrows_k,
        ncols_k,
        figsize=(6.4 * ncols_k, 4.8 * nrows_k),
        squeeze=False,
    )

    eig_parallel_methods = sorted(
        {row["method"] for row in rows if method_base(row["method"]) == "eig_par"}
    )

    for index, k in enumerate(ks):
        ax = axes[index // ncols_k][index % ncols_k]

        xs, ys, baseline = route_series(rows, "svd_ser", "svd_par", k)
        if xs and baseline is not None:
            style = style_for_method("svd_par")
            ax.plot(
                xs,
                [baseline / value for value in ys],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                label=style["label"],
            )

        for eig_method in eig_parallel_methods:
            xs, ys, baseline = route_series(rows, "eig_ser", eig_method, k)
            if not xs or baseline is None:
                continue
            style = style_for_method(eig_method)
            ax.plot(
                xs,
                [baseline / value for value in ys],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                label=style["label"],
            )

        x_ticks = sorted({1, *[row["num_procs"] for row in rows if row["k"] == k]})
        ax.set_title(f"Speedup vs processes (k={k})")
        ax.set_xlabel("Number of processes")
        ax.set_ylabel("Speedup")
        ax.set_xticks(x_ticks)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    for index in range(len(ks), nrows_k * ncols_k):
        axes[index // ncols_k][index % ncols_k].axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print("Wrote speedup plot to", out_png)


def plot(results_path: str, out_png: str | None = None) -> None:
    rows = read_results(results_path)
    if not rows:
        raise ValueError(f"No benchmark rows found in {results_path}")

    out_png = out_png or os.path.join(
        os.path.dirname(results_path), "benchmark_summary.png"
    )
    build_summary_figure(rows, out_png)

    root, ext = os.path.splitext(out_png)
    speedup_png = (
        f"{root.replace('summary', 'speedup')}{ext}"
        if "summary" in root
        else f"{root}_speedup{ext}"
    )
    build_speedup_figure(rows, speedup_png)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: plot.py results.csv [out.png]")
        sys.exit(1)
    plot(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
