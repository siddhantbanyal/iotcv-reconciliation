"""Plotting. Degrades gracefully when matplotlib is unavailable.

Every function ALWAYS writes a plot-ready CSV next to its figure, so results are
usable for the paper even in environments without matplotlib. If matplotlib is
importable, it also renders a PNG.

These plots characterize the IoT detectors: per-lane quality bars and IoT-side
ablation curves.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

try:  # optional dependency
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_MPL = False


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def bar_by_lane(summary: pd.DataFrame, metric: str, out_png: str,
                title: Optional[str] = None) -> str:
    """Bar chart of one metric (mean +/- std) across lanes. Writes CSV + PNG.

    ``summary`` is the per-lane summary frame (one row per lane, with
    ``<metric>_mean`` / ``<metric>_std`` columns and a ``lane`` column).
    """
    _ensure_dir(out_png)
    mean_col, std_col = f"{metric}_mean", f"{metric}_std"
    data = summary[["lane", mean_col, std_col]].copy()
    csv_path = out_png.replace(".png", ".csv")
    data.to_csv(csv_path, index=False)
    if not _HAVE_MPL:
        return csv_path
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(data["lane"], data[mean_col], yerr=data[std_col].fillna(0.0), capsize=4)
    ax.set_ylabel(metric)
    ax.set_title(title or metric)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


def ablation_curve(df: pd.DataFrame, metric: str, out_png: str,
                   title: Optional[str] = None) -> str:
    """Line plot of one metric vs an ablation knob (mean over seeds). CSV + PNG."""
    _ensure_dir(out_png)
    g = df.groupby("ablation_value")[metric].agg(["mean", "std"]).reset_index()
    csv_path = out_png.replace(".png", ".csv")
    g.to_csv(csv_path, index=False)
    if not _HAVE_MPL:
        return csv_path
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(g["ablation_value"], g["mean"], yerr=g["std"].fillna(0.0),
                marker="o", capsize=4)
    key = df["ablation_key"].iloc[0] if "ablation_key" in df else "value"
    ax.set_xlabel(key)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} vs {key}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


def have_matplotlib() -> bool:
    return _HAVE_MPL
