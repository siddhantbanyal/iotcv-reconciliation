#!/usr/bin/env python3
"""Render figures from results/ CSVs into figures/.

Always writes plot-ready CSVs; renders PNGs only if matplotlib is installed.
Plots the per-lane detector quality and the IoT-side ablation curves.

Example:
    python scripts/make_plots.py --lane all
"""

import argparse
import glob
import os

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from iotcv.plots import ablation_curve, bar_by_lane, have_matplotlib


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lane", default="all", help="lane tag used in result filenames")
    ap.add_argument("--resultsdir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--figdir", default=os.path.join(ROOT, "figures"))
    args = ap.parse_args()

    if not have_matplotlib():
        print("[warn] matplotlib not available: writing plot-ready CSVs only.")

    # Per-lane detector-quality bars.
    lane_summary_path = os.path.join(
        args.resultsdir, f"detector_by_lane_{args.lane}_summary.csv")
    if os.path.exists(lane_summary_path):
        summary = pd.read_csv(lane_summary_path)
        for metric, title in [
            ("f1", "Detector F1 by lane"),
            ("precision", "Detector precision by lane"),
            ("recall", "Detector recall by lane"),
            ("false_flag_rate", "False-flag rate by lane (lower is better)"),
            ("time_to_flag", "Time to flag by lane (ticks, lower is better)"),
            ("flags_per_tick", "Flag load per tick by lane"),
        ]:
            out = os.path.join(args.figdir, f"{args.lane}_{metric}.png")
            p = bar_by_lane(summary, metric, out, title=title)
            print(f"[ok] {p}")
    else:
        print(f"[warn] no per-lane summary at {lane_summary_path}; "
              "run run_experiment.py first")

    # IoT-side ablation curves: pick the most informative metric per sweep.
    ablation_metrics = {
        "weight_arl0": "false_flag_rate",
        "weight_noise": "f1",
        "location_alpha": "false_flag_rate",
        "dwell_eps": "false_flag_rate",
    }
    for path in glob.glob(os.path.join(args.resultsdir, f"ablation_*_{args.lane}.csv")):
        name = os.path.basename(path)[len("ablation_"):-len(f"_{args.lane}.csv")]
        df = pd.read_csv(path)
        metric = ablation_metrics.get(name, "f1")
        out = os.path.join(args.figdir, f"{args.lane}_ablation_{name}_{metric}.png")
        p = ablation_curve(df, metric, out, title=f"{metric} vs {name}")
        print(f"[ok] {p}")


if __name__ == "__main__":
    main()
