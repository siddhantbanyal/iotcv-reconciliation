#!/usr/bin/env python3
"""Run the IoT inventory-reconciliation experiment (sensing layer).

Evaluates the edge detectors against ground truth over N seeds and (optionally)
IoT-side ablation sweeps, writing tidy CSVs to results/. Also exports the
IoT->CV handoff flags as JSON Lines and a projected post-verification operating
point.

This script reports detector quality (precision / recall / F1 / false-flag rate /
time-to-flag / flag load). The joint IoT+CV evaluation is in run_composition.py.

Examples
--------
    python scripts/run_experiment.py --config configs/default.yaml --seeds 20
    python scripts/run_experiment.py --config configs/weight.yaml --seeds 15 --ablations
"""

import argparse
import os

import pandas as pd

import _bootstrap  # noqa: F401  (sets sys.path)
from _bootstrap import ROOT, load_config

from iotcv.handoff import write_flags_jsonl
from iotcv.metrics import project_post_cv
from iotcv.runner import (run_ablation, run_calibration_eval,
                          run_detector_eval, run_per_lane_eval,
                          run_triage_curve, summarize)
from iotcv.simulate import collect_flags, run_simulation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    ap.add_argument("--seeds", type=int, default=20, help="number of seeds (0..N-1)")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--ablations", action="store_true",
                    help="also run detector-threshold and sensor-noise sweeps")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = list(range(args.seeds))
    os.makedirs(args.outdir, exist_ok=True)
    lane = cfg.get("lane", "all")

    print(f"[run] config={os.path.basename(args.config)} lane={lane} "
          f"seeds={args.seeds} ticks={cfg['world']['ticks']}")

    # --- headline: detector quality, all lanes together --------------------
    df = run_detector_eval(cfg, seeds)
    per_seed_path = os.path.join(args.outdir, f"detector_{lane}_per_seed.csv")
    summary_path = os.path.join(args.outdir, f"detector_{lane}_summary.csv")
    df.to_csv(per_seed_path, index=False)
    summary = summarize(df)
    summary.to_csv(summary_path, index=False)
    print(f"[ok] wrote {per_seed_path}")
    print(f"[ok] wrote {summary_path}")

    show = ["precision_mean", "recall_mean", "f1_mean", "false_flag_rate_mean",
            "time_to_flag_mean", "exposure_per_loc_mean", "flags_per_tick_mean",
            "flags_per_truth_loc_mean"]
    show = [c for c in show if c in summary.columns]
    print("\n=== Detector quality (mean over seeds, all lanes) ===")
    print(summary[show].round(3).to_string(index=False))

    # --- per-lane detector quality -----------------------------------------
    lane_df = run_per_lane_eval(cfg, seeds)
    lane_per_seed = os.path.join(args.outdir, f"detector_by_lane_{lane}_per_seed.csv")
    lane_summary = summarize(lane_df, group="lane")
    lane_df.to_csv(lane_per_seed, index=False)
    lane_summary_path = os.path.join(args.outdir, f"detector_by_lane_{lane}_summary.csv")
    lane_summary.to_csv(lane_summary_path, index=False)
    print(f"[ok] wrote {lane_summary_path}")

    # --- projected post-CV operating point (Bayesian, not simulated) -------
    cvp = cfg["cv_projection"]
    pi = float(summary["precision_mean"].iloc[0])
    proj = project_post_cv(pi, cvp["accuracy"], cvp["fp_correction"])
    proj_row = dict(proj)
    proj_row.update({"cv_accuracy": cvp["accuracy"],
                     "cv_fp_correction": cvp["fp_correction"]})
    proj_path = os.path.join(args.outdir, f"cv_projection_{lane}.csv")
    pd.DataFrame([proj_row]).to_csv(proj_path, index=False)
    print(f"[ok] wrote {proj_path}")
    print(f"\n=== Projected CV effect (assumed a={cvp['accuracy']}, "
          f"f={cvp['fp_correction']}) ===")
    print(f"  pre-verify false-alert rate  : {proj['pre_false_alert_rate']:.3f}")
    print(f"  post-verify false-alert rate : {proj['post_false_alert_rate']:.3f}")
    print(f"  projected fold reduction     : {proj['false_alert_fold_reduction']:.1f}x")

    # --- confidence calibration: fit on held-out seeds, report ECE --------
    if len(seeds) >= 4:
        half = len(seeds) // 2
        cal_summary, reliability = run_calibration_eval(
            cfg, fit_seeds=seeds[:half], eval_seeds=seeds[half:])
        pd.DataFrame([cal_summary]).to_csv(
            os.path.join(args.outdir, f"calibration_{lane}.csv"), index=False)
        reliability.to_csv(
            os.path.join(args.outdir, f"reliability_{lane}.csv"), index=False)
        print(f"\n=== Confidence calibration (fit/eval split) ===")
        print(f"  ECE raw={cal_summary['ece_raw']:.4f}  "
              f"platt={cal_summary['ece_platt']:.4f}  "
              f"isotonic={cal_summary['ece_isotonic']:.4f}")

    # --- budgeted value-of-information triage curve -----------------------
    triage = run_triage_curve(cfg, seeds[: min(len(seeds), 10)],
                              budgets=[1, 2, 4, 8, 16, 32])
    triage.to_csv(os.path.join(args.outdir, f"triage_{lane}.csv"), index=False)
    print(f"\n=== Budgeted triage (expected resolved vs budget) ===")
    print(triage.round(3).to_string(index=False))

    # --- export handoff flags for the CV team ------------------------------
    run = run_simulation(cfg, seeds[0])
    sample = collect_flags(run)[:500]
    flags_path = os.path.join(args.outdir, f"handoff_flags_{lane}_seed0.jsonl")
    write_flags_jsonl(sample, flags_path)
    print(f"\n[ok] exported {len(sample)} sample handoff flags -> {flags_path}")

    # --- IoT-side ablation sweeps (over the new design parameters) --------
    if args.ablations:
        ab_seeds = seeds[: min(len(seeds), 10)]
        sweeps = {
            "weight_arl0": ("detectors.weight.target_arl0",
                            [100, 250, 500, 1000, 2000]),
            "weight_noise": ("sensors.weight.cell_noise_g",
                             [2.0, 5.0, 8.0, 12.0, 20.0]),
            "location_alpha": ("detectors.location.sprt_alpha",
                               [0.001, 0.005, 0.01, 0.05, 0.1]),
            "dwell_eps": ("detectors.dwell.survival_eps",
                          [0.005, 0.01, 0.02, 0.05, 0.1]),
        }
        for name, (key, grid) in sweeps.items():
            ab = run_ablation(cfg, key, grid, ab_seeds)
            path = os.path.join(args.outdir, f"ablation_{name}_{lane}.csv")
            ab.to_csv(path, index=False)
            print(f"[ok] wrote {path}")


if __name__ == "__main__":
    main()
