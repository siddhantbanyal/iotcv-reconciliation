#!/usr/bin/env python3
"""Run the IoT+CV composition study.

Joins the sensing and verification layers on one shared digital twin and scores
three arms on a common axis over N seeds: IoT-only, CV-only, and the composed
IoT-triggered-CV pipeline. The CV verifier is a calibrated stochastic oracle
seeded from the measured per-lane CV numbers (config ``cv_oracle``); the CV
model itself is not executed. See src/iotcv/compose.py.

Outputs:
  * results/composition_per_seed.csv    : one row per (seed, arm)
  * results/composition_summary.csv     : mean/std per arm
  * results/composition_budget.csv      : composed arm swept over CV budget
  * results/composition_sensitivity.csv : composed arm swept over CV accuracy

Example:
    python scripts/run_composition.py --config configs/default.yaml --seeds 20
"""

import argparse
import os

import pandas as pd

import _bootstrap  # noqa: F401  (sets sys.path)
from _bootstrap import ROOT, load_config

from iotcv.compose import run_composition

ARMS = ["iot_only", "cv_only", "composed"]
_METRICS = ["precision", "recall", "f1", "false_alert_rate",
            "alerts_per_tick", "cv_checks_per_tick", "n_alerts",
            "n_truth_locations"]


def _rows(results_by_seed):
    rows = []
    for seed, arms in results_by_seed:
        for arm in ARMS:
            r = arms[arm]
            rows.append({"seed": seed, "arm": arm,
                         **{m: getattr(r, m) for m in _METRICS}})
    return rows


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = {m: ["mean", "std"] for m in _METRICS}
    out = df.groupby("arm").agg(agg)
    out.columns = [f"{m}_{stat}" for m, stat in out.columns]
    # Preserve the IoT-only / CV-only / composed order.
    return out.reindex(ARMS).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = list(range(args.seeds))
    os.makedirs(args.outdir, exist_ok=True)
    budget = cfg["baselines"]["cv_budget_per_tick"]
    oc = cfg["cv_oracle"]
    print(f"[run] composition seeds={args.seeds} ticks={cfg['world']['ticks']} "
          f"budget={budget} cv_acc={oc['accuracy']} cv_fpc={oc['fp_correction']}")

    # --- three arms on the shared twin -------------------------------------
    results_by_seed = [(s, run_composition(cfg, s)) for s in seeds]
    df = pd.DataFrame(_rows(results_by_seed))
    per_seed_path = os.path.join(args.outdir, "composition_per_seed.csv")
    df.to_csv(per_seed_path, index=False)
    summary = _summarize(df)
    summary_path = os.path.join(args.outdir, "composition_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[ok] wrote {per_seed_path}")
    print(f"[ok] wrote {summary_path}")

    show = ["arm", "precision_mean", "recall_mean", "f1_mean",
            "false_alert_rate_mean", "alerts_per_tick_mean", "cv_checks_per_tick_mean"]
    print("\n=== Composition: three arms on the shared twin (mean over seeds) ===")
    print(summary[show].round(4).to_string(index=False))

    # --- composed arm swept over CV budget ---------------------------------
    budget_rows = []
    for b in [1, 2, 4, 8, 16, 32]:
        vals = [run_composition(cfg, s, budget=b)["composed"] for s in seeds]
        budget_rows.append({
            "budget": b,
            "precision_mean": sum(v.precision for v in vals) / len(vals),
            "recall_mean": sum(v.recall for v in vals) / len(vals),
            "f1_mean": sum(v.f1 for v in vals) / len(vals),
            "false_alert_rate_mean": sum(v.false_alert_rate for v in vals) / len(vals),
            "cv_checks_per_tick_mean": sum(v.cv_checks_per_tick for v in vals) / len(vals),
        })
    budget_path = os.path.join(args.outdir, "composition_budget.csv")
    pd.DataFrame(budget_rows).to_csv(budget_path, index=False)
    print(f"[ok] wrote {budget_path}")

    # --- composed arm sensitivity to the CV operating point ----------------
    # Do NOT hinge the headline on the fragile measured accuracy; show the
    # composed arm holds across a plausible band of verifier quality.
    sens_rows = []
    for acc, fpc in [(0.80, 0.90), (0.85, 0.93), (0.90, 0.96),
                     (0.92, 0.97), (0.95, 0.98)]:
        cfg2 = load_config(args.config)
        cfg2["cv_oracle"]["accuracy"] = acc
        cfg2["cv_oracle"]["fp_correction"] = fpc
        vals = [run_composition(cfg2, s)["composed"] for s in seeds]
        sens_rows.append({
            "cv_accuracy": acc, "cv_fp_correction": fpc,
            "precision_mean": sum(v.precision for v in vals) / len(vals),
            "recall_mean": sum(v.recall for v in vals) / len(vals),
            "f1_mean": sum(v.f1 for v in vals) / len(vals),
            "false_alert_rate_mean": sum(v.false_alert_rate for v in vals) / len(vals),
        })
    sens_path = os.path.join(args.outdir, "composition_sensitivity.csv")
    pd.DataFrame(sens_rows).to_csv(sens_path, index=False)
    print(f"[ok] wrote {sens_path}")


if __name__ == "__main__":
    main()
