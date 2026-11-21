"""Experiment runner: multi-seed detector evaluation + IoT-side ablation sweeps.

Produces tidy pandas DataFrames that ``scripts/`` write to CSV and ``plots.py``
renders. Everything is seed-driven and reproducible.

This runner evaluates the IoT detectors only. Entry points:

  * ``run_detector_eval``   : the headline table, detector quality (all lanes)
                              over N seeds, mean +/- std.
  * ``run_per_lane_eval``   : per-lane detector quality (weight/location/dwell).
  * ``run_ablation``        : sweep one IoT config knob (detector threshold or
                              sensor noise) over a grid, recomputing metrics.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Sequence

import pandas as pd

from .metrics import aggregate, score_run, score_run_by_lane  # noqa: F401
from .simulate import run_simulation


def _deep_set(cfg: dict, dotted_key: str, value: Any) -> dict:
    """Return a deep-copied cfg with cfg[a][b][c]=value for key 'a.b.c'."""
    out = copy.deepcopy(cfg)
    node = out
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = value
    return out


def run_detector_eval(cfg: dict, seeds: Sequence[int]) -> pd.DataFrame:
    """Score the detectors (all lanes together) over all seeds. Long-form rows."""
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        run = run_simulation(cfg, seed)
        m = score_run(run)
        m.update({"seed": seed})
        rows.append(m)
    return pd.DataFrame(rows)


def run_per_lane_eval(cfg: dict, seeds: Sequence[int]) -> pd.DataFrame:
    """Score each lane's detector separately over all seeds. Long-form rows."""
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        run = run_simulation(cfg, seed)
        by_lane = score_run_by_lane(run)
        for lane, m in by_lane.items():
            row = dict(m)
            row.update({"lane": lane, "seed": seed})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, group: str = None) -> pd.DataFrame:
    """Mean +/- std across seeds. If ``group`` given (e.g. 'lane'), group by it."""
    ignore = {"seed"}
    if group:
        ignore.add(group)
    metric_cols = [c for c in df.columns if c not in ignore]
    if group:
        g = df.groupby(group)[metric_cols]
    else:
        # single aggregate row
        summary = df[metric_cols].agg(["mean", "std"])
        flat = {}
        for c in metric_cols:
            flat[f"{c}_mean"] = summary.loc["mean", c]
            flat[f"{c}_std"] = summary.loc["std", c]
        return pd.DataFrame([flat])
    summary = g.agg(["mean", "std"])
    summary.columns = [f"{c}_{stat}" for c, stat in summary.columns]
    return summary.reset_index()


def run_ablation(cfg: dict, dotted_key: str, grid: Sequence[Any],
                 seeds: Sequence[int]) -> pd.DataFrame:
    """Sweep one IoT config knob over ``grid``; record detector metrics per seed."""
    rows: List[Dict[str, Any]] = []
    for value in grid:
        swept = _deep_set(cfg, dotted_key, value)
        for seed in seeds:
            run = run_simulation(swept, seed)
            m = score_run(run)
            m.update({"seed": seed, "ablation_key": dotted_key,
                      "ablation_value": value})
            rows.append(m)
    return pd.DataFrame(rows)


def run_calibration_eval(cfg: dict, fit_seeds: Sequence[int],
                         eval_seeds: Sequence[int]):
    """Fit confidence calibrators on held-out seeds and report ECE before/after.

    Detector confidences are pooled over ``fit_seeds`` to fit a Platt scaler and
    an isotonic map, then evaluated on disjoint ``eval_seeds``. Returns a summary
    dict plus the reliability table (raw) for a diagram.
    """
    import numpy as np

    from .calibration import (IsotonicCalibrator, PlattScaler,
                              expected_calibration_error)
    from .metrics import flag_score_label_pairs

    fit_s, fit_y = [], []
    for seed in fit_seeds:
        s, y = flag_score_label_pairs(run_simulation(cfg, seed))
        fit_s.append(s)
        fit_y.append(y)
    fit_s = np.concatenate(fit_s) if fit_s else np.array([])
    fit_y = np.concatenate(fit_y) if fit_y else np.array([])

    ev_s, ev_y = [], []
    for seed in eval_seeds:
        s, y = flag_score_label_pairs(run_simulation(cfg, seed))
        ev_s.append(s)
        ev_y.append(y)
    ev_s = np.concatenate(ev_s) if ev_s else np.array([])
    ev_y = np.concatenate(ev_y) if ev_y else np.array([])

    platt = PlattScaler().fit(fit_s, fit_y)
    iso = IsotonicCalibrator().fit(fit_s, fit_y)

    raw_ece, raw_rows = expected_calibration_error(ev_s, ev_y)
    platt_ece, _ = expected_calibration_error(platt.transform(ev_s), ev_y)
    iso_ece, iso_rows = expected_calibration_error(iso.transform(ev_s), ev_y)

    summary = {
        "n_fit_flags": int(len(fit_s)),
        "n_eval_flags": int(len(ev_s)),
        "ece_raw": raw_ece,
        "ece_platt": platt_ece,
        "ece_isotonic": iso_ece,
        "platt_a": platt.a,
        "platt_b": platt.b,
    }
    reliability = pd.DataFrame(
        raw_rows, columns=["bin_lo", "bin_hi", "count", "mean_conf", "empirical_acc"])
    return summary, reliability


def run_triage_curve(cfg: dict, seeds: Sequence[int],
                     budgets: Sequence[int]):
    """Per-tick budgeted forwarding: expected discrepancies resolved vs budget B.

    Compares the additive top-B policy (VoI-optimal) against the submodular
    per-zone coverage greedy, averaged over ticks and seeds. Demonstrates the
    value-of-information framing of the handoff.
    """
    import numpy as np

    from .triage import (expected_resolved, greedy_submodular_zones,
                         select_topk)

    rows = []
    for B in budgets:
        add_vals, cov_vals, forwarded = [], [], []
        for seed in seeds:
            run = run_simulation(cfg, seed)
            for flags in run.flags_per_tick:
                if not flags:
                    continue
                topk = select_topk(flags, B)
                add_vals.append(expected_resolved(topk))
                _, cov = greedy_submodular_zones(flags, B)
                cov_vals.append(cov)
                forwarded.append(min(B, len(flags)))
        rows.append({
            "budget": B,
            "expected_resolved_additive": float(np.mean(add_vals)) if add_vals else 0.0,
            "expected_zone_coverage": float(np.mean(cov_vals)) if cov_vals else 0.0,
            "mean_forwarded": float(np.mean(forwarded)) if forwarded else 0.0,
        })
    return pd.DataFrame(rows)
