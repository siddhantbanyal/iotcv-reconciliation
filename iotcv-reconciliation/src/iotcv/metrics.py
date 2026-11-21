"""Metrics: score the IoT detectors against ground truth and aggregate over seeds.

Characterizes the *raw flag stream* the detectors emit, before any CV
verification or acceptance step. Every flag is scored directly against the
recorded ground truth:

  * Flag quality  : precision, recall, F1, false-flag rate (flags that pointed
                    at a location that was not actually discrepant).
  * Timeliness    : time-to-flag (ticks from discrepancy onset to first flag).
  * Exposure      : location-ticks a discrepancy persisted before it was flagged.
  * Load          : flags per tick and flags per discrepant location: the
                    verification workload the CV layer would receive.

Two granularities matter:
  * flag-level precision (fraction of individual flags that hit a truly
    discrepant location) characterizes how noisy the stream handed to CV is;
  * location-episode recall (fraction of discrepant locations ever flagged)
    characterizes how much the detectors miss.

The ``project_post_cv`` helper turns measured detector precision into a
*projected* post-verification false-alert rate under a hypothetical CV verifier,
giving the expected end-to-end effect without executing CV.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .simulate import SimRun

# Map each detector lane to the location-key prefix it produces.
LANE_PREFIX = {"weight": "bin", "location": "asset", "dwell": "item"}


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _first_onset(truth_per_tick: List[set]) -> Dict[str, int]:
    """First tick each location appears as discrepant (episode start)."""
    onset: Dict[str, int] = {}
    for tick, truth in enumerate(truth_per_tick):
        for loc in truth:
            onset.setdefault(loc, tick)
    return onset


def score_run(run: SimRun, lane: Optional[str] = None) -> Dict[str, float]:
    """Detector-quality metrics for one run.

    If ``lane`` is given, restrict to that lane's flags and that lane's
    discrepancy universe (bin/asset/item); otherwise score all lanes together.
    """
    prefix = LANE_PREFIX.get(lane) if lane else None

    def in_scope(loc: str) -> bool:
        return prefix is None or loc.split(":", 1)[0] == prefix

    # --- flag-level precision: does each raised flag hit a real discrepancy? ---
    n_flags = 0
    n_flags_tp = 0
    # location-episode detection: first tick a location was flagged *while* truly
    # discrepant (a true-positive flag).
    first_flag_tp: Dict[str, int] = {}
    for tick, flags in enumerate(run.flags_per_tick):
        for f in flags:
            if not in_scope(f.location):
                continue
            n_flags += 1
            truly = run.truth_at_tick[tick].get(f.location, False)
            if truly:
                n_flags_tp += 1
                first_flag_tp.setdefault(f.location, tick)

    # --- ground-truth universe: locations discrepant at some point (in scope) ---
    truth_locations = set()
    for truth in run.truth_per_tick:
        truth_locations |= {loc for loc in truth if in_scope(loc)}
    n_truth = max(1, len(truth_locations))

    # Recall = fraction of discrepant locations the detectors ever caught.
    recall = len(first_flag_tp) / n_truth
    # Precision = fraction of raised flags that hit a real discrepancy.
    precision = (n_flags_tp / n_flags) if n_flags else 0.0
    f1 = _f1(precision, recall)
    false_flag_rate = (1.0 - precision) if n_flags else 0.0

    # --- time-to-flag: onset -> first true-positive flag at that location -------
    onset = _first_onset([{loc for loc in t if in_scope(loc)}
                          for t in run.truth_per_tick])
    ttf = [first_flag_tp[loc] - onset[loc]
           for loc in first_flag_tp
           if loc in onset and first_flag_tp[loc] >= onset[loc]]
    time_to_flag = float(np.mean(ttf)) if ttf else float("nan")

    # --- exposure: location-ticks discrepant AND not yet flagged ---------------
    exposure = 0
    for tick, truth in enumerate(run.truth_per_tick):
        for loc in truth:
            if not in_scope(loc):
                continue
            d = first_flag_tp.get(loc)
            if d is None or tick < d:
                exposure += 1
    exposure_per_loc = exposure / n_truth

    # --- load: verification workload the CV layer would receive ----------------
    n_ticks = max(1, len(run.flags_per_tick))
    flags_per_tick = n_flags / n_ticks
    flags_per_truth_loc = n_flags / n_truth

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_flag_rate": false_flag_rate,
        "time_to_flag": time_to_flag,
        "exposure_per_loc": exposure_per_loc,
        "flags_per_tick": flags_per_tick,
        "flags_per_truth_loc": flags_per_truth_loc,
        "n_flags": float(n_flags),
        "n_truth_locations": float(len(truth_locations)),
    }


def score_run_by_lane(run: SimRun) -> Dict[str, Dict[str, float]]:
    """Per-lane detector-quality metrics (weight / location / dwell)."""
    return {lane: score_run(run, lane=lane) for lane in LANE_PREFIX}


def aggregate(per_seed: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate a list of per-seed metric dicts into mean/std per metric."""
    if not per_seed:
        return {}
    keys = per_seed[0].keys()
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = np.array([d[k] for d in per_seed], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            out[k] = {"mean": float("nan"), "std": float("nan")}
        else:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def project_post_cv(precision: float, cv_accuracy: float,
                    cv_fp_correction: float) -> Dict[str, float]:
    """Project the post-verification operating point for a hypothetical CV layer.

    Given the measured detector precision ``pi`` (share of flags that are true
    discrepancies) and a CV verifier that confirms a true discrepancy with
    probability ``a`` and correctly rejects a false flag with probability ``f``,
    the accepted-set precision after keeping only CV-confirmed flags is (Bayes):

        P(true | confirm) = a*pi / ( a*pi + (1-f)*(1-pi) )

    The projected false-alert rate is its complement. This is an analytic
    projection, not a simulated result; see compose.py for the simulated joint
    evaluation.
    """
    pi = precision
    a, f = cv_accuracy, cv_fp_correction
    denom = a * pi + (1.0 - f) * (1.0 - pi)
    post_precision = (a * pi / denom) if denom > 0 else 0.0
    pre_far = 1.0 - pi
    post_far = 1.0 - post_precision
    fold = (pre_far / post_far) if post_far > 0 else float("inf")
    return {
        "pre_precision": pi,
        "pre_false_alert_rate": pre_far,
        "post_precision": post_precision,
        "post_false_alert_rate": post_far,
        "false_alert_fold_reduction": fold,
    }


def flag_score_label_pairs(run: SimRun):
    """Return (raw_confidence, is_true_positive) for every raised flag.

    Used to fit and evaluate the confidence calibrators: a flag's raw confidence
    is the detector's score, and its label is whether the flagged location was
    truly discrepant at the tick it fired.
    """
    scores: List[float] = []
    labels: List[int] = []
    for tick, flags in enumerate(run.flags_per_tick):
        lut = run.truth_at_tick[tick]
        for f in flags:
            scores.append(float(f.confidence))
            labels.append(1 if lut.get(f.location, False) else 0)
    return np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
