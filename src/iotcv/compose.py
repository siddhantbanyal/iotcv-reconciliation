"""Composition study: join the IoT and CV layers on one shared digital twin.

Cheap persistent sensing narrows the search space and targeted vision verifies
it. This module measures the two layers together on the same simulated world,
complementing the per-layer evaluations.

We reuse the sensing simulator (:func:`simulate.run_simulation`) so every arm
sees identical world state, events, sensor noise, and IoT flags per seed. The CV
verifier is a **calibrated stochastic oracle** whose confirm/reject rates are
seeded from the measured per-lane CV numbers (config ``cv_oracle``); the CV
model itself is not executed. Three arms are scored on one axis at equal
verification budget:

  * ``iot_only``: accept every IoT flag at face value (no verification).
  * ``cv_only``: signal-blind rotating CV scan of B locations/tick.
  * ``composed``: IoT flags -> top-B value-of-information handoff -> CV
                     verifies -> accept only CV-confirmed flags.

Scoring mirrors :func:`metrics.score_run`: an *alert* is an accepted (location,
tick); it is a true positive iff the location was truly discrepant at that tick.
Recall is the fraction of ever-discrepant locations that received at least one
true-positive alert; precision is the true-positive share of alerts; the
false-alert rate is its complement. Everything is driven by the run seed, so a
seed reproduces an arm exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .metrics import _f1
from .simulate import SimRun, run_simulation
from .triage import select_topk


@dataclass
class ArmResult:
    """Alert-level quality for one arm on one seed."""

    arm: str
    precision: float
    recall: float
    f1: float
    false_alert_rate: float
    alerts_per_tick: float
    cv_checks_per_tick: float
    n_alerts: int
    n_truth_locations: int


def _cv_confirms(rng: np.random.Generator, truly: bool,
                 accuracy: float, fp_correction: float) -> bool:
    """One stochastic CV verdict.

    A truly-discrepant location is confirmed with probability ``accuracy``; a
    non-discrepant location is (incorrectly) confirmed with probability
    ``1 - fp_correction``. This is the mechanism by which verification trims the
    IoT false-alert rate: most false flags are rejected before they become alerts.
    """
    roll = rng.random()
    if truly:
        return roll < accuracy
    return roll >= fp_correction


def _score(arm: str, accepted: List[tuple], first_tp: Dict[str, int],
           run: SimRun, cv_checks: int) -> ArmResult:
    """Turn an accepted-alert list into precision/recall/F1/false-alert-rate.

    ``accepted`` is a list of (tick, location, is_true_positive) triples; only
    accepted alerts appear. ``first_tp`` maps a location to the first tick it got
    a true-positive alert (drives recall).
    """
    n_alerts = len(accepted)
    n_tp = sum(1 for _, _, tp in accepted if tp)
    precision = (n_tp / n_alerts) if n_alerts else 0.0

    # Ground-truth universe: every location discrepant at some tick.
    truth_locations = set()
    for truth in run.truth_per_tick:
        truth_locations |= truth
    n_truth = max(1, len(truth_locations))
    recall = len(first_tp) / n_truth

    f1 = _f1(precision, recall)
    false_alert_rate = (1.0 - precision) if n_alerts else 0.0
    n_ticks = max(1, len(run.flags_per_tick))
    return ArmResult(
        arm=arm,
        precision=precision,
        recall=recall,
        f1=f1,
        false_alert_rate=false_alert_rate,
        alerts_per_tick=n_alerts / n_ticks,
        cv_checks_per_tick=cv_checks / n_ticks,
        n_alerts=n_alerts,
        n_truth_locations=len(truth_locations),
    )


def arm_iot_only(run: SimRun) -> ArmResult:
    """Accept every IoT flag at face value: high recall, but the raw false-flag
    rate flows straight through to the alert stream (no verification)."""
    accepted: List[tuple] = []
    first_tp: Dict[str, int] = {}
    for tick, flags in enumerate(run.flags_per_tick):
        lut = run.truth_at_tick[tick]
        for f in flags:
            truly = lut.get(f.location, False)
            accepted.append((tick, f.location, truly))
            if truly:
                first_tp.setdefault(f.location, tick)
    return _score("iot_only", accepted, first_tp, run, cv_checks=0)


def arm_cv_only(run: SimRun, seed: int, budget: int,
                accuracy: float, fp_correction: float) -> ArmResult:
    """Signal-blind CV: inspect a rotating sample of B locations/tick, with no
    IoT prior. Spends the same budget as the composed arm but wastes most of it
    on locations that are fine, so it catches discrepancies only when the
    rotation happens to land on one while it is active."""
    rng = np.random.default_rng(seed + 20_000)
    locs = ([f"bin:{b}" for b in run.final_world.bins]
            + [f"asset:{a}" for a in run.final_world.assets])
    accepted: List[tuple] = []
    first_tp: Dict[str, int] = {}
    cursor = 0
    cv_checks = 0
    for tick in range(len(run.flags_per_tick)):
        lut = run.truth_at_tick[tick]
        for _ in range(min(budget, len(locs)) if locs else 0):
            loc = locs[cursor % len(locs)]
            cursor += 1
            cv_checks += 1
            truly = lut.get(loc, False)
            if _cv_confirms(rng, truly, accuracy, fp_correction):
                accepted.append((tick, loc, truly))
                if truly:
                    first_tp.setdefault(loc, tick)
    return _score("cv_only", accepted, first_tp, run, cv_checks=cv_checks)


def arm_composed(run: SimRun, seed: int, budget: int,
                 accuracy: float, fp_correction: float) -> ArmResult:
    """The proposal: IoT flags -> top-B value-of-information handoff -> CV
    verifies each forwarded flag -> accept only CV-confirmed flags. The IoT prior
    concentrates the budget on likely-real locations; CV filters the residual
    false flags. Recall is retained (the flags come from the same high-recall
    detectors) while precision rises toward the verifier's."""
    rng = np.random.default_rng(seed + 30_000)
    accepted: List[tuple] = []
    first_tp: Dict[str, int] = {}
    cv_checks = 0
    for tick, flags in enumerate(run.flags_per_tick):
        lut = run.truth_at_tick[tick]
        # Budgeted value-of-information handoff: forward the top-B by confidence.
        forwarded = select_topk(flags, budget)
        for f in forwarded:
            cv_checks += 1
            truly = lut.get(f.location, False)
            if _cv_confirms(rng, truly, accuracy, fp_correction):
                accepted.append((tick, f.location, truly))
                if truly:
                    first_tp.setdefault(f.location, tick)
    return _score("composed", accepted, first_tp, run, cv_checks=cv_checks)


def run_composition(cfg: dict, seed: int,
                    budget: Optional[int] = None) -> Dict[str, ArmResult]:
    """Build one shared reality for ``seed`` and score all three arms on it."""
    if budget is None:
        budget = cfg["baselines"]["cv_budget_per_tick"]
    oc = cfg["cv_oracle"]
    acc, fpc = oc["accuracy"], oc["fp_correction"]
    run = run_simulation(cfg, seed)
    return {
        "iot_only": arm_iot_only(run),
        "cv_only": arm_cv_only(run, seed, budget, acc, fpc),
        "composed": arm_composed(run, seed, budget, acc, fpc),
    }
