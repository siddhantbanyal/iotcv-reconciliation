"""Budgeted handoff as a value-of-information selection problem.

Per tick the detectors raise a set of flags, but a downstream verifier can only
inspect B of them (a bounded budget). Which B should we forward? We treat this as
maximizing the expected number of *true* discrepancies resolved. If flag j has
calibrated posterior p_j = P(truly discrepant | evidence), forwarding a set S of
size <= B yields expected value

    V(S) = sum_{j in S} p_j                              (independent locations)

which is a modular (additive) set function; its constrained maximizer is exactly
the top-B flags by posterior. This is the value-of-information justification for
confidence-ranked triage: greedy-by-posterior is provably optimal here, not a
heuristic. ``select_topk`` implements it.

When flags share a physical zone, a single camera pass can cover several at once,
so marginal value has diminishing returns and the objective becomes submodular:

    V(S) = sum_{z} [ 1 - prod_{j in S, zone(j)=z} (1 - p_j) ]   (per-zone coverage)

i.e. the expected number of zones containing at least one real discrepancy that
we chose to inspect. Maximizing a monotone submodular function under a cardinality
constraint is NP-hard, but the greedy algorithm (Nemhauser, Wolsey & Fisher 1978)
achieves a (1 - 1/e) approximation. ``greedy_submodular_zones`` implements it and
returns both the chosen flags and the achieved objective value.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def select_topk(flags: Sequence, budget: int) -> List:
    """Modular VoI: forward the top-`budget` flags by (calibrated) confidence.

    Optimal for the additive objective V(S) = sum p_j under |S| <= budget.
    """
    if budget <= 0:
        return []
    ranked = sorted(flags, key=lambda f: f.confidence, reverse=True)
    return list(ranked[:budget])


def expected_resolved(flags: Sequence) -> float:
    """Expected true discrepancies among a set, under calibrated posteriors."""
    return float(sum(f.confidence for f in flags))


def greedy_submodular_zones(flags: Sequence, budget: int) -> Tuple[List, float]:
    """Greedy maximization of monotone submodular per-zone coverage under budget.

    Objective: expected number of zones with >=1 real discrepancy inspected,
        V(S) = sum_z [ 1 - prod_{j in S : zone(j)=z} (1 - p_j) ].
    Greedy picks, at each step, the flag with the largest marginal gain. By
    Nemhauser-Wolsey-Fisher this is within (1 - 1/e) ~ 0.632 of optimal.

    Returns (selected_flags, achieved_value).
    """
    if budget <= 0 or not flags:
        return [], 0.0
    remaining = list(flags)
    selected: List = []
    # per-zone "miss probability" product for currently selected flags
    zone_miss: Dict[int, float] = {}

    def value_with(zm: Dict[int, float]) -> float:
        return float(sum(1.0 - m for m in zm.values()))

    cur_val = 0.0
    while remaining and len(selected) < budget:
        best_gain = -1.0
        best_idx = -1
        for i, f in enumerate(remaining):
            z = getattr(f, "zone", -1)
            p = float(f.confidence)
            new_miss = zone_miss.get(z, 1.0) * (1.0 - p)
            # marginal gain = drop in that zone's miss probability
            gain = zone_miss.get(z, 1.0) - new_miss
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        f = remaining.pop(best_idx)
        z = getattr(f, "zone", -1)
        zone_miss[z] = zone_miss.get(z, 1.0) * (1.0 - float(f.confidence))
        selected.append(f)
        cur_val = value_with(zone_miss)
    return selected, cur_val
