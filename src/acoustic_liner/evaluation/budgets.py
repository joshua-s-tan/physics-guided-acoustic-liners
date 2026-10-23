"""Exact budget accounting: one charged oracle evaluation per new feasible geometry; duplicates and
infeasible proposals are recorded but never charged; a per-attempt outcome log.
"""
from __future__ import annotations

import time
from ..design.canonical import canonical_design
from ..design.canonical import geometry_key
from ..physics.oracle_evaluation import oracle_eval
from ..design import canonical as CAN
from ..physics.quality import quality_eval


ATTEMPT_CAP_FACTOR = 50


# --------------------------------------------------------------------------- #
#  Exact budget ledger
# --------------------------------------------------------------------------- #
class Ledger:
    """Charges exactly one oracle evaluation per NEW feasible geometry; records everything else. `trace` lists the
    charged evaluations in order (each with the attempt index at which it was charged)."""

    def __init__(self, cond, budget, attempt_cap=None):
        self.cond, self.budget = cond, int(budget)
        self.attempt_cap = int(attempt_cap if attempt_cap is not None else ATTEMPT_CAP_FACTOR * budget)
        self.cache = {}                      # key -> eval (feasible and infeasible)
        self.trace = []
        self.attempts = self.n_infeasible = self.n_duplicate = 0
        self.t0 = time.perf_counter()

    @property
    def exhausted(self):
        return len(self.trace) >= self.budget or self.attempts >= self.attempt_cap

    def propose(self, design, source):
        """Returns the eval dict (with 'charged'), or None if the ledger was already exhausted."""
        if self.exhausted:
            return None
        self.attempts += 1
        key = geometry_key(design)
        if key in self.cache:
            self.n_duplicate += 1
            return dict(self.cache[key], charged=False, duplicate=True)
        ev = oracle_eval(design, self.cond)
        ev["target"] = self.cond["dOASPL_target"]
        if not ev["feasible"]:
            self.n_infeasible += 1
            self.cache[key] = ev
            return dict(ev, charged=False, duplicate=False)
        ev.update(eval_index=len(self.trace) + 1, attempt_index=self.attempts, source=source,
                  t_s=round(time.perf_counter() - self.t0, 4), geometry=list(key),
                  design=canonical_design(design))
        self.cache[key] = ev
        self.trace.append(ev)
        return dict(ev, charged=True, duplicate=False)

    def summary(self):
        return dict(budget=self.budget, n_evals=len(self.trace), attempts=self.attempts,
                    n_infeasible=self.n_infeasible, n_duplicate=self.n_duplicate,
                    attempt_cap=self.attempt_cap, attempt_cap_reached=bool(self.attempts >= self.attempt_cap
                                                                           and len(self.trace) < self.budget),
                    wall_s=round(time.perf_counter() - self.t0, 3))


class LoggedLedger(Ledger):
    """Ledger that also keeps a per-attempt outcome log ('e' charged, 'i' infeasible, 'd' duplicate) so exact
    prefix budgets can be derived from one fixed-bank run."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.attempt_log = []

    def propose(self, design, source):
        r = super().propose(design, source)
        if r is not None:
            self.attempt_log.append("e" if r["charged"] else ("d" if r["duplicate"] else "i"))
        return r


# --------------------------------------------------------------------------- #
#  quality-CMA (prereg sec.2 arm 1): random-screened CMA under the sec.1 objective. Mirrors
#  the historical `study_core._run_cma_screened`/`_cma_restart` exactly (same K0 screening, restart/reinit rule, streams) but
#  ranks/optimizes by `quality_eval`/`quality_objective`/`quality_rank_key` throughout -- a FRESH run, never the
#  existing mass-min CMA cells the historical v2.4/v2.5 studies already computed (prereg sec.1: "NOT reused as the
#  quality baseline").
# --------------------------------------------------------------------------- #
class QualityLedger(LoggedLedger):
    """Like `evaluation.budgets.LoggedLedger` (dedup + exact budget accounting on UNIQUE geometries) but the cached
    evaluation is `quality_eval` (sec.1) rather than `physics.oracle_evaluation.oracle_eval`. A geometry is CHARGED iff it is
    `physically_feasible` (identical charging rule to the original Ledger's `ev['feasible']`) -- a physically
    feasible-but-over-budget design still consumes one oracle evaluation, exactly as a feasible-but-missed-
    target design does in the original study; only the RANKING differs."""

    def propose(self, design, source):
        if self.exhausted:
            return None
        self.attempts += 1
        key = CAN.geometry_key(design)
        if key in self.cache:
            self.n_duplicate += 1
            self.attempt_log.append("d")
            return dict(self.cache[key], charged=False, duplicate=True)
        ev = quality_eval(design, self.cond)
        if not ev["physically_feasible"]:
            self.n_infeasible += 1
            self.cache[key] = ev
            self.attempt_log.append("i")
            return dict(ev, charged=False, duplicate=False)
        ev = dict(ev, eval_index=len(self.trace) + 1, attempt_index=self.attempts, source=source,
                  t_s=round(time.perf_counter() - self.t0, 4), geometry=list(key),
                  design=CAN.canonical_design(design))
        self.cache[key] = ev
        self.trace.append(ev)
        self.attempt_log.append("e")
        return dict(ev, charged=True, duplicate=False)
