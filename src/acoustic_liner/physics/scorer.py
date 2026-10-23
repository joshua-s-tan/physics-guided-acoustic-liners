# -*- coding: utf-8 -*-
r"""PHYS-GEN-V1 AUTHORITATIVE scorer: `quality_eval_exact` (owner decision 2026-10-02, pre-seal correction).

The legacy `v26_objective.quality_eval` solves coverage by a 26-point grid argmax restricted to the mass budget plus a
golden refine whose result is kept only if it is in budget. It never evaluates the mass-cap coverage f_max itself, so
where the cap binds it under-scores a design by a design-dependent amount (observed up to 0.033 dB on training-only toy
cases -- the size of the +0.03 dB promotion floor).

`quality_eval_exact` is `quality_eval` with ONE change, the constrained coverage solve:
  * candidates: f = 0 (no treatment, exactly 0 dB, always admissible), every grid point with mass <= budget, and the
    exact mass-cap coverage f_max (the largest float coverage whose installed mass is <= budget);
  * golden-section refinement of the best candidate's bracket, confined to [0, f_max] and run to ~1e-11 in coverage;
  * the true constrained optimum among all of these is selected.
Feasibility, the robust worst-case objective, the mass model, the returned record and the unconstrained `raw_*`
diagnostics are unchanged. It drives every authoritative arm (corrected Arm A, Arms B, C, D) through
`ExactQualityLedger`; the legacy scorer is kept only as a continuity diagnostic. Earlier v2.x evidence is historical
and is not rescored.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..design import constraints as FEAS
from . import oracle as O2
from ..design import canonical as CAN
from ..evaluation import budgets as BUD
from . import quality as Q
from . import differentiable as VT

GOLDEN_TOL = 1e-11
GOLDEN_MAX_ITERS = 100
REFINE_MIN_GAIN = 1e-12        # dB; a refinement must beat the best candidate by more than round-off


def mass_cap_coverage(m_areal, cond):
    """Largest coverage in [0, 1] whose canonical installed mass is within budget (exact, float-safe)."""
    denom = (m_areal + VT._INSTALL_AREAL) * cond["A_wall"]
    f = min(1.0, float(cond["m_total_target"]) / denom) if denom > 0 else 1.0
    f = max(f, 0.0)
    while f > 0.0 and O2._total_mass(m_areal, f, cond["A_wall"]) > cond["m_total_target"]:
        f = float(np.nextafter(f, 0.0))
    for _ in range(64):                       # neighbouring floats can round to exactly the budget mass: take the largest
        nxt = float(np.nextafter(f, 1.0))
        if f >= 1.0 or O2._total_mass(m_areal, nxt, cond["A_wall"]) > cond["m_total_target"]:
            break
        f = nxt
    return f


def _golden(fn, a, b):
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a); d = a + gr * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(GOLDEN_MAX_ITERS):
        if b - a < GOLDEN_TOL:
            break
        if fc < fd:
            a, c, fc = c, d, fd
            d = a + gr * (b - a); fd = fn(d)
        else:
            b, d, fd = d, c, fc
            c = b - gr * (b - a); fc = fn(c)
    f = 0.5 * (a + b)
    return f, fn(f)


def quality_eval_exact(design, cond):
    """Same record as `physics.quality.quality_eval`; constrained coverage solved exactly (see module docstring)."""
    design = CAN.canonical_design(design)
    ok, _ = FEAS.is_feasible(design, 1.0)
    if not ok:
        return dict(physically_feasible=False, feasible_budget=False, best_val=None, best_cov=None, mass=None,
                    raw_best_val=None, raw_best_cov=None, min_mass_over_budget=None, hit=False, scorer="exact")
    a_band = O2._a_band(design)
    m_areal = O2._areal(design)
    W = lambda f: Q._worst(a_band, cond, float(f))
    fs = np.linspace(0.0, 1.0, Q.COV_GRID_N)
    wv = np.array([W(f) for f in fs])
    masses = np.array([O2._total_mass(m_areal, float(f), cond["A_wall"]) for f in fs])
    raw_j = int(wv.argmax())                                              # unconstrained diagnostic (unchanged)
    valid = masses <= cond["m_total_target"]
    if not valid.any():
        return dict(physically_feasible=True, feasible_budget=False, best_val=None, best_cov=None, mass=None,
                    raw_best_val=float(wv[raw_j]), raw_best_cov=float(fs[raw_j]),
                    min_mass_over_budget=float((masses - cond["m_total_target"]).min()), hit=False, scorer="exact")
    f_cap = mass_cap_coverage(m_areal, cond)
    cand = {float(f): float(v) for f, v, ok_ in zip(fs, wv, valid) if ok_}
    cand.setdefault(0.0, W(0.0))
    cand.setdefault(f_cap, W(f_cap))
    cf = np.array(sorted(cand)); cv = np.array([cand[f] for f in cf])
    j = int(cv.argmax())
    best_cov, best_val = float(cf[j]), float(cv[j])
    lo, hi = float(cf[max(0, j - 1)]), float(cf[min(len(cf) - 1, j + 1)])
    if hi > lo:
        f_g, v_g = _golden(W, lo, min(hi, f_cap))
        if v_g > best_val + REFINE_MIN_GAIN and f_g <= f_cap:
            best_cov, best_val = float(f_g), float(v_g)
    mass = O2._total_mass(m_areal, best_cov, cond["A_wall"])
    if mass > cond["m_total_target"]:
        raise Q.V26ObjectiveError(f"exact scorer selected an over-budget coverage {best_cov}")
    for name, val in (("best_val", best_val), ("best_cov", best_cov), ("mass", mass)):
        if not math.isfinite(val):
            raise Q.V26ObjectiveError(f"non-finite quality_eval_exact {name}")
    return dict(physically_feasible=True, feasible_budget=True, best_val=best_val, best_cov=best_cov, mass=float(mass),
                raw_best_val=float(wv[raw_j]), raw_best_cov=float(fs[raw_j]), min_mass_over_budget=None,
                hit=bool(best_val >= cond["dOASPL_target"]), f_cap=float(f_cap), scorer="exact")


class ExactQualityLedger(BUD.QualityLedger):
    """`evaluation.budgets.QualityLedger` verbatim (dedup, charging rule, attempt log, exact budget accounting) with the
    authoritative `quality_eval_exact` in place of the legacy `quality_eval`."""

    def propose(self, design, source):
        if self.exhausted:
            return None
        self.attempts += 1
        key = CAN.geometry_key(design)
        if key in self.cache:
            self.n_duplicate += 1
            self.attempt_log.append("d")
            return dict(self.cache[key], charged=False, duplicate=True)
        ev = quality_eval_exact(design, self.cond)
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
