"""The one oracle evaluation of a geometry under a condition (robust worst-case dOASPL with coverage
solved internally), the CMA objective built on it, and its ranking rule.
"""
from __future__ import annotations

import math
import numpy as np
from ..design import constraints as FEAS
from . import oracle as O2
from ..evaluation.errors import StudyError
from ..design.canonical import canonical_design


COV_GRID_N = 26                                   # = dataset._best_coverage grid
COV_REFINE_ITERS = 24                             # = dataset._COV_REFINE_ITERS


# CMA objective (minimized). Lexicographic, consistent with the ranking rule:
#   hit -> installed mass [kg] (< MISS_BASE);  feasible miss -> MISS_BASE + shortfall*MISS_SCALE;
#   infeasible -> INFEAS_BASE*(1 + n_violations) (graded, always worse than any feasible design).
MISS_BASE, MISS_SCALE, INFEAS_BASE = 1.0e4, 1.0e3, 1.0e7


# --------------------------------------------------------------------------- #
#  The one oracle evaluation
# --------------------------------------------------------------------------- #
def _worst(a_band, cond, f):
    return O2._worst(a_band, cond["spl"], f, cond["A_wall"], cond["kappa"])


def oracle_eval(design, cond):
    """Label ONE geometry under ONE condition (see module doc). Returns a dict; `feasible` False means it must NOT
    be charged. Physics non-evaluability (ValueError/FloatingPointError) is infeasible; anything else propagates."""
    design = canonical_design(design)
    ok, feas = FEAS.is_feasible(design, 1.0)
    if not ok:
        nv = len(feas["failures"]) if feas else 8
        return dict(feasible=False, n_violations=int(nv))
    try:
        a_band = O2._a_band(design)
        fs = np.linspace(0.0, 1.0, COV_GRID_N)
        wv = np.array([_worst(a_band, cond, f) for f in fs])
        j = int(wv.argmax())
        if fs[j] <= 0.0:
            best_cov, best_val = 0.0, float(wv[j])
        else:
            a, b = fs[max(0, j - 1)], fs[min(len(fs) - 1, j + 1)]
            best_cov, best_val = O2._golden_max(lambda f: _worst(a_band, cond, f), a, b)
            if best_val < wv[j]:
                best_cov, best_val = float(fs[j]), float(wv[j])
        target = cond["dOASPL_target"]
        hit, cov_min, mass, joint = False, None, None, False
        if best_val >= target:
            idx = np.nonzero(wv >= target)[0]
            if idx.size:
                cov_min = O2._refine_min_coverage(a_band, cond["spl"], cond["A_wall"], cond["kappa"], wv, fs,
                                                 target, iters=COV_REFINE_ITERS)
            else:                                      # max lies between grid nodes: bisect [grid node, argmax]
                lo, hi = float(fs[max(0, j - 1)]), float(best_cov)
                for _ in range(COV_REFINE_ITERS):
                    mid = 0.5 * (lo + hi)
                    if _worst(a_band, cond, mid) >= target:
                        hi = mid
                    else:
                        lo = mid
                cov_min = hi
            if cov_min is not None and cov_min > 0.0 and FEAS.is_feasible(design, cov_min)[0]:
                mass = O2._total_mass(O2._areal(design), cov_min, cond["A_wall"])
                hit = True
                joint = bool(mass <= cond["m_total_target"])
    except (ValueError, FloatingPointError):
        return dict(feasible=False, n_violations=0)
    out = dict(feasible=True, best_val=float(best_val), best_cov=float(best_cov), hit=bool(hit),
               cov_min=(None if cov_min is None else float(cov_min)), mass=(None if mass is None else float(mass)),
               joint=bool(joint))
    for k in ("best_val", "best_cov"):
        if not math.isfinite(out[k]):
            raise StudyError(f"non-finite oracle output {k}")
    return out


def objective(ev):
    """CMA objective (minimize), consistent with the frozen ranking rule."""
    if not ev["feasible"]:
        return INFEAS_BASE * (1.0 + ev.get("n_violations", 0))
    if ev["hit"]:
        return float(ev["mass"])
    return MISS_BASE + MISS_SCALE * max(0.0, ev["target"] - ev["best_val"])


def rank_key(ev):
    """Sort key: smaller is better (hits by mass, then misses by attenuation)."""
    return (0, ev["mass"]) if ev["hit"] else (1, -ev["best_val"])
