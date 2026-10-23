"""The constrained-quality evaluation: maximize the robust dOASPL subject to full feasibility and the
condition's installed-mass budget, with the matching CMA objective and ranking key.
"""
from __future__ import annotations

import math
import numpy as np
from ..design import constraints as FEAS
from . import oracle as O2
from ..design import canonical as CAN
from . import oracle_evaluation as OE


COV_GRID_N = OE.COV_GRID_N     # 26, same grid density as the frozen oracle


class V26ObjectiveError(RuntimeError):
    """Fail-closed: a non-finite or otherwise ill-formed quality evaluation."""


def _worst(a_band, cond, f):
    return O2._worst(a_band, cond["spl"], f, cond["A_wall"], cond["kappa"])


# --------------------------------------------------------------------------- #
#  sec.1 -- the constrained-quality oracle evaluation
# --------------------------------------------------------------------------- #
def quality_eval(design, cond):
    """The sec.1 constrained-quality evaluation of ONE geometry under ONE condition. Returns a dict:
      physically_feasible : bool  -- `design.schema`/`feasibility` gate at ANY coverage (coverage-independent,
                                     see module docstring) -- NOT charged/ranked further if False.
      feasible_budget      : bool  -- True iff >=1 coverage is BOTH physically feasible and within the mass
                                     budget -- a CANDIDATE for the sec.1 objective. A design infeasible or over
                                     budget at every coverage is NOT a candidate (prereg sec.1, verbatim).
      best_val / best_cov / mass : the constrained optimum (None if not a candidate).
      raw_best_val / raw_best_cov : the UNCONSTRAINED best (physics.oracle_evaluation.oracle_eval's own metric), reported for
                                     diagnostics only -- NEVER used to rank candidates, so a heavier/higher-
                                     coverage design can never be scored as a quality win.
      min_mass_over_budget : smallest (mass - m_total_target) over physically-feasible coverages, when the
                                     design is physically feasible but over budget at EVERY coverage (else None)
                                     -- a guiding signal for the CMA objective's graded penalty, nothing else.
      hit                    : best_val >= cond['dOASPL_target'] at the constrained optimum (reported, not used
                                     to rank -- sec.1's objective ranks by dOASPL alone among candidates).
    """
    design = CAN.canonical_design(design)
    ok, _ = FEAS.is_feasible(design, 1.0)          # coverage-independent (module docstring); one call suffices
    if not ok:
        return dict(physically_feasible=False, feasible_budget=False, best_val=None, best_cov=None, mass=None,
                   raw_best_val=None, raw_best_cov=None, min_mass_over_budget=None, hit=False)
    a_band = O2._a_band(design)
    m_areal = O2._areal(design)
    fs = np.linspace(0.0, 1.0, COV_GRID_N)
    wv = np.array([_worst(a_band, cond, float(f)) for f in fs])
    masses = np.array([O2._total_mass(m_areal, float(f), cond["A_wall"]) for f in fs])
    raw_j = int(wv.argmax())
    valid = masses <= cond["m_total_target"]
    if not valid.any():
        return dict(physically_feasible=True, feasible_budget=False, best_val=None, best_cov=None, mass=None,
                   raw_best_val=float(wv[raw_j]), raw_best_cov=float(fs[raw_j]),
                   min_mass_over_budget=float((masses - cond["m_total_target"]).min()), hit=False)
    masked = np.where(valid, wv, -np.inf)
    j = int(masked.argmax())
    best_cov, best_val, mass = float(fs[j]), float(masked[j]), float(masses[j])
    lo_i, hi_i = max(0, j - 1), min(COV_GRID_N - 1, j + 1)
    if hi_i > lo_i:
        fstar, vstar = O2._golden_max(lambda f: _worst(a_band, cond, f), float(fs[lo_i]), float(fs[hi_i]))
        if vstar > best_val:
            m2 = O2._total_mass(m_areal, float(fstar), cond["A_wall"])
            if m2 <= cond["m_total_target"]:
                best_cov, best_val, mass = float(fstar), float(vstar), float(m2)
    for name, val in (("best_val", best_val), ("best_cov", best_cov), ("mass", mass)):
        if not math.isfinite(val):
            raise V26ObjectiveError(f"non-finite quality_eval {name}")
    return dict(physically_feasible=True, feasible_budget=True, best_val=best_val, best_cov=best_cov, mass=mass,
               raw_best_val=float(wv[raw_j]), raw_best_cov=float(fs[raw_j]), min_mass_over_budget=None,
               hit=bool(best_val >= cond["dOASPL_target"]))


def quality_objective(ev):
    """CMA objective (MINIMIZE), consistent with sec.1: maximize `best_val` among candidates; non-candidates are
    penalized gradedly but ALWAYS worse than any candidate (mirrors `physics.oracle_evaluation.objective`'s own lexicographic
    structure -- physically infeasible is worse than feasible-but-over-budget, which still carries a gradient
    back toward the feasible-and-in-budget region)."""
    if ev["feasible_budget"]:
        return -float(ev["best_val"])
    if not ev["physically_feasible"]:
        return float(OE.INFEAS_BASE)
    over = ev.get("min_mass_over_budget") or 0.0
    return float(OE.MISS_BASE + OE.MISS_SCALE * max(0.0, over))


def quality_rank_key(ev):
    """Sort key: smaller is better. Candidates rank by -best_val; non-candidates always rank worse, physically
    infeasible worse than merely over-budget (mirrors `physics.oracle_evaluation.rank_key`'s two-tier structure)."""
    if ev["feasible_budget"]:
        return (0, -ev["best_val"])
    if ev["physically_feasible"]:
        return (1, float(ev.get("min_mass_over_budget") or 0.0))
    return (2, 0.0)
