# -*- coding: utf-8 -*-
"""cVAE v2.6 -- the shared constrained-quality objective (`acoustic_liner.physics.quality`; sec.1 of the
historical cVAE v2.6 pre-registration, which is not distributed): a design infeasible or over the mass budget at EVERY coverage is never a candidate,
and the objective/rank-key both penalize such designs strictly worse than any candidate."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_liner.design import schema as DS  # noqa: E402
from acoustic_liner.evaluation import conditions as COND  # noqa: E402
from acoustic_liner.design import canonical as CAN  # noqa: E402
from acoustic_liner.physics import oracle_evaluation as OE
from acoustic_liner.physics import quality as Q  # noqa: E402


def _cond(seed=1, **overrides):
    c = COND.synthetic_conditions(1, seed, "test")[0]
    c = dict(c, **overrides)
    return dict(c, spl=np.asarray(c["spl"], float))      # study_conditions.runtime_condition, inlined


def _feasible_design(rng):
    for _ in range(4000):
        d = DS.vector_to_design(DS.sample_random(rng))
        if CAN.canonical_design(d) and __import__("acoustic_liner.design.constraints", fromlist=["is_feasible"]).is_feasible(d, 1.0)[0]:
            return d
    raise AssertionError("could not draw a feasible design for the fixture")


def test_quality_eval_rejects_physically_infeasible_design():
    from acoustic_liner.design import constraints as FEAS
    # all four banks ACTIVE (nonzero f0, area_frac summing to 1) so `_enrich_groups`'s cavity-volume inversion
    # never divides by a zero target f0 -- only bank 0's r_neck is pushed far out of the manufacturing range.
    design = dict(D=0.03, face_t=0.001, wall_t=0.0008,
                 groups=[dict(f0=300.0, r_neck=0.2, L_neck=0.02, area_frac=0.4, Q=100.0),   # r_neck way OOB
                        dict(f0=400.0, r_neck=0.004, L_neck=0.01, area_frac=0.2, Q=100.0),
                        dict(f0=500.0, r_neck=0.004, L_neck=0.01, area_frac=0.2, Q=100.0),
                        dict(f0=600.0, r_neck=0.004, L_neck=0.01, area_frac=0.2, Q=100.0)])
    assert FEAS.is_feasible(design, 1.0)[0] is False          # sanity: the fixture really is infeasible
    cond = _cond()
    ev = Q.quality_eval(design, cond)
    assert ev["physically_feasible"] is False
    assert ev["feasible_budget"] is False
    assert ev["best_val"] is None and ev["mass"] is None
    assert Q.quality_objective(ev) == pytest.approx(OE.INFEAS_BASE)
    assert Q.quality_rank_key(ev) == (2, 0.0)


def test_quality_eval_rejects_feasible_design_over_every_coverage_mass_budget():
    rng = np.random.default_rng(0)
    design = _feasible_design(rng)
    # a NEGATIVE mass budget: even f_cov=0 (mass=0, per dataset._total_mass's own "f_cov=0 -> 0" contract) is
    # over budget, so EVERY coverage is over budget -- a genuinely unattainable constraint, unlike a merely
    # tiny positive budget (which f_cov=0 would trivially satisfy).
    cond = _cond(seed=2, m_total_target=-1.0)
    ev = Q.quality_eval(design, cond)
    assert ev["physically_feasible"] is True
    assert ev["feasible_budget"] is False
    assert ev["best_val"] is None and ev["mass"] is None
    assert ev["min_mass_over_budget"] is not None and ev["min_mass_over_budget"] > 0
    # over-budget is penalized, but STRICTLY better (smaller) than physically infeasible
    infeasible_obj = Q.quality_objective(dict(feasible_budget=False, physically_feasible=False))
    over_budget_obj = Q.quality_objective(ev)
    assert over_budget_obj < infeasible_obj
    assert Q.quality_rank_key(ev)[0] == 1


def test_quality_eval_candidate_mass_is_within_budget_and_matches_best_cov():
    rng = np.random.default_rng(0)
    design = _feasible_design(rng)
    cond = _cond(seed=3, m_total_target=1e6)           # a huge budget -- essentially unconstrained
    ev = Q.quality_eval(design, cond)
    assert ev["physically_feasible"] is True
    if ev["feasible_budget"]:
        assert ev["mass"] <= cond["m_total_target"]
        assert 0.0 <= ev["best_cov"] <= 1.0
        assert Q.quality_objective(ev) == pytest.approx(-ev["best_val"])
        assert Q.quality_rank_key(ev) == (0, -ev["best_val"])
        # an unconstrained budget's constrained optimum is close to the unconstrained grid best (a local
        # golden-section refine near the constrained optimum can exceed the COARSE grid's own unrefined max by
        # a small amount, so this is a tolerance check, not an exact bound)
        assert ev["best_val"] <= ev["raw_best_val"] + 0.01


