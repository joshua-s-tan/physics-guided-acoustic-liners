# -*- coding: utf-8 -*-
"""PHYS-GEN-V1 authoritative exact scorer (physgen_scorer.quality_eval_exact) vs the legacy quality_eval, and its use
by every authoritative arm. TOY / training-only designs and hand-made conditions only."""
import numpy as np
import pytest
import torch

from acoustic_liner.physics import oracle as O2
from acoustic_liner.evaluation import budgets as BUD
from acoustic_liner.physics import quality as Q
from acoustic_liner.optimization import baseline as PB
from acoustic_liner.design import coordinates as PC
from acoustic_liner.models import objective as PO
from acoustic_liner.physics import scorer as PS
from acoustic_liner.optimization import seeded_cma as HO
from test_physgen_objective import toy_conditions
from test_physgen_handoff import TOY_COND


@pytest.fixture(scope="module")
def scored():
    rng = np.random.default_rng(71)
    u = torch.as_tensor(HO.random_u(rng, 120))
    conds = toy_conditions(rng, 120)
    designs = [PC.decode_np(u[i].numpy())[0] for i in range(120)]
    ex = [PS.quality_eval_exact(d, c) for d, c in zip(designs, conds)]
    lg = [Q.quality_eval(d, c) for d, c in zip(designs, conds)]
    hard = PO.objective(PC.decode_t(u), PO.condition_tensors(conds), 0.0)
    return designs, conds, ex, lg, hard


def test_exact_never_below_legacy_and_fixes_cap(scored):
    designs, conds, ex, lg, hard = scored
    d = np.array([e["best_val"] - l["best_val"] for e, l in zip(ex, lg)])
    b = hard["binding"].numpy()
    assert d.min() >= -1e-12, d.min()                       # exact candidate set contains everything legacy can reach
    assert np.max(np.abs(d[~b])) <= 1e-5                    # off the cap: same optimum (legacy golden tol 1e-5)
    assert d[b].max() > 0.03                                 # on the cap: the legacy error reaches the promotion floor


def test_exact_matches_continuous_hard_objective(scored):
    designs, conds, ex, lg, hard = scored
    assert max(abs(e["best_val"] - float(j)) for e, j in zip(ex, hard["J"])) <= 1e-9


def test_exact_selected_coverage_within_budget_and_cap_is_exact(scored):
    designs, conds, ex, lg, hard = scored
    for d, c, e in zip(designs, conds, ex):
        assert e["mass"] <= c["m_total_target"] and 0.0 <= e["best_cov"] <= e["f_cap"] <= 1.0
        if e["f_cap"] < 1.0:                                # the cap is the LARGEST in-budget float coverage
            assert O2._total_mass(O2._areal(d), float(np.nextafter(e["f_cap"], 1.0)), c["A_wall"]) > c["m_total_target"]


def test_exact_ledger_accounting_identical_to_locked_ledger():
    rng = np.random.default_rng(3)
    props = HO.random_proposals(rng, 30)
    props = props + props[:5] + [dict(props[0], D=0.011)]  # duplicates and one infeasible design
    a, b = PS.ExactQualityLedger(TOY_COND, 64), BUD.QualityLedger(TOY_COND, 64)
    for p in props:
        a.propose(p, "s"); b.propose(p, "s")
    assert a.attempt_log == b.attempt_log and [e["geometry"] for e in a.trace] == [e["geometry"] for e in b.trace]
    assert (a.n_duplicate, a.n_infeasible) == (b.n_duplicate, b.n_infeasible) == (5, 1)


def test_every_authoritative_arm_scores_with_the_exact_scorer():
    runs = [PB.run_quality_cma_exact(TOY_COND, 0, 0, 48, 1234),
            HO.run_seeded_cma(TOY_COND, 0, 0, 48, 1234, HO.random_proposals, "armB"),
            HO.run_seeded_cma(TOY_COND, 0, 0, 48, 1234, HO.random_proposals, "armC", cma=False)]
    for L in runs:
        assert isinstance(L, PS.ExactQualityLedger)
        for e in L.trace:
            assert e["scorer"] == "exact"
            assert e["best_val"] == PS.quality_eval_exact(e["design"], TOY_COND)["best_val"]


def test_training_objective_is_never_an_acceptance_authority():
    """The soft training objective must not be reachable from the scoring / handoff / comparator modules."""
    import inspect
    for mod in (PS, HO, PB):
        assert "train_objective" not in inspect.getsource(mod)
