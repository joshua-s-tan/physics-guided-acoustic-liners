# -*- coding: utf-8 -*-
"""Corrected exact-gauge Arm A (physgen_baseline.run_quality_cma_exact) vs the locked legacy run_quality_cma.
TOY / training-only condition (hand-made, not a registered split)."""
import numpy as np
import pytest

from acoustic_liner.optimization import cma as CMA
from acoustic_liner.physics import quality as Q
from acoustic_liner.optimization import cma as V26
from acoustic_liner.optimization import baseline as PB
from acoustic_liner.optimization import seeded_cma as HO
from test_physgen_handoff import TOY_COND, _vec, _SCALE

B, BASE = 64, 4321


def _run_capturing_starts(fn, monkeypatch, seed):
    starts = []
    real = CMA._quality_cma_restart

    def spy(L, x0, *a, **k):
        starts.append(np.array(x0, float))
        return real(L, x0, *a, **k)

    monkeypatch.setattr(V26, "_quality_cma_restart", spy)
    L = fn(TOY_COND, 0, seed, B, BASE)
    monkeypatch.setattr(V26, "_quality_cma_restart", real)
    return L, starts


def _selected_seeds(L):
    return sorted(L.trace[:L.k0], key=Q.quality_rank_key)[:CMA.N_RESTARTS]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_corrected_arm_a_starts_exactly_at_selected_seeds(monkeypatch, seed):
    L, starts = _run_capturing_starts(PB.run_quality_cma_exact, monkeypatch, seed)
    sel = _selected_seeds(L)
    assert len(starts) == CMA.N_RESTARTS and len(sel) == CMA.N_RESTARTS
    for x0, ev in zip(starts, sel):
        assert np.max(np.abs(_vec(HO.from_cma01(x0)) - _vec(ev["design"])) / _SCALE) <= 1e-12


def test_corrected_arm_a_accounting_and_determinism():
    a = PB.run_quality_cma_exact(TOY_COND, 0, 4, B, BASE)
    b = PB.run_quality_cma_exact(TOY_COND, 0, 4, B, BASE)
    assert len(a.trace) == B and a.start_encoding == "exact_centered_logit"
    assert [(e["source"], e["geometry"], e.get("best_val")) for e in a.trace] == \
           [(e["source"], e["geometry"], e.get("best_val")) for e in b.trace]


def test_exact_quality_cma_smoke_stays_within_budget_and_charges_only_feasible_geometries():
    """Public replacement of test_v26_objective's smoke test of the superseded lossy CMA-ES: the same three
    assertions, on the exact-start CMA-ES and the same condition."""
    from acoustic_liner.evaluation import conditions as COND
    c = COND.synthetic_conditions(1, 4, "test")[0]
    cond = dict(c, spl=np.asarray(c["spl"], float))
    L = PB.run_quality_cma_exact(cond, 0, 0, 48, base_seed=999)
    assert len(L.trace) <= 48
    assert all(e["physically_feasible"] for e in L.trace)
    keys = [tuple(e["geometry"]) for e in L.trace]
    assert len(set(keys)) == len(keys)                 # no geometry charged twice
