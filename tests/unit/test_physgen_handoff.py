# -*- coding: utf-8 -*-
"""PHYS-GEN-V1 population -> CMA handoff conformance (prereg sec. 12), TOY / training-only inputs only.

Locks: (1) exact transform -> locked-CMA coordinates -> design invariance (geometry, area fractions, oracle score);
(2) exact budget accounting on the locked QualityLedger (proposals, fills, CMA remainder); (3) Arm B draws its random
proposals through the SAME feasibility transform. The toy condition is hand-made here and is not a registered split."""
import numpy as np
import pytest

from acoustic_liner.design import constraints as FEAS
from acoustic_liner.design import schema as DS
from acoustic_liner.evaluation.historical_splits import selection_rule as SEL
from acoustic_liner.design import canonical as CAN
from acoustic_liner.evaluation import random_streams as RS
from acoustic_liner.physics import quality as Q
from acoustic_liner.design import coordinates as PC
from acoustic_liner.optimization import seeded_cma as HO

LO, HI = DS.bounds()
_SPL = np.array([118.0, 122.0, 126.0, 128.0, 127.0, 124.0, 120.0, 116.0])
TOY_COND = dict(id="toy_physgen_handoff", group="toy", weights=list(SEL._weights_from_spl(_SPL)), spl=_SPL,
                dOASPL_target=0.5, m_total_target=150.0, A_wall=100.0, kappa=0.1)


def _vec(d):
    """Physical geometry + area fractions, banks sorted (the oracle is bank-permutation invariant)."""
    c = PC.canonicalize(d)
    return np.concatenate([[c["D"], c["face_t"], c["wall_t"]],
                           np.array([[g["f0"], g["r_neck"], g["L_neck"], g["area_frac"]] for g in c["groups"]]).ravel()])


_SCALE = np.concatenate([HI[:3] - LO[:3], np.tile([HI[3] - LO[3], HI[4] - LO[4], HI[5] - LO[5], 1.0], 4)])


def _transform_designs(n, seed):
    out = PC.decode_t(__import__("torch").as_tensor(HO.random_u(np.random.default_rng(seed), n)))
    rows = out["cma_vec"].detach().numpy()
    designs = []
    for i, row in enumerate(rows):
        groups = [dict(f0=float(out["f0"][i, g]), r_neck=float(out["r"][i, g]), L_neck=float(out["L"][i, g]),
                       area_frac=float(out["frac"][i, g]), Q=DS.Q_FIXED) for g in range(PC.NG)]
        designs.append(dict(D=float(out["D"][i]), groups=groups, face_t=float(out["face_t"][i]),
                            wall_t=float(out["wall_t"][i])))
    return designs


# ---- (1) representation invariance -------------------------------------------------------------------------------
def test_transform_to_cma_to_design_is_exact():
    worst = 0.0
    for d in _transform_designs(500, seed=21):
        y = HO.to_cma01(d)
        assert np.all((y >= 0.0) & (y <= 1.0))             # representable WITHOUT clipping
        worst = max(worst, float(np.max(np.abs(_vec(d) - _vec(HO.from_cma01(y))) / _SCALE)))
    assert worst <= 1e-12, worst


def test_handoff_preserves_oracle_score():
    """The canonical quality_eval score is unchanged by the transform -> CMA -> design conversion."""
    worst, n_cand = 0.0, 0
    for d in _transform_designs(40, seed=22):
        e0 = Q.quality_eval(d, TOY_COND)
        e1 = Q.quality_eval(HO.from_cma01(HO.to_cma01(d)), TOY_COND)
        assert e0["physically_feasible"] and e1["physically_feasible"]
        assert e0["feasible_budget"] == e1["feasible_budget"]
        if e0["feasible_budget"]:
            n_cand += 1
            worst = max(worst, abs(e0["best_val"] - e1["best_val"]))
    assert n_cand > 0
    assert worst <= 1e-9, worst


def test_locked_cma_designs_also_survive_handoff():
    """Every design the locked CMA itself can express (logits in [-3,3]) is representable exactly."""
    rng = np.random.default_rng(23)
    worst = 0.0
    for _ in range(300):
        d = DS.vector_to_design(DS.sample_random(rng))
        worst = max(worst, float(np.max(np.abs(_vec(d) - _vec(HO.from_cma01(HO.to_cma01(d)))) / _SCALE)))
    assert worst <= 1e-12, worst


def test_locked_design_to_vector_is_lossy_so_handoff_avoids_it():
    """Documents WHY the handoff uses centered logits: the locked DS.design_to_vector clips log(frac) to [-3,3]."""
    d = DS.vector_to_design(np.concatenate([[0.04, 0.001, 0.001],
                                            [150, 0.005, 0.03, -3.0, 180, 0.005, 0.03, 3.0,
                                             220, 0.005, 0.03, 3.0, 260, 0.005, 0.03, 3.0]]))
    lossy = DS.vector_to_design(DS.design_to_vector(d))
    assert abs(lossy["groups"][0]["area_frac"] - d["groups"][0]["area_frac"]) > 1e-3
    exact = HO.from_cma01(HO.to_cma01(d))
    assert np.max(np.abs(_vec(exact) - _vec(d)) / _SCALE) <= 1e-12


def test_to_cma01_rejects_unrepresentable_area_split():
    d = DS.vector_to_design(DS.sample_random(np.random.default_rng(24)))
    wide = [0.9, 0.09, 0.0098, 0.0002]                     # log-span 8.4 > 6: no logits in [-3,3] express it
    with pytest.raises(HO.HandoffError):
        HO.to_cma01(dict(d, groups=[dict(g, area_frac=a) for g, a in zip(d["groups"], wide)]))


# ---- (3) Arm-B comparator through the same transform -------------------------------------------------------------
def test_arm_b_proposals_feasible_and_deterministic():
    a = HO.random_proposals(np.random.default_rng(0), 200)
    b = HO.random_proposals(np.random.default_rng(0), 200)
    assert all(FEAS.is_feasible(d, 1.0)[0] for d in a)
    assert all(np.array_equal(_vec(x), _vec(y)) for x, y in zip(a, b))


def test_stream_tags_disjoint_from_locked_streams():
    import inspect
    locked = {1, 2, 3, 4}                                  # study_core.stream tags
    assert not (set(HO._TAGS.values()) & locked)
    assert "tags = {\"random\": 1, \"cvae\": 2, \"cma_start\": 3, \"cma\": 4}" in inspect.getsource(RS.stream)


# ---- (2) exact budget accounting ---------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def arm_b_run():
    return HO.run_seeded_cma(TOY_COND, 0, 0, 48, 1234, HO.random_proposals, "armB")


def test_accounting_proposals_then_cma(arm_b_run):
    L = arm_b_run
    assert len(L.trace) == 48
    src = [e["source"] for e in L.trace]
    assert src[:32] == ["armB_proposal"] * 32 and L.n_proposal_charged == 32 and L.n_fill_charged == 0
    assert all(s == "armB_cma" for s in src[32:])
    assert L.k_retained == HO.K_RETAIN and len(L.init_x0) == 19
    assert all(0.0 <= x <= 1.0 for x in L.init_x0)


def test_run_is_deterministic(arm_b_run):
    again = HO.run_seeded_cma(TOY_COND, 0, 0, 48, 1234, HO.random_proposals, "armB")
    key = lambda L: [(e["source"], CAN.key_hash(e["geometry"]), e.get("best_val")) for e in L.trace]
    assert key(again) == key(arm_b_run)


def test_duplicate_proposals_are_filled_and_charged():
    dup = HO.random_proposals(np.random.default_rng(5), 1)[0]
    L = HO.run_seeded_cma(TOY_COND, 0, 0, 40, 1234, lambda rng, n: [dup] * n, "armD")
    assert L.n_proposal_charged == 1 and L.n_fill_charged == 31
    assert [e["source"] for e in L.trace[:32]] == ["armD_proposal"] + ["armD_fill"] * 31
    assert len(L.trace) == 40


def test_generator_only_arm_c_spends_only_proposals():
    L = HO.run_seeded_cma(TOY_COND, 0, 0, 128, 1234, HO.random_proposals, "armC", cma=False)
    assert len(L.trace) == 32 and L.n_reinit == 0
