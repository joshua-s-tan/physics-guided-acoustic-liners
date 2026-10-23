# -*- coding: utf-8 -*-
"""PHYS-GEN-V1 coordinate-conformance gates (prereg sec. 4), TOY / training-only inputs only -- no pilot, development,
final, Falcon or Atlas condition is touched. Canonical checks (dataset.is_feasible, oracle) are the judges.

Gates: round-trip <= 1e-8 on >= 1,000 random feasible designs (drawn exactly as the locked CMA draws them); finite
outputs on >= 10,000 random u; finite Jacobians on those 10,000; full-rank Jacobian on 200 interior points; canonical
feasibility >= 0.98 (and exactly 1.00 under saturation); no bool/non-finite/shape coercion; no projection after
decoding; bank-order canonicalization invariants. Coverage is not a generator coordinate (owner decision 2026-10-02)."""
import numpy as np
import pytest
import torch

from acoustic_liner.design import constraints as FEAS
from acoustic_liner.design import schema as DS
from acoustic_liner.physics import oracle as O2
from acoustic_liner.design import coordinates as PC

N_ROUNDTRIP = 1000
N_RANDOM = 10000
LO, HI = DS.bounds()


def locked_cma_feasible_designs(rng, n):
    """Rejection-sample canonically feasible designs exactly as the locked CMA's random screen draws them."""
    out = []
    while len(out) < n:
        des = DS.vector_to_design(DS.sample_random(rng))
        if FEAS.is_feasible(des, 1.0)[0]:
            out.append(PC.canonicalize(des))
    return out


def batched_jacobian(u):
    """Per-sample Jacobian d cma_vec[b] / d u[b] through the PUBLIC (validated) decode_t: rows are independent, so one
    backward pass per output component yields that row of every sample's Jacobian."""
    u = u.clone().requires_grad_(True)
    out = PC.decode_t(u)["cma_vec"]
    rows = [torch.autograd.grad(out[:, k].sum(), u, retain_graph=True)[0] for k in range(out.shape[1])]
    return torch.stack(rows, dim=1)                         # [B, 19 (output), 18 (input)]


def _feasible(cma_vec_row):
    return FEAS.is_feasible(DS.vector_to_design(cma_vec_row), 1.0)[0]


# ---- static proof that the non-built-in constraints cannot bind --------------------------------------------------
def test_inactive_constraint_proof():
    p = PC.INACTIVE_PROOF
    assert p["sigma_max_in_box"] < 0.25 and p["sigma_max_in_box"] < p["sigma_cell_floor"]
    assert p["min_area_frac"] > 1e-6
    assert PC.D_MIN > LO[0] and PC.f0_min_np(PC.D_MIN) == pytest.approx(PC.F0_HI, rel=1e-9)
    assert 100.0 < PC.F0_FLOOR_AT_D_HI < 105.0           # frozen design-space limitation (~102 Hz), not a defect


def test_proof_survives_python_O():
    """The import-time proof must use explicit raises, not `assert` (stripped by python -O)."""
    import inspect
    src = inspect.getsource(PC)
    assert "\nassert " not in src and "    assert " not in src


# ---- round trip on >= 1,000 random feasible designs --------------------------------------------------------------
def test_roundtrip_feasible_designs():
    rng = np.random.default_rng(1)
    worst = 0.0
    scale = np.concatenate([HI[:3] - LO[:3], np.tile([HI[3] - LO[3], HI[4] - LO[4], HI[5] - LO[5], 1.0], 4)])
    for des in locked_cma_feasible_designs(rng, N_ROUNDTRIP):
        des2, _ = PC.decode_np(PC.encode_np(des))
        vec = lambda d: np.concatenate([[d["D"], d["face_t"], d["wall_t"]],
                                        np.array([[g["f0"], g["r_neck"], g["L_neck"], g["area_frac"]]
                                                  for g in d["groups"]]).ravel()])
        worst = max(worst, float(np.max(np.abs(vec(des) - vec(des2)) / scale)))
    assert worst <= 1e-8, worst


# ---- >= 10,000 random u: finite, inside the CMA box, canonically feasible -----------------------------------------
@pytest.fixture(scope="module")
def random_u():
    g = torch.Generator().manual_seed(7)
    u = torch.randn(N_RANDOM, PC.DIM, generator=g, dtype=torch.float64) * 3.0
    u[:500] *= 12.0                                         # saturated tails
    return u


def test_random_samples_finite_in_box_feasible(random_u):
    out = PC.decode_t(random_u)
    x = out["cma_vec"].numpy()
    assert np.isfinite(x).all()
    assert (x >= LO).all() and (x <= HI).all()              # already in the locked CMA box: nothing to project
    n_feas = sum(_feasible(row) for row in x)
    assert n_feas / len(x) >= 0.98, n_feas / len(x)
    w = out["logratio"].numpy()
    v = np.concatenate([w, np.zeros((len(w), 1))], axis=1)
    assert (v.max(1) - v.min(1) < PC.SPAN_MAX).all()        # CMA-representable area split


def test_random_samples_finite_jacobians(random_u):
    J = batched_jacobian(random_u)                          # all 10,000 samples
    assert J.shape == (N_RANDOM, 19, PC.DIM)
    assert bool(torch.isfinite(J).all())


def test_jacobian_full_rank_interior():
    """Bijection check on 200 interior points: the Jacobian onto the physical coordinates has rank 18."""
    g = torch.Generator().manual_seed(3)
    J = batched_jacobian(torch.randn(200, PC.DIM, generator=g, dtype=torch.float64))
    Jn = J / torch.tensor(HI - LO, dtype=torch.float64)[None, :, None]   # box-normalized rows (scale-free)
    assert int(torch.linalg.matrix_rank(Jn).min()) == PC.DIM   # 4 centered logits carry 3 free directions


def test_saturation_is_exactly_feasible():
    """Regression for the Phase-I corner round-off defect: at extreme |u| conditional intervals collapse toward a
    point; the nested margins must keep canonical feasibility at EXACTLY 1.00, not merely >= 0.98."""
    g = torch.Generator().manual_seed(9)
    x = PC.decode_t(torch.randn(3000, PC.DIM, generator=g, dtype=torch.float64) * 50.0)["cma_vec"].numpy()
    assert (x >= LO).all() and (x <= HI).all()
    bad = [i for i in range(len(x)) if not _feasible(x[i])]
    assert bad == [], len(bad)


# ---- input hygiene: no coercion ----------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["bool", "nan", "inf", "shape", "list"])
def test_decode_rejects_malformed(bad):
    u = torch.zeros(1, PC.DIM, dtype=torch.float64)
    if bad == "bool":
        u = torch.zeros(1, PC.DIM, dtype=torch.bool)
    elif bad == "nan":
        u[0, 3] = float("nan")
    elif bad == "inf":
        u[0, 0] = float("inf")
    elif bad == "shape":
        u = torch.zeros(1, PC.DIM + 1, dtype=torch.float64)
    elif bad == "list":
        u = [0.0] * PC.DIM
    with pytest.raises(PC.CoordinateDomainError):
        PC.decode_t(u)


def test_decode_np_rejects_bool():
    with pytest.raises(PC.CoordinateDomainError):
        PC.decode_np(np.zeros(PC.DIM, dtype=bool))


def test_encode_rejects_outside_domain():
    des = locked_cma_feasible_designs(np.random.default_rng(5), 1)[0]
    with pytest.raises(PC.CoordinateDomainError):           # below D_MIN: no feasible bank exists
        PC.encode_np(dict(des, D=0.011))
    with pytest.raises(PC.CoordinateDomainError):           # tied f0
        PC.encode_np(dict(des, groups=[dict(g, f0=150.0) for g in des["groups"]]))
    wide = [0.9, 0.09, 0.0098, 0.0002]                      # area log-span log(0.9/0.0002) = 8.4 > 6
    with pytest.raises(PC.CoordinateDomainError):
        PC.encode_np(dict(des, groups=[dict(g, area_frac=a) for g, a in zip(des["groups"], wide)]))


# ---- bank-order invariants ---------------------------------------------------------------------------------------
def test_decoded_f0_strictly_increasing(random_u):
    f0 = PC.decode_t(random_u[500:])["f0"]                  # non-saturated rows
    assert bool((f0[:, 1:] > f0[:, :-1]).all())


def test_permutation_canonicalization_invariance():
    """Permuting banks changes neither the canonical encoding nor the canonical oracle label."""
    des = locked_cma_feasible_designs(np.random.default_rng(11), 1)[0]
    perm = dict(des, groups=[des["groups"][i] for i in (2, 0, 3, 1)])
    assert np.array_equal(PC.encode_np(des), PC.encode_np(perm))
    spl = [120.0] * 8
    lab = lambda d: O2.label_v2(dict(d, f_cov=0.3, A_wall=100.0, kappa_nominal=0.1, spectrum=spl))["dOASPL_worst_dB"]
    assert lab(des) == pytest.approx(lab(perm), abs=1e-12)
