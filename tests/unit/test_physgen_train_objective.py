# -*- coding: utf-8 -*-
"""PHYS-GEN-V1 physics-only soft-coverage training objective (owner decision 2026-10-02). TOY / training-only only.

Required before the generator: dead-zone designs get finite nonzero gradients; gradient ascent improves the soft
objective on dead-zone cases; tau_cov -> 0 converges to the hard optimum over the coverage nodes; positive designs keep
the hard objective's improvement direction; no node exceeds the mass cap; the hard exact scorer stays the only
acceptance authority (tested in test_physgen_scorer.py); mass-only coordinates get no manufactured gradient."""
import math

import numpy as np
import pytest
import torch

from acoustic_liner.physics import oracle as O2
from acoustic_liner.design import coordinates as PC
from acoustic_liner.models import objective as PO
from acoustic_liner.physics import scorer as PS
from acoustic_liner.optimization import seeded_cma as HO
from test_physgen_objective import toy_conditions, x_from_u, geom_from_x, _SCALE


@pytest.fixture(scope="module")
def batch():
    rng = np.random.default_rng(71)
    u = torch.as_tensor(HO.random_u(rng, 120))
    conds = toy_conditions(rng, 120)
    ct = PO.condition_tensors(conds)
    hard = PO.objective(PC.decode_t(u), ct, 0.0)
    Jh = hard["J"].detach().numpy()
    return u, conds, ct, hard, Jh == 0.0, Jh > 1e-6, hard["binding"].numpy()


def _grad_x(fn, u):
    x = x_from_u(u).requires_grad_(True)
    return torch.autograd.grad(fn(geom_from_x(x)).sum(), x)[0].numpy() * _SCALE


@pytest.mark.parametrize("tau_cov", [PO.TAU_COV_START, PO.TAU_COV_END])
def test_dead_zone_gradients_finite_and_nonzero(batch, tau_cov):
    u, conds, ct, hard, dead, pos, b = batch
    assert dead.sum() >= 20
    uu = u.clone().requires_grad_(True)
    G = torch.autograd.grad(PO.train_objective(PC.decode_t(uu), ct, tau_cov)["J"].sum(), uu)[0].numpy()
    assert np.isfinite(G).all()
    assert (np.linalg.norm(G[dead], axis=1) > 0.0).all()


def test_gradient_ascent_improves_dead_zone_cases(batch):
    u, conds, ct, hard, dead, pos, b = batch
    dconds = [c for c, k in zip(conds, dead) if k]
    ctd = PO.condition_tensors(dconds)
    ud = u[dead].clone().requires_grad_(True)
    J0 = PO.train_objective(PC.decode_t(ud), ctd, PO.TAU_COV_START)["J"].detach().clone()
    opt = torch.optim.Adam([ud], lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        (-PO.train_objective(PC.decode_t(ud), ctd, PO.TAU_COV_START)["J"].sum()).backward()
        opt.step()
    J1 = PO.train_objective(PC.decode_t(ud), ctd, PO.TAU_COV_START)["J"].detach()
    assert bool((J1 > J0).all())
    # reported, judged by the HARD exact scorer: how many dead-zone designs became beneficial
    exact = [PS.quality_eval_exact(PC.decode_np(ud[i].detach().numpy())[0], c)["best_val"] for i, c in enumerate(dconds)]
    print(f"\ndead-zone ascent: {sum(e > 1e-9 for e in exact)}/{len(exact)} became beneficial under the exact scorer")
    assert sum(e > 1e-9 for e in exact) >= 1


def test_tau_cov_to_zero_converges_to_hard_node_optimum(batch):
    """J_train -> max_j W(f_j) with error <= tau_cov * log N (log-mean-exp); the node maximum is within the node-grid
    quantization of the continuous hard optimum (never above it: every node is an admissible coverage)."""
    u, conds, ct, hard, dead, pos, b = batch
    g = PC.decode_t(u)
    node_max = PO.train_objective(g, ct, 1.0)["W_nodes"].detach().max(1).values.numpy()
    errs = []
    for tc in (0.05, 0.01, 1e-3, 1e-4):
        J = PO.train_objective(g, ct, tc)["J"].detach().numpy()
        err = float(np.max(np.abs(J - node_max)))
        assert err <= tc * math.log(PO.COV_GRID_N) + 1e-12
        errs.append(err)
    assert errs == sorted(errs, reverse=True)
    gap = hard["J"].detach().numpy() - node_max
    assert gap.min() >= -1e-12
    print(f"\nnode-grid quantization vs exact hard optimum: max {gap.max():.2e} dB")


@pytest.mark.parametrize("tau_cov", [PO.TAU_COV_START, PO.TAU_COV_END])
def test_positive_designs_keep_the_hard_improvement_direction(batch, tau_cov):
    u, conds, ct, hard, dead, pos, b = batch
    Gh = _grad_x(lambda g: PO.objective(g, ct, 0.0)["J"], u)
    Gs = _grad_x(lambda g: PO.train_objective(g, ct, tau_cov)["J"], u)
    cos = np.array([Gh[i] @ Gs[i] / (np.linalg.norm(Gh[i]) * np.linalg.norm(Gs[i])) for i in np.flatnonzero(pos)])
    print(f"\ntau_cov={tau_cov}: cos(soft, hard) on {pos.sum()} positive designs: min {cos.min():.3f} median {np.median(cos):.4f}")
    assert (cos > 0.0).all() and np.median(cos) >= 0.99


def test_coverage_nodes_never_exceed_the_mass_cap(batch):
    u, conds, ct, hard, dead, pos, b = batch
    out = PO.train_objective(PC.decode_t(u), ct, PO.TAU_COV_END)
    f = out["f_nodes"].detach().numpy()
    assert (f <= out["f_max"].detach().numpy()[:, None]).all() and (f >= 0.0).all()
    for i, c in enumerate(conds):
        d = PC.decode_np(u[i].numpy())[0]
        m_areal = O2._areal(d)
        assert max(O2._total_mass(m_areal, float(fj), c["A_wall"]) for fj in f[i]) <= c["m_total_target"] * (1 + 1e-12)


def test_no_manufactured_mass_only_gradients(batch):
    """Off the mass cap, facesheet / wall thickness enter only through mass: the hard objective gives them exactly zero
    gradient, and the default (cap-only) soft objective must not manufacture one through interior node positions."""
    u, conds, ct, hard, dead, pos, b = batch
    interior = pos & ~b
    Gh = _grad_x(lambda g: PO.objective(g, ct, 0.0)["J"], u)
    assert np.abs(Gh[interior][:, 1:3]).max() == 0.0
    Gs = _grad_x(lambda g: PO.train_objective(g, ct, PO.TAU_COV_END)["J"], u)
    rel = np.abs(Gs[interior][:, 1:3]).max(1) / np.abs(Gs[interior]).max(1)
    Ga = _grad_x(lambda g: PO.train_objective(g, ct, PO.TAU_COV_END, node_grad="all")["J"], u)
    rel_all = np.abs(Ga[interior][:, 1:3]).max(1) / np.abs(Ga[interior]).max(1)
    print(f"\nface/wall |grad|/max on interior positives: cap_only median {np.median(rel):.1e}; all-nodes median {np.median(rel_all):.1e}")
    assert np.median(rel) <= 1e-6 < np.median(rel_all)


def test_value_identical_across_node_gradient_modes(batch):
    u, conds, ct, *_ = batch
    g = PC.decode_t(u)
    a = PO.train_objective(g, ct, 0.02)["J"]; b2 = PO.train_objective(g, ct, 0.02, node_grad="all")["J"]
    assert torch.equal(a, b2)


def test_tau_schedule_endpoints():
    assert PO.tau_cov_schedule(0.0) == pytest.approx(0.05) and PO.tau_cov_schedule(1.0) == pytest.approx(0.01)
    with pytest.raises(PO.ObjectiveInputError):
        PO.train_objective(PC.decode_t(torch.zeros(1, PC.DIM, dtype=torch.float64)),
                           PO.condition_tensors(toy_conditions(np.random.default_rng(0), 1)), 0.0)
