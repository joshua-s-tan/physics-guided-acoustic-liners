# -*- coding: utf-8 -*-
"""PHYS-GEN-V1 differentiable objective: forward agreement with the canonical numpy oracle and autograd-vs-central-
finite-difference gradients (owner Phase-I item 3; prereg sec. 10 thresholds applied to the objective alone).
TOY / training-only designs and hand-made conditions only."""
import numpy as np
import pytest
import torch

from acoustic_liner.physics import oracle as O2
from acoustic_liner.design import schema as DS
from acoustic_liner.evaluation.historical_splits import selection_rule as SEL
from acoustic_liner.physics import quality as Q
from acoustic_liner.design import coordinates as PC
from acoustic_liner.models import objective as PO
from acoustic_liner.optimization import seeded_cma as HO

LO, HI = DS.bounds()
# physical geometry layout x = [D, face, wall, f0 x4, r x4, L x4, w x3]; normalization scale per coordinate
_SCALE = np.concatenate([HI[:3] - LO[:3], [HI[3] - LO[3]] * 4, [HI[4] - LO[4]] * 4, [HI[5] - LO[5]] * 4, [1.0] * 3])


def toy_conditions(rng, n, kappa=None):
    out = []
    for _ in range(n):
        spl = 112.0 + 18.0 * rng.random(8); A = float(rng.uniform(50.0, 200.0))
        out.append(dict(weights=list(SEL._weights_from_spl(spl)), spl=spl, dOASPL_target=0.5,
                        m_total_target=float(rng.uniform(0.3, 4.0) * A), A_wall=A,
                        kappa=float(rng.uniform(0.0, 0.3) if kappa is None else kappa)))
    return out


def geom_from_x(x):
    """x [B, 18] physical -> geom dict for PO.objective."""
    w = x[:, 15:18]
    frac = torch.softmax(torch.cat([w, torch.zeros_like(w[:, :1])], 1), 1)
    return dict(D=x[:, 0], face_t=x[:, 1], wall_t=x[:, 2], f0=x[:, 3:7], r=x[:, 7:11], L=x[:, 11:15], frac=frac)


def x_from_u(u):
    g = PC.decode_t(u)
    return torch.cat([g["D"][:, None], g["face_t"][:, None], g["wall_t"][:, None], g["f0"], g["r"], g["L"],
                      g["logratio"]], 1).detach()


def _case_set(seed, n):
    rng = np.random.default_rng(seed)
    u = torch.as_tensor(HO.random_u(rng, n))
    return u, toy_conditions(rng, n)


# ---- forward agreement ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def forward_batch():
    u, conds = _case_set(41, 64)
    g = PC.decode_t(u)
    return u, conds, g, PO.objective(g, PO.condition_tensors(conds), 0.0)


def test_batched_physics_matches_canonical(forward_batch):
    u, conds, g, out = forward_batch
    ab = PO.alpha_band(g["D"], g["f0"], g["r"], g["L"], g["frac"]).numpy()
    ma = PO.areal_mass(g["D"], g["face_t"], g["wall_t"], g["f0"], g["r"], g["L"], g["frac"]).numpy()
    for i, c in enumerate(conds):
        des, _ = PC.decode_np(u[i].numpy())
        assert np.max(np.abs(ab[i] - O2._a_band(des))) <= 1e-12
        assert abs(ma[i] - O2._areal(des)) <= 1e-12 * O2._areal(des)
        w = O2.robust_delta_oaspl(O2._a_band(des), c["spl"], float(out["f_star"][i]), c["A_wall"], c["kappa"],
                                  O2.DELTA_KAPPA_DEFAULT)["worst"]
        assert abs(w - float(out["worst_hard"][i])) <= 1e-12


def test_objective_vs_canonical_quality_eval(forward_batch):
    """Interior coverage optimum: equal to canonical within its own golden tolerance. Mass cap binding: the continuous
    solve is >= canonical (canonical only tries grid/golden points strictly inside the budget, never the cap)."""
    u, conds, g, out = forward_batch
    for i, c in enumerate(conds):
        des, _ = PC.decode_np(u[i].numpy())
        q = Q.quality_eval(des, c)
        dJ = float(out["J"][i]) - q["best_val"]
        if bool(out["binding"][i]):
            assert dJ >= -1e-9
        else:
            assert abs(dJ) <= 1e-5, (i, dJ)
    assert bool(out["binding"].any()) and bool((~out["binding"]).any())


def test_soft_min_converges_to_hard_min(forward_batch):
    u, conds, g, out0 = forward_batch
    ct = PO.condition_tensors(conds)
    gaps = [float((PO.objective(g, ct, tau)["J"] - out0["J"]).abs().max()) for tau in (0.1, 0.01, 0.001)]
    assert gaps[0] > gaps[1] > gaps[2] and gaps[2] < 1e-2


def test_objective_rejects_bad_inputs():
    with pytest.raises(PO.ObjectiveInputError):
        PO.condition_tensors([dict(spl=[120.0] * 8, A_wall=-1.0, m_total_target=10.0, kappa=0.1)])
    u, conds = _case_set(3, 2)
    with pytest.raises(PO.ObjectiveInputError):
        PO.objective(PC.decode_t(u), PO.condition_tensors(conds), tau=-0.1)


# ---- autograd vs central finite differences ------------------------------------------------------------------------
def gradient_audit(x, conds, tau, h=1e-6, n_dirs=8, step=1e-3, seed=0):
    """Per design: autograd dJ/d(normalized x) vs central FD (coverage RE-SOLVED at every perturbed point, so FD sees
    the full envelope derivative). Returns per-design cosine, sign agreement, Spearman rank correlation, false-
    improvement rate over random small steps, and whether every value was finite."""
    from scipy.stats import spearmanr
    ct_all = PO.condition_tensors(conds)
    xg = x.clone().requires_grad_(True)
    out = PO.objective(geom_from_x(xg), ct_all, tau)
    G = torch.autograd.grad(out["J"].sum(), xg)[0].numpy() * _SCALE          # d J / d(normalized x)
    rng = np.random.default_rng(seed)
    res = []
    for i in range(x.shape[0]):
        if float(out["f_star"][i]) == 0.0:                                     # no treatment: J == 0, gradient 0
            res.append(None); continue
        n = x.shape[1]
        E = np.eye(n) * (h * _SCALE)
        dirs = rng.standard_normal((n_dirs, n)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        P = np.concatenate([x[i].numpy() + E, x[i].numpy() - E, x[i].numpy() + step * dirs * _SCALE])
        ct = PO.condition_tensors([conds[i]] * len(P))
        Jp = PO.objective(geom_from_x(torch.as_tensor(P)), ct, tau)["J"].detach().numpy()
        fd = (Jp[:n] - Jp[n:2 * n]) / (2.0 * h)
        g = G[i]
        finite = bool(np.isfinite(g).all() and np.isfinite(fd).all())
        cos = float(g @ fd / (np.linalg.norm(g) * np.linalg.norm(fd) + 1e-300))
        big = np.abs(fd) > 1e-3 * np.abs(fd).max()
        sign = float(np.mean(np.sign(g[big]) == np.sign(fd[big])))
        # rank correlation over the NON-NEGLIGIBLE components (same set as sign agreement): e.g. face/wall thickness
        # enter only through mass, so off the mass cap autograd is exactly 0 while FD is +-1e-10 round-off noise, and
        # ranking that noise is meaningless. The all-component value is kept for the record.
        rank = float(spearmanr(g[big], fd[big]).correlation) if big.sum() >= 3 else 1.0
        rank_all = float(spearmanr(g, fd).correlation)
        actual = Jp[2 * n:] - float(out["J"][i])
        pred = step * (dirs @ g)
        sel = np.abs(pred) > 1e-9
        false_imp = float(np.mean((pred[sel] > 0) & (actual[sel] < 0))) if sel.any() else 0.0
        res.append(dict(cos=cos, sign=sign, rank=rank, rank_all=rank_all, false_imp=false_imp, finite=finite,
                        binding=bool(out["binding"][i])))
    return res


def _summary(res):
    r = [x for x in res if x is not None]
    return dict(n=len(r), n_zero=sum(x is None for x in res), n_binding=sum(x["binding"] for x in r),
                cos=float(np.median([x["cos"] for x in r])), sign=float(np.median([x["sign"] for x in r])),
                rank=float(np.median([x["rank"] for x in r])),
                rank_all=float(np.nanmedian([x["rank_all"] for x in r])),
                false_imp=float(np.mean([x["false_imp"] for x in r])), finite=all(x["finite"] for x in r))


@pytest.mark.parametrize("tau", [0.0, 0.05])
def test_gradient_matches_finite_difference(tau):
    u, conds = _case_set(51, 60)
    s = _summary(gradient_audit(x_from_u(u), conds, tau))
    assert s["n"] >= 25 and s["finite"], s
    assert s["cos"] >= 0.99 and s["sign"] >= 0.95 and s["rank"] >= 0.90 and s["false_imp"] <= 0.05, s


@pytest.mark.parametrize("regime", ["low_kappa", "high_kappa", "tight_mass", "near_boundary"])
def test_gradient_regimes(regime):
    rng = np.random.default_rng(61)
    u = torch.as_tensor(HO.random_u(rng, 40))
    if regime == "near_boundary":
        u = u * 4.0                                                            # pushes coordinates toward their limits
    conds = toy_conditions(rng, 40, kappa={"low_kappa": 0.0, "high_kappa": 0.3}.get(regime))
    if regime == "tight_mass":
        for c in conds:
            c["m_total_target"] = 0.4 * c["A_wall"]                             # cap binds for most designs
    s = _summary(gradient_audit(x_from_u(u), conds, 0.0, seed=7))
    assert s["n"] >= 10 and s["finite"], s
    assert s["cos"] >= 0.99 and s["sign"] >= 0.95 and s["rank"] >= 0.90 and s["false_imp"] <= 0.05, s
